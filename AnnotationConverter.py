#%%
import pandas as pd
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import json
import numpy as np
import logging
from typing import List, Optional


#%%
class AnnotationConverter:
    def __init__(self, schema: Optional[List[str]] = None):
        self.schema = schema or [
            "info", "img_id", "img_width", "img_height", "img_image_name", "img_file_name",
            "licenses", "cat_id", "cat_name", "cat_supercategory", "errors", "ann_id",
            "ann_segmentation", "ann_image_id", "ann_category_id", "ann_area", "ann_bbox",
            "ann_iscrowd", "labels", "classifications", "augmentation_settings",
            "tile_settings", "False_positive"
        ]
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def voc_to_dataframe(self, directory_path: str) -> pd.DataFrame:
        """
        Converts Pascal VOC annotations from XML files in a directory to a pandas DataFrame.

        Args:
            directory_path (str): The path to the directory with VOC XML annotation files.

        Returns:
            pd.DataFrame: DataFrame containing the aggregated annotations.
        """
        self.logger.info(f"Converting VOC annotations in {directory_path} to DataFrame")

        img_data = []
        ann_data = []
        cat_data = []

        img_id = 0
        cat_id = 0
        categories = {}
        annotation_id = 0

        try:
            xml_files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]

            for xml_file in tqdm(xml_files, desc="Processing XML files"):
                tree = ET.parse(os.path.join(directory_path, xml_file))
                root = tree.getroot()

                img_info = self._parse_voc_image(root, img_id)
                img_data.append(img_info)

                for obj in root.findall('object'):
                    cat_name = obj.find('name').text
                    if cat_name not in categories:
                        categories[cat_name] = cat_id
                        cat_data.append({
                            'cat_id': cat_id,
                            'cat_name': cat_name
                        })
                        cat_id += 1

                    ann_info = self._parse_voc_annotation(obj, img_id, annotation_id, categories)
                    ann_data.append(ann_info)
                    annotation_id += 1

                img_id += 1

                img_df = pd.DataFrame(img_data)
                ann_df = pd.DataFrame(ann_data)
                cat_df = pd.DataFrame(cat_data)

            return self._prepare_dataframe(img_df, ann_df, cat_df)
        except Exception as e:
            self.logger.error(f"Error converting VOC to DataFrame: {e}")
            raise

    @staticmethod
    def _parse_voc_image(root, img_id):
        return {
            'img_id': img_id,
            'img_width': int(root.find('size/width').text),
            'img_height': int(root.find('size/height').text),
            'img_image_name': root.find('filename').text,
            'img_file_name': root.find('path').text if root.find('path') is not None else ""
        }

    @staticmethod
    def _parse_voc_annotation(obj, img_id, annotation_id, categories):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        return {
            'ann_id': annotation_id,
            'ann_segmentation': [],
            'ann_bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'ann_area': (xmax - xmin) * (ymax - ymin),
            'ann_image_id': img_id,
            'ann_category_id': categories[obj.find('name').text],
            'ann_category_name': obj.find('name').text,
            'ann_iscrowd': 0
        }

    def coco_to_dataframe(self, folder: str, encoding: str = "utf-8") -> pd.DataFrame:
        self.logger.info(f"Converting COCO annotations in folder {folder} to DataFrame")

        all_images = []
        all_categories = []
        all_annotations = []

        try:
            json_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')]
            for json_file in json_files:
                with open(json_file, encoding=encoding) as cocojson:
                    annotations_json = json.load(cocojson)

                images = pd.json_normalize(annotations_json["images"]).add_prefix("img_")
                categories = pd.json_normalize(annotations_json["categories"]).add_prefix("cat_")
                annotations = pd.json_normalize(annotations_json["annotations"]).add_prefix("ann_")

                all_images.append(images)
                all_categories.append(categories)
                all_annotations.append(annotations)

            # Concatenate all DataFrames
            images_df = pd.concat(all_images, ignore_index=True)
            categories_df = pd.concat(all_categories, ignore_index=True)
            annotations_df = pd.concat(all_annotations, ignore_index=True)

            return self._prepare_dataframe(images_df, annotations_df, categories_df)
        except Exception as e:
            self.logger.error(f"Error converting COCO to DataFrame: {e}")
            raise

    def _prepare_dataframe(self, images_df, annotations_df, categories_df):
        for col in ["img_width", "img_height"]:
            if col in images_df.columns:
                images_df[col] = images_df[col].astype("int64", errors='ignore').replace("", 0).fillna(0)

        categories_df["cat_id"] = pd.to_numeric(categories_df["cat_id"]).fillna(0).astype(int)

        # Handling error
        annotations_df["ann_category_id"] = annotations_df["ann_category_id"].astype(str)

        annotations_df.fillna("", inplace=True)
        images_df.fillna("", inplace=True)
        categories_df.index.name = "id"

        df = pd.concat([images_df, annotations_df, categories_df], axis=1)
        df = df.reindex(columns=self.schema, fill_value="")

        return df

    @staticmethod
    def dataframe_to_bina_coco(dataframe, output_path=None, cat_id_index=None):
        """
        Writes COCO annotation files to disk (in JSON format) and returns the path to files.
        """
        df = dataframe.copy(deep=True)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        df["ann_iscrowd"] = df["ann_iscrowd"].fillna(0)

        df_outputI = []
        df_outputA = []
        df_outputC = []

        pbar = tqdm(desc="Exporting to COCO file...", total=df.shape[0])
        for i in range(0, df.shape[0]):
            if not pd.isna(df["img_id"][i]):
                images = [
                    {
                        "id": int(df["img_id"][i]),
                        "width": df["img_width"][i],
                        "height": df["img_height"][i],
                        "image_name": df["img_file_name"][i],
                        "file_name": df["img_file_name"][i],
                    }
                ]
                df_outputI.append(pd.DataFrame([images]))

            if not pd.isna(df["cat_id"][i]):
                categories = [
                    {
                        "id": int(df["cat_id"][i]),
                        "name": df["cat_name"][i],
                        "supercategory": df["cat_supercategory"][i],
                    }
                ]
                df_outputC.append(pd.DataFrame([categories]))

            if not pd.isna(df["ann_id"][i]):
                annotations = [
                    {
                        "id": int(df["ann_id"][i]),
                        "segmentation": df["ann_segmentation"][i],
                        "image_id": int(df["ann_image_id"][i]),
                        "category_id": int(df["ann_category_id"][i]),
                        "area": df["ann_area"][i],
                        "bbox": df["ann_bbox"][i],
                        "iscrowd": df["ann_iscrowd"][i],
                    }
                ]
                df_outputA.append(pd.DataFrame([annotations]))

            pbar.update()

        mergedI = pd.concat(df_outputI, ignore_index=True)
        mergedA = pd.concat(df_outputA, ignore_index=True)
        mergedC = pd.concat(df_outputC, ignore_index=True)

        resultI = mergedI[0].to_json(orient="split", default_handler=str)
        resultA = mergedA[0].to_json(orient="split", default_handler=str)
        resultC = mergedC[0].to_json(orient="split", default_handler=str)

        parsedI = json.loads(resultI)
        parsedA = json.loads(resultA)
        parsedC = json.loads(resultC)

        parsedI.update({
            "info": {},
            "images": parsedI.pop("data"),
            "categories": parsedC.pop("data"),
            "licenses": [],
            "errors": [],
            "annotations": parsedA.pop("data"),
            "labels": [],
            "classifications": [],
            "augmentation_settings": {},
            "tile_settings": {},
            "False_positive": {}
        })
        del parsedI["index"]
        del parsedI["name"]

        json_output = parsedI

        with open(output_path, "w") as outfile:
            json.dump(obj=json_output, fp=outfile, indent=4)
        return [str(output_path)]


#%%
# Example usage
directory_path = 'converted_Dental_1.v4i.coco.zip/annotations/coco/_annotations.coco.json'
annotation_converter = AnnotationConverter()
voc_df = annotation_converter.coco_to_dataframe(directory_path)
print(voc_df.head())
voc_df.to_csv('mine.csv', index=False)
annotation_converter.dataframe_to_bina_coco(voc_df,
                                            output_path="converted_Dental_1.v4i.coco.zip/cocos/train_coco.json", )
