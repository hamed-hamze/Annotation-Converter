#%%
from pylabel import importer
from Df2coco import export_to_bina_coco
import pandas as pd
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

#%%

path_to_annotations = "Example Datasets/valid"

# #Identify the path to get from the annotations to the images
# path_to_images = "Example Datasets/valid"

dataset = importer.ImportVOC(path=path_to_annotations, name="BCCD_Dataset")
dataset.df.head(5)
# dataset.df.to_csv('mine.csv', index=False)

#%%
schema = [
    "info", "img_id", "img_width", "img_height", "img_image_name", "img_file_name",
    "licenses", "cat_id", "cat_name", "cat_supercategory", "errors", "ann_id",
    "ann_segmentation", "ann_image_id", "ann_category_id", "ann_area", "ann_bbox",
    "ann_iscrowd", "labels", "classifications", "augmentation_settings",
    "tile_settings", "False_positive"
]


def voc_directory_to_dataframe(directory_path: str) -> pd.DataFrame:
    """
    Converts Pascal VOC annotations from XML files in a directory to a pandas DataFrame.

    Args:
        directory_path (str): The path to the directory with VOC XML annotation files.

    Returns:
        pd.DataFrame: DataFrame containing the aggregated annotations.
    """
    img_data = []
    ann_data = []
    cat_data = []

    img_id = 0
    cat_id = 0
    categories = {}
    annotation_id = 0

    xml_files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        tree = ET.parse(os.path.join(directory_path, xml_file))
        root = tree.getroot()

        img_info = {
            'img_id': img_id,
            'img_width': int(root.find('size/width').text),
            'img_height': int(root.find('size/height').text),
            'img_image_name': root.find('filename').text,
            'img_file_name': root.find('path').text if root.find('path') is not None else ""
        }
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

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            annotation_info = {
                'ann_id': annotation_id,
                'ann_segmentation': [],
                'ann_bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'ann_area': (xmax - xmin) * (ymax - ymin),
                'ann_image_id': img_id,
                'ann_category_id': categories[cat_name],
                'ann_category_name': cat_name,
                'ann_iscrowd': 0
            }
            ann_data.append(annotation_info)
            annotation_id += 1

        img_id += 1

    images_df = pd.DataFrame(img_data)
    annotations_df = pd.DataFrame(ann_data)
    categories_df = pd.DataFrame(cat_data)

    for col in ["img_width", "img_height"]:
        if col in images_df.columns:
            images_df[col] = images_df[col].astype("int64", errors='ignore')

    categories_df["cat_id"] = categories_df["cat_id"].astype(str)
    annotations_df["ann_category_id"] = annotations_df["ann_category_id"].astype(str)

    annotations_df.fillna("", inplace=True)
    images_df.fillna("", inplace=True)
    images_df["img_width"] = images_df["img_width"].replace("", 0).astype(int)
    images_df["img_height"] = images_df["img_height"].replace("", 0).astype(int)
    categories_df["cat_id"] = categories_df["cat_id"].astype(str)
    categories_df.index.name = "id"
    # df["annotated"] = 1

    # Concatenate the DataFrames horizontally if necessary
    df = pd.concat([images_df, annotations_df, categories_df], axis=1)

    df = df.reindex(columns=schema, fill_value="")

    return df


#%%
directory_path = 'organized_files_1/annotations/xml'
df = voc_directory_to_dataframe(directory_path)
df.to_csv('mine.csv', index=False)
print(df.head())

#%%
export_to_bina_coco(df, output_path='organized_files_1/annotations/cocoss.json')
