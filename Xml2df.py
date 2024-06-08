#%%
"""
VOC converter function Documentation
---------------------
This function processes XML files in the specified directory to extract image information,
annotation data, and category details, and then aggregates this data into a pandas DataFrame.

Steps Involved:
1. Initialize Containers and Variables:
   - img_data: List to store image information.
   - ann_data: List to store annotation details.
   - cat_data: List to store category information.
   - img_id, cat_id, annotation_id: Counters for image IDs, category IDs, and annotation IDs, respectively.
   - categories: Dictionary to map category names to category IDs.

2. Read XML Files:
   - Identify and list all XML files in the provided directory.

3. Process Each XML File:
   - Parse the XML file to extract the root element.
   - Extract image information (img_id, img_width, img_height, img_image_name, img_file_name).
   - For each object (annotation) in the XML:
     - Extract category name and assign a unique category ID if not already present.
     - Extract bounding box coordinates and calculate the area.
     - Store annotation details (ann_id, ann_segmentation, ann_bbox, ann_area, ann_image_id,
       ann_category_id, ann_category_name, ann_iscrowd).

4. Create DataFrames:
   - Convert lists (img_data, ann_data, cat_data) to pandas DataFrames (images_df, annotations_df, categories_df).

5. Data Type Adjustments:
   - Ensure certain columns have the correct data types (e.g., integer for dimensions, string for category IDs).

6. Handle Missing Values:
   - Fill any missing values in the DataFrames with appropriate defaults.

7. Concatenate DataFrames:
   - Optionally concatenate the DataFrames horizontally if needed.

8. Reindex Columns:
   - Reindex the DataFrame to match a predefined schema if necessary.

Example Usage:
--------------
directory_path = "path/to/voc/xml/files"
annotations_df = voc_to_dataframe(directory_path)
print(annotations_df.head())
"""


#%%
import pandas as pd
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from Df2coco import dataframe_to_bina_coco

#%%
schema = [
    "info", "img_id", "img_width", "img_height", "img_image_name", "img_file_name",
    "licenses", "cat_id", "cat_name", "cat_supercategory", "errors", "ann_id",
    "ann_segmentation", "ann_image_id", "ann_category_id", "ann_area", "ann_bbox",
    "ann_iscrowd", "labels", "classifications", "augmentation_settings",
    "tile_settings", "False_positive"
]


def voc_to_dataframe(directory_path: str) -> pd.DataFrame:
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
df = voc_to_dataframe(directory_path)
# df.to_csv('mine.csv', index=False)
print(df.head())
