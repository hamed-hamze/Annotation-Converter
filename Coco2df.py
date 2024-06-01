#%% Coco Importer to pandas dataframe
import json
import pandas as pd

schema = [
    "info", "img_id", "img_width", "img_height", "img_image_name", "img_file_name",
    "licenses", "cat_id", "cat_name", "cat_supercategory", "errors", "ann_id",
    "ann_segmentation", "ann_image_id", "ann_category_id", "ann_area", "ann_bbox",
    "ann_iscrowd", "labels", "classifications", "augmentation_settings",
    "tile_settings", "False_positive"
]

def coco_to_dataframe(
    path: str,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Converts COCO annotations from a JSON file to a pandas DataFrame.

    Args:
        path (str): The path to the JSON file with the COCO annotations.
        encoding (str): Encoding of the annotations file. Default is 'utf-8'.

    Returns:
        pd.DataFrame: DataFrame containing the annotations.
    """
    with open(path, encoding=encoding) as cocojson:
        annotations_json = json.load(cocojson)

    info = pd.json_normalize(annotations_json.get("info", {}))
    images = pd.json_normalize(annotations_json["images"]).add_prefix("img_")
    categories = pd.json_normalize(annotations_json["categories"]).add_prefix("cat_")
    license = pd.json_normalize(annotations_json.get("licenses", []))
    errors = pd.json_normalize(annotations_json.get("errors", []))
    annotations = pd.json_normalize(annotations_json["annotations"]).add_prefix("ann_")

    # Ensuring the columns exist before casting types
    for col in ["img_width", "img_height"]:
        if col in images.columns:
            images[col] = images[col].astype("int64", errors='ignore')

    categories["cat_id"] = categories["cat_id"].astype(str)
    annotations["ann_category_id"] = annotations["ann_category_id"].astype(str)

    df_list = [info, images, categories, license, errors, annotations]
    df = pd.concat(df_list, axis=1, join="outer")

    # Ensure schema compliance
    df = df.reindex(columns=schema, fill_value="")

    # Handle missing values and type conversions
    df.fillna("", inplace=True)
    df["img_width"] = df["img_width"].replace("", 0).astype(int)
    df["img_height"] = df["img_height"].replace("", 0).astype(int)
    df["cat_id"] = df["cat_id"].astype(str)
    df.index.name = "id"
    df["annotated"] = 1

    return df


#%%Example usage
data = coco_to_dataframe(path="dataset.json", path_to_images="train", name="tt")

data.to_csv('mine.csv', index=False)
#%%
from Exporter import export_to_bina_coco

export_to_bina_coco(data, output_path='train_coco.json')
