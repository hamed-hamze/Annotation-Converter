import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from Coco2df import coco_to_dataframe

#%%
# df = coco_to_dataframe(path="train/_annotations.coco.json", path_to_images="train", name="")


#%%
def dataframe_to_bina_coco(dataframe, output_path=None, cat_id_index=None):
    """
    Writes COCO annotation files to disk (in JSON format) and returns the path to files.

    Args:
        output_path (str):
            This is where the annotation files will be written. If not-specified then the path will be derived from the path_to_annotations and
            name properties of the dataset object.
        cat_id_index (int):
            Reindex the cat_id values so that they start from an int (usually 0 or 1) and
            then increment the cat_ids to index + number of categories continuously.
            It's useful if the cat_ids are not continuous in the original dataset.
            Some models like Yolo require starting from 0 and others like Detectron require starting from 1.

    Returns:
        A list with 1 or more paths (strings) to annotations files.

    Example:

    """

    # Copy the dataframe in the dataset so the original dataset doesn't change when you apply the export tranformations
    df = dataframe.copy(deep=True)
    # Replace empty string values with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    pd.to_numeric(df["cat_id"])

    df["ann_iscrowd"] = df["ann_iscrowd"].fillna(0)

    df_outputI = []
    df_outputA = []
    df_outputC = []

    pbar = tqdm(desc="Exporting to COCO file...", total=df.shape[0])
    for i in range(0, df.shape[0]):

        if not pd.isna(df["img_id"][i]):
            images = [
                {
                    "id": df["img_id"][i],
                    "width": df["img_width"][i],
                    "height": df["img_height"][i],
                    "image_name": df["img_file_name"][i],
                    "file_name": df["img_file_name"][i],
                    # "depth": df["img_depth"][i],
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
                    "id": df["ann_id"][i],
                    "segmentation": df["ann_segmentation"][i],
                    "image_id": df["ann_image_id"][i],
                    "category_id": int(df["ann_category_id"][i]),
                    "area": df["ann_area"][i],
                    "bbox": df["ann_bbox"][i],
                    "iscrowd": df["ann_iscrowd"][i],
                    # "pose": df["ann_pose"][i],
                    # "truncated": df["ann_truncated"][i],
                    # "difficult": df["ann_difficult"][i],
                }
            ]
            df_outputA.append(pd.DataFrame([annotations]))

        # Include Keypoints, if available
        # if "ann_keypoints" in df.keys() and (not np.isnan(df["ann_keypoints"][i]).all()):
        #     keypoints = df["ann_keypoints"][i]
        #     if isinstance(keypoints, list):
        #         n_keypoints = int(len(keypoints) / 3)  # 3 numbers per keypoint: x,y,visibility
        #     elif isinstance(keypoints, np.ndarray):
        #         n_keypoints = int(keypoints.size / 3)  # 3 numbers per keypoint: x,y,visibility
        #     else:
        #         raise TypeError('The keypoints array is expected to be either a list or a numpy array')
        #     annotations[0]["num_keypoints"] = n_keypoints
        #     annotations[0]["keypoints"] = keypoints
        # else:
        #     pass

        # if list_c: # Check if the list is empty
        #     if categories[0]["id"] in list_c:
        #         pass
        #     else:
        #         categories[0]["id"] = int(categories[0]["id"])
        #         df_outputC.append(pd.DataFrame([categories]))
        # elif not pd.isna(categories[0]["id"]):
        #     categories[0]["id"] = int(categories[0]["id"])
        #     df_outputC.append(pd.DataFrame([categories]))
        # else:
        #     pass
        # list_c.append(categories[0]["id"])
        #
        # if list_i:
        #     if images[0]["id"] in list_i or np.isnan(images[0]["id"]):
        #         pass
        #     else:
        #         df_outputI.append(pd.DataFrame([images]))
        # elif ~np.isnan(images[0]["id"]):
        #     df_outputI.append(pd.DataFrame([images]))
        # else:
        #     pass
        # list_i.append(images[0]["id"])
        #
        # # If the class id is blank, then there is no annotation to add
        # if not pd.isna(categories[0]["id"]):
        #     df_outputA.append(pd.DataFrame([annotations]))

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

    # Update keys
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
# export_to_bina_coco(df, output_path='cocoss.json')
