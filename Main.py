#%%
import shutil

from AnnotationExplorer import AnnotationExplorer
from AnnotationConverter import AnnotationConverter
import os


def explore_and_convert(zip_path):
    explorer = AnnotationExplorer(zip_path)
    result = explorer.explore_and_organize()
    converter = AnnotationConverter()
    output_path = os.path.join(explorer.cocos_dir, 'train_coco.json')

    if explorer.identified_format == 'Pascal VOC':
        xml_path = os.path.join(explorer.annotations_dir, 'xml')
        voc_df = converter.voc_to_dataframe(xml_path)
        converter.dataframe_to_bina_coco(voc_df, output_path=output_path)
        
    elif explorer.identified_format == 'COCO':
        coco_path = os.path.join(explorer.annotations_dir, 'coco')
        coco_df = converter.coco_to_dataframe(coco_path)
        # coco_df.to_csv('mine.csv', index=False)
        converter.dataframe_to_bina_coco(coco_df, output_path=output_path)
        
    #TODO Yolo converter is not written yet
    elif explorer.identified_format == 'YOLO':
        print("TODO")

    else:
        print("Failed")

    delete_folder(explorer.annotations_dir)


def delete_folder(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

#%%
explore_and_convert(zip_path = "Example Datasets/empty.zip")