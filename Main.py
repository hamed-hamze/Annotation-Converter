#%%
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
        converter.dataframe_to_bina_coco(coco_df, output_path=output_path)
    #TODO Yolo converter is not written yet
    elif explorer.identified_format == 'YOLO':
        print("TODO")
    else:
        print("Failed")

    #TODO delete the annotation folder
    #TODO zip the final folder



#%%
explore_and_convert(zip_path = "Example Datasets/pi for reza.v1i.coco.zip")