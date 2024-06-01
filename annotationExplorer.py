#%%
"""
AnnotationExplorer Class Documentation
---------------------------------------

This class is designed to explore a zip file containing annotation files for object detection tasks.
It identifies the format of the annotation files within the zip archive, supporting three common formats:
Pascal VOC XML, YOLO, and COCO JSON. The class organizes the files into separate folders based on their type and
 handles nested folders within the zip file structure, ensuring that annotation files located at any level of nesting are properly detected.

Class Usage:
    explorer = AnnotationExplorer(zip_path)
    results = explorer.explore_and_organize()
    print(f"Annotation format detected: {results['annotation_format']}")
    print(f"Number of images: {results['num_images']}")
    print(f"Number of annotation files: {results['num_annotations_files']}")

Class Methods:
    1. __init__(self, zip_path: str)
        - Initializes the AnnotationExplorer object with the path to the zip file containing annotation data.
        - Ensures that the organized directory has a unique name to avoid conflicts.
        - Initializes attributes to store the identified format, and the number of images and annotation files.

    2. _ensure_unique_organized_dir(self)
        - Ensures that the organized directory has a unique name by appending a counter if necessary.

    3. extract_zip(self)
        - Extracts the contents of the zip file to a temporary directory for further processing.

    4. organize_files_and_identify_format(self)
        - Identifies the annotation format and organizes files into separate folders.
        - Creates separate folders for annotation files based on their format:
          - 'annotations/xml' for Pascal VOC
          - 'annotations/coco' for COCO
          - 'annotations/yolo' for YOLO
          - 'images' for all image files
        - Updates the identified format and counts the number of images and annotation files.
        - Returns the identified annotation format.

    5. _move_file(self, file_path: str, destination_folder: str)
        - Moves a file to the specified destination folder, creating the folder if it does not exist.

    6. _is_pascal_voc(self, file_path: str) -> bool
        - Checks if the given file is a Pascal VOC XML annotation file.
        - Returns True if the file is in Pascal VOC XML format, False otherwise.

    7. _is_yolo(self, file_path: str) -> bool
        - Checks if the given file is a YOLO annotation file.
        - Returns True if the file is in YOLO format, False otherwise.

    8. _is_coco(self, file_path: str) -> bool
        - Checks if the given file is a COCO JSON annotation file.
        - Returns True if the file is in COCO JSON format, False otherwise.

    9. cleanup(self)
        - Cleans up the extracted files while keeping organized images and annotations folders.

    10. explore_and_organize(self)
        - Main method to extract, organize, identify format, and cleanup.
        - Returns a dictionary containing the identified annotation format, the number of images, and the number of annotation files.

Note:
    - The AnnotationExplorer class is designed to be instantiated with the path to a zip file containing annotation data.
    - It provides methods to explore the zip file, identify the annotation format, and clean up temporary files.
    - The class is robust and handles nested folders within the zip file structure.
"""

#%%
import zipfile
import os
 
import shutil
import json
import xml.etree.ElementTree as ET


class AnnotationExplorer:
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.extract_dir = "extracted_files"
        self.organized_dir = "organized_files"
        self._ensure_unique_organized_dir()
        self.identified_format = None
        self.num_images = 0
        self.num_annotations = 0

    def _ensure_unique_organized_dir(self):
        """Ensures that the organized directory has a unique name."""
        base_dir = self.organized_dir
        counter = 1
        while os.path.exists(self.organized_dir):
            self.organized_dir = f"{base_dir}_{counter}"
            counter += 1

    def extract_zip(self):
        """Extracts the zip file to a temporary directory."""
        os.makedirs(self.extract_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)

    def organize_files_and_identify_format(self):
        """
        Identifies the annotation format and organizes files into separate folders.

        Creates separate folders for annotation files based on their format:
        - 'annotations/xml' for Pascal VOC
        - 'annotations/coco' for COCO
        - 'annotations/yolo' for YOLO
        - 'images' for all image files

        Returns:
            str: The identified annotation format ('Pascal VOC', 'COCO', 'YOLO') or None if no format is recognized.
        """

        annotations_dir = os.path.join(self.organized_dir, 'annotations')
        images_dir = os.path.join(self.organized_dir, 'images')

        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        for root, _, files in os.walk(self.extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.move(file_path, os.path.join(images_dir, file))
                    self.num_images += 1
                elif file.endswith('.xml'):
                    if self._is_pascal_voc(file_path):
                        self._move_file(file_path, os.path.join(annotations_dir, 'xml'))
                        self.identified_format = 'Pascal VOC'
                        self.num_annotations += 1
                elif file.endswith('.txt'):
                    if self._is_yolo(file_path):
                        self._move_file(file_path, os.path.join(annotations_dir, 'yolo'))
                        self.identified_format = 'YOLO'
                        self.num_annotations += 1
                elif file.endswith('.json'):
                    if self._is_coco(file_path):
                        self._move_file(file_path, os.path.join(annotations_dir, 'coco'))
                        self.identified_format = 'COCO'
                        self.num_annotations += 1

    def _move_file(self, file_path: str, destination_folder: str):
        """Moves a file to the specified destination folder."""
        os.makedirs(destination_folder, exist_ok=True)
        shutil.move(file_path, destination_folder)

    def _is_pascal_voc(self, file_path: str) -> bool:
        """Checks if the file is a Pascal VOC XML annotation."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            if root.tag == 'annotation' and root.find('object') is not None:
                return True
        except ET.ParseError:
            pass
        return False

    def _is_yolo(self, file_path: str) -> bool:
        """Checks if the file is a YOLO annotation."""
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5 and all(part.replace('.', '', 1).isdigit() for part in parts):
                        return True
        except Exception:
            pass
        return False

    def _is_coco(self, file_path: str) -> bool:
        """Checks if the file is a COCO JSON annotation."""
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                if 'annotations' in data and 'images' in data and 'categories' in data:
                    return True
        except json.JSONDecodeError:
            pass
        return False

    def cleanup(self):
        """Cleans up the extracted files while keeping organized images and annotations folders."""
        for root, dirs, files in os.walk(self.extract_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
        if os.path.exists(self.extract_dir):
            os.rmdir(self.extract_dir)

    def explore_and_organize(self):
        """Main method to extract, organize, identify format, and cleanup."""
        self.extract_zip()
        self.organize_files_and_identify_format()
        self.cleanup()
        return {
            'annotation_format': self.identified_format,
            'num_images': self.num_images,
            'num_annotations_files': self.num_annotations
        }


#%% Example usage:
explorer = AnnotationExplorer("Dental_1.v4i.coco.zip")
results = explorer.explore_and_organize()
print(results)
