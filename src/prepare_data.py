import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from config import DATA_DIR

def organize_dataset_from_xml():
    """
    Reads XML annotations in Pascal VOC format to determine the class of each image,
    then organizes images into a class-based directory structure suitable for PyTorch's ImageFolder.
    
    Structure required:
    data/raw/IMAGES/*.jpg
    data/raw/ANNOTATIONS/*.xml
    """
    
    # Define paths
    raw_dir = os.path.join(DATA_DIR, "raw")
    images_dir = os.path.join(raw_dir, "IMAGES")
    annotations_dir = os.path.join(raw_dir, "ANNOTATIONS")
    processed_dir = os.path.join(DATA_DIR, "processed")

    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"Error: Could not find IMAGES or ANNOTATIONS directories in {raw_dir}")
        return

    print(f"Starting dataset organization from {annotations_dir}...")

    # Counters for statistics
    processed_count = 0
    errors = 0

    # Iterate over all XML files
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"):
            continue

        try:
            # 1. Parse XML to find the class label
            xml_path = os.path.join(annotations_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract filename from XML (e.g., "crazing_1.jpg")
            filename = root.find('filename').text
            
            # Extract the class name from the first object tag
            # We assume the image represents the class of the first object found
            obj = root.find('object')
            if obj is None:
                print(f"Warning: No object found in {xml_file}. Skipping.")
                continue
                
            class_name = obj.find('name').text.lower().strip()
            
            # 2. Define source and destination
            src_image_path = os.path.join(images_dir, filename)
            
            # Handle potential case mismatch in extensions (jpg vs JPG) if necessary
            if not os.path.exists(src_image_path):
                # Try to find the file if the XML filename extension is wrong
                base_name = os.path.splitext(filename)[0]
                possible_exts = ['.jpg', '.JPG', '.bmp', '.BMP', '.png']
                for ext in possible_exts:
                    if os.path.exists(os.path.join(images_dir, base_name + ext)):
                        src_image_path = os.path.join(images_dir, base_name + ext)
                        filename = base_name + ext
                        break

            if not os.path.exists(src_image_path):
                print(f"Error: Image {filename} not found in {images_dir}")
                errors += 1
                continue

            # Create class directory if it doesn't exist (e.g., data/processed/crazing)
            class_dir = os.path.join(processed_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # 3. Copy the image
            dst_image_path = os.path.join(class_dir, filename)
            shutil.copy2(src_image_path, dst_image_path)
            
            processed_count += 1

        except Exception as e:
            print(f"Failed to process {xml_file}: {e}")
            errors += 1

    print(f"\nProcessing complete.")
    print(f"Successfully organized: {processed_count} images.")
    print(f"Errors encountered: {errors}")
    print(f"Data is ready at: {processed_dir}")

if __name__ == "__main__":
    organize_dataset_from_xml()