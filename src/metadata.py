import exifread
import os
from datetime import datetime

def extract_exif_data(image_path):
    """
    Extracts EXIF data from an image file.
    Args:
        image_path (str): The path to the image file.
    Returns:
        dict: A dictionary containing extracted EXIF tags and their values,
              or an empty dictionary if no EXIF data is found or an error occurs.
    """
    metadata = {}
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            for tag_name, tag_value in tags.items():
                # Convert tag value to string for easier storage
                metadata[tag_name] = str(tag_value)
    except Exception as e:
        print(f"Error extracting EXIF data from {image_path}: {e}")
    return metadata

def get_file_properties(image_path):
    """
    Extracts basic file properties from an image.
    Args:
        image_path (str): The path to the image file.
    Returns:
        dict: A dictionary containing file properties.
    """
    file_properties = {}
    try:
        stat = os.stat(image_path)
        file_properties['file_size'] = stat.st_size
        file_properties['created_time'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        file_properties['modified_time'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        file_properties['file_format'] = os.path.splitext(image_path)[1].lower()
    except Exception as e:
        print(f"Error getting file properties for {image_path}: {e}")
    return file_properties

def get_image_metadata(image_path):
    """
    Combines EXIF data and basic file properties.
    Args:
        image_path (str): The path to the image file.
    Returns:
        dict: A dictionary containing all extracted metadata.
    """
    metadata = get_file_properties(image_path)
    exif_data = extract_exif_data(image_path)
    metadata.update(exif_data)
    return metadata
