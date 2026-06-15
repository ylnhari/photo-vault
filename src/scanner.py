import os
import json
import exifread
from pathlib import Path
from tqdm import tqdm

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp'}

def get_metadata(file_path):
    """Extract EXIF data if possible."""
    metadata = {}
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if tags:
                if 'EXIF DateTimeOriginal' in tags:
                    metadata['date'] = str(tags['EXIF DateTimeOriginal'])
                if 'Image Make' in tags:
                    metadata['camera_make'] = str(tags['Image Make'])
    except Exception:
        pass
    return metadata

def load_existing_data(output_file):
    """Load existing catalog and return (images_dict, seen_paths_set)."""
    images = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                images = data.get("images", {})
            elif isinstance(data, list):
                # Old list format — migrate to dict keyed by path
                images = {img["path"]: img for img in data if "path" in img}
            print(f"Resuming from existing database: {len(images)} images already scanned.")
        except Exception as e:
            print(f"Warning: Could not load existing database ({e}). Starting fresh.")
    return images, set(images.keys())

def save_data(images_dict, output_file):
    """Atomically save catalog as {\"images\": {path: data}} to JSON file."""
    temp_file = output_file + ".tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump({"images": images_dict}, f, indent=2)
        os.replace(temp_file, output_file)
    except Exception as e:
        print(f"Error saving database: {e}")

def scan_directory(root_dir, output_file, checkpoint_interval=100):
    """Recursively scan directory with checkpointing and resume support."""
    images, seen_paths = load_existing_data(output_file)
    root = Path(root_dir)

    if not root.exists():
        print(f"Error: Root directory {root_dir} does not exist.")
        return

    print(f"Starting scan in: {root_dir}")
    try:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                str_path = str(path.absolute())
                if str_path not in seen_paths:
                    try:
                        stats = path.stat()
                        img_data = {
                            'path': str_path,
                            'filename': path.name,
                            'extension': path.suffix,
                            'size_bytes': stats.st_size,
                            'created_at': stats.st_ctime,
                            'metadata': get_metadata(path)
                        }
                        images[str_path] = img_data
                        seen_paths.add(str_path)

                        if len(images) % checkpoint_interval == 0:
                            print(f"Checkpoint: {len(images)} images scanned...")
                            save_data(images, output_file)
                    except Exception:
                        continue

        save_data(images, output_file)
        print(f"Scan complete. Total images in database: {len(images)}")

    except KeyboardInterrupt:
        print(f"\nScan interrupted. Progress saved ({len(images)} images).")
        save_data(images, output_file)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        save_data(images, output_file)

if __name__ == "__main__":
    # Configuration
    TARGET_DIR = r"C:\Users\ylnha\Pictures"
    OUTPUT_JSON = r"C:\Users\ylnha\Projects\local-image-search\data\images.json"
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    scan_directory(TARGET_DIR, OUTPUT_JSON)
