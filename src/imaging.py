"""
Centralized image-decoding setup. Importing this module:
  1. Registers the HEIF/HEIC opener so PIL can read iPhone photos.
  2. Caps the maximum decoded pixel count to defuse decompression bombs.

Import this once, early, anywhere a PIL Image.open may run on user files
(api.py for thumbnails, vision.py for captioning).
"""
from PIL import Image

# ~200 megapixels. Real consumer photos top out around 50MP; anything far above
# this is almost certainly a crafted or pathological file. PIL raises
# Image.DecompressionBombError above this, which callers treat as a bad file.
Image.MAX_IMAGE_PIXELS = 200_000_000

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except Exception as e:  # pragma: no cover
    print(f"[imaging] HEIF support unavailable: {e}")
    HEIF_SUPPORTED = False


def safe_open(path):
    """
    Open an image with bomb protection already configured. Raises on
    corrupt/malformed/oversized files — callers must handle the exception.
    """
    return Image.open(path)
