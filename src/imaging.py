"""
Centralized image-decoding setup. Importing this module:
  1. Registers the HEIF/HEIC opener so PIL can read iPhone photos.
  2. Caps the maximum decoded pixel count to defuse decompression bombs.

Import this once, early, anywhere a PIL Image.open may run on user files
(api.py for thumbnails, vision.py for captioning).
"""
import hashlib
import os

from PIL import Image

from constants import THUMB_DIR

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


# ── derivative (thumb / medium) generation ────────────────────────────────────
# One place for the API's on-demand serving and the "thumbs" pregeneration job.
# New derivatives are WebP (~40% smaller than the old JPEGs at like quality);
# pre-existing .jpg derivatives keep being served until regenerated.

THUMB_PX = 400
MEDIUM_PX = 1600

os.makedirs(THUMB_DIR, exist_ok=True)


def derivative_key(img_id: str) -> str:
    return hashlib.sha1(img_id.encode("utf-8")).hexdigest()


def derivative_path(img_id: str, suffix: str = "") -> str:
    return os.path.join(THUMB_DIR, f"{derivative_key(img_id)}{suffix}.webp")


def legacy_derivative_path(img_id: str, suffix: str = "") -> str:
    """Path of the pre-WebP .jpg derivative (served if it already exists)."""
    return os.path.join(THUMB_DIR, f"{derivative_key(img_id)}{suffix}.jpg")


def ensure_derivative(src_path: str, out_path: str, max_px: int) -> bool:
    """Generate a downscaled WebP derivative if missing. False on failure."""
    if os.path.exists(out_path):
        return True
    try:
        with safe_open(src_path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            im.save(out_path, "WEBP", quality=80, method=4)
        return True
    except Exception as e:
        print(f"[imaging] derivative ({max_px}px) failed for {src_path}: {e}")
        return False
