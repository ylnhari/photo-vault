"""
Centralized image-decoding setup. Importing this module:
  1. Registers the HEIF/HEIC opener so PIL can read iPhone photos.
  2. Caps the maximum decoded pixel count to defuse decompression bombs.

Import this once, early, anywhere a PIL Image.open may run on user files
(api.py for thumbnails, vision.py for captioning).
"""
import hashlib
import os

from PIL import Image, ImageOps

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
    """Generate a downscaled WebP derivative if missing. False on failure.

    For a video source there is no still to decode, so we first pull a poster
    frame via ffmpeg into a temp JPEG and downscale that — the grid/lightbox
    thumbnail of a video is its poster frame. Same call site works for both
    media types, so nothing above here special-cases video thumbnails."""
    if os.path.exists(out_path):
        return True
    tmp_poster = None
    try:
        from scanner import is_video_path
        actual_src = src_path
        if is_video_path(src_path):
            import tempfile
            import video
            fd, tmp_poster = tempfile.mkstemp(suffix=".jpg", prefix="pv_poster_")
            os.close(fd)
            if not video.poster_frame(src_path, tmp_poster):
                return False
            actual_src = tmp_poster
        with safe_open(actual_src) as im:
            # Bake the EXIF orientation into the pixels now — the source's
            # rotation tag doesn't survive into the resized WebP otherwise,
            # so a portrait photo shot with a rotated sensor would render
            # sideways forever with no metadata left to correct it.
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            im.save(out_path, "WEBP", quality=80, method=4)
        return True
    except Exception as e:
        print(f"[imaging] derivative ({max_px}px) failed for {src_path}: {e}")
        return False
    finally:
        if tmp_poster and os.path.exists(tmp_poster):
            try:
                os.remove(tmp_poster)
            except OSError:
                pass
