"""
Centralized image-decoding setup. Importing this module:
  1. Registers the HEIF/HEIC opener so PIL can read iPhone photos.
  2. Caps the maximum decoded pixel count to defuse decompression bombs.

Import this once, early, anywhere a PIL Image.open may run on user files
(api.py for thumbnails, vision.py for captioning).
"""
import hashlib
import os
import tempfile
import threading

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

_derivative_locks_guard = threading.Lock()
_derivative_locks: dict[str, list] = {}


class _KeyedLock:
    """Reference-counted per-output lock; concurrent requests decode once."""
    def __init__(self, key: str):
        self.key = key

    def __enter__(self):
        with _derivative_locks_guard:
            entry = _derivative_locks.get(self.key)
            if entry is None:
                entry = [threading.Lock(), 0]
                _derivative_locks[self.key] = entry
            entry[1] += 1
            self.lock = entry[0]
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.lock.release()
        with _derivative_locks_guard:
            entry = _derivative_locks.get(self.key)
            if entry is not None:
                entry[1] -= 1
                if entry[1] == 0:
                    _derivative_locks.pop(self.key, None)


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
    lock_key = os.path.normcase(os.path.abspath(out_path))
    with _KeyedLock(lock_key):
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return True
        tmp_poster = None
        tmp_out = None
        try:
            from scanner import is_video_path
            actual_src = src_path
            if is_video_path(src_path):
                import video
                fd, tmp_poster = tempfile.mkstemp(suffix=".jpg", prefix="pv_poster_")
                os.close(fd)
                if not video.poster_frame(src_path, tmp_poster):
                    return False
                actual_src = tmp_poster
            out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
            os.makedirs(out_dir, exist_ok=True)
            fd, tmp_out = tempfile.mkstemp(suffix=".webp", prefix=".pv_derivative_", dir=out_dir)
            os.close(fd)
            with safe_open(actual_src) as im:
                # Let JPEG codecs decode a smaller draft before materializing
                # the full raster. Apply EXIF rotation afterwards, then enforce
                # the requested final bound in display orientation.
                if getattr(im, "format", None) == "JPEG":
                    im.draft("RGB", (max_px, max_px))
                # Bake the EXIF orientation into pixels before sizing so the
                # derivative matches the face detector and browser display.
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                im.thumbnail((max_px, max_px))
                im.save(tmp_out, "WEBP", quality=80, method=4)
            if os.path.getsize(tmp_out) <= 0:
                return False
            # Same-directory replacement is atomic: readers see either the
            # old complete derivative or the newly completed one.
            os.replace(tmp_out, out_path)
            tmp_out = None
            return True
        except Exception as e:
            print(f"[imaging] derivative ({max_px}px) failed for {src_path}: {e}")
            return False
        finally:
            for path in (tmp_poster, tmp_out):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
