import os
import sys
import json
import hashlib
import exifread
from pathlib import Path
from datetime import datetime

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp'}

# Windows file attributes for cloud/offline files (OneDrive, SharePoint, etc.)
_FILE_ATTRIBUTE_OFFLINE = 0x1000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000


def _is_locally_available(path) -> bool:
    """
    On Windows, return False for cloud-placeholder files (OneDrive offline, SharePoint
    not-yet-synced) so we don't trigger a download of every cloud-only photo.
    Always True on non-Windows platforms.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == 0xFFFFFFFF:   # INVALID_FILE_ATTRIBUTES — can't tell, try anyway
            return True
        return not (attrs & (_FILE_ATTRIBUTE_OFFLINE | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))
    except Exception:
        return True


def _ratio_to_float(r) -> float:
    try:
        return float(r.num) / float(r.den) if getattr(r, "den", 0) else float(r)
    except Exception:
        return 0.0


def _gps_to_decimal(values, ref) -> float | None:
    """Convert EXIF GPS DMS rationals + hemisphere ref to a signed decimal degree."""
    try:
        vals = values.values if hasattr(values, "values") else values
        d, m, s = (_ratio_to_float(vals[0]), _ratio_to_float(vals[1]), _ratio_to_float(vals[2]))
        dec = d + m / 60.0 + s / 3600.0
        if str(ref).upper().startswith(("S", "W")):
            dec = -dec
        return round(dec, 6)
    except Exception:
        return None


def get_metadata(file_path):
    """Extract a rich-but-cheap EXIF subset (no full image decode)."""
    metadata = {}
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        if not tags:
            return metadata

        def s(key):
            return str(tags[key]).strip() if key in tags else None

        # Core
        for key, field in [
            ('EXIF DateTimeOriginal', 'date'),
            ('Image Make', 'camera_make'),
            ('Image Model', 'camera_model'),
            ('EXIF LensModel', 'lens'),
            ('EXIF ISOSpeedRatings', 'iso'),
            ('EXIF FNumber', 'f_number'),
            ('EXIF ExposureTime', 'exposure'),
            ('EXIF FocalLength', 'focal_length'),
            ('Image Orientation', 'orientation'),
            ('EXIF ExifImageWidth', 'width'),
            ('EXIF ExifImageLength', 'height'),
        ]:
            val = s(key)
            if val:
                metadata[field] = val

        # GPS → decimal lat/lon
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat = _gps_to_decimal(tags['GPS GPSLatitude'], tags.get('GPS GPSLatitudeRef', 'N'))
            lon = _gps_to_decimal(tags['GPS GPSLongitude'], tags.get('GPS GPSLongitudeRef', 'E'))
            if lat is not None and lon is not None:
                metadata['gps_lat'] = lat
                metadata['gps_lon'] = lon
    except Exception:
        pass
    return metadata


def content_uid(path) -> str:
    """SHA-1 of file bytes — the canonical photo id. Survives moves/renames, so the
    same photo at a new path is recognized as the same photo (not delete + new)."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _sig(stat) -> str:
    """Cheap change-signature so unchanged files are skipped without re-hashing."""
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _build_excluded_set(excluded_paths) -> set[str]:
    """Resolve and normalize excluded paths for fast prefix matching."""
    result = set()
    for p in (excluded_paths or []):
        try:
            result.add(os.path.normcase(str(Path(p).resolve())))
        except Exception:
            pass
    return result


def _is_excluded(abs_path_norm: str, excluded: set[str]) -> bool:
    """True if abs_path_norm is or is inside any excluded directory."""
    for ex in excluded:
        if abs_path_norm == ex or abs_path_norm.startswith(ex + os.sep):
            return True
    return False


def _iter_images(root: Path, excluded: set[str]):
    """
    Yield image Paths under root, skipping excluded dirs and handling
    errors gracefully (broken symlinks, permission errors, circular symlinks).
    Uses os.walk so we can skip entire subtrees efficiently.
    """
    root_str = str(root)
    for dirpath, dirnames, filenames in os.walk(root_str, followlinks=True, onerror=None):
        dir_norm = os.path.normcase(dirpath)

        # Prune excluded directories in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d for d in dirnames
            if not _is_excluded(os.path.normcase(os.path.join(dirpath, d)), excluded)
        ]

        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            # Skip circular-symlink targets (already-visited inodes on Unix)
            if p.is_symlink() and not p.exists():
                continue
            yield p


def load_existing_data(output_file):
    """Load catalog. Returns (images_by_uid, folders)."""
    images, folders = {}, {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                images = data.get("images", {})
                folders = data.get("folders", {})
        except Exception as e:
            print(f"Warning: could not load catalog ({e}). Starting fresh.")
    return images, folders


def save_data(images, folders, output_file):
    """Atomically save catalog as {images: {uid: data}, folders: {root: meta}}."""
    temp = output_file + ".tmp"
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(temp, 'w') as f:
            json.dump({"images": images, "folders": folders}, f, indent=2)
        os.replace(temp, output_file)
    except Exception as e:
        print(f"Error saving catalog: {e}")


def scan_directory(
    root_dir,
    output_file,
    checkpoint_interval=200,
    excluded_paths=None,
    progress_callback=None,
) -> dict:
    """
    Recursively scan a folder. Identifies photos by content hash so moves are
    detected. Skips paths under excluded_paths, cloud-only (offline) files, and
    handles permission errors gracefully.

    Returns {added, moved, unchanged, total, scanned}
      scanned  = image files walked in this specific directory
      total    = cumulative total across the entire catalog
    """
    images, folders = load_existing_data(output_file)
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: {root_dir} does not exist.")
        return {"added": 0, "moved": 0, "unchanged": 0, "total": len(images), "scanned": 0}
    if not root.is_dir():
        print(f"Error: {root_dir} is not a directory.")
        return {"added": 0, "moved": 0, "unchanged": 0, "total": len(images), "scanned": 0}

    excluded = _build_excluded_set(excluded_paths)
    by_path = {d.get("path"): uid for uid, d in images.items()}
    added = moved = unchanged = seen = 0

    print(f"Scanning: {root_dir}")
    for path in _iter_images(root, excluded):
        # Skip Windows cloud-placeholder files to avoid triggering downloads
        if not _is_locally_available(path):
            continue

        str_path = str(path.absolute())
        try:
            stats = path.stat()
        except OSError:
            continue
        sig = _sig(stats)

        # Fast path: same path + same signature → unchanged, no hashing.
        existing_uid = by_path.get(str_path)
        if existing_uid and images.get(existing_uid, {}).get("sig") == sig:
            unchanged += 1
            seen += 1
            if progress_callback:
                progress_callback(str(root_dir), seen)
            continue

        try:
            uid = content_uid(path)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue

        # A file edited in place keeps its path but gets a new content uid; the
        # old uid entry would linger pointing at the same (existing) path — a
        # permanent duplicate that never shows up as orphaned. Retire it.
        if existing_uid and existing_uid != uid and existing_uid in images:
            if images[existing_uid].get("path") == str_path:
                del images[existing_uid]

        if uid in images:
            entry = images[uid]
            if entry.get("path") != str_path:
                moved += 1
                entry["path"] = str_path
                entry["filename"] = path.name
            else:
                unchanged += 1
            entry["sig"] = sig
            entry["size_bytes"] = stats.st_size
        else:
            images[uid] = {
                "uid": uid,
                "path": str_path,
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": stats.st_size,
                "created_at": stats.st_ctime,
                "sig": sig,
                "metadata": get_metadata(path),
            }
            added += 1

        by_path[str_path] = uid
        seen += 1
        if progress_callback:
            progress_callback(str(root_dir), seen)
        if seen % checkpoint_interval == 0:
            print(f"  …{seen} files ({added} new, {moved} moved)")
            save_data(images, folders, output_file)

    folders[str(root.absolute())] = {
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "count": seen,
    }
    save_data(images, folders, output_file)
    summary = {
        "added": added,
        "moved": moved,
        "unchanged": unchanged,
        "total": len(images),
        "scanned": seen,
    }
    print(f"Scan complete: {summary}")
    return summary
