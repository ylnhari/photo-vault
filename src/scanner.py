import math
import os
import sys
import json
import hashlib
import exifread
from pathlib import Path
from datetime import datetime
import catalog_db

# Supported still-image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp'}
# Supported video container formats. Videos are catalogued as first-class media
# (media_type="video") alongside photos; ingest.py imports this same set so
# import-dedup and scanning agree on what counts as a video.
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm',
                    '.3gp', '.mts', '.m2ts', '.wmv'}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def is_video_path(path) -> bool:
    """True when a path's extension names a video container (case-insensitive)."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS

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
    """Convert EXIF GPS DMS rationals + hemisphere ref to a signed decimal degree.
    Returns None (instead of a garbage value) if malformed EXIF produces a
    decimal degree outside the valid range for the axis (lat: [-90,90],
    lon: [-180,180]) or a non-finite value — bad GPS should never silently
    flow into geocoding."""
    try:
        vals = values.values if hasattr(values, "values") else values
        d, m, s = (_ratio_to_float(vals[0]), _ratio_to_float(vals[1]), _ratio_to_float(vals[2]))
        dec = d + m / 60.0 + s / 3600.0
        ref_up = str(ref).upper()
        if ref_up.startswith(("S", "W")):
            dec = -dec
        dec = round(dec, 6)
        if not math.isfinite(dec):
            return None
        axis_limit = 90.0 if ref_up.startswith(("N", "S")) else 180.0
        if abs(dec) > axis_limit:
            return None
        return dec
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


def _note_dup_path(entry: dict, path: str):
    """Remember a byte-identical extra copy of this photo (see the duplicate
    branch in scan_directory). Consumed by indexer.get_redundant_copies()."""
    dups = entry.setdefault("dup_paths", [])
    if path not in dups and path != entry.get("path"):
        dups.append(path)


def _sig(stat) -> str:
    """Cheap change-signature so unchanged files are skipped without re-hashing.
    Uses sub-second mtime precision (rounded to milliseconds for stable
    comparison) — truncating to whole seconds would miss a same-second edit
    that leaves size unchanged."""
    return f"{stat.st_size}:{round(stat.st_mtime, 3)}"


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


def _iter_media(root: Path, excluded: set[str]):
    """
    Yield image and video Paths under root, skipping excluded dirs and handling
    errors gracefully (broken symlinks, permission errors, circular symlinks).
    Uses os.walk so we can skip entire subtrees efficiently.

    Two things require resolving each directory the walk visits to its real
    (symlink-free, canonical) path, the same way `excluded` was built:
      - excluded-dir matching: comparing raw os.walk paths (normcase only)
        against fully resolved excluded paths lets a symlink/junction,
        trailing separator, or casing difference slip an excluded folder
        through unpruned.
      - cycle prevention: os.walk(followlinks=True) will traverse a
        self-referential directory symlink/junction forever unless we track
        which real directories we've already descended into and refuse to
        re-enter them (broken-symlink skipping alone does not catch this).
    """
    visited_real_dirs: set[str] = set()
    root_str = str(root)
    for dirpath, dirnames, filenames in os.walk(root_str, followlinks=True, onerror=None):
        try:
            real_dir = os.path.normcase(str(Path(dirpath).resolve()))
        except OSError:
            real_dir = None

        if real_dir is not None:
            if real_dir in visited_real_dirs:
                # Already descended into this real directory in this scan —
                # a symlink/junction cycle. Don't recurse further.
                dirnames[:] = []
                continue
            visited_real_dirs.add(real_dir)

        # Prune excluded directories in-place so os.walk doesn't descend into
        # them. Resolve each candidate child the same way the excluded set
        # was built so the comparison is apples-to-apples.
        kept = []
        for d in dirnames:
            try:
                d_real = os.path.normcase(str(Path(dirpath, d).resolve()))
            except OSError:
                kept.append(d)  # can't resolve — err on the side of walking it
                continue
            if not _is_excluded(d_real, excluded):
                kept.append(d)
        dirnames[:] = kept

        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            # Skip circular-symlink targets (already-visited inodes on Unix)
            if p.is_symlink() and not p.exists():
                continue
            yield p


def _media_fields(path: Path) -> dict:
    """media_type plus, for videos, probe metadata (duration/dims/codec) and a
    capture date derived from the container's creation_time. Images keep the
    existing EXIF path. Probe failures degrade to a video row with null
    fields — the file is still catalogued, browsable, and re-probable later."""
    if is_video_path(path):
        import video
        info = video.probe(str(path)) or {}
        meta: dict = {}
        if info.get("capture_time"):
            meta["date"] = info["capture_time"]
        if info.get("width"):
            meta["width"] = str(info["width"])
        if info.get("height"):
            meta["height"] = str(info["height"])
        return {
            "media_type": "video",
            "duration_s": info.get("duration_s"),
            "width": info.get("width"),
            "height": info.get("height"),
            "codec": info.get("codec"),
            "metadata": meta,
        }
    return {"media_type": "image", "metadata": get_metadata(path)}


def load_existing_data(output_file):
    """Load catalog. Returns (images_by_uid, folders).

    Deliberately does NOT catch-and-return-empty on a load failure (locked
    DB, one corrupted row, transient I/O). scan_directory()'s checkpoint save
    is a full sync that deletes every catalog row absent from the in-memory
    dict — if a transient read failure were silently treated as "empty
    catalog", the very next checkpoint would wipe the entire catalog. Let the
    exception propagate so the scan aborts loudly instead; a scan can simply
    be retried once the transient issue (e.g. a momentary lock) clears."""
    data = catalog_db.load_all(output_file)
    return data.get("images", {}), data.get("folders", {})


def save_data(images, folders, output_file):
    """Save catalog (SQLite-backed — see catalog_db.py). Upserts every row
    currently in `images`/`folders`; scan checkpoints are infrequent (every
    `checkpoint_interval` files) so dirty-tracking isn't worth the complexity
    here, unlike the per-batch saves in the job loop (indexer.py)."""
    try:
        catalog_db.save_all(output_file, images, folders)
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
    by_path: dict = {}
    for uid, d in images.items():
        p = d.get("path")
        prior_uid = by_path.get(p)
        if prior_uid is not None and prior_uid != uid:
            # Two catalog rows claim the same path — the dict comprehension
            # this replaced would silently let one overwrite the other with
            # no trace. Surface it; the underlying data-model limitation
            # (one physical path can only map to one uid) is documented on
            # the duplicate-path handling below.
            print(f"Warning: catalog path collision for '{p}' — uids {prior_uid!r} and {uid!r} both claim it; keeping {uid!r}.")
        by_path[p] = uid
    added = moved = unchanged = seen = 0
    moved_ids: list[str] = []

    print(f"Scanning: {root_dir}")
    for path in _iter_media(root, excluded):
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
            old_path = entry.get("path")
            if old_path == str_path:
                unchanged += 1
            else:
                old_exists = bool(old_path) and os.path.exists(old_path)
                if old_exists and str_path >= old_path:
                    # Byte-identical duplicate: two live paths hash to the
                    # same content uid. The catalog tracks ONE canonical path
                    # per uid; rather than flip-flopping to "whichever path
                    # os.walk visited last" (walk order isn't stable across
                    # rescans), deterministically keep the lexicographically-
                    # first path. The other copy is recorded in dup_paths so
                    # the "dedupe" job can physically reclaim it — entries are
                    # re-verified (exists + hash) before anything is trashed,
                    # so a stale record here is harmless.
                    _note_dup_path(entry, str_path)
                    unchanged += 1
                else:
                    # Either a genuine move (old_path no longer exists) or a
                    # duplicate where the new path sorts first — both are the
                    # right cases to adopt the new path. In the duplicate case
                    # the old (still existing) copy becomes the redundant one.
                    if old_exists:
                        _note_dup_path(entry, old_path)
                    moved += 1
                    moved_ids.append(uid)
                    entry["path"] = str_path
                    entry["filename"] = path.name
                    if str_path in entry.get("dup_paths", []):
                        entry["dup_paths"].remove(str_path)
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
                **_media_fields(path),
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
        "moved_ids": moved_ids,
    }
    print(f"Scan complete: {summary}")
    return summary
