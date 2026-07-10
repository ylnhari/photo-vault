"""Import & consolidate: merge a staging folder (Google Takeout extract,
pen-drive dump, phone download, SD-card folder) into the canonical library.

Every file is identified by SHA-1 of its bytes — the same identity the
catalog uses — so content the library has EVER seen is skipped, no matter
how many times it was copied around or what it was renamed to. Only genuinely
new files are copied in, organized <dest>/YYYY/MM/ by EXIF date (file mtime
fallback). Videos ride along: they're deduped and copied like images (tracked
in a persistent media-hash cache, since the image catalog doesn't hold them)
but never captioned — vision is image-only.

The staging source is read-only to us: originals are never deleted here, so
a botched run costs nothing. After an ingest, run Scan → the pipeline; only
the newly imported images will be pending.
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import exifread

from constants import DATA_DIR
from scanner import IMAGE_EXTENSIONS, content_uid, _sig, _is_locally_available

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm',
                    '.3gp', '.mts', '.m2ts', '.wmv'}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Persistent hash cache for media the catalog doesn't track (videos) and for
# everything ever ingested. by_path holds a cheap size+mtime signature so
# library videos only get re-hashed when they actually change.
MEDIA_HASHES_PATH = os.path.join(DATA_DIR, "media_hashes.json")


def _load_media_cache() -> dict:
    if os.path.exists(MEDIA_HASHES_PATH):
        try:
            with open(MEDIA_HASHES_PATH) as f:
                data = json.load(f)
            if isinstance(data.get("by_path"), dict):
                return data
        except Exception:
            pass
    return {"by_path": {}}


def _save_media_cache(cache: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = MEDIA_HASHES_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, MEDIA_HASHES_PATH)


def default_dest() -> str | None:
    """Where imports land when no explicit ingest_dest is configured: an
    Imported/ folder inside the first included scan folder — inside, so the
    normal Scan job picks new photos up with zero extra configuration."""
    import settings as settings_mod
    dest = settings_mod.load().get("ingest_dest")
    if dest:
        return dest
    from folders import get_effective_scan_dirs
    dirs = get_effective_scan_dirs()
    return os.path.join(dirs[0], "Imported") if dirs else None


def list_staging_files(source: str) -> list[str]:
    """All media files under the staging folder (recursive), skipping cloud
    placeholders. Sorted for deterministic run order."""
    root = Path(source)
    if not root.is_dir():
        raise ValueError(f"staging folder not found: {source}")
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Never descend into a recycle bin / hidden system dirs on the card.
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(('$', 'System Volume Information'))]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in MEDIA_EXTENSIONS and _is_locally_available(p):
                out.append(str(p))
    return sorted(out)


def _media_date(path: str) -> datetime:
    """Best-available capture date: EXIF DateTimeOriginal for images, file
    mtime otherwise. Drives only the YYYY/MM destination folder, so a wrong
    guess merely mis-files — it can't corrupt anything."""
    if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
        try:
            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False,
                                             stop_tag="EXIF DateTimeOriginal")
            raw = str(tags.get("EXIF DateTimeOriginal", "")).strip()
            if raw:
                return datetime.strptime(raw[:19], "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return datetime.now()


class IngestSession:
    """One ingest run. Builds the seen-hash set once (catalog image ids +
    media-hash cache, refreshing video hashes for library folders), then
    ingest_one() per staging file. close() persists the cache."""

    def __init__(self, source: str, catalog_images: dict, dest: str = None):
        self.source = str(Path(source).resolve())
        self.dest = dest or default_dest()
        if not self.dest:
            raise ValueError("no destination: configure a scan folder or ingest_dest")
        self.dest = str(Path(self.dest).resolve())
        self.cache = _load_media_cache()
        self._dirty = 0
        # Image content the catalog already tracks — the big dedupe net.
        self.seen: set[str] = set(catalog_images.keys())
        self._refresh_library_video_hashes()
        self.seen.update(
            rec["sha1"] for rec in self.cache["by_path"].values() if rec.get("sha1")
        )

    def _refresh_library_video_hashes(self):
        """Hash library videos not yet in the cache (or changed since), so a
        staging video that already lives in the library is recognized as a
        duplicate. One-time cost on first run; sig fast-path afterwards."""
        from folders import get_effective_scan_dirs
        by_path = self.cache["by_path"]
        for root in get_effective_scan_dirs():
            root_p = Path(root)
            if not root_p.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root_p):
                dirnames[:] = [d for d in dirnames if not d.startswith('$')]
                for name in filenames:
                    p = Path(dirpath) / name
                    if p.suffix.lower() not in VIDEO_EXTENSIONS:
                        continue
                    if not _is_locally_available(p):
                        continue
                    sp = str(p)
                    try:
                        sig = _sig(p.stat())
                    except OSError:
                        continue
                    rec = by_path.get(sp)
                    if rec and rec.get("sig") == sig:
                        continue
                    try:
                        by_path[sp] = {"sig": sig, "sha1": content_uid(p)}
                        self._dirty += 1
                    except Exception:
                        continue
        if self._dirty:
            _save_media_cache(self.cache)
            self._dirty = 0

    def _dest_path(self, src: str, uid: str) -> Path:
        d = _media_date(src)
        folder = Path(self.dest) / f"{d.year:04d}" / f"{d.month:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        base = Path(src).name
        target = folder / base
        if target.exists():
            # Same name, different content (same content was already caught
            # by the hash check) — qualify with a short content-hash suffix.
            target = folder / f"{Path(base).stem}-{uid[:8]}{Path(base).suffix}"
        return target

    def ingest_one(self, src: str) -> str:
        """Ingest one staging file. Returns a job-log note; notes containing
        'skipped' land in the job's skipped bucket, not ok/fail."""
        uid = content_uid(src)
        if uid in self.seen:
            return "skipped (duplicate — already in library)"
        target = self._dest_path(src, uid)
        shutil.copy2(src, target)
        self.seen.add(uid)
        self.cache["by_path"][str(target)] = {
            "sig": _sig(os.stat(target)), "sha1": uid,
        }
        self._dirty += 1
        if self._dirty >= 25:
            _save_media_cache(self.cache)
            self._dirty = 0
        rel = os.path.relpath(target, self.dest)
        return f"imported → {rel}"

    def close(self):
        if self._dirty:
            _save_media_cache(self.cache)
            self._dirty = 0
