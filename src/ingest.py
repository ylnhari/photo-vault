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

import platformfs
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
    Imported/ folder inside an included scan folder — inside, so the normal
    Scan job picks new photos up with zero extra configuration. Prefers a
    non-cloud-synced folder: scan dirs are sorted alphabetically, and
    'OneDrive' sorting before 'Pictures' once silently routed imports into
    OneDrive — gigabytes of videos syncing to the cloud is never the
    expected outcome of a local import."""
    import settings as settings_mod
    dest = settings_mod.load().get("ingest_dest")
    if dest:
        return dest
    from folders import get_effective_scan_dirs
    dirs = get_effective_scan_dirs()
    if not dirs:
        return None
    local = [d for d in dirs if "onedrive" not in d.lower()]
    return os.path.join((local or dirs)[0], "Imported")


def _norm(p: str) -> str:
    return os.path.normcase(str(Path(p).resolve()))


def _under(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent + os.sep)


def validate_source(src: str) -> dict:
    """Pre-flight check for an import SOURCE folder. Returns {ok, reason} —
    reason is a friendly, complete sentence the UI shows verbatim.
    Rules (user-defined): the source must exist, must not already be part of
    the scanned library (that would import the library into itself), and
    must not be inside an excluded folder (excluded means 'not my photos' —
    importing from there is almost certainly a mistake)."""
    from folders import get_effective_scan_dirs, get_excluded_paths
    src = (src or "").strip()
    if not src:
        return {"ok": False, "reason": "Pick a folder to import from."}
    if not os.path.isdir(src):
        return {"ok": False, "reason": f"That folder doesn't exist or isn't accessible: {src}"}
    s = _norm(src)
    for ex in get_excluded_paths():
        if _under(s, _norm(ex)):
            return {"ok": False, "reason":
                    f"Can't import from here — it's inside an excluded folder ({ex}). "
                    "Excluded means these files were deliberately left out of your library; "
                    "remove the exclusion first if you actually want them."}
    for inc in get_effective_scan_dirs():
        if _under(s, _norm(inc)):
            return {"ok": False, "reason":
                    f"Can't import from here — it's already part of your scanned library ({inc}). "
                    "Importing it would only create duplicate copies of photos you already have."}
    return {"ok": True, "reason": None}


def validate_dest(dest: str) -> dict:
    """Pre-flight check for the import DESTINATION. It must be inside an
    included scan folder (or photos land where they'd never be indexed) and
    not inside an excluded one (same invisibility problem)."""
    from folders import get_effective_scan_dirs, get_excluded_paths
    dest = (dest or "").strip()
    if not dest:
        return {"ok": False, "reason": "Pick a destination folder for imports."}
    d = _norm(dest)
    for ex in get_excluded_paths():
        if _under(d, _norm(ex)):
            return {"ok": False, "reason":
                    f"Can't import into here — it's inside an excluded folder ({ex}), "
                    "so imported photos would never be scanned or captioned."}
    if not any(_under(d, _norm(inc)) for inc in get_effective_scan_dirs()):
        return {"ok": False, "reason":
                "The destination must be inside one of your included scan folders — "
                "otherwise imported photos would sit invisible, never scanned or captioned. "
                "Pick a folder (or subfolder) of the folders listed in Folder Management."}
    return {"ok": True, "reason": None}


def source_stats(src: str) -> dict:
    """What an import would look at: media file count and total size (plus
    how many non-media files will be ignored). Powers the pre-flight preview
    so the user sees the scope before committing."""
    media_files = 0
    media_bytes = 0
    other_files = 0
    for dirpath, dirnames, filenames in os.walk(src):
        platformfs.skip_system_dirs(dirnames)
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in MEDIA_EXTENSIONS:
                media_files += 1
                try:
                    media_bytes += p.stat().st_size
                except OSError:
                    pass
            else:
                other_files += 1
    return {"media_files": media_files, "media_bytes": media_bytes,
            "other_files": other_files}


def list_staging_files(source: str) -> list[str]:
    """All media files under the staging folder (recursive), skipping cloud
    placeholders. Sorted for deterministic run order."""
    root = Path(source)
    if not root.is_dir():
        raise ValueError(f"staging folder not found: {source}")
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Never descend into a recycle bin / trash / system dirs on the card
        # (any OS — deleted photos must not resurrect as "new" imports).
        platformfs.skip_system_dirs(dirnames)
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
        # Self-heal: drop cache entries whose file no longer exists. Without
        # this, a file deleted after ingest would keep its hash in the "seen"
        # set forever and re-ingesting its source would silently skip it —
        # claiming it's "in the library" when it isn't. (Deliberate deletions
        # are re-importable this way; that's the lesser evil vs. phantom
        # library claims.)
        stale = [p for p in self.cache["by_path"] if not os.path.exists(p)]
        for p in stale:
            del self.cache["by_path"][p]
            self._dirty += 1
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
