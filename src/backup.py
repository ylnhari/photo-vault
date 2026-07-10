"""Opportunistic mirror of the photo library + photo-vault's own data/ to a
backup destination (external drive / SD card / any connected folder).

Model: the laptop is the single source of truth; the backup destination is a
one-way mirror. The destination isn't always plugged in, so this is
deliberately opportunistic: status() reports whether the destination is
currently available and how stale the last successful backup is, and the UI
nags instead of failing.

data/ rides along so a restore brings back captions, faces, embeddings and
settings — not just pixels. Backup runs as a normal job (one job at a time),
so nothing else is writing the catalog/ChromaDB mid-copy.

Copy engine per OS: on Windows, robocopy (built in, incremental by
timestamp+size, unattended-safe with /R:1 /W:1); everywhere else, a stdlib
incremental mirror with the same semantics (size + mtime-within-2s compare —
the 2-second tolerance matches FAT/exFAT timestamp granularity, robocopy's
/FFT, so an NTFS/ext4→exFAT mirror doesn't re-copy the world every run).
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import platformfs
from constants import DATA_DIR

STATE_PATH = os.path.join(DATA_DIR, "backup_state.json")

# Video files are invisible to photo-root mirroring — videos are out of the
# app's scope for now, and some live INSIDE destination photo trees where a
# purge would delete them as extras. Kept in sync with ingest's video set.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
                    ".3gp", ".mts", ".m2ts", ".wmv"}

# robocopy exit codes are a bitmask; < 8 means "no failures" (0 = nothing to
# do, 1 = files copied, 2 = extras deleted at dest, 4 = mismatches fixed).
_ROBOCOPY_OK_BELOW = 8

_SUMMARY_RE = re.compile(
    r"Files\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
)

# FAT/exFAT stores mtimes at 2-second granularity — treat timestamps within
# this window as equal or every NTFS/ext4→exFAT run re-copies everything.
_MTIME_TOLERANCE = 2.0


def get_dest() -> str | None:
    import settings as settings_mod
    return settings_mod.load().get("backup_dest") or None


def _label(src: str) -> str:
    """Folder-name-safe label for a source root, e.g.
    'C:\\Users\\ylnha\\Pictures' → 'C_Users_ylnha_Pictures'."""
    return re.sub(r"[:\\/]+", "_", src).strip("_")


def backup_roots() -> list[tuple[str, str]]:
    """[(src, dest)] pairs to mirror: every included scan folder lands at
    <dest>/<its basename> (so C:\\Users\\ylnha\\Pictures backed up to D:\\
    becomes simply D:\\Pictures — Hari's requested layout), plus data/ under
    <dest>/photo-vault-data so captions/faces/index restore along with the
    pixels. If two included folders share a basename, the full sanitized
    path label is used for both to keep them apart."""
    dest = get_dest()
    if not dest:
        return []
    from folders import get_effective_scan_dirs
    dirs = get_effective_scan_dirs()
    basenames = [os.path.basename(os.path.normpath(d)) or _label(d) for d in dirs]
    pairs = []
    for src, base in zip(dirs, basenames):
        sub = base if basenames.count(base) == 1 else _label(src)
        pairs.append((src, os.path.join(dest, sub)))
    pairs.append((DATA_DIR, os.path.join(dest, "photo-vault-data")))
    return pairs


def _drive_available(dest: str) -> bool:
    return platformfs.dest_available(dest)


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def record_success():
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"last_backup_at": time.time()}, f)
    os.replace(tmp, STATE_PATH)


def status() -> dict:
    """For the Backup card: is a destination configured, is its drive
    currently plugged in, and how stale is the last successful run."""
    dest = get_dest()
    last = _load_state().get("last_backup_at")
    days = round((time.time() - last) / 86400, 1) if last else None
    return {
        "configured": bool(dest),
        "dest": dest,
        "available": _drive_available(dest) if dest else False,
        "last_backup_at": last,
        "days_since": days,
        "roots": [src for src, _ in backup_roots()],
    }


def validate_dest(dest: str) -> dict:
    """Pre-flight check for the backup destination. It must not overlap the
    scanned library in either direction: inside an included folder the mirror
    would recursively back itself up; containing an included folder means the
    scan would index your own backup, doubling every photo in the catalog.
    Anything else — SD card, pen drive, another internal drive — is fine."""
    from folders import get_effective_scan_dirs, get_excluded_paths

    def _norm(p):
        return os.path.normcase(str(Path(p).resolve()))

    def _under(child, parent):
        return child == parent or child.startswith(parent + os.sep)

    dest = (dest or "").strip()
    if not dest:
        return {"ok": False, "reason": "Pick a backup destination folder."}
    if not platformfs.dest_available(dest):
        return {"ok": False, "reason": platformfs.unavailable_reason(dest)}
    b = _norm(dest)
    for inc in get_effective_scan_dirs():
        i = _norm(inc)
        if _under(b, i):
            return {"ok": False, "reason":
                    f"Can't back up into here — it's inside your scanned library ({inc}), "
                    "so the mirror would recursively back itself up. Pick a folder outside "
                    "the library, ideally on another drive or removable storage."}
        if _under(i, b):
            return {"ok": False, "reason":
                    f"Can't back up into here — it contains your scanned folder ({inc}), "
                    "so the next scan would index your own backup and double every photo. "
                    "Pick a separate folder."}
    for ex in get_excluded_paths():
        if _under(b, _norm(ex)) or _under(_norm(ex), b):
            return {"ok": False, "reason":
                    f"Can't back up into here — it overlaps an excluded folder ({ex}). "
                    "Pick a clean, separate folder for the backup."}
    return {"ok": True, "reason": None}


def backup_one(src: str) -> str:
    """Mirror one source root to its destination. Returns a job-log note.
    Raises on copy failure so the job counts it as a fail.

    Two mirroring modes, same on every OS:
      - Photo folders: additive (no purge) with video files invisible —
        videos are out of the app's scope for now, and some live INSIDE the
        destination photo tree; a purge would delete them as extras. No purge
        also means a photo deleted on the laptop lingers in the backup — the
        safer failure mode until video handling lands and strict mirroring
        returns.
      - photo-vault's own data/: strict mirror (purge extras) — it has no
        videos, and stale ChromaDB segment files from older runs must never
        mix into a restore.
    """
    dest_map = dict(backup_roots())
    dst = dest_map.get(src)
    if not dst:
        raise RuntimeError(f"no backup destination mapped for {src}")
    if not os.path.isdir(src):
        return "skipped (source folder missing)"
    os.makedirs(dst, exist_ok=True)
    purge = src == DATA_DIR
    if os.name == "nt":
        note = _mirror_robocopy(src, dst, purge)
    else:
        note = _mirror_python(src, dst, purge)
    record_success()
    return note


def _mirror_robocopy(src: str, dst: str, purge: bool) -> str:
    """Windows engine: robocopy — built in, incremental, unattended-safe."""
    video_globs = [f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS)]
    mode = ["/MIR"] if purge else (["/E", "/XF"] + video_globs)
    cmd = [
        "robocopy", src, dst, *mode, "/R:1", "/W:1",
        # /FFT: FAT-style 2-second timestamp granularity, /DST: tolerate DST
        # offsets — without these an NTFS→exFAT mirror (SD cards are exFAT)
        # sees every file as "changed" and re-copies the whole library each run.
        "/FFT", "/DST",
        "/NP", "/NDL", "/NFL",
        "/XD", "$RECYCLE.BIN", "System Volume Information",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode >= _ROBOCOPY_OK_BELOW:
        tail = (proc.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(
            f"robocopy failed (exit {proc.returncode}): {' | '.join(tail)}"
        )
    m = _SUMMARY_RE.search(proc.stdout or "")
    if m:
        total, copied, skipped, _mismatch, failed, extras = (int(g) for g in m.groups())
        return (f"mirrored — {copied} copied · {skipped} unchanged"
                + (f" · {extras} removed at dest" if extras else "")
                + (f" · {failed} FAILED" if failed else ""))
    return "mirrored"


def _unchanged(src_st: os.stat_result, dst_path: str) -> bool:
    """robocopy-style incremental compare: same size and mtime within the
    FAT 2-second window means 'already backed up'."""
    try:
        dst_st = os.stat(dst_path)
    except OSError:
        return False
    return (src_st.st_size == dst_st.st_size
            and abs(src_st.st_mtime - dst_st.st_mtime) <= _MTIME_TOLERANCE)


def _mirror_python(src: str, dst: str, purge: bool) -> str:
    """POSIX engine: stdlib incremental mirror with the same semantics as the
    robocopy invocation. copy2 preserves mtimes so the next run's compare
    works; per-file failures are counted (and reported) rather than aborting
    the whole root, mirroring robocopy's keep-going behavior — but a run with
    ANY failure still raises so the job never records a partial mirror as a
    clean success."""
    copied = unchanged = removed = failed = 0
    first_error = None
    skip_videos = not purge  # photo roots skip videos; data/ has none

    for dirpath, dirnames, filenames in os.walk(src):
        platformfs.skip_system_dirs(dirnames)
        rel = os.path.relpath(dirpath, src)
        out_dir = dst if rel == "." else os.path.join(dst, rel)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            failed += len(filenames)
            first_error = first_error or f"{out_dir}: {e}"
            dirnames[:] = []
            continue
        for name in filenames:
            if skip_videos and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                continue
            sp = os.path.join(dirpath, name)
            dp = os.path.join(out_dir, name)
            try:
                st = os.stat(sp)
                if _unchanged(st, dp):
                    unchanged += 1
                    continue
                shutil.copy2(sp, dp)
                copied += 1
            except OSError as e:
                failed += 1
                first_error = first_error or f"{sp}: {e}"

    if purge:
        # Walk the DESTINATION bottom-up and drop anything the source no
        # longer has — the /MIR half. Bottom-up so emptied dirs delete cleanly.
        for dirpath, dirnames, filenames in os.walk(dst, topdown=False):
            rel = os.path.relpath(dirpath, dst)
            src_dir = src if rel == "." else os.path.join(src, rel)
            for name in filenames:
                if not os.path.exists(os.path.join(src_dir, name)):
                    try:
                        os.remove(os.path.join(dirpath, name))
                        removed += 1
                    except OSError as e:
                        failed += 1
                        first_error = first_error or f"{dirpath}: {e}"
            if rel != "." and not os.path.isdir(src_dir):
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass  # not empty (a failed remove above) — leave it

    if failed:
        raise RuntimeError(
            f"mirror had {failed} failure(s) (first: {first_error}) — "
            f"{copied} copied · {unchanged} unchanged before/despite the errors"
        )
    return (f"mirrored — {copied} copied · {unchanged} unchanged"
            + (f" · {removed} removed at dest" if removed else ""))
