"""Opportunistic mirror of the photo library + photo-vault's own data/ to a
backup destination (the 2TB SD card).

Model: the laptop is the single source of truth; the backup destination is a
one-way /MIR mirror — after a run it's byte-identical, including deletions.
The card isn't always plugged in, so this is deliberately opportunistic:
status() reports whether the destination drive is currently available and how
stale the last successful backup is, and the UI nags instead of failing.

data/ rides along so a restore brings back captions, faces, embeddings and
settings — not just pixels. Backup runs as a normal job (one job at a time),
so nothing else is writing the catalog/ChromaDB mid-copy.

robocopy does the heavy lifting: built into Windows, incremental by
timestamp+size, and unattended-safe with /R:1 /W:1.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

from constants import DATA_DIR

STATE_PATH = os.path.join(DATA_DIR, "backup_state.json")

# robocopy exit codes are a bitmask; < 8 means "no failures" (0 = nothing to
# do, 1 = files copied, 2 = extras deleted at dest, 4 = mismatches fixed).
_ROBOCOPY_OK_BELOW = 8

_SUMMARY_RE = re.compile(
    r"Files\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
)


def get_dest() -> str | None:
    import settings as settings_mod
    return settings_mod.load().get("backup_dest") or None


def _label(src: str) -> str:
    """Folder-name-safe label for a source root, e.g.
    'C:\\Users\\ylnha\\Pictures' → 'C_Users_ylnha_Pictures'."""
    return re.sub(r"[:\\/]+", "_", src).strip("_")


def backup_roots() -> list[tuple[str, str]]:
    """[(src, dest)] pairs to mirror: every included scan folder under
    <dest>/library/, plus data/ under <dest>/photo-vault-data."""
    dest = get_dest()
    if not dest:
        return []
    from folders import get_effective_scan_dirs
    pairs = [
        (src, os.path.join(dest, "library", _label(src)))
        for src in get_effective_scan_dirs()
    ]
    pairs.append((DATA_DIR, os.path.join(dest, "photo-vault-data")))
    return pairs


def _drive_available(dest: str) -> bool:
    drive = os.path.splitdrive(dest)[0]
    return os.path.exists(drive + os.sep) if drive else os.path.isdir(dest)


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


def backup_one(src: str) -> str:
    """Mirror one source root to its destination. Returns a job-log note.
    Raises on robocopy failure (exit >= 8) so the job counts it as a fail."""
    dest_map = dict(backup_roots())
    dst = dest_map.get(src)
    if not dst:
        raise RuntimeError(f"no backup destination mapped for {src}")
    if not os.path.isdir(src):
        return "skipped (source folder missing)"
    os.makedirs(dst, exist_ok=True)
    cmd = [
        "robocopy", src, dst, "/MIR", "/R:1", "/W:1",
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
    record_success()
    if m:
        total, copied, skipped, _mismatch, failed, extras = (int(g) for g in m.groups())
        return (f"mirrored — {copied} copied · {skipped} unchanged"
                + (f" · {extras} removed at dest" if extras else "")
                + (f" · {failed} FAILED" if failed else ""))
    return "mirrored"
