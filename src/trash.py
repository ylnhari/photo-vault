"""
Soft-delete for indexed photos. "Removing" a photo moves its catalog entry to
data/trash.json (search vectors are dropped so results stay clean, but the
caption and derived files are kept, so restore is cheap: the photo re-enters
the catalog and just needs re-embedding). Purging the trash does the permanent
cleanup. File deletion uses the OS trash (Windows Recycle Bin, macOS Finder
Trash, freedesktop Trash on Linux — see platformfs) so even that is
recoverable outside the app.
"""
import json
import os
import threading
import time

import platformfs
from constants import DATA_DIR

TRASH_PATH = os.path.join(DATA_DIR, "trash.json")

# Serializes the read-modify-write cycle around TRASH_PATH. Concurrent add()
# calls for different image ids (e.g. a multi-select bulk delete racing
# against a single-image delete, or the API thread vs. a job) can otherwise
# lose one call's trash record entirely, defeating undo. Same-process only,
# per this app's single-user-local-tool scope.
_lock = threading.Lock()


def _load() -> dict:
    if os.path.exists(TRASH_PATH):
        try:
            with open(TRASH_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = TRASH_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, TRASH_PATH)


def add(img_id: str, img_data: dict, file_deleted: bool = False):
    with _lock:
        data = _load()
        data[img_id] = {
            "entry": img_data,
            "deleted_at": time.time(),
            "file_deleted": file_deleted,
        }
        _save(data)


def list_items() -> dict:
    return _load()


def take(img_ids: list[str]) -> dict[str, dict]:
    """Remove the given ids from the trash and return {id: catalog entry}."""
    with _lock:
        data = _load()
        out = {}
        for iid in img_ids:
            item = data.pop(iid, None)
            if item:
                out[iid] = item["entry"]
        _save(data)
        return out


def purge(img_ids: list[str] | None = None) -> list[str]:
    """Drop entries permanently (all when img_ids is None). Returns dropped ids."""
    with _lock:
        data = _load()
        ids = list(data.keys()) if img_ids is None else [i for i in img_ids if i in data]
        for iid in ids:
            del data[iid]
        _save(data)
        return ids


def delete_file_to_recycle_bin(path: str) -> bool:
    """
    Delete a file recoverably via the OS trash (Recycle Bin / Finder Trash /
    freedesktop Trash — platformfs picks the implementation). Returns True
    when the trash path was used; False means the caller should fall back to
    os.remove. Name kept for the existing call sites; behavior is now
    cross-platform.
    """
    return platformfs.move_to_trash(path)
