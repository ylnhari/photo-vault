"""
Persistent folder configuration — which directories to include/exclude from scanning.
Stored in data/folders.json. Single source of truth for all scan targets.
"""
import os
import json
import platform
from datetime import datetime
from pathlib import Path

from constants import FOLDERS_CONFIG_PATH, DATA_DIR


# ── persistence ──────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(FOLDERS_CONFIG_PATH):
        try:
            with open(FOLDERS_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"included": [], "excluded": []}


def _save(cfg: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = FOLDERS_CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, FOLDERS_CONFIG_PATH)


# ── path helpers ─────────────────────────────────────────────────────────────

def _norm(path: str) -> str:
    """Resolve to absolute path. Preserves OS-native casing."""
    return str(Path(path).resolve())


def _normkey(path: str) -> str:
    """Case-folded comparison key (lowercase on Windows, unchanged on Unix)."""
    return os.path.normcase(_norm(path))


def _is_child(child: str, parent: str) -> bool:
    """True if child is strictly inside parent (not the same path). Case-insensitive on Windows."""
    ck = _normkey(child)
    pk = _normkey(parent)
    return ck != pk and (ck.startswith(pk + os.sep) or ck.startswith(pk + "/"))


# ── public read API ───────────────────────────────────────────────────────────

def get_config() -> dict:
    return _load()


def get_included() -> list[dict]:
    return _load().get("included", [])


def get_excluded_paths() -> list[str]:
    return [e["path"] for e in _load().get("excluded", [])]


def get_effective_scan_dirs() -> list[str]:
    """
    Minimal set of dirs to scan. Removes any folder already covered by a parent
    in the included list to prevent walking the same files twice.
    """
    paths = sorted([e["path"] for e in _load().get("included", [])])
    effective = []
    for p in paths:
        if not any(_is_child(p, parent) for parent in effective):
            effective.append(p)
    return effective


def get_defaults() -> list[str]:
    """OS-appropriate default photo directories that actually exist on this machine."""
    try:
        home = str(Path.home())
    except Exception:
        return []
    system = platform.system()

    if system == "Windows":
        candidates = [
            os.path.join(home, "Pictures"),
            os.path.join(home, "OneDrive", "Pictures"),
            os.path.join(home, "OneDrive", "Camera Roll"),
        ]
    elif system == "Darwin":
        candidates = [
            os.path.join(home, "Pictures"),
        ]
    else:  # Linux / other
        candidates = [
            os.path.join(home, "Pictures"),
        ]

    return [p for p in candidates if os.path.isdir(p)]


# ── bootstrap ─────────────────────────────────────────────────────────────────

def ensure_defaults() -> dict:
    """If no folders configured yet, seed with OS-default photo dirs that exist."""
    cfg = _load()
    if not cfg.get("included"):
        for p in get_defaults():
            _add_to_cfg(cfg, p)
        if cfg.get("included"):
            _save(cfg)
    return cfg


# ── mutations ─────────────────────────────────────────────────────────────────

def _add_to_cfg(cfg: dict, path: str) -> dict:
    """
    Add path to cfg["included"] with overlap resolution. Modifies cfg in-place.

    Returns a result dict:
      status   "added" | "duplicate" | "redundant" | "not_found"
      path     resolved path
      replaced list of child paths that were made redundant by this new parent
      covered_by  path of existing parent (when status=="redundant")
    """
    path = _norm(path)
    if not os.path.isdir(path):
        return {"status": "not_found", "path": path, "replaced": [], "covered_by": None}

    included = cfg.setdefault("included", [])
    existing_paths = [e["path"] for e in included]

    # Case-insensitive duplicate check
    if any(_normkey(path) == _normkey(e) for e in existing_paths):
        return {"status": "duplicate", "path": path, "replaced": [], "covered_by": None}

    # An existing folder already covers this one → adding it is redundant
    covering = next((e for e in existing_paths if _is_child(path, e)), None)
    if covering:
        return {"status": "redundant", "path": path, "replaced": [], "covered_by": covering}

    # Remove existing children now covered by this new parent
    redundant = [e for e in included if _is_child(e["path"], path)]
    cfg["included"] = [e for e in included if e not in redundant]

    cfg["included"].append({
        "path": path,
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "last_scanned_at": None,
        "image_count": 0,
    })

    return {
        "status": "added",
        "path": path,
        "replaced": [e["path"] for e in redundant],
        "covered_by": None,
    }


def add_included(path: str) -> dict:
    """Add a folder to the included list. Returns result + updated config."""
    cfg = _load()
    result = _add_to_cfg(cfg, path)
    if result["status"] == "added":
        _save(cfg)
    return {**result, "config": cfg}


def remove_included(path: str) -> dict:
    """Remove a folder from the included list (does NOT purge indexed data)."""
    path = _norm(path)
    pkey = _normkey(path)
    cfg = _load()
    cfg["included"] = [e for e in cfg.get("included", []) if _normkey(e["path"]) != pkey]
    _save(cfg)
    return {"path": path, "config": cfg}


def add_excluded(path: str) -> dict:
    """Mark a subfolder as excluded from scanning."""
    path = _norm(path)
    if not os.path.isdir(path):
        return {"status": "not_found", "path": path}
    pkey = _normkey(path)
    cfg = _load()
    excluded = cfg.setdefault("excluded", [])
    if any(_normkey(e["path"]) == pkey for e in excluded):
        return {"status": "duplicate", "path": path, "config": cfg}
    excluded.append({"path": path, "excluded_at": datetime.now().isoformat(timespec="seconds")})
    _save(cfg)
    return {"status": "added", "path": path, "config": cfg}


def remove_excluded(path: str) -> dict:
    """Remove a folder from the excluded list (re-enables scanning it)."""
    path = _norm(path)
    pkey = _normkey(path)
    cfg = _load()
    cfg["excluded"] = [e for e in cfg.get("excluded", []) if _normkey(e["path"]) != pkey]
    _save(cfg)
    return {"path": path, "config": cfg}


def update_scan_result(folder_path: str, image_count: int):
    """Record last_scanned_at and image_count after a successful scan of a folder."""
    folder_path = _norm(folder_path)
    fpkey = _normkey(folder_path)
    cfg = _load()
    for e in cfg.get("included", []):
        if _normkey(e["path"]) == fpkey:
            e["last_scanned_at"] = datetime.now().isoformat(timespec="seconds")
            e["image_count"] = image_count
            _save(cfg)
            return
