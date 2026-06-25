"""
Albums: named, ordered collections of photos (by image id). Stored in
data/albums.json. Membership is just ids — an image can be in many albums, and
deleting an image removes it from all of them.
"""
import os
import json
import uuid
from datetime import datetime

from constants import ALBUMS_PATH, DATA_DIR


def _load() -> dict:
    if os.path.exists(ALBUMS_PATH):
        try:
            with open(ALBUMS_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict) and "albums" in data:
                return data
        except Exception:
            pass
    return {"albums": {}}


def _save(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = ALBUMS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, ALBUMS_PATH)


def list_albums() -> list[dict]:
    data = _load()
    out = []
    for aid, a in data["albums"].items():
        ids = a.get("image_ids", [])
        out.append({
            "id": aid,
            "name": a.get("name", ""),
            "count": len(ids),
            "cover": ids[0] if ids else None,
            "created_at": a.get("created_at"),
        })
    out.sort(key=lambda x: x.get("created_at") or "")
    return out


def get_album(album_id: str) -> dict | None:
    return _load()["albums"].get(album_id)


def create_album(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("album name is required")
    data = _load()
    aid = uuid.uuid4().hex[:12]
    data["albums"][aid] = {
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_ids": [],
    }
    _save(data)
    return {"id": aid, "name": name, "count": 0, "cover": None}


def rename_album(album_id: str, name: str):
    name = (name or "").strip()
    if not name:
        raise ValueError("album name is required")
    data = _load()
    if album_id not in data["albums"]:
        raise KeyError(album_id)
    data["albums"][album_id]["name"] = name
    _save(data)


def delete_album(album_id: str) -> bool:
    data = _load()
    if album_id in data["albums"]:
        del data["albums"][album_id]
        _save(data)
        return True
    return False


def add_to_album(album_id: str, image_ids: list[str]) -> int:
    data = _load()
    if album_id not in data["albums"]:
        raise KeyError(album_id)
    current = data["albums"][album_id]["image_ids"]
    seen = set(current)
    for iid in image_ids:
        if iid not in seen:
            current.append(iid)
            seen.add(iid)
    _save(data)
    return len(current)


def remove_from_album(album_id: str, image_ids: list[str]) -> int:
    data = _load()
    if album_id not in data["albums"]:
        raise KeyError(album_id)
    rm = set(image_ids)
    kept = [i for i in data["albums"][album_id]["image_ids"] if i not in rm]
    data["albums"][album_id]["image_ids"] = kept
    _save(data)
    return len(kept)


def remove_image_from_all(image_id: str):
    """Drop an image from every album (called when the image is deleted)."""
    data = _load()
    changed = False
    for a in data["albums"].values():
        if image_id in a["image_ids"]:
            a["image_ids"] = [i for i in a["image_ids"] if i != image_id]
            changed = True
    if changed:
        _save(data)
