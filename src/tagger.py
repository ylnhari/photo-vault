import os
import json
import threading
import numpy as np
from faces import detect_and_embed_faces
from constants import PERSON_MAP_PATH

# Guards every person_map.json read-modify-write cycle below. Without it,
# two concurrent edits (e.g. register + rename in quick succession) can race:
# both read the same on-disk state, and whichever writes last silently wins,
# dropping the other's change.
_map_lock = threading.Lock()


def add_person_reference(person_name, image_dir):
    """Register a person from a folder of reference photos.

    Each reference image is expected to show exactly one person. An image
    where more than one face is detected (e.g. a group photo used by
    mistake) is skipped entirely rather than folded into the average — the
    safe default, since blending in a stranger's face would silently
    corrupt the resulting mean embedding with no way to tell afterward.

    Returns a structured result instead of a bare count so a caller can
    tell registration succeeded from registration being skipped, and can
    surface which reference images were skipped and why:
        {"registered": bool, "faces_used": int, "skipped_multi_face": [path, ...]}
    """
    person_name = (person_name or "").strip()
    embeddings = []
    skipped_multi_face = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.heic')):
            path = os.path.join(image_dir, f)
            faces = detect_and_embed_faces(path)
            if len(faces) > 1:
                print(f"Skipping {path}: {len(faces)} faces detected "
                      f"(reference images must show exactly one person).")
                skipped_multi_face.append(path)
                continue
            for face in faces:
                embeddings.append(face['embedding'])

    if not embeddings:
        print(f"No faces found for {person_name}.")
        return {"registered": False, "faces_used": 0, "skipped_multi_face": skipped_multi_face}

    mean_embedding = np.mean(embeddings, axis=0).tolist()
    with _map_lock:
        person_map = _load_map()
        person_map[person_name] = mean_embedding
        _save_map(person_map)
    print(f"Registered: {person_name}")
    return {"registered": True, "faces_used": len(embeddings), "skipped_multi_face": skipped_multi_face}

def add_person_embedding(person_name, embedding):
    """Register a person directly from a precomputed mean embedding (e.g. from a
    reviewed face cluster). Returns True on success."""
    person_name = (person_name or "").strip()
    if not person_name or embedding is None:
        return False
    with _map_lock:
        person_map = _load_map()
        person_map[person_name] = list(embedding)
        _save_map(person_map)
    print(f"Registered (from cluster): {person_name}")
    return True


def rename_person(old_name, new_name):
    """Rename a registered person. Raises KeyError/ValueError on bad input."""
    new_name = (new_name or "").strip()
    with _map_lock:
        person_map = _load_map()
        if old_name not in person_map:
            raise KeyError(old_name)
        if not new_name:
            raise ValueError("new name required")
        if new_name in person_map:
            raise ValueError(f"'{new_name}' already exists")
        person_map[new_name] = person_map.pop(old_name)
        _save_map(person_map)


def delete_person(person_name) -> bool:
    """Unregister a person (their photos stay indexed). True if they existed."""
    with _map_lock:
        person_map = _load_map()
        existed = person_name in person_map
        person_map.pop(person_name, None)
        if existed:
            _save_map(person_map)
    return existed


def get_person_embedding(person_name):
    return _load_map().get(person_name)

def get_all_persons():
    return list(_load_map().keys())

def _load_map():
    if os.path.exists(PERSON_MAP_PATH):
        with open(PERSON_MAP_PATH, 'r') as f:
            return json.load(f)
    return {}

def _save_map(person_map):
    os.makedirs(os.path.dirname(PERSON_MAP_PATH), exist_ok=True)
    with open(PERSON_MAP_PATH, 'w') as f:
        json.dump(person_map, f, indent=4)
