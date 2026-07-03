import os
import json
import numpy as np
from faces import detect_and_embed_faces
from constants import PERSON_MAP_PATH

def add_person_reference(person_name, image_dir):
    """Register a person from a folder of reference photos.
    Returns the number of faces used (0 → nothing registered)."""
    embeddings = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.heic')):
            path = os.path.join(image_dir, f)
            for face in detect_and_embed_faces(path):
                embeddings.append(face['embedding'])

    if not embeddings:
        print(f"No faces found for {person_name}.")
        return 0

    mean_embedding = np.mean(embeddings, axis=0).tolist()
    person_map = _load_map()
    person_map[person_name] = mean_embedding
    _save_map(person_map)
    print(f"Registered: {person_name}")
    return len(embeddings)

def add_person_embedding(person_name, embedding):
    """Register a person directly from a precomputed mean embedding (e.g. from a
    reviewed face cluster). Returns True on success."""
    if not person_name or embedding is None:
        return False
    person_map = _load_map()
    person_map[person_name] = list(embedding)
    _save_map(person_map)
    print(f"Registered (from cluster): {person_name}")
    return True


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
