import insightface
import numpy as np
import cv2
import os
import json
import db
from constants import FACE_DIR, SIMILARITY_THRESHOLD

os.makedirs(FACE_DIR, exist_ok=True)

# CPU + GPU fallback — ONNX picks best available provider automatically
_face_app = None

def _get_app():
    global _face_app
    if _face_app is None:
        _face_app = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app

def detect_and_embed_faces(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        faces = _get_app().get(img)
        return [{"bbox": f.bbox.tolist(), "embedding": f.embedding.tolist()} for f in faces]
    except Exception as e:
        print(f"Face detection error {image_path}: {e}")
        return []

def save_face_data(image_id, face_data):
    face_file = os.path.join(FACE_DIR, f"{image_id}.json")
    with open(face_file, 'w') as f:
        json.dump(face_data, f)

def load_face_data(image_id):
    face_file = os.path.join(FACE_DIR, f"{image_id}.json")
    if os.path.exists(face_file):
        with open(face_file, 'r') as f:
            return json.load(f)
    return []


# ── ANN face index (ChromaDB) ─────────────────────────────────────────────────
# Each detected face is one entry id'd "{image_id}:{face_index}" so person search
# is a vector query across ALL faces, not a per-file cosine scan over results.

def index_faces(image_id, face_data):
    """Upsert one image's faces into the ANN index (replacing any prior entries)."""
    col = db.faces_collection()
    try:
        col.delete(where={"image_id": image_id})
    except Exception:
        pass
    if not face_data:
        return
    ids, embs, metas = [], [], []
    for i, f in enumerate(face_data):
        emb = f.get("embedding")
        if not emb:
            continue
        ids.append(f"{image_id}:{i}")
        embs.append(emb)
        metas.append({"image_id": image_id, "face_index": i})
    if ids:
        col.upsert(ids=ids, embeddings=embs, metadatas=metas)


def delete_faces_for_image(image_id):
    try:
        db.faces_collection().delete(where={"image_id": image_id})
    except Exception:
        pass


def face_index_count():
    try:
        return db.faces_collection().count()
    except Exception:
        return 0


def query_person_faces(embedding, max_results=3000, distance_threshold=None):
    """Image ids whose best matching face is within the similarity threshold."""
    if distance_threshold is None:
        distance_threshold = 1.0 - SIMILARITY_THRESHOLD  # cosine distance = 1 - similarity
    col = db.faces_collection()
    n = col.count()
    if n == 0:
        # Backfill once from JSON for libraries indexed before the ANN index existed.
        if rebuild_face_index() == 0:
            return set()
        n = col.count()
        if n == 0:
            return set()
    res = col.query(query_embeddings=[embedding], n_results=min(max_results, n))
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    matched = set()
    for meta, dist in zip(metas, dists):
        if dist <= distance_threshold:
            matched.add(meta["image_id"])
    return matched


def rebuild_face_index(batch=500):
    """Backfill/rebuild the ANN index from the on-disk face JSON files."""
    col = db.faces_collection()
    try:
        existing = col.get()
        if existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass
    if not os.path.isdir(FACE_DIR):
        return 0
    ids, embs, metas, total = [], [], [], 0
    for fname in os.listdir(FACE_DIR):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        image_id = fname[:-5]
        try:
            with open(os.path.join(FACE_DIR, fname)) as fh:
                data = json.load(fh)
        except Exception:
            continue
        for i, item in enumerate(data):
            emb = item.get("embedding")
            if not emb:
                continue
            ids.append(f"{image_id}:{i}")
            embs.append(emb)
            metas.append({"image_id": image_id, "face_index": i})
            if len(ids) >= batch:
                col.upsert(ids=ids, embeddings=embs, metadatas=metas)
                total += len(ids); ids, embs, metas = [], [], []
    if ids:
        col.upsert(ids=ids, embeddings=embs, metadatas=metas)
        total += len(ids)
    return total
