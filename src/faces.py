import insightface
import numpy as np
import cv2
import os
import json
from PIL import Image, ImageOps

import db
import catalog_db
from constants import FACE_DIR, SIMILARITY_THRESHOLD, IMAGE_CATALOG_PATH

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


# EXIF orientation tag (274) → the cv2 rotation that reaches display-correct
# orientation, mirroring what PIL.ImageOps.exif_transpose()/cv2.imread already
# do for the other two decode paths below.
_EXIF_ROTATE_FOR_ORIENTATION = {
    3: cv2.ROTATE_180,
    6: cv2.ROTATE_90_CLOCKWISE,
    8: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _apply_exif_rotation_cv2(img, image_path):
    """Rotate a RAW (orientation-ignored) cv2.imdecode() array to match the
    EXIF orientation tag. img must come from a decode that did NOT already
    apply orientation correction (see _read_image_bgr, which decodes with
    IMREAD_IGNORE_ORIENTATION specifically so this step is deterministic —
    whether cv2.imdecode auto-corrects EXIF orientation on its own varies by
    OpenCV build/version, so relying on that default would make the result
    depend on which OpenCV happens to be installed)."""
    try:
        orientation = Image.open(image_path).getexif().get(274)
    except Exception:
        orientation = None
    rotate_code = _EXIF_ROTATE_FOR_ORIENTATION.get(orientation)
    if rotate_code is not None:
        img = cv2.rotate(img, rotate_code)
    return img


def _read_image_bgr(image_path):
    """cv2.imread returns None for non-ASCII Windows paths and formats cv2
    can't decode (HEIC) — fall back to byte-decode, then PIL.

    All three decode paths below MUST agree on "rotated to display-correct
    orientation" as the coordinate space face bounding boxes are computed
    in — the same orientation PIL.ImageOps.exif_transpose() would produce.
    cv2.imread applies EXIF orientation automatically. The imdecode fallback
    decodes with IMREAD_IGNORE_ORIENTATION and applies the rotation
    ourselves instead of trusting the decoder's own default (whether
    cv2.imdecode auto-corrects varies by OpenCV build), and the PIL fallback
    calls exif_transpose() explicitly since PIL never rotates on its own.
    This is a contract callers that later re-derive a face crop from the
    original file (e.g. the API's face-crop endpoint) must also honor.
    """
    img = cv2.imread(image_path)
    if img is not None:
        return img
    try:
        data = np.fromfile(image_path, dtype=np.uint8)  # unicode-path safe
        # IMREAD_IGNORE_ORIENTATION forces a raw (un-rotated) decode so the
        # EXIF correction below is applied exactly once, deterministically —
        # not dependent on whether this OpenCV build's default already does it.
        img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            return _apply_exif_rotation_cv2(img, image_path)
    except Exception:
        pass
    try:
        from imaging import safe_open  # registers the HEIF opener
        with safe_open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB → BGR
    except Exception:
        return None


def detect_and_embed_faces(image_path):
    """Detect faces and return their bboxes/embeddings (display-correct,
    EXIF-rotated coordinate space — see _read_image_bgr).

    Returns [] only for genuinely expected "this file can't be processed"
    situations (missing/corrupt/unreadable image, or a recognized detector
    failure on a pathological image). Anything else propagates so the
    caller's per-image failure handling (indexer.py/jobs.py) can count a
    truly unexpected error as a real failure instead of silently recording
    "processed, 0 faces"."""
    img = _read_image_bgr(image_path)
    if img is None:
        return []
    try:
        faces = _get_app().get(img)
    except (cv2.error, RuntimeError, ValueError) as e:
        # Expected failure modes from OpenCV/InsightFace/onnxruntime on a
        # malformed or pathological (but readable) image.
        print(f"Face detection error {image_path}: {e}")
        return []
    return [{"bbox": f.bbox.tolist(), "embedding": f.embedding.tolist()} for f in faces]

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
    except Exception as e:
        # Not "nothing to delete" (Chroma no-ops that silently) — a genuine
        # failure here risks stale + new face vectors coexisting for this
        # image once we upsert below, so make it loud instead of swallowed.
        print(f"[faces] index_faces: delete of prior entries for {image_id} "
              f"failed ({e}); old and new face vectors may coexist")
    else:
        # Cheap verification that the delete actually took effect, rather
        # than trusting a non-raising call blindly.
        try:
            leftover = col.get(where={"image_id": image_id}, include=[])
            leftover_ids = leftover.get("ids") or []
            if leftover_ids:
                print(f"[faces] index_faces: {len(leftover_ids)} stale face "
                      f"entries still present for {image_id} after delete")
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
    except Exception as e:
        print(f"[faces] delete_faces_for_image: failed to delete faces for {image_id}: {e}")


def face_index_count():
    try:
        return db.faces_collection().count()
    except Exception:
        return 0


# Practical cap on how many nearest-neighbor face matches a person query
# considers. This is a performance guard, not a correctness boundary: for a
# personal-library-scale face index (tens of thousands of faces, not
# millions) holding this many ids in memory is cheap, so the cap is set high
# enough that a genuine match ranked outside it should be very rare. Raise
# further if a library's face count approaches this value.
MAX_PERSON_QUERY_RESULTS = 10000


def query_person_faces(embedding, max_results=MAX_PERSON_QUERY_RESULTS, distance_threshold=None):
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


def _live_catalog_ids():
    """Read-only lookup of every image id currently in the catalog, used to
    detect orphaned face JSON files (left behind by deleted/moved photos)
    during a rebuild. Returns None if the catalog can't be read at all
    (rather than an empty set), so callers can distinguish "no photos in the
    catalog" from "couldn't check" and skip the orphan check safely."""
    try:
        return set(catalog_db.load_all(IMAGE_CATALOG_PATH).get("images", {}).keys())
    except Exception as e:
        print(f"[faces] rebuild_face_index: could not read catalog for orphan "
              f"cross-check ({e}); skipping orphan detection this run")
        return None


def rebuild_face_index(batch=500):
    """Backfill/rebuild the ANN index from the on-disk face JSON files.

    Cross-checks each face file's image id against the live catalog first —
    without this, a face JSON left behind by a deleted/moved photo (a "ghost
    face") would resurrect into the ANN index and into clustering on every
    rebuild, even though the photo it came from no longer exists.
    """
    col = db.faces_collection()
    try:
        existing = col.get()
        if existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass
    if not os.path.isdir(FACE_DIR):
        return 0

    catalog_ids = _live_catalog_ids()

    ids, embs, metas, total, orphaned = [], [], [], 0, 0
    for fname in os.listdir(FACE_DIR):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        image_id = fname[:-5]
        if catalog_ids is not None and image_id not in catalog_ids:
            orphaned += 1
            try:
                os.remove(os.path.join(FACE_DIR, fname))
            except Exception as e:
                print(f"[faces] rebuild_face_index: could not remove orphaned "
                      f"face file {fname}: {e}")
            continue
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
    if orphaned:
        print(f"[faces] rebuild_face_index: removed {orphaned} orphaned face "
              f"file(s) with no matching catalog entry")
    return total
