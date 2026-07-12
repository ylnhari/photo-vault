import insightface
import onnxruntime as ort
import numpy as np
import cv2
import os
import json
from PIL import Image, ImageOps

import db
import catalog_db
import settings as settings_mod
from constants import FACE_DIR, SIMILARITY_THRESHOLD, IMAGE_CATALOG_PATH, DATA_DIR

os.makedirs(FACE_DIR, exist_ok=True)

# Optional Intel GPU/NPU acceleration: if the OpenVINO runtime is installed
# (`openvino` package, an opt-in extra — see README), importing it HERE, before
# any onnxruntime InferenceSession is built, registers its DLL directory so
# onnxruntime's OpenVINO execution-provider bridge can load openvino.dll. It's a
# harmless no-op when OpenVINO isn't installed (the CPU-only default). This must
# run before _get_app(); onnxruntime resolves the provider DLL's dependencies at
# session-creation time, not import time.
try:
    import openvino as _openvino  # noqa: F401
except Exception:
    _openvino = None

# OpenVINO compiles each model for the target device on first use (tens of
# seconds for GPU/NPU); a persistent cache turns that into a one-time cost.
_OV_CACHE_DIR = os.path.join(DATA_DIR, "ov_cache")

# ── Execution-provider (accelerator) selection ───────────────────────────────
# Face detection runs on whatever execution providers the INSTALLED onnxruntime
# wheel exposes — which varies by machine (plain CPU wheel, onnxruntime-gpu,
# onnxruntime-directml, onnxruntime-openvino, …). We don't hardcode a provider
# anymore: we enumerate what's actually available at runtime (ort.get_available_
# providers() + OpenVINO's own device list) and let the user pick one in
# Settings → "face_provider". Whatever they pick, CPU is always appended as a
# guaranteed fallback so an unavailable/failed accelerator degrades instead of
# crashing the faces job. See available_accelerators() for the option list the
# UI renders and _resolve_providers() for choice → onnxruntime args.

# Raw EPs that are never offered as a user choice: Azure is a cloud model-serving
# EP (not a local accelerator), and TensorRT needs an engine-build step
# InsightFace doesn't perform. Everything else the wheel reports is fair game.
_HIDDEN_PROVIDERS = {"AzureExecutionProvider", "TensorrtExecutionProvider"}

# Friendly labels for OpenVINO physical devices (it exposes ONE onnxruntime EP
# but several devices — iGPU / NPU / CPU — each of which we surface separately).
_OPENVINO_DEVICE_LABELS = {
    "GPU": "Intel GPU (OpenVINO)",
    "NPU": "Intel NPU (OpenVINO)",
    "CPU": "CPU (OpenVINO)",
    "AUTO": "OpenVINO Auto-select",
}


def _openvino_devices():
    """Physical OpenVINO devices (e.g. ['CPU','GPU','NPU']). [] when OpenVINO
    isn't installed or can't enumerate — callers treat that as 'no OpenVINO'."""
    try:
        import openvino as ov  # only present with onnxruntime-openvino / openvino
        return list(ov.Core().available_devices)
    except Exception:
        return []


def available_accelerators():
    """The accelerators a user can choose for face detection, derived from the
    installed onnxruntime build + OpenVINO device list. Always starts with
    'auto' and always includes 'cpu' (the guaranteed fallback). Each entry:
    {id, label, provider, device}. Pure detection — no 'selected'/'active'
    state (the API layer adds that)."""
    eps = set(ort.get_available_providers())
    opts = [{"id": "auto", "label": "Auto (fastest available)",
             "provider": None, "device": None}]
    if "OpenVINOExecutionProvider" in eps:
        for dev in _openvino_devices():
            base = dev.split(".")[0]  # 'GPU.0' → 'GPU'
            opts.append({
                "id": f"openvino:{dev}",
                "label": _OPENVINO_DEVICE_LABELS.get(base, f"OpenVINO {dev}"),
                "provider": "OpenVINOExecutionProvider", "device": dev,
            })
    if "CUDAExecutionProvider" in eps:
        opts.append({"id": "cuda", "label": "NVIDIA GPU (CUDA)",
                     "provider": "CUDAExecutionProvider", "device": None})
    if "DmlExecutionProvider" in eps:
        opts.append({"id": "dml", "label": "GPU (DirectML)",
                     "provider": "DmlExecutionProvider", "device": None})
    # Any other real EP the wheel exposes (e.g. ROCm) — offer it generically so
    # detection stays future-proof rather than silently dropping it.
    known = {"OpenVINOExecutionProvider", "CUDAExecutionProvider",
             "DmlExecutionProvider", "CPUExecutionProvider"} | _HIDDEN_PROVIDERS
    for ep in ort.get_available_providers():
        if ep not in known:
            opts.append({"id": ep, "label": ep.replace("ExecutionProvider", ""),
                         "provider": ep, "device": None})
    opts.append({"id": "cpu", "label": "CPU",
                 "provider": "CPUExecutionProvider", "device": None})
    return opts


_CPU = "CPUExecutionProvider"


def _auto_choice():
    """Best available accelerator id for 'auto': a dedicated NVIDIA GPU first,
    then the Intel NPU, then DirectML, then the Intel iGPU, else CPU.

    The NPU outranks the iGPU here on purpose: on Intel Core Ultra hardware the
    NPU (AI Boost) is purpose-built for sustained CNN inference and measured
    ~14× faster than CPU for InsightFace detection, while the integrated GPU was
    actually SLOWER than CPU for these small models (kernel-launch/transfer
    overhead dominates, plus a long first-run compile). OpenVINO handles the
    FP32 models fine on the NPU. A machine where the iGPU wins can still pick it
    explicitly in Settings."""
    accels = {a["id"]: a for a in available_accelerators()}
    ov = "OpenVINOExecutionProvider" in ort.get_available_providers()
    ov_devs = _openvino_devices()
    if "cuda" in accels:
        return "cuda"
    if ov and "NPU" in ov_devs:
        return "openvino:NPU"
    if "dml" in accels:
        return "dml"
    if ov and "GPU" in ov_devs:
        return "openvino:GPU"
    return "cpu"


def _resolve_providers(choice):
    """Map a saved face_provider id → (providers, provider_options) for
    onnxruntime, always ending in CPU as a guaranteed fallback. Unknown or
    unavailable choices degrade to CPU rather than raising."""
    if not choice or choice == "auto":
        choice = _auto_choice()
    eps = set(ort.get_available_providers())
    if choice.startswith("openvino:") and "OpenVINOExecutionProvider" in eps:
        dev = choice.split(":", 1)[1]
        if dev in _openvino_devices():
            try:
                os.makedirs(_OV_CACHE_DIR, exist_ok=True)
            except OSError:
                pass
            return (["OpenVINOExecutionProvider", _CPU],
                    [{"device_type": dev, "cache_dir": _OV_CACHE_DIR}, {}])
    elif choice == "cuda" and "CUDAExecutionProvider" in eps:
        return (["CUDAExecutionProvider", _CPU], [{}, {}])
    elif choice == "dml" and "DmlExecutionProvider" in eps:
        return (["DmlExecutionProvider", _CPU], [{}, {}])
    elif choice not in ("cpu", "auto") and choice in eps:
        return ([choice, _CPU], [{}, {}])
    return ([_CPU], [{}])


def resolved_provider_label(choice=None):
    """Human-readable name of the accelerator a given choice (default: the
    saved setting) would actually run on — for the UI to show without paying
    the cost of building the model. Reflects fallback: an unavailable choice
    reports 'CPU'."""
    if choice is None:
        choice = settings_mod.load().get("face_provider", "auto")
    providers, options = _resolve_providers(choice)
    primary = providers[0]
    if primary == "OpenVINOExecutionProvider":
        dev = options[0].get("device_type", "")
        return _OPENVINO_DEVICE_LABELS.get(dev.split(".")[0], f"OpenVINO {dev}")
    for a in available_accelerators():
        if a["provider"] == primary and a["device"] is None:
            return a["label"]
    return primary.replace("ExecutionProvider", "")


# The built FaceAnalysis app is cached, keyed by the accelerator choice it was
# built with — if the user changes face_provider, _get_app() rebuilds on the
# next call instead of serving a stale session on the old device.
_face_app = None
_face_app_choice = None


def reset_face_app():
    """Drop the cached model so the next _get_app() rebuilds (e.g. after the
    accelerator setting changed)."""
    global _face_app, _face_app_choice
    _face_app = None
    _face_app_choice = None


def _get_app():
    global _face_app, _face_app_choice
    desired = settings_mod.load().get("face_provider", "auto")
    if _face_app is not None and _face_app_choice == desired:
        return _face_app
    providers, provider_options = _resolve_providers(desired)
    app = insightface.app.FaceAnalysis(
        name='buffalo_l',
        providers=providers,
        provider_options=provider_options,
    )
    # ctx_id=0 targets the first accelerator device; onnxruntime still honors
    # the providers list above, so a CPU-only resolution runs on CPU regardless.
    app.prepare(ctx_id=0, det_size=(640, 640))
    _face_app = app
    _face_app_choice = desired
    try:
        actual = app.models[next(iter(app.models))].session.get_providers()
        print(f"[faces] face detection running on: {actual} (choice={desired!r})")
    except Exception:
        pass
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

# Within-video face grouping threshold (cosine similarity of embeddings). Lower
# than the person-search SIMILARITY_THRESHOLD on purpose: within one clip we want
# to MERGE the same person seen across frames/angles (so the per-video face set
# reflects distinct PEOPLE, not raw detections), and merging a bit too eagerly is
# safer here than splitting one person into several.
VIDEO_FACE_GROUP_SIM = 0.45


def _bbox_area(face: dict) -> float:
    b = face.get("bbox") or [0, 0, 0, 0]
    try:
        return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    except Exception:
        return 0.0


def group_faces(faces: list[dict], threshold: float = VIDEO_FACE_GROUP_SIM) -> list[dict]:
    """Collapse many face detections (e.g. the same people across a video's
    keyframes) into ONE representative per distinct person, by greedily
    clustering on embedding cosine similarity. The representative is the
    largest-bbox detection in each group (usually the sharpest, most frontal).
    This is the sparse-keyframe stand-in for face tracking: it turns 'N raw
    detections' into 'the distinct people in this clip'. Faces without a usable
    embedding are dropped."""
    valid = [f for f in faces if f.get("embedding")]
    if len(valid) <= 1:
        return valid
    embs = np.array([f["embedding"] for f in valid], dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embs / norms

    groups: list[dict] = []  # {"members": [idx...], "centroid": unit-vector}
    for i in range(len(valid)):
        v = unit[i]
        best_sim, best_g = -1.0, -1
        for gi, g in enumerate(groups):
            sim = float(np.dot(v, g["centroid"]))
            if sim > best_sim:
                best_sim, best_g = sim, gi
        if best_g >= 0 and best_sim >= threshold:
            g = groups[best_g]
            g["members"].append(i)
            m = unit[g["members"]].mean(axis=0)
            n = np.linalg.norm(m) or 1.0
            g["centroid"] = m / n
        else:
            groups.append({"members": [i], "centroid": v})

    reps = []
    for g in groups:
        members = [valid[i] for i in g["members"]]
        reps.append(max(members, key=_bbox_area))
    return reps


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
