import insightface
import numpy as np
import cv2
import os
import json
from constants import FACE_DIR

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
