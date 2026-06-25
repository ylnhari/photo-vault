"""
Face clustering: group detected face embeddings into candidate people using
DBSCAN, so the user can review the groups and name them. The cluster file is a
transient working set (regenerated each run); the durable outcome of naming a
cluster is a person added to person_map.json.
"""
import os
import json
import numpy as np
from sklearn.cluster import DBSCAN

from constants import FACE_DIR, DATA_DIR
from faces import load_face_data

CLUSTERS_FILE = os.path.join(DATA_DIR, "face_clusters.json")


def _load_all_faces():
    """Return [(image_id, face_index, embedding, bbox), ...] across all face files."""
    items = []
    if not os.path.isdir(FACE_DIR):
        return items
    for f in os.listdir(FACE_DIR):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        image_id = f[:-5]
        try:
            with open(os.path.join(FACE_DIR, f)) as fh:
                data = json.load(fh)
        except Exception:
            continue
        for idx, item in enumerate(data):
            emb = item.get("embedding")
            if emb:
                items.append((image_id, idx, emb, item.get("bbox")))
    return items


def _save_clusters(payload: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = CLUSTERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, CLUSTERS_FILE)


def load_clusters() -> dict:
    if os.path.exists(CLUSTERS_FILE):
        try:
            with open(CLUSTERS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"clusters": [], "params": {}, "total_faces": 0}


def cluster_faces(eps: float = 0.5, min_samples: int = 3) -> dict:
    """
    Run DBSCAN over all face embeddings (cosine). Noise (label -1) is dropped.
    Persists clusters sorted largest-first. Returns a summary.
    """
    items = _load_all_faces()
    if not items:
        _save_clusters({"clusters": [], "params": {"eps": eps, "min_samples": min_samples},
                        "total_faces": 0})
        return {"clusters": 0, "faces": 0, "noise": 0}

    X = np.array([it[2] for it in items])
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X).labels_

    groups: dict[int, list] = {}
    for it, label in zip(items, labels):
        label = int(label)
        if label == -1:
            continue  # unclustered / noise
        groups.setdefault(label, []).append(
            {"image_id": it[0], "face_index": it[1], "bbox": it[3]}
        )

    clusters = []
    for new_id, (_, members) in enumerate(
        sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ):
        clusters.append({
            "cluster_id": new_id,
            "size": len(members),
            "members": members,
            "name": None,
            "status": "new",   # new | named | ignored
        })

    _save_clusters({
        "clusters": clusters,
        "params": {"eps": eps, "min_samples": min_samples},
        "total_faces": len(items),
    })
    noise = int(np.sum(labels == -1))
    return {"clusters": len(clusters), "faces": len(items), "noise": noise}


def get_cluster(cluster_id: int) -> dict | None:
    for c in load_clusters().get("clusters", []):
        if c["cluster_id"] == cluster_id:
            return c
    return None


def cluster_mean_embedding(cluster_id: int) -> list | None:
    """Mean face embedding across a cluster's members (for person registration)."""
    cluster = get_cluster(cluster_id)
    if not cluster:
        return None
    embs = []
    for m in cluster["members"]:
        faces = load_face_data(m["image_id"])
        if 0 <= m["face_index"] < len(faces):
            embs.append(faces[m["face_index"]]["embedding"])
    if not embs:
        return None
    return np.mean(np.array(embs), axis=0).tolist()


def set_cluster_status(cluster_id: int, status: str, name: str | None = None):
    data = load_clusters()
    for c in data.get("clusters", []):
        if c["cluster_id"] == cluster_id:
            c["status"] = status
            if name is not None:
                c["name"] = name
            break
    _save_clusters(data)
