"""
Face clustering: group detected face embeddings into candidate people using
DBSCAN, so the user can review the groups and name them. The cluster file is a
transient working set (regenerated each run); the durable outcome of naming a
cluster is a person added to person_map.json.
"""
import hashlib
import os
import json
import numpy as np
from sklearn.cluster import DBSCAN

from constants import FACE_DIR, DATA_DIR
from faces import load_face_data

CLUSTERS_FILE = os.path.join(DATA_DIR, "face_clusters.json")

# How much of a newly-computed cluster's membership must already have been
# grouped together in the PREVIOUS run for it to inherit that previous
# cluster's name/status, instead of starting over as "new" (see
# _match_previous_cluster). A plain majority (>50%) is the threshold: if most
# of what's grouped together now was already reviewed together, it's almost
# certainly the same person even though DBSCAN's exact boundaries shifted
# (new photos indexed, eps/min_samples changed, etc).
CLUSTER_MATCH_MEMBER_OVERLAP = 0.5

# Fallback when member overlap isn't conclusive (e.g. a cluster gained/lost
# most of its members between runs): if the new cluster's mean embedding is
# within this cosine distance of a previous cluster's mean embedding, treat
# it as the same person. Tight on purpose — a false merge here would
# silently relabel one person's cluster with another's name.
CLUSTER_MATCH_CENTROID_DISTANCE = 0.1


class ClusterMembersStaleError(Exception):
    """Raised by cluster_mean_embedding when a cluster HAS recorded members,
    but every one of them is now stale — the face at that (image_id,
    face_index) position no longer matches what was originally clustered
    (see _member_is_stale), most likely because the image was re-processed
    with a different number/order of detected faces since the cluster was
    built. This is distinct from a cluster that genuinely has zero members:
    callers (api.py) should catch this and tell the user their review is out
    of date and a re-cluster is needed, rather than a bare "not found"."""

    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        super().__init__(
            f"Cluster {cluster_id} has recorded members, but all of them "
            "are stale (re-cluster to refresh)"
        )


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


def _member_key(member: dict):
    return (member["image_id"], member["face_index"])


def _cosine_distance(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def _embedding_fingerprint(embedding) -> str:
    """Cheap, non-cryptographic fingerprint of a face embedding, stored per
    cluster member at build time. Storing the full ~512-float vector for
    every member of every cluster (rather than just once per cluster, as the
    mean embedding) would bloat face_clusters.json a lot for not much
    benefit — a fingerprint is enough to detect "the embedding at this
    (image_id, face_index) position changed since we clustered it" (#18)
    without persisting the vector itself."""
    rounded = ",".join(f"{v:.6f}" for v in embedding)
    return hashlib.sha1(rounded.encode("utf-8")).hexdigest()


def _match_previous_cluster(new_members: list, new_centroid, previous_clusters: list):
    """Decide whether a newly-computed cluster should inherit an existing
    previous cluster's review state (name + status), so re-running
    clustering doesn't discard every "named"/"ignored" decision the user
    already made.

    Matching heuristic (first confident match wins):
      1. Member-set overlap — if more than CLUSTER_MATCH_MEMBER_OVERLAP of
         the new cluster's (image_id, face_index) members were already
         grouped together under some previous cluster, treat that as the
         same cluster. This is robust to DBSCAN's labeling changing between
         runs (cluster ids are not stable) and to a cluster picking up a
         handful of new photos.
      2. Centroid distance — if member overlap doesn't produce a confident
         match (e.g. the previous cluster file predates this field, or
         membership shifted a lot), fall back to comparing mean embeddings:
         a previous cluster whose centroid is within
         CLUSTER_MATCH_CENTROID_DISTANCE cosine distance is still very
         likely the same person.

    Returns the matching previous cluster dict, or None if nothing crosses
    either threshold — a genuinely new cluster with status "new".
    """
    new_keys = {_member_key(m) for m in new_members}

    best_overlap_cluster, best_overlap = None, 0.0
    for prev in previous_clusters:
        prev_keys = {_member_key(m) for m in prev.get("members", [])}
        if not prev_keys:
            continue
        shared = len(new_keys & prev_keys)
        if not shared:
            continue
        overlap = shared / len(new_keys)
        if overlap > best_overlap:
            best_overlap, best_overlap_cluster = overlap, prev
    if best_overlap_cluster is not None and best_overlap > CLUSTER_MATCH_MEMBER_OVERLAP:
        return best_overlap_cluster

    if new_centroid is None:
        return None
    best_centroid_cluster, best_distance = None, None
    for prev in previous_clusters:
        prev_centroid = prev.get("mean_embedding")
        if not prev_centroid:
            continue
        dist = _cosine_distance(new_centroid, prev_centroid)
        if best_distance is None or dist < best_distance:
            best_distance, best_centroid_cluster = dist, prev
    if best_centroid_cluster is not None and best_distance is not None \
            and best_distance <= CLUSTER_MATCH_CENTROID_DISTANCE:
        return best_centroid_cluster
    return None


def cluster_faces(eps: float = 0.5, min_samples: int = 3) -> dict:
    """
    Run DBSCAN over all face embeddings (cosine). Noise (label -1) is dropped.
    Persists clusters sorted largest-first. Returns a summary.

    Before overwriting, loads the PREVIOUS cluster file and carries over each
    previous cluster's name/status onto whichever new cluster matches it
    (see _match_previous_cluster) — otherwise every re-cluster would revert
    all review progress (named/ignored) back to "new", which defeats the
    point of persisting review state at all.
    """
    previous_clusters = load_clusters().get("clusters", [])

    items = _load_all_faces()
    if not items:
        _save_clusters({"clusters": [], "params": {"eps": eps, "min_samples": min_samples},
                        "total_faces": 0})
        return {"clusters": 0, "faces": 0, "noise": 0}

    X = np.array([it[2] for it in items])
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X).labels_

    groups: dict[int, list] = {}
    group_embeddings: dict[int, list] = {}
    for it, label, emb in zip(items, labels, X):
        label = int(label)
        if label == -1:
            continue  # unclustered / noise
        groups.setdefault(label, []).append(
            {
                "image_id": it[0],
                "face_index": it[1],
                "bbox": it[3],
                "fp": _embedding_fingerprint(it[2]),
            }
        )
        group_embeddings.setdefault(label, []).append(emb)

    clusters = []
    for new_id, (label, members) in enumerate(
        sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ):
        centroid = np.mean(np.array(group_embeddings[label]), axis=0).tolist()
        matched = _match_previous_cluster(members, centroid, previous_clusters)
        clusters.append({
            "cluster_id": new_id,
            "size": len(members),
            "members": members,
            "name": matched["name"] if matched else None,
            "status": matched["status"] if matched else "new",   # new | named | ignored
            "mean_embedding": centroid,
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
    """Mean face embedding across a cluster's members (for person registration).

    Cluster membership is stored as (image_id, positional face_index), which
    goes stale if that image is later re-processed with a different number
    or order of detected faces — position no longer means the same face. To
    catch that without a full stable-face-id redesign (a deeper data-model
    change than fits here — see module docstring / fix report), each member
    also carries a fingerprint of the embedding it had at cluster-build
    time; a member is skipped if the embedding currently at that position no
    longer matches.

    Returns None if the cluster has no members at all (or doesn't exist).
    Raises ClusterMembersStaleError if the cluster DOES have members but
    every one of them is now stale — a different situation the caller should
    surface differently (re-cluster needed) than a plain "nothing here".
    """
    cluster = get_cluster(cluster_id)
    if not cluster:
        return None
    if not cluster["members"]:
        return None

    embs = []
    for m in cluster["members"]:
        faces = load_face_data(m["image_id"])
        idx = m["face_index"]
        if not (0 <= idx < len(faces)):
            continue
        current_emb = faces[idx]["embedding"]
        expected_fp = m.get("fp")
        if expected_fp and _embedding_fingerprint(current_emb) != expected_fp:
            # The face now at this position isn't the one that was
            # clustered — don't silently blend an unrelated face in.
            continue
        embs.append(current_emb)

    if not embs:
        raise ClusterMembersStaleError(cluster_id)
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
