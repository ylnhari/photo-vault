import json
from unittest.mock import patch

import pytest


def _write_face(face_dir, image_id, faces):
    (face_dir / f"{image_id}.json").write_text(json.dumps(faces))


def _patched(face_dir, clusters_file):
    """Patch both clustering's own FACE_DIR (used by _load_all_faces) and
    faces.FACE_DIR (used by faces.load_face_data, which cluster_mean_embedding
    calls) to the same tmp directory, plus clustering's CLUSTERS_FILE."""
    return (
        patch("clustering.FACE_DIR", str(face_dir)),
        patch("faces.FACE_DIR", str(face_dir)),
        patch("clustering.CLUSTERS_FILE", str(clusters_file)),
    )


# ── cluster_faces basics ─────────────────────────────────────────────────────

def test_cluster_faces_no_faces_returns_empty(tmp_path):
    import clustering
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    clusters_file = tmp_path / "face_clusters.json"

    p1, p2, p3 = _patched(face_dir, clusters_file)
    with p1, p2, p3:
        summary = clustering.cluster_faces(eps=0.5, min_samples=1)

    assert summary == {"clusters": 0, "faces": 0, "noise": 0}


def test_cluster_faces_groups_similar_embeddings(tmp_path):
    import clustering
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    clusters_file = tmp_path / "face_clusters.json"
    _write_face(face_dir, "imgA1", [{"embedding": [1.0, 0.0, 0.0], "bbox": [0, 0, 1, 1]}])
    _write_face(face_dir, "imgA2", [{"embedding": [0.95, 0.05, 0.0], "bbox": [0, 0, 1, 1]}])
    _write_face(face_dir, "imgB1", [{"embedding": [0.0, 1.0, 0.0], "bbox": [0, 0, 1, 1]}])
    _write_face(face_dir, "imgB2", [{"embedding": [0.0, 0.95, 0.05], "bbox": [0, 0, 1, 1]}])

    p1, p2, p3 = _patched(face_dir, clusters_file)
    with p1, p2, p3:
        summary = clustering.cluster_faces(eps=0.3, min_samples=1)
        clusters = clustering.load_clusters()["clusters"]

    assert summary["clusters"] == 2
    assert summary["faces"] == 4
    sizes = sorted(c["size"] for c in clusters)
    assert sizes == [2, 2]
    for c in clusters:
        assert c["status"] == "new"
        assert c["name"] is None
        assert "mean_embedding" in c
        assert all("fp" in m for m in c["members"])


# ── #17: re-clustering preserves review state ───────────────────────────────

def test_reclustering_preserves_name_and_status_across_runs(tmp_path):
    """The most important fix in clustering.py: naming/ignoring a cluster
    must survive a later re-cluster, even though DBSCAN's cluster ids are
    not stable and the cluster's membership can grow."""
    import clustering
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    clusters_file = tmp_path / "face_clusters.json"
    _write_face(face_dir, "imgA1", [{"embedding": [1.0, 0.0, 0.0], "bbox": [0, 0, 1, 1]}])
    _write_face(face_dir, "imgA2", [{"embedding": [0.95, 0.05, 0.0], "bbox": [0, 0, 1, 1]}])
    _write_face(face_dir, "imgB1", [{"embedding": [0.0, 1.0, 0.0], "bbox": [0, 0, 1, 1]}])
    _write_face(face_dir, "imgB2", [{"embedding": [0.0, 0.95, 0.05], "bbox": [0, 0, 1, 1]}])

    p1, p2, p3 = _patched(face_dir, clusters_file)
    with p1, p2, p3:
        clustering.cluster_faces(eps=0.3, min_samples=1)
        clusters = clustering.load_clusters()["clusters"]
        a_cluster = next(c for c in clusters if
                          {"imgA1", "imgA2"} <= {m["image_id"] for m in c["members"]})
        clustering.set_cluster_status(a_cluster["cluster_id"], "named", name="Alice")

        # A third "A" photo gets indexed, and the library is re-clustered —
        # this used to blow away the "Alice" naming above.
        _write_face(face_dir, "imgA3", [{"embedding": [0.97, 0.03, 0.0], "bbox": [0, 0, 1, 1]}])
        clustering.cluster_faces(eps=0.3, min_samples=1)
        clusters_after = clustering.load_clusters()["clusters"]

    new_a_cluster = next(c for c in clusters_after if
                          {"imgA1", "imgA2", "imgA3"} <= {m["image_id"] for m in c["members"]})
    new_b_cluster = next(c for c in clusters_after if
                          {"imgB1", "imgB2"} <= {m["image_id"] for m in c["members"]})

    assert new_a_cluster["name"] == "Alice"
    assert new_a_cluster["status"] == "named"
    assert new_a_cluster["size"] == 3
    # The untouched "B" cluster must not have been mistakenly matched/renamed.
    assert new_b_cluster["name"] is None
    assert new_b_cluster["status"] == "new"


def test_reclustering_new_unrelated_cluster_stays_new(tmp_path):
    import clustering
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    clusters_file = tmp_path / "face_clusters.json"
    _write_face(face_dir, "imgA1", [{"embedding": [1.0, 0.0, 0.0], "bbox": [0, 0, 1, 1]}])

    p1, p2, p3 = _patched(face_dir, clusters_file)
    with p1, p2, p3:
        clustering.cluster_faces(eps=0.3, min_samples=1)
        clusters = clustering.load_clusters()["clusters"]
        clustering.set_cluster_status(clusters[0]["cluster_id"], "named", name="Alice")

        # A completely unrelated person appears for the first time.
        _write_face(face_dir, "imgC1", [{"embedding": [0.0, 0.0, 1.0], "bbox": [0, 0, 1, 1]}])
        clustering.cluster_faces(eps=0.3, min_samples=1)
        clusters_after = clustering.load_clusters()["clusters"]

    c_cluster = next(c for c in clusters_after if
                      any(m["image_id"] == "imgC1" for m in c["members"]))
    assert c_cluster["status"] == "new"
    assert c_cluster["name"] is None


# ── get_cluster / cluster_mean_embedding ────────────────────────────────────

def test_get_cluster_unknown_id_returns_none(tmp_path):
    import clustering
    clusters_file = tmp_path / "face_clusters.json"
    clusters_file.write_text(json.dumps({"clusters": [], "params": {}, "total_faces": 0}))
    with patch("clustering.CLUSTERS_FILE", str(clusters_file)):
        assert clustering.get_cluster(0) is None


def test_cluster_mean_embedding_zero_members_returns_none(tmp_path):
    """A cluster that genuinely has no members (as opposed to members that
    are all stale) returns None, not an exception (#19)."""
    import clustering
    clusters_file = tmp_path / "face_clusters.json"
    clusters_file.write_text(json.dumps({
        "clusters": [{"cluster_id": 0, "size": 0, "members": [], "name": None, "status": "new"}],
        "params": {}, "total_faces": 0,
    }))
    with patch("clustering.CLUSTERS_FILE", str(clusters_file)):
        assert clustering.cluster_mean_embedding(0) is None


def test_cluster_mean_embedding_averages_members(tmp_path):
    import clustering
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    clusters_file = tmp_path / "face_clusters.json"
    _write_face(face_dir, "imgA1", [{"embedding": [1.0, 0.0], "bbox": [0, 0, 1, 1]}])
    _write_face(face_dir, "imgA2", [{"embedding": [0.9, 0.1], "bbox": [0, 0, 1, 1]}])

    p1, p2, p3 = _patched(face_dir, clusters_file)
    with p1, p2, p3:
        clustering.cluster_faces(eps=0.3, min_samples=1)
        cluster_id = clustering.load_clusters()["clusters"][0]["cluster_id"]
        mean = clustering.cluster_mean_embedding(cluster_id)

    assert mean == pytest.approx([0.95, 0.05])


def test_cluster_mean_embedding_stale_all_members_raises(tmp_path):
    """#18: if the image behind a cluster's only member was later
    re-processed with a different face at that position, that must be
    distinguishable from a real zero-member cluster (#19) — the caller
    (api.py) needs to tell the user to re-cluster, not just "not found"."""
    import clustering
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    clusters_file = tmp_path / "face_clusters.json"
    _write_face(face_dir, "imgX", [{"embedding": [1.0, 0.0, 0.0], "bbox": [0, 0, 1, 1]}])

    p1, p2, p3 = _patched(face_dir, clusters_file)
    with p1, p2, p3:
        clustering.cluster_faces(eps=0.3, min_samples=1)
        cluster_id = clustering.load_clusters()["clusters"][0]["cluster_id"]

        # Simulate imgX being re-indexed with a different face at the same
        # position — the fingerprint recorded at cluster-build time no
        # longer matches what's on disk now.
        _write_face(face_dir, "imgX", [{"embedding": [0.0, 1.0, 0.0], "bbox": [0, 0, 1, 1]}])

        with pytest.raises(clustering.ClusterMembersStaleError):
            clustering.cluster_mean_embedding(cluster_id)


def test_cluster_mean_embedding_missing_face_index_returns_none_or_raises(tmp_path):
    """If the member's face_index is out of range for the image's current
    face data (fewer faces now than when clustered), that member is skipped
    just like a fingerprint mismatch."""
    import clustering
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    clusters_file = tmp_path / "face_clusters.json"
    _write_face(face_dir, "imgY", [{"embedding": [1.0, 0.0], "bbox": [0, 0, 1, 1]}])

    p1, p2, p3 = _patched(face_dir, clusters_file)
    with p1, p2, p3:
        clustering.cluster_faces(eps=0.3, min_samples=1)
        cluster_id = clustering.load_clusters()["clusters"][0]["cluster_id"]

        # imgY re-processed with zero faces now.
        _write_face(face_dir, "imgY", [])

        with pytest.raises(clustering.ClusterMembersStaleError):
            clustering.cluster_mean_embedding(cluster_id)


# ── set_cluster_status ───────────────────────────────────────────────────────

def test_set_cluster_status_updates_name_and_status(tmp_path):
    import clustering
    clusters_file = tmp_path / "face_clusters.json"
    clusters_file.write_text(json.dumps({
        "clusters": [{"cluster_id": 0, "size": 1, "members": [], "name": None, "status": "new"}],
        "params": {}, "total_faces": 1,
    }))
    with patch("clustering.CLUSTERS_FILE", str(clusters_file)):
        clustering.set_cluster_status(0, "ignored")
        assert clustering.get_cluster(0)["status"] == "ignored"

        clustering.set_cluster_status(0, "named", name="Bob")
        c = clustering.get_cluster(0)
        assert c["status"] == "named"
        assert c["name"] == "Bob"
