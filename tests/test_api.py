from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import api

    return TestClient(api.app)


def test_health(client):
    with patch(
        "api.service_status",
        return_value={"lm_studio": True, "gemini": False, "gemini_key_set": False},
    ):
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["lm_studio"] is True


def test_status(client):
    fake = MagicMock()
    fake.get_stage_stats.return_value = {"total_scanned": 5, "vision_done": 3}
    fake.get_vision_pending.return_value = [("a", {})]
    fake.get_embed_pending.return_value = []
    fake.get_missing_attributes.return_value = []
    fake.get_missing.return_value = [("a", {})]
    fake.get_missing_files.return_value = []
    fake.get_vision_pending_for_model.return_value = []
    fake.get_embed_eligible_ids.return_value = []
    fake.get_vision_model_summary.return_value = {}
    fake.get_faces_stats.return_value = {"total": 0, "detected": 0, "pending": 0}
    fake.get_video_faces_stats.return_value = {"total": 0, "detected": 0, "pending": 0}
    with (
        patch("api.Indexer", return_value=fake),
        patch("api.settings_mgr.load", return_value={}),
        patch("api.settings_mgr.vision_model_label", return_value=None),
    ):
        r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["stage"]["total_scanned"] == 5
    assert body["vision_pending"] == 1


def test_search_post(client):
    res = {
        "ids": [["a", "b"]],
        "metadatas": [
            [
                {
                    "path": "/p/a.jpg",
                    "caption": "beach",
                    "year": "2024",
                    "occasion": "vacation",
                },
                {
                    "path": "/p/b.jpg",
                    "caption": "party",
                    "year": "2023",
                    "occasion": "birthday",
                },
            ]
        ],
    }
    with (
        patch("api.search_images", return_value=res),
        patch("api.os.path.exists", return_value=True),
    ):
        r = client.post("/api/search", json={"q": "beach", "filters": {"year": "2024"}})
    assert r.status_code == 200
    cards = r.json()["results"]
    assert len(cards) == 2
    assert cards[0]["caption"] == "beach"
    assert cards[0]["filename"] == "a.jpg"


def test_search_get_forwards_json_filters(client):
    with patch("api.search_images", return_value={"ids": [[]], "metadatas": [[]]}) as si:
        r = client.get("/api/search", params={"q": "x", "filters": '{"year": "2024"}'})
    assert r.status_code == 200
    assert si.call_args.kwargs["filters"] == {"year": "2024"}


def test_search_get_rejects_malformed_filters_json(client):
    r = client.get("/api/search", params={"q": "x", "filters": "not json"})
    assert r.status_code == 400


def test_search_unavailable_returns_503(client):
    from search import SearchUnavailableError

    with patch("api.search_images", side_effect=SearchUnavailableError("both providers down")):
        r = client.get("/api/search", params={"q": "x"})
    assert r.status_code == 503

    with patch("api.search_images", side_effect=SearchUnavailableError("both providers down")):
        r = client.post("/api/search", json={"q": "x"})
    assert r.status_code == 503


def test_search_surfaces_person_not_found_and_filter_error_flags(client):
    res = {"ids": [[]], "metadatas": [[]], "person_not_found": True, "filter_error": True}
    with patch("api.search_images", return_value=res):
        r = client.post("/api/search", json={"q": "", "person": "Nobody"})
    body = r.json()
    assert body["person_not_found"] is True
    assert body["filter_error"] is True


def test_faces_name_stale_cluster_returns_409(client):
    from clustering import ClusterMembersStaleError

    with patch("api.clustering.cluster_mean_embedding",
               side_effect=ClusterMembersStaleError(1)):
        r = client.post("/api/faces/name", json={"cluster_id": 1, "name": "Alice"})
    assert r.status_code == 409


def test_filters(client):
    with patch(
        "api.get_available_filter_values",
        return_value={"year": ["2023", "2024"], "weather": ["sunny"]},
    ):
        r = client.get("/api/filters")
    assert r.status_code == 200
    assert r.json()["year"] == ["2023", "2024"]


def test_index_start_unknown_type(client):
    r = client.post("/api/index/start", json={"type": "bogus"})
    assert r.status_code == 400


def test_index_start_conflict_when_running(client):
    with patch(
        "api.manager.start", side_effect=RuntimeError("a job is already running")
    ):
        r = client.post("/api/index/start", json={"type": "vision"})
    assert r.status_code == 409


def test_scan_invalid_dir(client):
    with patch("api.os.path.isdir", return_value=False):
        r = client.post("/api/scan", json={"dirs": ["/no/such"]})
    assert r.status_code == 400


def test_people_list(client):
    with patch("api.get_all_persons", return_value=["Alice", "Bob"]):
        r = client.get("/api/people")
    assert r.json()["people"] == ["Alice", "Bob"]


def test_models(client):
    reg = {
        "active_model": "m1",
        "models": {"m1": {"source": "lm_studio", "dimension": 768}},
    }
    fake_col = MagicMock()
    fake_col.count.return_value = 42
    with (
        patch("api.get_registry", return_value=reg),
        patch("api.db.collection", return_value=fake_col),
    ):
        r = client.get("/api/models")
    body = r.json()
    assert body["active"] == "m1"
    assert body["models"]["m1"]["indexed_count"] == 42


def test_delete_image(client):
    fake = MagicMock()
    fake.delete_image.return_value = "/p/a.jpg"
    with (
        patch("api.Indexer", return_value=fake),
        patch("api.os.path.exists", return_value=False),
    ):
        r = client.request("DELETE", "/api/image", params={"id": "/p/a.jpg"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_image_rejects_arbitrary_path_traversal(client):
    """A client id that isn't an indexed photo must never be used as a filesystem
    path — this is the arbitrary-file-read fix."""
    with (
        patch("api._chroma_meta", return_value={}),
        patch("api.catalog_path_for", return_value=None),
    ):
        r = client.get("/api/image", params={"id": "C:\\Windows\\win.ini"})
    assert r.status_code == 404


def test_image_serves_only_resolved_indexed_path(tmp_path, client):
    real = tmp_path / "photo.jpg"
    real.write_bytes(b"\xff\xd8\xff\xe0jpegdata")
    with patch("api._chroma_meta", return_value={"path": str(real)}):
        r = client.get("/api/image", params={"id": "abc123"})
    assert r.status_code == 200


def test_auth_enforced_when_enabled(client, monkeypatch):
    monkeypatch.setenv("PV_REQUIRE_AUTH", "1")
    # No Authorization header → rejected
    r = client.get("/api/status")
    assert r.status_code == 401
    # Correct bearer token → allowed through (status itself may 200 with empty data)
    import security

    r2 = client.get(
        "/api/status", headers={"Authorization": f"Bearer {security.get_token()}"}
    )
    assert r2.status_code != 401


def test_token_endpoint_exempt_from_auth(client, monkeypatch):
    monkeypatch.setenv("PV_REQUIRE_AUTH", "1")
    r = client.get("/api/token")
    assert r.status_code == 200
    assert r.json()["token"]


def test_health_exempt_from_auth(client, monkeypatch):
    monkeypatch.setenv("PV_REQUIRE_AUTH", "1")
    with patch(
        "api.service_status",
        return_value={"lm_studio": False, "gemini": False, "gemini_key_set": False},
    ):
        r = client.get("/api/health")
    assert r.status_code == 200


def test_scan_blocked_while_job_active(client):
    with patch("api.manager.status", return_value={"active": True}):
        r = client.post("/api/scan", json={"dirs": []})
    assert r.status_code == 409


def test_status_empty_catalog(client):
    """All done/remaining counts are 0 when nothing is scanned (edge case)."""
    fake = MagicMock()
    fake.get_stage_stats.return_value = {
        "total_scanned": 0,
        "vision_done": 0,
        "active_model": None,
        "active_model_embedded": 0,
        "models": {},
    }
    fake.get_vision_pending.return_value = []
    fake.get_embed_pending.return_value = []
    fake.get_missing_attributes.return_value = []
    fake.get_missing.return_value = []
    fake.get_missing_files.return_value = []
    fake.get_embed_eligible_ids.return_value = []
    fake.get_vision_model_summary.return_value = {}
    fake.get_faces_stats.return_value = {"total": 0, "detected": 0, "pending": 0}
    fake.get_video_faces_stats.return_value = {"total": 0, "detected": 0, "pending": 0}
    with patch("api.Indexer", return_value=fake), patch("api.folder_mgr") as fm:
        fm.get_effective_scan_dirs.return_value = []
        r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["stage"]["total_scanned"] == 0
    assert body["vision_pending"] == 0
    assert body["embed_pending"] == 0
    assert body["missing_attrs"] == 0
    assert body["missing_full"] == 0
    assert body["missing_files"] == 0
    assert body["faces_pending"] == 0


def test_provider_models(client):
    # Mock all provider listings so the test is hermetic (no live Gemini call).
    with (
        patch("api.list_lm_studio_models", return_value=["qwen-vl", "nomic-embed"]),
        patch("api.list_gemini_vision_models", return_value=["gemini-2.0-flash"]),
        patch("api.list_gemini_embed_models", return_value=["gemini-embedding-001"]),
    ):
        r = client.get("/api/provider-models")
    assert r.status_code == 200
    b = r.json()
    assert b["lm_studio"] == ["qwen-vl", "nomic-embed"]
    assert b["gemini_embed"] == ["gemini-embedding-001"]
    assert b["gemini_vision"] == ["gemini-2.0-flash"]


def test_provider_models_gemini_unreachable(client):
    """Gemini model lists are empty when no models can be fetched (no hardcoded fallback)."""
    with (
        patch("api.list_lm_studio_models", return_value=[]),
        patch("api.list_gemini_vision_models", return_value=[]),
        patch("api.list_gemini_embed_models", return_value=[]),
    ):
        r = client.get("/api/provider-models")
    assert r.status_code == 200
    b = r.json()
    assert b["gemini_vision"] == []
    assert b["gemini_embed"] == []


def test_index_start_passes_model_config(client):
    captured = {}

    def fake_start(jtype, **kw):
        captured.update(kw)
        captured["jtype"] = jtype
        return {"active": False, "finished": True}

    with patch("api.manager.start", side_effect=fake_start):
        r = client.post(
            "/api/index/start",
            json={
                "type": "vision",
                "vision_provider": "gemini",
                "vision_model": "gemini-2.0-flash",
            },
        )
    assert r.status_code == 200
    assert captured["jtype"] == "vision"
    assert captured["vision_provider"] == "gemini"
    assert captured["vision_model"] == "gemini-2.0-flash"


def test_faces_cluster_runs(client):
    with (
        patch(
            "api.clustering.cluster_faces",
            return_value={"clusters": 3, "faces": 40, "noise": 5},
        ) as cf,
        patch("api.manager.status", return_value={"active": False}),
    ):
        r = client.post("/api/faces/cluster", json={})
    assert r.status_code == 200
    assert r.json()["clusters"] == 3
    cf.assert_called_once()


def test_faces_cluster_blocked_while_job_active(client):
    with patch("api.manager.status", return_value={"active": True}):
        r = client.post("/api/faces/cluster", json={})
    assert r.status_code == 409


def test_faces_name_registers_person(client):
    with (
        patch("api.clustering.cluster_mean_embedding", return_value=[0.1, 0.2]),
        patch("api.clustering.set_cluster_status") as setst,
        patch("api.add_person_embedding", return_value=True) as addp,
    ):
        r = client.post("/api/faces/name", json={"cluster_id": 2, "name": "Alice"})
    assert r.status_code == 200
    addp.assert_called_once()
    setst.assert_called_once()


def test_faces_name_missing_cluster_404(client):
    with patch("api.clustering.cluster_mean_embedding", return_value=None):
        r = client.post("/api/faces/name", json={"cluster_id": 99, "name": "Bob"})
    assert r.status_code == 404


def test_albums_create_and_list(client):
    with patch(
        "api.albums_mgr.create_album",
        return_value={"id": "abc", "name": "Trip", "count": 0, "cover": None},
    ) as cr:
        r = client.post("/api/albums", json={"name": "Trip"})
    assert r.status_code == 200
    assert r.json()["id"] == "abc"
    cr.assert_called_once_with("Trip")


def test_albums_add(client):
    with patch("api.albums_mgr.add_to_album", return_value=4) as add:
        r = client.post("/api/albums/abc/add", json={"ids": ["x", "y"]})
    assert r.status_code == 200
    assert r.json()["count"] == 4
    add.assert_called_once_with("abc", ["x", "y"])


def test_albums_add_missing_404(client):
    with patch("api.albums_mgr.add_to_album", side_effect=KeyError("abc")):
        r = client.post("/api/albums/abc/add", json={"ids": ["x"]})
    assert r.status_code == 404


def test_albums_get_returns_cards(client):
    with (
        patch(
            "api.albums_mgr.get_album",
            return_value={"name": "Trip", "image_ids": ["x"]},
        ),
        patch("api._cards_for_ids", return_value=[{"id": "x", "filename": "x.jpg"}]),
    ):
        r = client.get("/api/albums/abc")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Trip"
    assert body["photos"][0]["id"] == "x"


def test_map_returns_only_geotagged(client):
    catalog = {
        "images": {
            "a": {
                "filename": "a.jpg",
                "path": "/a.jpg",
                "metadata": {"gps_lat": 37.7, "gps_lon": -122.4},
            },
            "b": {
                "filename": "b.jpg",
                "path": "/b.jpg",
                "metadata": {"date": "2024:01:01"},
            },  # no GPS
            "c": {
                "filename": "c.jpg",
                "path": "/c.jpg",
                "metadata": {"gps_lat": 51.5, "gps_lon": -0.1},
            },
        }
    }
    with (
        patch("api.load_catalog_cached", return_value=catalog),
        patch("api.os.path.exists", return_value=True),
    ):
        r = client.get("/api/map")
    assert r.status_code == 200
    pts = r.json()["points"]
    assert {p["id"] for p in pts} == {"a", "c"}
    assert all("lat" in p and "lon" in p for p in pts)


def test_faces_reindex(client):
    with (
        patch("api.rebuild_face_index", return_value=42) as rb,
        patch("api.manager.status", return_value={"active": False}),
    ):
        r = client.post("/api/faces/reindex")
    assert r.status_code == 200
    assert r.json()["indexed"] == 42
    rb.assert_called_once()


def test_faces_reindex_blocked_while_job_active(client):
    with patch("api.manager.status", return_value={"active": True}):
        r = client.post("/api/faces/reindex")
    assert r.status_code == 409


def test_faces_clusters_excludes_ignored(client):
    data = {
        "clusters": [
            {
                "cluster_id": 0,
                "size": 5,
                "status": "new",
                "name": None,
                "members": [{"image_id": "a", "face_index": 0}],
            },
            {
                "cluster_id": 1,
                "size": 3,
                "status": "ignored",
                "name": None,
                "members": [],
            },
        ]
    }
    with patch("api.clustering.load_clusters", return_value=data):
        r = client.get("/api/faces/clusters")
    ids = [c["cluster_id"] for c in r.json()["clusters"]]
    assert ids == [0]


def test_batch_delete_removes_each(client):
    fake = MagicMock()
    fake.delete_image.return_value = None
    with patch("api.Indexer", return_value=fake):
        r = client.post("/api/images/delete", json={"ids": ["a", "b", "c"]})
    assert r.status_code == 200
    assert r.json()["removed"] == 3
    assert fake.delete_image.call_count == 3


def test_batch_delete_empty_is_noop(client):
    with patch("api.Indexer") as Idx:
        r = client.post("/api/images/delete", json={"ids": []})
    assert r.json() == {"removed": 0, "files_removed": 0, "files_failed": []}
    Idx.assert_not_called()


def test_batch_delete_reports_failed_files_without_aborting(client):
    """A failed on-disk delete for one item must not abort the whole batch
    (that's the single-delete endpoint's stricter behavior) — it should soft-
    delete the catalog entry anyway and surface the failure via files_failed."""
    fake = MagicMock()
    fake.image_catalog = {"images": {"a": {"path": "/p/a.jpg"}}}
    fake.delete_image.return_value = None
    with (
        patch("api.Indexer", return_value=fake),
        patch("api.os.path.exists", return_value=True),
        patch("api._delete_file_recoverably", return_value=False),
    ):
        r = client.post(
            "/api/images/delete", json={"ids": ["a"], "delete_file": True}
        )
    assert r.status_code == 200
    body = r.json()
    assert body["removed"] == 1
    assert body["files_removed"] == 0
    assert body["files_failed"] == ["a"]
    fake.delete_image.assert_called_once()


def test_cleanup_missing(client):
    fake = MagicMock()
    fake.get_missing_files.return_value = [("a", {}), ("b", {})]
    with patch("api.Indexer", return_value=fake):
        r = client.post("/api/cleanup-missing")
    assert r.json()["removed"] == 2
    assert fake.delete_image.call_count == 2


def test_search_post_empty_q_preserved_for_person_browse(client):
    """q='' with a person must stay empty so search_images takes the
    person-only browse path (coercing to 'photo' disabled it entirely)."""
    res = {"ids": [["i1"]], "metadatas": [[{"path": "/x.jpg"}]]}
    with patch("api.search_images", return_value=res) as si:
        r = client.post("/api/search", json={"q": "", "person": "Alice"})
    assert r.status_code == 200
    assert si.call_args[0][0] == ""
    assert si.call_args[1]["person"] == "Alice"


def test_add_person_no_faces_is_400(client):
    with patch("api.os.path.isdir", return_value=True), \
         patch("api.add_person_reference",
                return_value={"registered": False, "faces_used": 0, "skipped_multi_face": []}):
        r = client.post("/api/people", json={"name": "Bob", "ref_dir": "C:/refs"})
    assert r.status_code == 400
    assert "no faces" in r.json()["detail"]


def test_delete_image_rejected_while_job_active(client):
    with patch("api.manager.status", return_value={"active": True}):
        r = client.delete("/api/image?id=abc")
    assert r.status_code == 409


def test_similar_returns_neighbors_excluding_self(client):
    col = MagicMock()
    col.get.return_value = {"ids": ["x"], "embeddings": [[0.1, 0.2]]}
    col.count.return_value = 3
    col.query.return_value = {
        "ids": [["x", "y", "z"]],
        "metadatas": [[{"path": "/x.jpg"}, {"path": "/y.jpg"}, {"path": "/z.jpg"}]],
    }
    with patch("api.get_active_model", return_value="m"), \
         patch("api.db.collection", return_value=col):
        r = client.get("/api/similar?id=x&top_k=5")
    assert r.status_code == 200
    ids = [c["id"] for c in r.json()["results"]]
    assert ids == ["y", "z"]  # self excluded


def test_similar_404_when_not_indexed(client):
    col = MagicMock()
    col.get.return_value = {"ids": [], "embeddings": []}
    with patch("api.get_active_model", return_value="m"), \
         patch("api.db.collection", return_value=col):
        r = client.get("/api/similar?id=nope")
    assert r.status_code == 404


# ── audit fixes: bounded top_k / limit / samples query params (422, not 500) ──


def test_search_get_rejects_non_positive_top_k(client):
    r = client.get("/api/search", params={"q": "x", "top_k": 0})
    assert r.status_code == 422
    r2 = client.get("/api/search", params={"q": "x", "top_k": -5})
    assert r2.status_code == 422


def test_search_post_rejects_non_positive_top_k(client):
    r = client.post("/api/search", json={"q": "x", "top_k": 0})
    assert r.status_code == 422


def test_similar_rejects_non_positive_top_k(client):
    r = client.get("/api/similar", params={"id": "x", "top_k": -1})
    assert r.status_code == 422


def test_recent_rejects_non_positive_limit(client):
    r = client.get("/api/recent", params={"limit": 0})
    assert r.status_code == 422


def test_duplicates_rejects_non_positive_limit(client):
    r = client.get("/api/duplicates", params={"limit": -3})
    assert r.status_code == 422


def test_faces_clusters_rejects_out_of_range_samples(client):
    assert client.get("/api/faces/clusters", params={"samples": 0}).status_code == 422
    assert client.get("/api/faces/clusters", params={"samples": 21}).status_code == 422


# ── audit fixes: /api/models/active validates against the registry ───────────


def test_set_model_rejects_unknown_model(client):
    with patch("api.set_active_model", side_effect=ValueError("not in registry")):
        r = client.post("/api/models/active", json={"model": "typo-model"})
    assert r.status_code == 400


def test_set_model_accepts_known_model(client):
    with patch("api.set_active_model") as sam:
        r = client.post("/api/models/active", json={"model": "m1"})
    assert r.status_code == 200
    sam.assert_called_once_with("m1")


# ── audit fixes: auth hardening ───────────────────────────────────────────────


def test_docs_blocked_when_auth_enabled(client, monkeypatch):
    monkeypatch.setenv("PV_REQUIRE_AUTH", "1")
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/docs").status_code == 401


def test_token_endpoint_rejects_non_loopback_client(monkeypatch):
    """A request whose actual connection isn't loopback must not receive the
    real bearer token even though /api/token is exempt from the bearer check
    itself — that exemption exists only to bootstrap a loopback SPA."""
    import api

    monkeypatch.setenv("PV_REQUIRE_AUTH", "1")
    remote = TestClient(api.app, client=("203.0.113.5", 40000))
    r = remote.get("/api/token")
    assert r.status_code == 403


def test_token_endpoint_null_when_auth_disabled_regardless_of_client():
    import api

    remote = TestClient(api.app, client=("203.0.113.5", 40000))
    r = remote.get("/api/token")
    assert r.status_code == 200
    assert r.json()["token"] is None


# ── audit fixes: destructive-op defaults / consistency ────────────────────────


def test_remove_folder_defaults_to_no_purge(client):
    with (
        patch("api.folder_mgr.remove_included", return_value={"status": "ok"}) as ri,
        patch("api.Indexer") as Idx,
    ):
        r = client.request(
            "DELETE", "/api/folders/include", params={"path": "/some/dir"}
        )
    assert r.status_code == 200
    assert r.json()["images_purged"] == 0
    Idx.assert_not_called()
    ri.assert_called_once()


def test_put_settings_explicit_null_clears_field(client):
    """exclude_unset must let an explicit `null` clear a field, while an
    omitted field leaves the stored value untouched — filtering by
    `is not None` (the old behavior) could never distinguish the two."""
    captured = {}
    with patch("api.settings_mgr.update", side_effect=lambda p: captured.update(p) or p):
        r = client.put("/api/settings", json={"vision_model": None})
    assert r.status_code == 200
    assert "vision_model" in captured
    assert captured["vision_model"] is None
    # A field never mentioned in the body must not appear in the patch at all.
    assert "embed_model" not in captured


def test_add_person_strips_whitespace(client):
    with (
        patch("api.os.path.isdir", return_value=True),
        patch("api.add_person_reference",
              return_value={"registered": True, "faces_used": 3, "skipped_multi_face": []}) as apr,
    ):
        r = client.post(
            "/api/people", json={"name": "  Bob  ", "ref_dir": "  C:/refs  "}
        )
    assert r.status_code == 200
    assert r.json()["name"] == "Bob"
    apr.assert_called_once_with("Bob", "C:/refs")


# ── audit fixes: orphaned/trash cleanup moved off DELETE-with-body ────────────


def test_cleanup_orphaned_is_now_a_post(client):
    fake = MagicMock()
    fake.get_missing_files.return_value = [("a", {}), ("b", {})]
    with patch("api.Indexer", return_value=fake):
        r = client.post("/api/orphaned/cleanup", json={"ids": ["a"]})
    assert r.status_code == 200
    assert r.json()["removed"] == 1
    # The old DELETE route must be gone, not silently still working.
    assert client.delete("/api/orphaned").status_code == 405


def test_trash_purge_is_now_a_post(client):
    fake = MagicMock()
    fake.purge_trash.return_value = 2
    with patch("api.Indexer", return_value=fake):
        r = client.post("/api/trash/purge", json={"ids": []})
    assert r.status_code == 200
    assert r.json()["purged"] == 2
    assert client.delete("/api/trash").status_code == 405


# ── audit fixes: timeline offset only meaningful with a year ─────────────────


def test_timeline_offset_without_year_is_rejected(client):
    with patch("api.load_catalog_cached", return_value={"images": {}}):
        r = client.get("/api/timeline", params={"offset": 10})
    assert r.status_code == 400


def test_timeline_offset_with_year_still_works(client):
    catalog = {
        "images": {
            "a": {"filename": "a.jpg", "path": "/a.jpg", "metadata": {"date": "2024:01:01"}},
        }
    }
    with (
        patch("api.load_catalog_cached", return_value=catalog),
        patch("api.os.path.exists", return_value=True),
    ):
        r = client.get("/api/timeline", params={"year": "2024", "offset": 0})
    assert r.status_code == 200
    assert r.json()["year"] == "2024"


# ── audit fix: face-crop applies EXIF transpose before cropping by bbox ───────


def test_face_crop_applies_exif_transpose(client, tmp_path):
    out_path = str(tmp_path / "face.webp")
    legacy_path = str(tmp_path / "legacy.jpg")

    class FakeImg:
        size = (100, 100)

        def convert(self, mode):
            return self

        def crop(self, box):
            return self

        def thumbnail(self, size):
            pass

        def save(self, path, fmt, quality=80):
            with open(path, "wb") as f:
                f.write(b"fake-webp-bytes")

    fake_img = FakeImg()
    cm = MagicMock()
    cm.__enter__.return_value = fake_img
    cm.__exit__.return_value = False

    with (
        patch("api.derivative_path", return_value=out_path),
        patch("api.legacy_derivative_path", return_value=legacy_path),
        patch("api._resolve_indexed_path", return_value="/fake/source.jpg"),
        patch("api.load_face_data", return_value=[{"bbox": [10, 10, 50, 50]}]),
        patch("api.safe_open", return_value=cm),
        patch("api.ImageOps.exif_transpose", return_value=fake_img) as et,
    ):
        r = client.get(
            "/api/faces/crop", params={"image_id": "abc", "face_index": 0}
        )
    assert r.status_code == 200
    et.assert_called_once_with(fake_img)


# ── audit fix: /api/image sets a real media_type instead of defaulting to text/plain ──


def test_image_full_sets_webp_media_type(tmp_path, client):
    real = tmp_path / "photo.webp"
    real.write_bytes(b"RIFFfakewebpdata")
    with patch("api._chroma_meta", return_value={"path": str(real)}):
        r = client.get("/api/image", params={"id": "abc123", "size": "full"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"


# ── audit fix: _chroma_meta logs unexpected failures instead of silently
#    collapsing them into "not indexed" ──────────────────────────────────────


def test_meta_404_when_truly_not_indexed(client):
    with patch("api.db.collection", side_effect=Exception("boom")):
        r = client.get("/api/meta", params={"id": "nope"})
    # Still surfaces as 404 (no distinct "not found" exception type exists in
    # this Chroma usage pattern), but the failure is now logged server-side —
    # exercised here via a plain call to confirm it doesn't crash the request.
    assert r.status_code == 404


# ── 9Router integration ───────────────────────────────────────────────────────

def test_index_start_9router_vision_without_model_rejected(client):
    with patch("api.manager.start") as start:
        r = client.post("/api/index/start", json={
            "type": "vision", "vision_provider": "9router",
        })
    assert r.status_code == 422
    assert "vision model" in r.json()["detail"]
    start.assert_not_called()


def test_index_start_9router_embed_without_model_rejected(client):
    with patch("api.manager.start") as start:
        r = client.post("/api/index/start", json={
            "type": "embed", "embed_provider": "9router",
        })
    assert r.status_code == 422
    start.assert_not_called()


def test_index_start_9router_with_model_accepted(client):
    captured = {}
    def fake_start(jtype, **kw):
        captured.update(kw); captured["jtype"] = jtype
        return {"active": False, "finished": True}
    with patch("api.manager.start", side_effect=fake_start):
        r = client.post("/api/index/start", json={
            "type": "vision", "vision_provider": "9router",
            "vision_model": "gc/gemini-2.5-flash-lite",
        })
    assert r.status_code == 200
    assert captured["vision_provider"] == "9router"
    assert captured["vision_model"] == "gc/gemini-2.5-flash-lite"


def test_index_start_faces_job_not_blocked_by_incomplete_9router_settings(client):
    """A job that uses neither vision nor embed must not be rejected just
    because a 9Router provider is saved without a model."""
    with patch("api.manager.start", return_value={"active": False, "finished": True}):
        r = client.post("/api/index/start", json={
            "type": "faces", "vision_provider": "9router", "embed_provider": "9router",
        })
    assert r.status_code == 200


def test_provider_models_includes_9router_lists(client):
    with patch("api.list_lm_studio_models", return_value=[]), \
         patch("api.list_lm_studio_models_v0", return_value=[]), \
         patch("api.list_gemini_vision_models", return_value=[]), \
         patch("api.list_gemini_embed_models", return_value=[]), \
         patch("api.list_9router_vision_models", return_value=["gc/gemini-2.5-flash"]), \
         patch("api.list_9router_embed_models", return_value=["gemini/gemini-embedding-001"]), \
         patch("api.ninerouter_cooldowns", return_value={}), \
         patch("api.ninerouter_embed_cooldowns", return_value={"gemini/gemini-embedding-001": 42.0}):
        r = client.get("/api/provider-models")
    assert r.status_code == 200
    body = r.json()
    assert body["ninerouter_vision"] == ["gc/gemini-2.5-flash"]
    assert body["ninerouter_embed"] == ["gemini/gemini-embedding-001"]
    assert body["ninerouter_cooldowns"]["gemini/gemini-embedding-001"] == 42.0


def test_vision_model_label_none_for_9router():
    """9Router captions are stored under the SERVED model's label, so the
    pre-run label is unpredictable → None, which makes a 9Router vision run
    target photos with no caption at all (coverage semantics)."""
    import settings as settings_mod
    assert settings_mod.vision_model_label(
        {"vision_provider": "9router", "vision_model": "gc/gemini-2.5-flash"}
    ) is None
    assert settings_mod.vision_model_label(
        {"vision_provider": "lm_studio", "vision_model": "m"}
    ) == "lm_studio:m"


# ── video streaming + range requests ──────────────────────────────────────────

def test_parse_range_variants():
    import api
    assert api._parse_range("bytes=0-99", 1000) == (0, 99)
    assert api._parse_range("bytes=100-", 1000) == (100, 999)   # open-ended
    assert api._parse_range("bytes=-50", 1000) == (950, 999)    # suffix range
    assert api._parse_range("bytes=0-100000", 1000) == (0, 999)  # clamped to size
    assert api._parse_range("bytes=2000-3000", 1000) is None     # past EOF
    assert api._parse_range("bytes=abc", 1000) is None
    assert api._parse_range("kbytes=0-1", 1000) is None


def test_video_full_request_advertises_ranges(client, tmp_path):
    import api
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"VIDEODATA" * 100)
    with patch("api._resolve_indexed_path", return_value=str(vid)):
        r = client.get("/api/video?id=abc")
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-type"] == "video/mp4"


def test_video_range_request_returns_206_partial(client, tmp_path):
    import api
    data = bytes(range(256)) * 8  # 2048 bytes
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(data)
    with patch("api._resolve_indexed_path", return_value=str(vid)):
        r = client.get("/api/video?id=abc", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 10-19/{len(data)}"
    assert r.headers["content-length"] == "10"
    assert r.content == data[10:20]


def test_video_404_when_not_indexed(client):
    with patch("api._resolve_indexed_path", return_value=None):
        r = client.get("/api/video?id=nope")
    assert r.status_code == 404


def test_timeline_card_carries_media_type(client):
    catalog = {"images": {
        "vid1": {"path": "/x/movie.mp4", "filename": "movie.mp4",
                 "media_type": "video", "duration_s": 12.5,
                 "metadata": {"date": "2023:06:01 10:00:00"}, "created_at": 1},
        "pic1": {"path": "/x/photo.jpg", "filename": "photo.jpg",
                 "media_type": "image", "metadata": {"date": "2023:06:02 10:00:00"},
                 "created_at": 2},
    }}
    with patch("api.load_catalog_cached", return_value=catalog):
        r = client.get("/api/timeline?year=2023")
    assert r.status_code == 200
    cards = {c["id"]: c for c in r.json()["photos"]}
    assert cards["vid1"]["media_type"] == "video"
    assert cards["vid1"]["duration_s"] == 12.5
    assert cards["pic1"]["media_type"] == "image"
