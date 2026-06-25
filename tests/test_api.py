from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import api
    return TestClient(api.app)


def test_health(client):
    with patch("api.service_status", return_value={"lm_studio": True, "gemini": False,
                                                   "gemini_key_set": False}):
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
    with patch("api.Indexer", return_value=fake):
        r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["stage"]["total_scanned"] == 5
    assert body["vision_pending"] == 1


def test_search_post(client):
    res = {"ids": [["a", "b"]], "metadatas": [[
        {"path": "/p/a.jpg", "caption": "beach", "year": "2024", "occasion": "vacation"},
        {"path": "/p/b.jpg", "caption": "party", "year": "2023", "occasion": "birthday"},
    ]]}
    with patch("api.search_images", return_value=res), \
         patch("api.os.path.exists", return_value=True):
        r = client.post("/api/search", json={"q": "beach", "filters": {"year": "2024"}})
    assert r.status_code == 200
    cards = r.json()["results"]
    assert len(cards) == 2
    assert cards[0]["caption"] == "beach"
    assert cards[0]["filename"] == "a.jpg"


def test_filters(client):
    with patch("api.get_available_filter_values",
               return_value={"year": ["2023", "2024"], "weather": ["sunny"]}):
        r = client.get("/api/filters")
    assert r.status_code == 200
    assert r.json()["year"] == ["2023", "2024"]


def test_index_start_unknown_type(client):
    r = client.post("/api/index/start", json={"type": "bogus"})
    assert r.status_code == 400


def test_index_start_conflict_when_running(client):
    with patch("api.manager.start", side_effect=RuntimeError("a job is already running")):
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
    reg = {"active_model": "m1", "models": {"m1": {"source": "lm_studio", "dimension": 768}}}
    with patch("api.get_registry", return_value=reg):
        r = client.get("/api/models")
    assert r.json()["active"] == "m1"


def test_delete_image(client):
    fake = MagicMock()
    fake.delete_image.return_value = "/p/a.jpg"
    with patch("api.Indexer", return_value=fake), \
         patch("api.os.path.exists", return_value=False):
        r = client.request("DELETE", "/api/image", params={"id": "/p/a.jpg"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_image_rejects_arbitrary_path_traversal(client):
    """A client id that isn't an indexed photo must never be used as a filesystem
    path — this is the arbitrary-file-read fix."""
    with patch("api._chroma_meta", return_value={}), \
         patch("api.catalog_path_for", return_value=None):
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
    r2 = client.get("/api/status", headers={"Authorization": f"Bearer {security.get_token()}"})
    assert r2.status_code != 401


def test_token_endpoint_exempt_from_auth(client, monkeypatch):
    monkeypatch.setenv("PV_REQUIRE_AUTH", "1")
    r = client.get("/api/token")
    assert r.status_code == 200
    assert r.json()["token"]


def test_health_exempt_from_auth(client, monkeypatch):
    monkeypatch.setenv("PV_REQUIRE_AUTH", "1")
    with patch("api.service_status", return_value={"lm_studio": False, "gemini": False,
                                                   "gemini_key_set": False}):
        r = client.get("/api/health")
    assert r.status_code == 200


def test_scan_blocked_while_job_active(client):
    with patch("api.manager.status", return_value={"active": True}):
        r = client.post("/api/scan", json={"dirs": []})
    assert r.status_code == 409


def test_provider_models(client):
    # Mock all provider listings so the test is hermetic (no live Gemini call).
    with patch("api.list_lm_studio_models", return_value=["qwen-vl", "nomic-embed"]), \
         patch("api.list_gemini_vision_models", return_value=["gemini-2.0-flash"]), \
         patch("api.list_gemini_embed_models", return_value=["gemini-embedding-001"]):
        r = client.get("/api/provider-models")
    assert r.status_code == 200
    b = r.json()
    assert b["lm_studio"] == ["qwen-vl", "nomic-embed"]
    assert b["gemini_embed"] == ["gemini-embedding-001"]
    assert len(b["gemini_vision"]) > 0


def test_index_start_passes_model_config(client):
    captured = {}
    def fake_start(jtype, **kw):
        captured.update(kw); captured["jtype"] = jtype
        return {"active": False, "finished": True}
    with patch("api.manager.start", side_effect=fake_start):
        r = client.post("/api/index/start", json={
            "type": "vision", "vision_provider": "gemini", "vision_model": "gemini-2.0-flash",
        })
    assert r.status_code == 200
    assert captured["jtype"] == "vision"
    assert captured["vision_provider"] == "gemini"
    assert captured["vision_model"] == "gemini-2.0-flash"


def test_faces_cluster_runs(client):
    with patch("api.clustering.cluster_faces", return_value={"clusters": 3, "faces": 40, "noise": 5}) as cf, \
         patch("api.manager.status", return_value={"active": False}):
        r = client.post("/api/faces/cluster", json={})
    assert r.status_code == 200
    assert r.json()["clusters"] == 3
    cf.assert_called_once()


def test_faces_cluster_blocked_while_job_active(client):
    with patch("api.manager.status", return_value={"active": True}):
        r = client.post("/api/faces/cluster", json={})
    assert r.status_code == 409


def test_faces_name_registers_person(client):
    with patch("api.clustering.cluster_mean_embedding", return_value=[0.1, 0.2]), \
         patch("api.clustering.set_cluster_status") as setst, \
         patch("api.add_person_embedding", return_value=True) as addp:
        r = client.post("/api/faces/name", json={"cluster_id": 2, "name": "Alice"})
    assert r.status_code == 200
    addp.assert_called_once()
    setst.assert_called_once()


def test_faces_name_missing_cluster_404(client):
    with patch("api.clustering.cluster_mean_embedding", return_value=None):
        r = client.post("/api/faces/name", json={"cluster_id": 99, "name": "Bob"})
    assert r.status_code == 404


def test_albums_create_and_list(client):
    with patch("api.albums_mgr.create_album", return_value={"id": "abc", "name": "Trip", "count": 0, "cover": None}) as cr:
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
    with patch("api.albums_mgr.get_album", return_value={"name": "Trip", "image_ids": ["x"]}), \
         patch("api._cards_for_ids", return_value=[{"id": "x", "filename": "x.jpg"}]):
        r = client.get("/api/albums/abc")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Trip"
    assert body["photos"][0]["id"] == "x"


def test_map_returns_only_geotagged(client):
    catalog = {"images": {
        "a": {"filename": "a.jpg", "path": "/a.jpg", "metadata": {"gps_lat": 37.7, "gps_lon": -122.4}},
        "b": {"filename": "b.jpg", "path": "/b.jpg", "metadata": {"date": "2024:01:01"}},  # no GPS
        "c": {"filename": "c.jpg", "path": "/c.jpg", "metadata": {"gps_lat": 51.5, "gps_lon": -0.1}},
    }}
    with patch("api.load_catalog_cached", return_value=catalog), \
         patch("api.os.path.exists", return_value=True):
        r = client.get("/api/map")
    assert r.status_code == 200
    pts = r.json()["points"]
    assert {p["id"] for p in pts} == {"a", "c"}
    assert all("lat" in p and "lon" in p for p in pts)


def test_faces_reindex(client):
    with patch("api.rebuild_face_index", return_value=42) as rb, \
         patch("api.manager.status", return_value={"active": False}):
        r = client.post("/api/faces/reindex")
    assert r.status_code == 200
    assert r.json()["indexed"] == 42
    rb.assert_called_once()


def test_faces_reindex_blocked_while_job_active(client):
    with patch("api.manager.status", return_value={"active": True}):
        r = client.post("/api/faces/reindex")
    assert r.status_code == 409


def test_faces_clusters_excludes_ignored(client):
    data = {"clusters": [
        {"cluster_id": 0, "size": 5, "status": "new", "name": None, "members": [{"image_id": "a", "face_index": 0}]},
        {"cluster_id": 1, "size": 3, "status": "ignored", "name": None, "members": []},
    ]}
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
    assert r.json() == {"removed": 0, "files_removed": 0}
    Idx.assert_not_called()


def test_cleanup_missing(client):
    fake = MagicMock()
    fake.get_missing_files.return_value = [("a", {}), ("b", {})]
    with patch("api.Indexer", return_value=fake):
        r = client.post("/api/cleanup-missing")
    assert r.json()["removed"] == 2
    assert fake.delete_image.call_count == 2
