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


def test_provider_models(client):
    with patch("api.list_lm_studio_models", return_value=["qwen-vl", "nomic-embed"]):
        r = client.get("/api/provider-models")
    assert r.status_code == 200
    b = r.json()
    assert b["lm_studio"] == ["qwen-vl", "nomic-embed"]
    assert "text-embedding-004" in b["gemini_embed"]
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


def test_cleanup_missing(client):
    fake = MagicMock()
    fake.get_missing_files.return_value = [("a", {}), ("b", {})]
    with patch("api.Indexer", return_value=fake):
        r = client.post("/api/cleanup-missing")
    assert r.json()["removed"] == 2
    assert fake.delete_image.call_count == 2
