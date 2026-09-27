import json
import os
import subprocess
import sys

import catalog_db


def _image(name, *, date="", media_type="image", created_at=1, caption="", occasion=""):
    return {
        "path": f"/synthetic/{name}",
        "filename": name,
        "created_at": created_at,
        "media_type": media_type,
        "duration_s": 12.5 if media_type == "video" else 0,
        "metadata": {"date": date} if date else {},
        "caption_json": json.dumps({"caption": caption, "occasion": occasion}),
    }


def test_catalog_projection_migrates_legacy_rows_and_indexes_browse(tmp_path):
    path = str(tmp_path / "legacy.db")
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE images (id TEXT PRIMARY KEY, data_json TEXT NOT NULL)")
    conn.execute("CREATE TABLE folders (root TEXT PRIMARY KEY, data_json TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO images VALUES (?, ?)",
        ("a", json.dumps(_image("IMG-20240506-a.jpg", caption="Beach walk", occasion="vacation"))),
    )
    conn.commit()
    conn.close()

    page = catalog_db.library_page(path, limit=10, offset=0, year="2024")
    assert page["total"] == 1
    assert page["results"][0]["date"].startswith("2024:05:06")
    assert page["results"][0]["caption"] == "Beach walk"
    assert catalog_db.library_summary(path)["years"] == [{"year": "2024", "count": 1}]


def test_failed_projection_migration_rolls_back_and_next_write_recovers(tmp_path):
    path = str(tmp_path / "broken-then-recovered.db")
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE images (id TEXT PRIMARY KEY, data_json TEXT NOT NULL)")
    conn.execute("CREATE TABLE folders (root TEXT PRIMARY KEY, data_json TEXT NOT NULL)")
    conn.execute("INSERT INTO images VALUES ('bad', '{not-json')")
    conn.commit()
    conn.close()

    import pytest
    with pytest.raises(json.JSONDecodeError):
        catalog_db.library_page(path, limit=10, offset=0)
    check = sqlite3.connect(path)
    assert check.execute("SELECT COUNT(*) FROM library_projection").fetchone()[0] == 0
    assert check.execute("SELECT COUNT(*) FROM catalog_meta").fetchone()[0] == 0
    check.execute("UPDATE images SET data_json=? WHERE id='bad'", (json.dumps(_image("fixed.jpg")),))
    check.commit()
    check.close()

    catalog_db.upsert_images(path, {"new": _image("new.jpg", date="2025:01:02 00:00:00")})
    result = catalog_db.library_page(path, limit=10, offset=0)
    assert result["total"] == 2
    assert {row["id"] for row in result["results"]} == {"bad", "new"}


def test_projection_updates_and_deletes_with_catalog_writes(tmp_path):
    path = str(tmp_path / "catalog.db")
    catalog_db.upsert_images(path, {
        "photo": _image("summer.jpg", date="2022:07:03 11:00:00", caption="Sunny beach at sunset", occasion="vacation"),
        "clip": _image("clip.mp4", date="2023:08:04 12:00:00", media_type="video"),
    })
    assert catalog_db.library_page(path, limit=10, offset=0, q="beach sunset")["total"] == 1
    assert catalog_db.library_page(path, limit=10, offset=0, media_type="video")["total"] == 1

    catalog_db.upsert_images(path, {"photo": _image("renamed.jpg", date="2024:01:02 00:00:00", caption="New caption")})
    page = catalog_db.library_page(path, limit=10, offset=0, year="2024")
    assert page["total"] == 1
    assert page["results"][0]["filename"] == "renamed.jpg"
    assert catalog_db.image_path(path, "photo") == "/synthetic/renamed.jpg"

    catalog_db.delete_images(path, ["photo"])
    assert catalog_db.library_summary(path) == {
        "total": 1, "photos": 0, "videos": 1, "captioned": 0,
        "years": [{"year": "2023", "count": 1}],
    }


def test_library_api_pages_filters_and_summary(tmp_path, monkeypatch):
    import api

    path = str(tmp_path / "catalog.db")
    catalog_db.save_all(path, {
        "a": _image("IMG-20230101-a.jpg", caption="Train by the beach", created_at=3),
        "b": _image("party.jpg", date="2024:02:03 00:00:00", caption="Birthday", occasion="birthday"),
        "c": _image("clip.mp4", date="2025:04:05 00:00:00", media_type="video", created_at=4),
    }, {})
    monkeypatch.setattr(api, "IMAGE_CATALOG_PATH", path)
    from fastapi.testclient import TestClient

    with TestClient(api.app) as client:
        page = client.get("/api/library", params={"limit": 2})
        assert page.status_code == 200
        body = page.json()
        assert body["total"] == 3 and body["has_more"] is True
        assert len(body["results"]) == 2
        assert body["results"][0]["id"] == "c"
        assert body["results"][0]["exists"] is True
        assert client.get("/api/library", params={"q": "beach"}).json()["total"] == 1
        assert client.get("/api/library", params={"media_type": "video"}).json()["total"] == 1
        assert client.get("/api/library", params={"year": "2024"}).json()["total"] == 1
        summary = client.get("/api/library/summary").json()
        assert summary["total"] == 3
        assert summary["photos"] == 2 and summary["videos"] == 1
        assert summary["years"] == [
            {"year": "2025", "count": 1},
            {"year": "2024", "count": 1},
            {"year": "2023", "count": 1},
        ]


def test_api_module_import_does_not_load_ml_or_vector_stack(tmp_path):
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    script = (
        "import sys; sys.path.insert(0, " + repr(src) + "); import api; "
        "forbidden = {'insightface','onnxruntime','cv2','sklearn','chromadb','openai','vision','embeddings','faces','clustering','jobs','indexer','search'}; "
        "loaded = forbidden.intersection(sys.modules); "
        "assert not loaded, f'heavy modules imported: {sorted(loaded)}'"
    )
    env = os.environ.copy()
    env["PHOTO_VAULT_ENV_FILE"] = "-"
    env["PHOTO_VAULT_DATA_DIR"] = str(tmp_path / "isolated-data")
    run = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=40)
    assert run.returncode == 0, run.stderr
