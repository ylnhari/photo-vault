import json
import os
import pytest
from unittest.mock import patch, MagicMock, call
import catalog_db


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_catalog(image_ids):
    return {"images": {
        img_id: {"path": f"/photos/{img_id}.jpg", "filename": f"{img_id}.jpg", "metadata": {}}
        for img_id in image_ids
    }}


def _seed_catalog_db(tmp_path, catalog: dict) -> str:
    """Seed a temp SQLite catalog file with catalog ({'images': {...}}) and
    return its path, for patching indexer.IMAGE_CATALOG_PATH in tests."""
    path = str(tmp_path / "catalog.db")
    catalog_db.save_all(path, catalog.get("images", {}), catalog.get("folders", {}))
    return path


def _mock_chromadb(existing_ids=None, metadata_rows=None):
    col = MagicMock()
    existing_ids = existing_ids or []
    col.get.return_value = {
        "ids": existing_ids,
        "metadatas": metadata_rows or [{} for _ in existing_ids],
    }
    col.count.return_value = len(existing_ids)
    client = MagicMock()
    client.get_or_create_collection.return_value = col
    return client, col


# ── Indexer.get_missing ────────────────────────────────────────────────────────

def test_get_missing_all_new(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a", "b", "c"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    mock_client, _ = _mock_chromadb(existing_ids=[])

    with patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_active_model", return_value="test-model"):
        idx = Indexer()
        missing = idx.get_missing()

    assert len(missing) == 3
    assert {img_id for img_id, _ in missing} == {"a", "b", "c"}


def test_get_missing_none_when_all_indexed(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a", "b"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    mock_client, _ = _mock_chromadb(existing_ids=["a", "b"])

    with patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_active_model", return_value="test-model"):
        idx = Indexer()
        missing = idx.get_missing()

    assert missing == []


def test_get_missing_partial(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a", "b", "c"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    mock_client, _ = _mock_chromadb(existing_ids=["a"])

    with patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_active_model", return_value="test-model"):
        idx = Indexer()
        missing = idx.get_missing()

    assert {img_id for img_id, _ in missing} == {"b", "c"}


def test_get_missing_degrades_gracefully_with_no_active_model(tmp_path):
    """db.collection() raises ValueError when no active embedding model is
    configured (so embedding/indexing writes can't silently land in an
    ungoverned fallback collection) — but read-only "what's already
    embedded" checks like get_missing/get_missing_attributes/
    get_embed_pending must still degrade to "nothing embedded yet" for a
    fresh install, not 500. Regression test for that exact crash."""
    from indexer import Indexer
    catalog = _make_catalog(["a", "b"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    mock_client, _ = _mock_chromadb(existing_ids=[])

    with patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("db.get_active_model", return_value=None):
        idx = Indexer()
        missing = idx.get_missing()

    assert {img_id for img_id, _ in missing} == {"a", "b"}


# ── Indexer.get_missing_files ──────────────────────────────────────────────────

def test_get_missing_files_detects_deleted(tmp_path):
    from indexer import Indexer
    real = tmp_path / "real.jpg"
    real.write_text("x")
    catalog = {"images": {
        "real": {"path": str(real), "filename": "real.jpg", "metadata": {}},
        "gone": {"path": str(tmp_path / "gone.jpg"), "filename": "gone.jpg", "metadata": {}},
    }}
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)):
        idx = Indexer()
        missing = idx.get_missing_files()

    assert {img_id for img_id, _ in missing} == {"gone"}


# ── Indexer.get_missing_attributes ────────────────────────────────────────────

def test_get_missing_attributes_all_stale(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a", "b"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    stale_meta = [{"weather": "unknown", "occasion": "unknown", "location_type": "unknown",
                   "scene": "unknown", "mood": "unknown"}] * 2
    mock_client, _ = _mock_chromadb(existing_ids=["a", "b"], metadata_rows=stale_meta)

    with patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_active_model", return_value="test-model"):
        idx = Indexer()
        stale = idx.get_missing_attributes()

    assert len(stale) == 2


def test_get_missing_attributes_none_when_rich(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    rich_meta = [{"weather": "sunny", "occasion": "vacation", "location_type": "beach",
                  "scene": "outdoor", "mood": "happy"}]
    mock_client, _ = _mock_chromadb(existing_ids=["a"], metadata_rows=rich_meta)

    with patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_active_model", return_value="test-model"):
        idx = Indexer()
        stale = idx.get_missing_attributes()

    assert stale == []


# ── _index_one ────────────────────────────────────────────────────────────────

def _make_mock_client():
    col = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = col
    return client, col


def test_index_one_raises_on_vision_error():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}}

    with patch("indexer.get_image_caption", return_value=(json.dumps({"error": "LM Studio down"}), "error")), \
         patch("indexer.parse_vision_attributes", return_value={"caption": "", "scene": "unknown",
               "location_type": "unknown", "weather": "unknown", "season": "unknown",
               "time_of_day": "unknown", "occasion": "unknown", "group_size": "unknown",
               "clothing_style": "unknown", "mood": "unknown", "objects": "", "people_description": ""}), \
         patch("indexer.get_embedding", return_value=([0.1], "test-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        with pytest.raises(RuntimeError, match="error"):
            _index_one("img1", img_data)


def test_index_one_raises_when_embedding_none():
    from indexer import _index_one
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}}
    full_caption = ('{"caption":"test","scene":"outdoor","occasion":"everyday",'
                     '"weather":"sunny","group_size":"solo"}')

    with patch("indexer.get_image_caption", return_value=(full_caption, "lm_studio:m")), \
         patch("indexer.parse_vision_attributes", return_value={"caption": "test", "scene": "outdoor",
               "location_type": "unknown", "weather": "unknown", "season": "unknown",
               "time_of_day": "unknown", "occasion": "unknown", "group_size": "unknown",
               "clothing_style": "unknown", "mood": "unknown", "objects": "", "people_description": ""}), \
         patch("indexer.get_embedding", return_value=(None, "", "error")), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        with pytest.raises(RuntimeError, match="embedding failed"):
            _index_one("img1", img_data)


def test_index_one_success_returns_note():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {"date": "2024-01-01"}}
    attrs = {"caption": "beach", "scene": "outdoor", "location_type": "beach",
             "weather": "sunny", "season": "summer", "time_of_day": "afternoon",
             "occasion": "vacation", "festival_name": "", "group_size": "couple", "person_count": 2,
             "clothing_style": "swimwear", "mood": "happy", "objects": "umbrella",
             "animals": "", "vehicles": "", "food_items": "", "activities": "",
             "photo_type": "photo", "text_in_image": "", "landmark": "", "dominant_colors": "",
             "people_description": "two people"}

    full_caption = ('{"caption":"beach","scene":"outdoor","occasion":"vacation",'
                     '"weather":"sunny","group_size":"couple"}')
    with patch("indexer.get_image_caption", return_value=(full_caption, "gemini")), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1, 0.2], "text-embedding-004", "gemini")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        note = _index_one("img1", img_data, upsert=False)

    assert note == "embed:gemini"
    col.add.assert_called_once()
    col.upsert.assert_not_called()


def test_index_one_upsert_path():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}}
    attrs = {"caption": "", "scene": "unknown", "location_type": "unknown",
             "weather": "unknown", "season": "unknown", "time_of_day": "unknown",
             "occasion": "unknown", "festival_name": "", "group_size": "unknown", "person_count": 0,
             "clothing_style": "unknown", "mood": "unknown", "objects": "",
             "animals": "", "vehicles": "", "food_items": "", "activities": "",
             "photo_type": "unknown", "text_in_image": "", "landmark": "", "dominant_colors": "",
             "people_description": ""}

    full_caption = ('{"caption":"x","scene":"unknown","occasion":"unknown",'
                     '"weather":"unknown","group_size":"unknown"}')
    with patch("indexer.get_image_caption", return_value=(full_caption, "lm_studio:m")), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1], "lm-embed-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        _index_one("img1", img_data, upsert=True)

    col.upsert.assert_called_once()
    col.add.assert_not_called()


def test_index_one_stores_embedding_model_in_metadata():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}}
    attrs = {k: "unknown" for k in ["caption", "scene", "location_type", "weather", "season",
             "time_of_day", "occasion", "festival_name", "group_size", "person_count", "clothing_style",
             "mood", "objects", "animals", "vehicles", "food_items", "activities", "photo_type",
             "text_in_image", "landmark", "dominant_colors", "people_description"]}

    full_caption = ('{"caption":"x","scene":"unknown","occasion":"unknown",'
                     '"weather":"unknown","group_size":"unknown"}')
    with patch("indexer.get_image_caption", return_value=(full_caption, "lm_studio:m")), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1], "my-embed-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        _index_one("img1", img_data, upsert=False)

    call_kwargs = col.add.call_args[1]
    assert call_kwargs["metadatas"][0]["embedding_model"] == "my-embed-model"
    assert call_kwargs["metadatas"][0]["embedding_source"] == "lm_studio"


def test_index_one_stores_caption_in_img_data():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}}
    attrs = {k: "unknown" for k in ["caption", "scene", "location_type", "weather", "season",
             "time_of_day", "occasion", "festival_name", "group_size", "person_count", "clothing_style",
             "mood", "objects", "animals", "vehicles", "food_items", "activities", "photo_type",
             "text_in_image", "landmark", "dominant_colors", "people_description"]}

    full_caption = ('{"caption":"beach photo","scene":"outdoor","occasion":"everyday",'
                     '"weather":"sunny","group_size":"solo"}')
    with patch("indexer.get_image_caption", return_value=(full_caption, "lm_studio:m")), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1], "my-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        _index_one("img1", img_data, upsert=False)

    assert img_data["caption_json"] == full_caption


def test_index_one_reuses_cached_caption():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {},
                "caption_json": '{"caption":"cached caption"}'}
    attrs = {k: "unknown" for k in ["caption", "scene", "location_type", "weather", "season",
             "time_of_day", "occasion", "festival_name", "group_size", "person_count", "clothing_style",
             "mood", "objects", "animals", "vehicles", "food_items", "activities", "photo_type",
             "text_in_image", "landmark", "dominant_colors", "people_description"]}
    mock_vision = MagicMock(return_value='{"caption":"fresh"}')

    with patch("indexer.get_image_caption", mock_vision), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1], "my-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        _index_one("img1", img_data, upsert=False, use_cached=True)

    mock_vision.assert_not_called()


def test_index_one_skips_cache_when_use_cached_false():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {},
                "caption_json": '{"caption":"old cached caption"}'}
    attrs = {k: "unknown" for k in ["caption", "scene", "location_type", "weather", "season",
             "time_of_day", "occasion", "festival_name", "group_size", "person_count", "clothing_style",
             "mood", "objects", "animals", "vehicles", "food_items", "activities", "photo_type",
             "text_in_image", "landmark", "dominant_colors", "people_description"]}
    fresh_caption = ('{"caption":"fresh caption","scene":"outdoor","occasion":"everyday",'
                      '"weather":"sunny","group_size":"solo"}')
    mock_vision = MagicMock(return_value=(fresh_caption, "lm_studio:m"))

    with patch("indexer.get_image_caption", mock_vision), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1], "my-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        _index_one("img1", img_data, upsert=False, use_cached=False)

    mock_vision.assert_called_once()
    assert img_data["caption_json"] == fresh_caption


# ── get_stage_stats ───────────────────────────────────────────────────────────

def test_get_stage_stats_counts_captioned(tmp_path):
    from indexer import Indexer
    catalog = {"images": {
        "a": {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}, "caption_json": '{"c":"x"}'},
        "b": {"path": "/b.jpg", "filename": "b.jpg", "metadata": {}},
        "c": {"path": "/c.jpg", "filename": "c.jpg", "metadata": {}, "caption_json": '{"c":"y"}'},
    }}
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    mock_client, col = _mock_chromadb(existing_ids=["a"])
    reg = {"active_model": "m", "models": {"m": {"source": "lm_studio", "dimension": 3}}}

    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.get_registry", return_value=reg), \
         patch("indexer.get_active_model", return_value="m"):
        idx = Indexer()
        stats = idx.get_stage_stats()

    assert stats["total_scanned"] == 3
    assert stats["vision_done"] == 2
    assert stats["vision_pending"] == 1
    assert stats["active_model_embedded"] == 1


def test_get_stage_stats_embed_pending_never_negative(tmp_path):
    """Regression: if the active collection has MORE embedded ids than the
    catalog has captions for (e.g. stale entries after a failed/partial Chroma
    delete), embed_pending must clamp to 0, not go negative — mirrors the
    max(0, ...) pattern count_thumbs_missing already uses."""
    from indexer import Indexer
    catalog = {"images": {
        "a": {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}, "caption_json": '{"c":"x"}'},
    }}
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    # Active collection reports MORE indexed items than the catalog has
    # captions for.
    mock_client, col = _mock_chromadb(existing_ids=["a", "stale-b", "stale-c"])
    reg = {"active_model": "m", "models": {"m": {"source": "lm_studio", "dimension": 3}}}

    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)),          patch("indexer.db.client", return_value=mock_client),          patch("indexer.get_registry", return_value=reg),          patch("indexer.get_active_model", return_value="m"):
        idx = Indexer()
        stats = idx.get_stage_stats()

    assert stats["vision_done"] == 1
    assert stats["active_model_embedded"] == 3
    assert stats["embed_pending"] == 0


# ── caption history / model tracking ───────────────────────────────────────────

def test_record_caption_history_appends_different_models():
    from indexer import _record_caption_history
    img_data = {}
    _record_caption_history(img_data, "lm_studio:qwen", '{"caption":"a"}')
    _record_caption_history(img_data, "gemini", '{"caption":"b"}')
    assert len(img_data["caption_history"]) == 2
    assert img_data["caption_json"] == '{"caption":"b"}'       # latest used for embedding
    assert img_data["caption_model"] == "gemini"
    models = {h["model"] for h in img_data["caption_history"]}
    assert models == {"lm_studio:qwen", "gemini"}


def test_record_caption_history_replaces_same_model():
    from indexer import _record_caption_history
    img_data = {}
    _record_caption_history(img_data, "lm_studio:qwen", '{"caption":"old"}')
    _record_caption_history(img_data, "lm_studio:qwen", '{"caption":"new"}')
    assert len(img_data["caption_history"]) == 1
    assert img_data["caption_history"][0]["caption_json"] == '{"caption":"new"}'
    assert img_data["caption_json"] == '{"caption":"new"}'


def test_vision_one_raises_on_error_json(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_image_caption", return_value=(json.dumps({"error": "down"}), "error")), \
         patch("indexer.Indexer._save_catalog"):
        idx = Indexer()
        with pytest.raises(RuntimeError, match="down"):
            idx.vision_one("a")


def test_vision_one_stores_and_returns_model(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    full_caption = json.dumps({
        "caption": "ok", "scene": "outdoor", "occasion": "everyday",
        "weather": "sunny", "group_size": "solo",
    })
    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_image_caption", return_value=(full_caption, "gemini")), \
         patch("indexer.Indexer._save_catalog"):
        idx = Indexer()
        note = idx.vision_one("a", force_provider="gemini")

    assert note == "vision:gemini"
    assert idx.image_catalog["images"]["a"]["caption_model"] == "gemini"


# ── dirty-tracking / incremental catalog saves ─────────────────────────────────

def test_save_catalog_only_writes_dirty_rows(tmp_path):
    """_save_catalog must upsert only the touched id(s), not rewrite every row
    in the catalog — the whole point of the SQLite migration (images.json was
    rewritten in full on every job batch)."""
    from indexer import Indexer
    catalog = _make_catalog(["a", "b", "c"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    full_caption = json.dumps({
        "caption": "x", "scene": "outdoor", "occasion": "everyday",
        "weather": "sunny", "group_size": "solo",
    })
    with patch("indexer.IMAGE_CATALOG_PATH", catalog_path), \
         patch("indexer.get_image_caption", return_value=(full_caption, "lm_studio:m")):
        idx = Indexer()
        idx.vision_one("a")

    # Reload straight from the DB (bypassing the in-memory copy) to prove the
    # write actually landed, and that b/c were never touched.
    reloaded = catalog_db.load_all(catalog_path)["images"]
    assert reloaded["a"].get("caption_json")
    assert "caption_json" not in reloaded["b"]
    assert "caption_json" not in reloaded["c"]


def test_delete_image_removes_row_from_db_not_just_memory(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a", "b"])
    catalog_path = _seed_catalog_db(tmp_path, catalog)

    with patch("indexer.IMAGE_CATALOG_PATH", catalog_path), \
         patch("indexer.Indexer._drop_from_collections"):
        idx = Indexer()
        idx.delete_image("a", to_trash=False)

    reloaded = catalog_db.load_all(catalog_path)["images"]
    assert set(reloaded) == {"b"}


def test_purge_folder_removes_rows_from_db(tmp_path):
    from indexer import Indexer
    sub = tmp_path / "sub"
    other = tmp_path / "other"
    catalog = {"images": {
        "a": {"path": str(sub / "a.jpg"), "filename": "a.jpg", "metadata": {}},
        "b": {"path": str(other / "b.jpg"), "filename": "b.jpg", "metadata": {}},
    }}
    catalog_path = _seed_catalog_db(tmp_path / "cat", catalog)
    mock_client, _ = _mock_chromadb(existing_ids=[])

    with patch("indexer.IMAGE_CATALOG_PATH", catalog_path), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.get_registry", return_value={"active_model": None, "models": {}}), \
         patch("indexer._remove_derived_files"):
        idx = Indexer()
        removed = idx.purge_folder(str(sub))

    assert removed == 1
    reloaded = catalog_db.load_all(catalog_path)["images"]
    assert set(reloaded) == {"b"}


# ── _path_under (drive-root fix) ────────────────────────────────────────────

def test_path_under_matches_normal_subfolder():
    from indexer import _path_under
    assert _path_under(r"C:\Users\foo\photos\a.jpg", r"C:\Users\foo\photos")


def test_path_under_rejects_sibling_with_shared_prefix():
    from indexer import _path_under
    assert not _path_under(r"C:\Users\foobar\a.jpg", r"C:\Users\foo")


def test_path_under_drive_root_matches_children():
    """Regression: Path('D:\\\\').resolve() keeps the trailing separator, so a
    naive f + os.sep comparison doubled up ('D:\\\\\\\\') and never matched
    any real child path — a whole-drive scan folder silently purged/counted 0."""
    from indexer import _path_under
    assert _path_under(r"D:\photos\a.jpg", "D:\\")
    assert _path_under(r"D:\a.jpg", "D:\\")


def test_path_under_drive_root_rejects_other_drive():
    from indexer import _path_under
    assert not _path_under(r"C:\other\a.jpg", "D:\\")


# ── detect_faces_one / thumb_one / dhash_one missing-file consistency ───────

def test_detect_faces_one_returns_skipped_not_raise_on_missing_file(tmp_path):
    """detect_faces_one used to raise FileNotFoundError for a missing file
    while thumb_one/dhash_one returned a 'skipped' string — now all three
    agree, so a job runner can treat them the same way (neither a hard
    failure nor an indistinguishable success)."""
    from indexer import Indexer
    catalog = {"images": {"a": {"path": str(tmp_path / "gone.jpg"), "filename": "gone.jpg",
                                 "metadata": {}}}}
    catalog_path = _seed_catalog_db(tmp_path, catalog)
    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)):
        idx = Indexer()
        note = idx.detect_faces_one("a")
    assert "skipped" in note


# ── _embed_one guards face detection (item 3) ───────────────────────────────

def test_embed_one_succeeds_when_face_detection_raises():
    """A corrupt/unreadable image must not discard an already-successful text
    embedding — face detection failure should be logged, not fatal."""
    from indexer import _embed_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {},
                "caption_json": '{"caption":"x","scene":"outdoor"}'}

    with patch("indexer.get_embedding", return_value=([0.1], "my-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", side_effect=RuntimeError("corrupt image")):
        note = _embed_one("img1", img_data, detect_faces=True)

    assert note == "embed:lm_studio"
    col.add.assert_called_once()


# ── resolve_caption_json (item 15) ──────────────────────────────────────────

def test_resolve_caption_json_raises_on_invalid_json():
    """Invalid JSON must raise, not be handed back for the caller to embed as
    garbage text."""
    from indexer import resolve_caption_json
    img_data = {"caption_json": "not valid json {{{"}
    with pytest.raises(RuntimeError, match="not valid JSON"):
        resolve_caption_json(img_data)


def test_resolve_caption_json_returns_valid_json_unchanged():
    from indexer import resolve_caption_json
    img_data = {"caption_json": '{"caption":"ok"}'}
    assert resolve_caption_json(img_data) == '{"caption":"ok"}'


# ── reconcile_paths targeted fetch (item 19) ────────────────────────────────

def test_reconcile_paths_fetches_only_moved_ids_when_known():
    from indexer import Indexer
    col = MagicMock()
    col.get.return_value = {"ids": ["a"], "metadatas": [{"path": "/old/a.jpg", "filename": "a.jpg"}]}
    client = MagicMock()
    client.get_or_create_collection.return_value = col
    reg = {"active_model": "m", "models": {"m": {}}}

    with patch("indexer.db.client", return_value=client), \
         patch("indexer.get_registry", return_value=reg):
        idx = Indexer.__new__(Indexer)
        idx.image_catalog = {"images": {"a": {"path": "/new/a.jpg", "filename": "a.jpg"}}}
        fixed = idx.reconcile_paths(moved_ids=["a"])

    col.get.assert_called_once_with(ids=["a"], include=["metadatas"])
    assert fixed == 1


def test_reconcile_paths_batches_update_into_one_call():
    """One col.update() call carrying every changed id/metadata pair, not one
    call per changed id — turns an O(n) sequence of Chroma round-trips into
    O(1) per collection per scan."""
    from indexer import Indexer
    col = MagicMock()
    col.get.return_value = {
        "ids": ["a", "b", "c"],
        "metadatas": [
            {"path": "/old/a.jpg", "filename": "a.jpg"},
            {"path": "/new/b.jpg", "filename": "b.jpg"},  # unchanged, no update needed
            {"path": "/old/c.jpg", "filename": "c.jpg"},
        ],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = col
    reg = {"active_model": "m", "models": {"m": {}}}

    with patch("indexer.db.client", return_value=client),          patch("indexer.get_registry", return_value=reg):
        idx = Indexer.__new__(Indexer)
        idx.image_catalog = {"images": {
            "a": {"path": "/new/a.jpg", "filename": "a.jpg"},
            "b": {"path": "/new/b.jpg", "filename": "b.jpg"},
            "c": {"path": "/new/c.jpg", "filename": "c.jpg"},
        }}
        fixed = idx.reconcile_paths()

    assert fixed == 2
    col.update.assert_called_once()
    assert set(col.update.call_args.kwargs["ids"]) == {"a", "c"}


# ── physical dedupe of byte-identical extra copies ───────────────────────────

def _bare_indexer(images: dict):
    """Indexer without loading the real catalog: dedupe_copy_one only touches
    image_catalog and _mark_dirty."""
    from indexer import Indexer
    idx = Indexer.__new__(Indexer)
    idx.image_catalog = {"images": images}
    idx._mark_dirty = lambda *a, **k: None
    return idx


def test_dedupe_copy_one_trashes_verified_duplicate(tmp_path, monkeypatch):
    from scanner import content_uid
    canonical = tmp_path / "keep.jpg"
    extra = tmp_path / "extra.jpg"
    canonical.write_bytes(b"same-bytes")
    extra.write_bytes(b"same-bytes")
    uid = content_uid(canonical)
    idx = _bare_indexer({uid: {"path": str(canonical), "dup_paths": [str(extra)]}})

    trashed = []
    monkeypatch.setattr("trash.delete_file_to_recycle_bin",
                        lambda p: trashed.append(p) or True)
    note = idx.dedupe_copy_one(f"{uid}::{extra}")
    assert "Recycle Bin" in note
    assert trashed == [str(extra)]
    assert idx.image_catalog["images"][uid]["dup_paths"] == []


def test_dedupe_copy_one_skips_missing_and_changed_files(tmp_path, monkeypatch):
    from scanner import content_uid
    canonical = tmp_path / "keep.jpg"
    canonical.write_bytes(b"same-bytes")
    uid = content_uid(canonical)
    gone = str(tmp_path / "gone.jpg")
    changed = tmp_path / "changed.jpg"
    changed.write_bytes(b"DIFFERENT-bytes")
    idx = _bare_indexer({uid: {"path": str(canonical),
                               "dup_paths": [gone, str(changed)]}})
    monkeypatch.setattr("trash.delete_file_to_recycle_bin",
                        lambda p: pytest.fail("must not trash"))
    assert "already gone" in idx.dedupe_copy_one(f"{uid}::{gone}")
    assert "not a duplicate" in idx.dedupe_copy_one(f"{uid}::{changed}")
    assert changed.exists()
    assert idx.image_catalog["images"][uid]["dup_paths"] == []


def test_dedupe_copy_one_refuses_when_canonical_missing(tmp_path, monkeypatch):
    from scanner import content_uid
    extra = tmp_path / "only_copy.jpg"
    extra.write_bytes(b"same-bytes")
    uid = content_uid(extra)
    idx = _bare_indexer({uid: {"path": str(tmp_path / "vanished.jpg"),
                               "dup_paths": [str(extra)]}})
    monkeypatch.setattr("trash.delete_file_to_recycle_bin",
                        lambda p: pytest.fail("must not trash the last copy"))
    note = idx.dedupe_copy_one(f"{uid}::{extra}")
    assert "canonical copy missing" in note
    assert extra.exists()
    # NOT forgotten — it should be retried once the canonical is back.
    assert idx.image_catalog["images"][uid]["dup_paths"] == [str(extra)]


def test_get_redundant_copies_lists_uid_path_items():
    idx = _bare_indexer({
        "u1": {"path": "p1", "dup_paths": ["x", "y"]},
        "u2": {"path": "p2"},
    })
    assert idx.get_redundant_copies() == ["u1::x", "u1::y"]
