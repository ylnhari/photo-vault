import json
import os
import pytest
from unittest.mock import patch, MagicMock, call


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_catalog(image_ids):
    return {"images": {
        img_id: {"path": f"/photos/{img_id}.jpg", "filename": f"{img_id}.jpg", "metadata": {}}
        for img_id in image_ids
    }}


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
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

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
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

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
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

    mock_client, _ = _mock_chromadb(existing_ids=["a"])

    with patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_active_model", return_value="test-model"):
        idx = Indexer()
        missing = idx.get_missing()

    assert {img_id for img_id, _ in missing} == {"b", "c"}


# ── Indexer.get_missing_files ──────────────────────────────────────────────────

def test_get_missing_files_detects_deleted(tmp_path):
    from indexer import Indexer
    real = tmp_path / "real.jpg"
    real.write_text("x")
    catalog = {"images": {
        "real": {"path": str(real), "filename": "real.jpg", "metadata": {}},
        "gone": {"path": str(tmp_path / "gone.jpg"), "filename": "gone.jpg", "metadata": {}},
    }}
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)):
        idx = Indexer()
        missing = idx.get_missing_files()

    assert {img_id for img_id, _ in missing} == {"gone"}


# ── Indexer.get_missing_attributes ────────────────────────────────────────────

def test_get_missing_attributes_all_stale(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a", "b"])
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

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
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

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
        with pytest.raises(RuntimeError, match="vision error"):
            _index_one("img1", img_data)


def test_index_one_raises_when_embedding_none():
    from indexer import _index_one
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}}

    with patch("indexer.get_image_caption", return_value=('{"caption":"test","scene":"outdoor"}', "lm_studio:m")), \
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
             "occasion": "vacation", "group_size": "couple", "clothing_style": "swimwear",
             "mood": "happy", "objects": "umbrella", "people_description": "two people"}

    with patch("indexer.get_image_caption", return_value=('{"caption":"beach"}', "gemini")), \
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
             "occasion": "unknown", "group_size": "unknown", "clothing_style": "unknown",
             "mood": "unknown", "objects": "", "people_description": ""}

    with patch("indexer.get_image_caption", return_value=('{"caption":"x"}', "lm_studio:m")), \
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
             "time_of_day", "occasion", "group_size", "clothing_style", "mood", "objects", "people_description"]}

    with patch("indexer.get_image_caption", return_value=('{"caption":"x"}', "lm_studio:m")), \
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
             "time_of_day", "occasion", "group_size", "clothing_style", "mood", "objects", "people_description"]}

    with patch("indexer.get_image_caption", return_value=('{"caption":"beach photo"}', "lm_studio:m")), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1], "my-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        _index_one("img1", img_data, upsert=False)

    assert img_data["caption_json"] == '{"caption":"beach photo"}'


def test_index_one_reuses_cached_caption():
    from indexer import _index_one
    mock_client, col = _make_mock_client()
    img_data = {"path": "/a.jpg", "filename": "a.jpg", "metadata": {},
                "caption_json": '{"caption":"cached caption"}'}
    attrs = {k: "unknown" for k in ["caption", "scene", "location_type", "weather", "season",
             "time_of_day", "occasion", "group_size", "clothing_style", "mood", "objects", "people_description"]}
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
             "time_of_day", "occasion", "group_size", "clothing_style", "mood", "objects", "people_description"]}
    mock_vision = MagicMock(return_value=('{"caption":"fresh caption"}', "lm_studio:m"))

    with patch("indexer.get_image_caption", mock_vision), \
         patch("indexer.parse_vision_attributes", return_value=attrs), \
         patch("indexer.get_embedding", return_value=([0.1], "my-model", "lm_studio")), \
         patch("indexer.db.client", return_value=mock_client), \
         patch("indexer.detect_and_embed_faces", return_value=[]), \
         patch("indexer.save_face_data"):
        _index_one("img1", img_data, upsert=False, use_cached=False)

    mock_vision.assert_called_once()
    assert img_data["caption_json"] == '{"caption":"fresh caption"}'


# ── get_stage_stats ───────────────────────────────────────────────────────────

def test_get_stage_stats_counts_captioned(tmp_path):
    from indexer import Indexer
    catalog = {"images": {
        "a": {"path": "/a.jpg", "filename": "a.jpg", "metadata": {}, "caption_json": '{"c":"x"}'},
        "b": {"path": "/b.jpg", "filename": "b.jpg", "metadata": {}},
        "c": {"path": "/c.jpg", "filename": "c.jpg", "metadata": {}, "caption_json": '{"c":"y"}'},
    }}
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

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
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_image_caption", return_value=(json.dumps({"error": "down"}), "error")), \
         patch("indexer.Indexer._save_catalog"):
        idx = Indexer()
        with pytest.raises(RuntimeError, match="down"):
            idx.vision_one("a")


def test_vision_one_stores_and_returns_model(tmp_path):
    from indexer import Indexer
    catalog = _make_catalog(["a"])
    catalog_path = tmp_path / "images.json"
    catalog_path.write_text(json.dumps(catalog))

    with patch("indexer.IMAGE_CATALOG_PATH", str(catalog_path)), \
         patch("indexer.get_image_caption", return_value=('{"caption":"ok"}', "gemini")), \
         patch("indexer.Indexer._save_catalog"):
        idx = Indexer()
        note = idx.vision_one("a", force_provider="gemini")

    assert note == "vision:gemini"
    assert idx.image_catalog["images"]["a"]["caption_model"] == "gemini"
