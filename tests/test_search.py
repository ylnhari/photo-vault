import json
import pytest
from unittest.mock import patch, MagicMock


# ── build_where_clause ────────────────────────────────────────────────────────

def test_where_empty_dict_returns_none():
    from search import build_where_clause
    assert build_where_clause({}) is None


def test_where_all_values_returns_none():
    from search import build_where_clause
    assert build_where_clause({"weather": "All", "scene": "All"}) is None


def test_where_single_filter_no_and():
    from search import build_where_clause
    result = build_where_clause({"weather": "sunny"})
    assert result == {"weather": {"$eq": "sunny"}}
    assert "$and" not in result


def test_where_multiple_filters_uses_and():
    from search import build_where_clause
    result = build_where_clause({"weather": "sunny", "scene": "outdoor"})
    assert "$and" in result
    assert len(result["$and"]) == 2


def test_where_skips_empty_string_values():
    from search import build_where_clause
    result = build_where_clause({"weather": "sunny", "scene": ""})
    assert result == {"weather": {"$eq": "sunny"}}


def test_where_skips_none_values():
    from search import build_where_clause
    result = build_where_clause({"weather": "sunny", "scene": None})
    assert result == {"weather": {"$eq": "sunny"}}


def test_where_three_filters():
    from search import build_where_clause
    result = build_where_clause({"weather": "sunny", "scene": "outdoor", "mood": "happy"})
    assert "$and" in result
    assert len(result["$and"]) == 3


def test_where_coerces_person_count_to_int():
    from search import build_where_clause
    result = build_where_clause({"person_count": "2"})
    assert result == {"person_count": {"$eq": 2}}


def test_where_skips_unparseable_person_count():
    from search import build_where_clause
    assert build_where_clause({"person_count": "many"}) is None


# ── search_images ─────────────────────────────────────────────────────────────

def _mock_collection(count=0, query_result=None):
    col = MagicMock()
    col.count.return_value = count
    col.query.return_value = query_result or {"ids": [[]], "metadatas": [[]]}
    return col


def test_search_empty_collection():
    from search import search_images
    mock_col = _mock_collection(count=0)
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_active_model", return_value="test-model"), \
         patch("search.get_embedding", return_value=([0.1, 0.2], "test-model", "lm_studio")):
        result = search_images("beach")

    assert result == {"ids": [[]], "metadatas": [[]]}
    mock_col.query.assert_not_called()


def test_search_returns_none_when_embedding_fails():
    from search import search_images
    mock_col = _mock_collection(count=5)
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_active_model", return_value="test-model"), \
         patch("search.get_embedding", return_value=(None, "", "error")):
        result = search_images("beach")

    assert result is None


def test_search_passes_where_clause():
    from search import search_images
    mock_col = _mock_collection(count=10, query_result={"ids": [["id1"]], "metadatas": [[{"path": "/a.jpg"}]]})
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_active_model", return_value="test-model"), \
         patch("search.get_embedding", return_value=([0.1], "test-model", "lm_studio")):
        search_images("beach", filters={"weather": "sunny"})

    call_kwargs = mock_col.query.call_args[1]
    assert call_kwargs.get("where") == {"weather": {"$eq": "sunny"}}


def test_search_no_where_when_all_filters():
    from search import search_images
    mock_col = _mock_collection(count=10, query_result={"ids": [["id1"]], "metadatas": [[{"path": "/a.jpg"}]]})
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_active_model", return_value="test-model"), \
         patch("search.get_embedding", return_value=([0.1], "test-model", "lm_studio")):
        search_images("beach", filters={"weather": "All"})

    call_kwargs = mock_col.query.call_args[1]
    assert call_kwargs.get("where") is None


def test_search_falls_back_when_where_clause_fails():
    from search import search_images
    mock_col = MagicMock()
    mock_col.count.return_value = 5
    mock_col.query.side_effect = [Exception("invalid where"), {"ids": [["id1"]], "metadatas": [[{}]]}]
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_active_model", return_value="test-model"), \
         patch("search.get_embedding", return_value=([0.1], "test-model", "lm_studio")):
        result = search_images("beach", filters={"weather": "sunny"})

    assert mock_col.query.call_count == 2
    assert result is not None


def test_search_filters_by_person():
    """With a text query + person, results are intersected with the person's ANN matches."""
    from search import search_images
    ids = ["img1", "img2"]
    metas = [{"path": "/img1.jpg"}, {"path": "/img2.jpg"}]
    mock_col = _mock_collection(count=2, query_result={"ids": [ids], "metadatas": [metas]})

    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_active_model", return_value="test-model"), \
         patch("search.get_embedding", return_value=([0.1], "test-model", "lm_studio")), \
         patch("search.get_person_embedding", return_value=[1.0, 0.0]), \
         patch("search.query_person_faces", return_value={"img1"}):
        result = search_images("beach", person="Alice")

    assert result["ids"][0] == ["img1"]
    assert len(result["metadatas"][0]) == 1


def test_search_person_only_returns_all_matches():
    """Person + no query/filters → fetch all the person's photos from the face index."""
    from search import search_images
    mock_col = _mock_collection(count=5)
    mock_col.get.return_value = {"ids": ["img1", "img3"], "metadatas": [{"path": "/1"}, {"path": "/3"}]}

    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_active_model", return_value="test-model"), \
         patch("search.get_person_embedding", return_value=[1.0, 0.0]), \
         patch("search.query_person_faces", return_value={"img1", "img3"}):
        result = search_images("", person="Alice")

    assert set(result["ids"][0]) == {"img1", "img3"}
    mock_col.get.assert_called_once()


def test_search_uses_active_model_collection():
    from search import search_images
    mock_col = _mock_collection(count=3, query_result={"ids": [["id1"]], "metadatas": [[{"path": "/a.jpg"}]]})
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    # Let the real db.collection() run so the model→collection-name derivation
    # is exercised; only the underlying client is mocked.
    with patch("db.client", return_value=mock_client), \
         patch("db.get_active_model", return_value="my-embed-model"), \
         patch("search.get_embedding", return_value=([0.1], "my-embed-model", "lm_studio")):
        search_images("test")

    col_name_used = mock_client.get_or_create_collection.call_args[1].get("name") or \
                    mock_client.get_or_create_collection.call_args[0][0]
    assert "my_embed_model" in col_name_used


def test_query_embedded_with_active_model():
    """The query vector must come from the ACTIVE model's provider/model, not
    whatever the auto chain would pick — mixed vector spaces break search."""
    from search import search_images
    mock_col = _mock_collection(count=2, query_result={"ids": [["a"]], "metadatas": [[{"path": "/a"}]]})
    reg = {"active_model": "my-embed", "models": {"my-embed": {"source": "lm_studio"}}}
    with patch("search.db.collection", return_value=mock_col), \
         patch("search.get_registry", return_value=reg), \
         patch("search.get_embedding", return_value=([0.1], "my-embed", "lm_studio")) as ge:
        search_images("beach")
    ge.assert_called_once_with("beach", force_provider="lm_studio", model="my-embed")
