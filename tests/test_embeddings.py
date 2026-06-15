import json
import pytest
from unittest.mock import patch, MagicMock
import urllib.error


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_urlopen_lm_studio_models(model_id: str):
    class _Resp:
        def read(self): return json.dumps({"data": [{"id": model_id}]}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()


def _fake_urlopen_lm_studio_embed(embedding: list):
    class _Resp:
        def read(self): return json.dumps({"data": [{"embedding": embedding}]}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()


def _fake_urlopen_gemini(values: list):
    class _Resp:
        def read(self): return json.dumps({"embedding": {"values": values}}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()


def _http_error(code: int):
    err = urllib.error.HTTPError(url="", code=code, msg="", hdrs=None, fp=None)
    err.read = lambda: b"error"
    return err


# ── collection_name_for ───────────────────────────────────────────────────────

def test_collection_name_for_simple():
    from embeddings import collection_name_for
    assert collection_name_for("nomic-embed-text") == "img_nomic_embed_text"


def test_collection_name_for_gemini():
    from embeddings import collection_name_for
    assert collection_name_for("text-embedding-004") == "img_text_embedding_004"


def test_collection_name_for_truncates_long_names():
    from embeddings import collection_name_for
    long_name = "a" * 100
    result = collection_name_for(long_name)
    assert len(result) <= 63


# ── _lm_studio_embed ──────────────────────────────────────────────────────────

def test_lm_studio_embed_success():
    from embeddings import _lm_studio_embed
    expected = [0.3, 0.4, 0.5]
    responses = [
        _fake_urlopen_lm_studio_models("test-embed-model"),
        _fake_urlopen_lm_studio_embed(expected),
    ]
    call_count = [0]
    def fake_urlopen(req, timeout=None):
        r = responses[call_count[0]]
        call_count[0] += 1
        return r
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        vec, model = _lm_studio_embed("test")
    assert vec == expected
    assert model == "test-embed-model"


def test_lm_studio_embed_falls_back_to_default_model_name_on_models_error():
    from embeddings import _lm_studio_embed
    expected = [0.1, 0.2]
    call_count = [0]
    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ConnectionRefusedError("refused")  # /models call fails
        return _fake_urlopen_lm_studio_embed(expected)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        vec, model = _lm_studio_embed("test")
    assert vec == expected
    assert model == "lm_studio_embed"  # default fallback name


# ── _gemini_embed ─────────────────────────────────────────────────────────────

def test_gemini_embed_success():
    from embeddings import _gemini_embed
    expected = [0.5, 0.6, 0.7]
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_gemini(expected)), \
         patch("embeddings.GEMINI_API_KEY", "fake-key"):
        vec, model = _gemini_embed("test")
    assert vec == expected
    assert "embedding" in model


def test_gemini_embed_no_key_raises():
    from embeddings import _gemini_embed
    with patch("embeddings.GEMINI_API_KEY", ""):
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            _gemini_embed("test")


def test_gemini_embed_http_error_raises():
    from embeddings import _gemini_embed
    with patch("urllib.request.urlopen", side_effect=_http_error(429)), \
         patch("embeddings.GEMINI_API_KEY", "fake-key"):
        with pytest.raises(RuntimeError, match="429"):
            _gemini_embed("test")


# ── get_embedding ─────────────────────────────────────────────────────────────

def test_get_embedding_lm_studio_primary():
    from embeddings import get_embedding
    vec = [0.1, 0.2]
    with patch("embeddings._lm_studio_embed", return_value=(vec, "test-model")), \
         patch("embeddings.register_model"):
        result, model_name, source = get_embedding("text")
    assert result == vec
    assert source == "lm_studio"
    assert model_name == "test-model"


def test_get_embedding_falls_back_to_gemini_on_connection_error():
    from embeddings import get_embedding
    vec = [0.9, 0.8]
    with patch("embeddings._lm_studio_embed", side_effect=ConnectionRefusedError("refused")), \
         patch("embeddings._gemini_embed", return_value=(vec, "text-embedding-004")), \
         patch("embeddings.register_model"):
        result, model_name, source = get_embedding("text")
    assert result == vec
    assert source == "gemini"


def test_get_embedding_returns_none_when_all_fail():
    from embeddings import get_embedding
    with patch("embeddings._lm_studio_embed", side_effect=ConnectionRefusedError("refused")), \
         patch("embeddings._gemini_embed", side_effect=RuntimeError("quota")):
        result, model_name, source = get_embedding("text")
    assert result is None
    assert model_name == ""
    assert source == "error"


def test_get_embedding_timeout_triggers_fallback():
    from embeddings import get_embedding
    vec = [0.3]
    with patch("embeddings._lm_studio_embed", side_effect=TimeoutError("timeout")), \
         patch("embeddings._gemini_embed", return_value=(vec, "text-embedding-004")), \
         patch("embeddings.register_model"):
        result, model_name, source = get_embedding("text")
    assert source == "gemini"


def test_get_embedding_lm_studio_error_falls_back_to_gemini():
    from embeddings import get_embedding
    vec = [0.5]
    with patch("embeddings._lm_studio_embed", side_effect=ValueError("bad response")), \
         patch("embeddings._gemini_embed", return_value=(vec, "text-embedding-004")), \
         patch("embeddings.register_model"):
        result, model_name, source = get_embedding("text")
    assert result == vec
    assert source == "gemini"


def test_get_embedding_calls_register_model():
    from embeddings import get_embedding
    vec = [0.1, 0.2]
    with patch("embeddings._lm_studio_embed", return_value=(vec, "my-model")), \
         patch("embeddings.register_model") as mock_reg:
        get_embedding("text")
    mock_reg.assert_called_once_with("lm_studio", "my-model", 2)


# ── registry functions ────────────────────────────────────────────────────────

def test_register_model_sets_active_if_first(tmp_path):
    from embeddings import register_model
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", str(tmp_path / "reg.json")):
        register_model("lm_studio", "test-model", 768)
        from embeddings import get_active_model
        with patch("embeddings.EMBEDDING_REGISTRY_PATH", str(tmp_path / "reg.json")):
            active = get_active_model()
    assert active == "test-model"


def test_register_model_does_not_overwrite_active(tmp_path):
    from embeddings import register_model, get_active_model
    reg_path = str(tmp_path / "reg.json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", reg_path):
        register_model("lm_studio", "model-a", 768)
        register_model("gemini", "model-b", 256)
        active = get_active_model()
    assert active == "model-a"


def test_set_active_model_changes_active(tmp_path):
    from embeddings import register_model, set_active_model, get_active_model
    reg_path = str(tmp_path / "reg.json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", reg_path):
        register_model("lm_studio", "model-a", 768)
        register_model("gemini", "model-b", 256)
        set_active_model("model-b")
        active = get_active_model()
    assert active == "model-b"


def test_set_active_model_raises_if_not_registered(tmp_path):
    from embeddings import set_active_model
    reg_path = str(tmp_path / "reg.json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", reg_path):
        with pytest.raises(ValueError, match="not in registry"):
            set_active_model("unknown-model")


# ── _is_connection_error ──────────────────────────────────────────────────────

def test_is_connection_error_true():
    from embeddings import _is_connection_error
    assert _is_connection_error(ConnectionRefusedError("Connection refused"))
    assert _is_connection_error(OSError("Cannot connect to host"))
    assert _is_connection_error(TimeoutError("timeout"))


def test_is_connection_error_false():
    from embeddings import _is_connection_error
    assert not _is_connection_error(ValueError("bad input"))
    assert not _is_connection_error(KeyError("missing key"))
