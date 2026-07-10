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
    """No loaded embeddings model reported by the v0 API → falls back to the
    plain /v1/models heuristic (first entry)."""
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
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("embeddings.list_lm_studio_models_v0", return_value=[]):
        vec, model = _lm_studio_embed("test")
    assert vec == expected
    assert model == "test-embed-model"


def test_lm_studio_embed_prefers_v0_loaded_embedding_model():
    """When the v0 API reports a loaded embeddings model, use it directly —
    no /v1/models heuristic call needed at all."""
    from embeddings import _lm_studio_embed
    expected = [0.7, 0.8]
    v0_models = [
        {"id": "some-vlm", "type": "vlm", "state": "loaded"},
        {"id": "nomic-embed-text", "type": "embeddings", "state": "loaded"},
    ]
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_lm_studio_embed(expected)), \
         patch("embeddings.list_lm_studio_models_v0", return_value=v0_models):
        vec, model = _lm_studio_embed("test")
    assert vec == expected
    assert model == "nomic-embed-text"


def test_lm_studio_embed_falls_back_to_default_model_name_on_models_error():
    from embeddings import _lm_studio_embed
    expected = [0.1, 0.2]
    call_count = [0]
    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ConnectionRefusedError("refused")  # /models call fails
        return _fake_urlopen_lm_studio_embed(expected)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("embeddings.list_lm_studio_models_v0", return_value=[]):
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


def test_get_embedding_forced_gemini_skips_lm_studio():
    from embeddings import get_embedding
    with patch("embeddings._lm_studio_embed") as lm, \
         patch("embeddings._gemini_embed", return_value=([0.1], "text-embedding-004")), \
         patch("embeddings.register_model"):
        _, _, source = get_embedding("text", force_provider="gemini")
    lm.assert_not_called()
    assert source == "gemini"


def test_get_embedding_forwards_model_to_lm_studio():
    from embeddings import get_embedding
    captured = {}
    def fake_lm(t, model=None):
        captured["model"] = model
        return ([0.1], model or "auto")
    with patch("embeddings._lm_studio_embed", side_effect=fake_lm), \
         patch("embeddings.register_model"):
        get_embedding("text", force_provider="lm_studio", model="nomic-embed")
    assert captured["model"] == "nomic-embed"


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


def test_register_model_raises_on_dimension_change(tmp_path):
    """A model name that suddenly reports a different vector dimension would
    silently corrupt the collection (mixed vector sizes) if allowed through —
    must raise instead of quietly updating the stored dimension."""
    from embeddings import register_model, get_registry
    reg_path = str(tmp_path / "reg.json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", reg_path):
        register_model("lm_studio", "model-a", 768)
        with pytest.raises(RuntimeError, match="dimension"):
            register_model("lm_studio", "model-a", 384)
        # The stored dimension must not have been silently overwritten.
        reg = get_registry()
        assert reg["models"]["model-a"]["dimension"] == 768


def test_register_model_same_dimension_again_does_not_raise(tmp_path):
    from embeddings import register_model
    reg_path = str(tmp_path / "reg.json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", reg_path):
        register_model("lm_studio", "model-a", 768)
        register_model("lm_studio", "model-a", 768)  # same dimension: no raise


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


def test_batch_embed_one_request_in_order():
    """get_embeddings_batch sends all texts in ONE LM Studio request and
    returns vectors in input order (rows may arrive index-shuffled)."""
    from embeddings import get_embeddings_batch
    import json as _json

    calls = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("/models"):
            return _fake_urlopen_lm_studio_models("test-embed-model")
        calls.append(_json.loads(req.data))
        body = {"data": [
            {"index": 1, "embedding": [0.2]},
            {"index": 0, "embedding": [0.1]},
        ]}

        class _R:
            def read(self): return _json.dumps(body).encode()
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _R()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("embeddings.register_model"):
        vectors, model, source = get_embeddings_batch(["a", "b"])

    assert vectors == [[0.1], [0.2]]          # re-sorted to input order
    assert source == "lm_studio"
    assert len(calls) == 1                     # one request for the whole batch
    assert calls[0]["input"] == ["a", "b"]


def test_batch_embed_empty_input():
    from embeddings import get_embeddings_batch
    assert get_embeddings_batch([]) == ([], "", "")


def test_batch_embed_gemini_partial_failure_keeps_successful_vectors():
    """One bad text in a Gemini batch must not discard the whole chunk — the
    failing slot comes back None, the rest keep their real vectors."""
    from embeddings import get_embeddings_batch

    def fake_gemini_embed(text, model=None):
        if text == "bad":
            raise RuntimeError("content blocked")
        return ([0.1] if text == "good1" else [0.2]), "text-embedding-004"

    with patch("embeddings._lm_studio_embed_batch", side_effect=ConnectionRefusedError("refused")), \
         patch("embeddings._gemini_embed", side_effect=fake_gemini_embed), \
         patch("embeddings.register_model"):
        vectors, model, source = get_embeddings_batch(["good1", "bad", "good2"])

    assert source == "gemini"
    assert vectors[0] == [0.1]
    assert vectors[1] is None
    assert vectors[2] == [0.2]


def test_batch_embed_gemini_all_fail_returns_none():
    from embeddings import get_embeddings_batch

    with patch("embeddings._lm_studio_embed_batch", side_effect=ConnectionRefusedError("refused")), \
         patch("embeddings._gemini_embed", side_effect=RuntimeError("quota exceeded")):
        vectors, model, source = get_embeddings_batch(["a", "b"], force_provider="gemini")

    assert vectors is None
    assert source == "error"


# ── registry persistence resilience (items 7, 8, 20) ────────────────────────

def test_load_registry_recovers_from_corrupt_file(tmp_path):
    """A truncated/corrupt registry file (e.g. from a crash mid-write) must
    not crash every subsequent registry read — start fresh instead."""
    from embeddings import _load_registry
    reg_path = tmp_path / "reg.json"
    reg_path.write_text("{not valid json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", str(reg_path)):
        reg = _load_registry()
    assert reg == {"active_model": None, "models": {}}


def test_save_registry_is_atomic_no_leftover_tmp_file(tmp_path):
    from embeddings import _save_registry
    reg_path = tmp_path / "reg.json"
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", str(reg_path)):
        _save_registry({"active_model": "m", "models": {}})
    assert reg_path.exists()
    assert not (tmp_path / "reg.json.tmp").exists()
    import json as _json
    assert _json.loads(reg_path.read_text())["active_model"] == "m"


def test_register_model_raises_on_dimension_mismatch(tmp_path):
    """Re-registering a model name with a different embedding dimension must
    not silently upsert a mismatched vector into the same collection — it
    raises a clear error instead (callers already catch exceptions from
    register_model and surface them as a normal per-item/per-batch failure,
    same as any other embedding-provider error)."""
    from embeddings import register_model, get_registry
    reg_path = str(tmp_path / "reg.json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", reg_path):
        register_model("lm_studio", "model-a", 768)
        with pytest.raises(RuntimeError, match="dimension"):
            register_model("lm_studio", "model-a", 384)  # dimension changed
        reg = get_registry()
    # Dimension recorded at first registration is left as-is (the mismatched
    # second call never reached _save_registry).
    assert reg["models"]["model-a"]["dimension"] == 768


def test_register_model_same_dimension_repeated_call_does_not_raise(tmp_path):
    """The normal case: every successful embed re-registers the same model at
    the same dimension — must not raise or otherwise regress."""
    from embeddings import register_model, get_registry
    reg_path = str(tmp_path / "reg.json")
    with patch("embeddings.EMBEDDING_REGISTRY_PATH", reg_path):
        register_model("lm_studio", "model-a", 768)
        register_model("lm_studio", "model-a", 768)
        reg = get_registry()
    assert reg["models"]["model-a"]["dimension"] == 768


# ── Gemini embed rate-limit cooldown (item 9) ───────────────────────────────

def test_gemini_embed_429_sets_cooldown_and_skips_retry():
    from embeddings import _gemini_embed, gemini_embed_cooldowns
    import embeddings as embeddings_mod
    embeddings_mod._gemini_embed_cooldown.clear()

    with patch("urllib.request.urlopen", side_effect=_http_error(429)), \
         patch("embeddings.GEMINI_API_KEY", "fake-key"):
        with pytest.raises(RuntimeError, match="429"):
            _gemini_embed("test", model="text-embedding-004")

    cooldowns = gemini_embed_cooldowns()
    assert "text-embedding-004" in cooldowns
    assert cooldowns["text-embedding-004"] > 0

    # A second call while in cooldown must not hit the network at all.
    with patch("urllib.request.urlopen") as mock_open:
        with pytest.raises(RuntimeError, match="cooldown"):
            _gemini_embed("test", model="text-embedding-004")
        mock_open.assert_not_called()
    embeddings_mod._gemini_embed_cooldown.clear()


# ── 9Router ───────────────────────────────────────────────────────────────────

def _fake_urlopen_9r_embed(rows: list):
    class _Resp:
        def read(self): return json.dumps({"data": rows}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()


def test_9router_embed_requires_model():
    from embeddings import _9router_embed
    with pytest.raises(ValueError, match="explicit embedding model"):
        _9router_embed("text", None)


def test_9router_embed_success():
    import embeddings
    rows = [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_9r_embed(rows)), \
         patch.dict(embeddings._9r_embed_cooldown, {}, clear=True):
        vec, model = embeddings._9router_embed("text", "gemini/gemini-embedding-001")
    assert vec == [0.1, 0.2, 0.3]
    assert model == "gemini/gemini-embedding-001"


def test_9router_embed_batch_orders_by_index():
    import embeddings
    rows = [
        {"index": 2, "embedding": [3.0]},
        {"index": 0, "embedding": [1.0]},
        {"index": 1, "embedding": [2.0]},
    ]
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_9r_embed(rows)), \
         patch.dict(embeddings._9r_embed_cooldown, {}, clear=True):
        vecs, model = embeddings._9router_embed_batch(["a", "b", "c"], "gemini/gemini-embedding-001")
    assert vecs == [[1.0], [2.0], [3.0]]


def test_9router_embed_429_sets_cooldown():
    import embeddings
    with patch("urllib.request.urlopen", side_effect=_http_error(429)), \
         patch.dict(embeddings._9r_embed_cooldown, {}, clear=True):
        with pytest.raises(RuntimeError, match="429"):
            embeddings._9router_embed("text", "gemini/gemini-embedding-001")
        assert "gemini/gemini-embedding-001" in embeddings._9r_embed_cooldown
        with pytest.raises(RuntimeError, match="cooldown"):
            embeddings._9router_embed("text", "gemini/gemini-embedding-001")


def test_get_embedding_9router_no_cross_provider_fallback():
    """force_provider='9router' failure must NOT silently fall back to
    LM Studio/Gemini — different model = different vector space."""
    from embeddings import get_embedding
    lm = MagicMock(); gem = MagicMock()
    with patch("embeddings._9router_embed", side_effect=RuntimeError("exhausted")), \
         patch("embeddings._lm_studio_embed", lm), patch("embeddings._gemini_embed", gem):
        result, model_name, source = get_embedding("text", force_provider="9router",
                                                   model="gemini/gemini-embedding-001")
    assert result is None
    assert source == "error"
    lm.assert_not_called()
    gem.assert_not_called()


def test_get_embedding_9router_registers_with_source():
    from embeddings import get_embedding
    with patch("embeddings._9router_embed", return_value=([0.1] * 4, "gemini/gemini-embedding-001")), \
         patch("embeddings.register_model") as reg:
        result, model_name, source = get_embedding("text", force_provider="9router",
                                                   model="gemini/gemini-embedding-001")
    assert source == "9router"
    assert model_name == "gemini/gemini-embedding-001"
    reg.assert_called_once_with("9router", "gemini/gemini-embedding-001", 4)


def test_get_embeddings_batch_9router():
    from embeddings import get_embeddings_batch
    with patch("embeddings._9router_embed_batch", return_value=([[0.1], [0.2]], "gemini/gemini-embedding-001")), \
         patch("embeddings.register_model"):
        vecs, model_name, source = get_embeddings_batch(["a", "b"], force_provider="9router",
                                                        model="gemini/gemini-embedding-001")
    assert vecs == [[0.1], [0.2]]
    assert source == "9router"


def test_list_9router_embed_models_offline_returns_empty():
    import embeddings
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")), \
         patch.object(embeddings, "_9r_embed_models_cache", None):
        assert embeddings.list_9router_embed_models() == []


# ── /v1/models fallback never grabs an arbitrary (possibly chat) model ────────

def test_lm_studio_embed_v1_fallback_skips_chat_models():
    """Old-LM-Studio path (/v1/models heuristic): pick the first EMBED-named
    model, not [0] — [0] can be a several-GB chat model that JIT-loads."""
    from embeddings import _lm_studio_embed
    models = {"data": [{"id": "gemma-4-e4b-it"}, {"id": "text-embedding-nomic-embed-text-v1.5"}]}
    expected = [0.5, 0.6]
    class _ModelsResp:
        def read(self): return json.dumps(models).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def fake(req, timeout=None):
        url = getattr(req, "full_url", str(req))
        if url.endswith("/models"):
            return _ModelsResp()
        return _fake_urlopen_lm_studio_embed(expected)
    with patch("urllib.request.urlopen", side_effect=fake), \
         patch("embeddings.list_lm_studio_models_v0", return_value=[]):
        vec, model = _lm_studio_embed("test")
    assert model == "text-embedding-nomic-embed-text-v1.5"
    assert vec == expected


def _fake_urlopen_9r_embed_with_model(rows: list, served_model: str):
    class _Resp:
        def read(self): return json.dumps({"data": rows, "model": served_model}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()


def test_9router_embed_rejects_substituted_model():
    """Same-dimension substitution (e.g. gemini-embedding-2-preview is also
    3072-d) would silently poison the collection — reject, never store."""
    import embeddings
    rows = [{"index": 0, "embedding": [0.1] * 4}]
    with patch("urllib.request.urlopen",
               return_value=_fake_urlopen_9r_embed_with_model(rows, "gemini-embedding-2-preview")), \
         patch.dict(embeddings._9r_embed_cooldown, {}, clear=True):
        with pytest.raises(RuntimeError, match="substituted"):
            embeddings._9router_embed("text", "gemini/gemini-embedding-001")


def test_9router_embed_accepts_prefix_stripped_echo():
    """The normal case: response echoes the requested id without the provider
    prefix (verified live) — that's the same model, accept it."""
    import embeddings
    rows = [{"index": 0, "embedding": [0.1, 0.2]}]
    with patch("urllib.request.urlopen",
               return_value=_fake_urlopen_9r_embed_with_model(rows, "gemini-embedding-001")), \
         patch.dict(embeddings._9r_embed_cooldown, {}, clear=True):
        vec, model = embeddings._9router_embed("text", "gemini/gemini-embedding-001")
    assert vec == [0.1, 0.2]
    assert model == "gemini/gemini-embedding-001"


def test_9router_embed_batch_rejects_substituted_model():
    import embeddings
    rows = [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [0.2]}]
    with patch("urllib.request.urlopen",
               return_value=_fake_urlopen_9r_embed_with_model(rows, "some-other-model")), \
         patch.dict(embeddings._9r_embed_cooldown, {}, clear=True):
        with pytest.raises(RuntimeError, match="substituted"):
            embeddings._9router_embed_batch(["a", "b"], "gemini/gemini-embedding-001")


def test_batch_full_failure_records_reason_for_job_log():
    """The (None,'','error') return can't carry WHY — last_embed_error() must,
    so the job log can tell 'pick a different model' from 'service offline'."""
    import embeddings
    with patch("embeddings._9router_embed_batch",
               side_effect=RuntimeError("9Router substituted embedding model (a → b) — rejected")):
        vecs, _, source = embeddings.get_embeddings_batch(
            ["x"], force_provider="9router", model="gemini/gemini-embedding-001")
    assert vecs is None and source == "error"
    assert "substituted" in embeddings.last_embed_error()


def test_batch_success_clears_last_error():
    import embeddings
    embeddings._last_error = "stale"
    embeddings._last_substitution = {"requested": "a", "served": "b"}
    with patch("embeddings._lm_studio_embed_batch", return_value=([[0.1]], "m")), \
         patch("embeddings.register_model"):
        vecs, _, source = embeddings.get_embeddings_batch(["x"], force_provider="lm_studio")
    assert vecs == [[0.1]]
    assert embeddings.last_embed_error() is None
    assert embeddings.last_substitution() is None


def test_substitution_rejection_records_structured_details():
    """The rejection must leave {requested, served} behind so the job manager
    can surface a one-click 'switch to the served model' recovery."""
    import embeddings
    rows = [{"index": 0, "embedding": [0.1] * 4}]
    with patch("urllib.request.urlopen",
               return_value=_fake_urlopen_9r_embed_with_model(rows, "gemini-embedding-2-preview")), \
         patch.dict(embeddings._9r_embed_cooldown, {}, clear=True):
        vecs, _, source = embeddings.get_embeddings_batch(
            ["x"], force_provider="9router", model="gemini/gemini-embedding-001")
    assert vecs is None and source == "error"
    assert embeddings.last_substitution() == {
        "requested": "gemini/gemini-embedding-001",
        "served": "gemini-embedding-2-preview",
    }


def test_get_embedding_single_also_records_reason():
    """The single-embed path (full/reanalyze jobs) surfaces the same honest
    reason as the batch path."""
    import embeddings
    with patch("embeddings._9router_embed",
               side_effect=RuntimeError("9Router substituted embedding model (a → b) — rejected")):
        vec, _, source = embeddings.get_embedding(
            "x", force_provider="9router", model="gemini/gemini-embedding-001")
    assert vec is None and source == "error"
    assert "substituted" in embeddings.last_embed_error()


def test_resolve_9router_embed_id_prefers_live_list():
    import embeddings
    with patch("embeddings.list_9router_embed_models",
               return_value=["gemini/gemini-embedding-001", "gemini/gemini-embedding-2-preview"]):
        assert embeddings.resolve_9router_embed_id(
            "gemini-embedding-2-preview", "gemini/gemini-embedding-001"
        ) == "gemini/gemini-embedding-2-preview"


def test_resolve_9router_embed_id_falls_back_to_requested_prefix():
    import embeddings
    with patch("embeddings.list_9router_embed_models", return_value=[]):
        assert embeddings.resolve_9router_embed_id(
            "gemini-embedding-2-preview", "gemini/gemini-embedding-001"
        ) == "gemini/gemini-embedding-2-preview"
