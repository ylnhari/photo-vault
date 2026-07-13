import json
import pytest
import urllib.error
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _reset_service_status_memo():
    """service_status() memoizes its probe result for a few seconds (so the
    dashboard's on-load fetch is instant). That module-level cache would leak a
    prior test's mocked service state into the next call, so clear it around
    every test — each test drives its own mocked probes and expects them read."""
    import validator
    validator._status_cache = {"at": 0.0, "data": None}
    yield
    validator._status_cache = {"at": 0.0, "data": None}


def _http_ok():
    class _Resp:
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()


def _http_error(code: int):
    err = urllib.error.HTTPError(url="", code=code, msg="", hdrs=None, fp=None)
    err.read = lambda: b""
    return err


# ── check_lm_studio ───────────────────────────────────────────────────────────

def test_check_lm_studio_online():
    from validator import check_lm_studio
    with patch("urllib.request.urlopen", return_value=_http_ok()):
        assert check_lm_studio() is True


def test_check_lm_studio_offline():
    from validator import check_lm_studio
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
        assert check_lm_studio() is False


def test_check_lm_studio_timeout():
    from validator import check_lm_studio
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
        assert check_lm_studio() is False


# ── check_gemini ─────────────────────────────────────────────────────────────

def test_check_gemini_no_key_returns_false():
    from validator import check_gemini
    with patch("validator.GEMINI_API_KEY", ""):
        assert check_gemini() is False


def test_check_gemini_with_key_online():
    from validator import check_gemini
    with patch("validator.GEMINI_API_KEY", "fake-key"), \
         patch("urllib.request.urlopen", return_value=_http_ok()):
        assert check_gemini() is True


def test_check_gemini_with_key_offline():
    from validator import check_gemini
    with patch("validator.GEMINI_API_KEY", "fake-key"), \
         patch("urllib.request.urlopen", side_effect=_http_error(403)):
        assert check_gemini() is False


# ── service_status ────────────────────────────────────────────────────────────

_LM_STATE_LOADED = {"known": True, "vision_loaded": "gemma-4-e4b-it", "embed_loaded": None}


def test_service_status_all_online():
    from validator import service_status
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.lm_studio_loaded_state", return_value=_LM_STATE_LOADED), \
         patch("validator.check_gemini", return_value=True), \
         patch("validator.check_9router", return_value=True), \
         patch("validator.GEMINI_API_KEY", "fake-key"):
        status = service_status()
    assert status["lm_studio"] is True
    assert status["gemini"] is True
    assert status["gemini_key_set"] is True
    assert status["ninerouter"] is True
    assert status["lm_studio_state"]["vision_loaded"] == "gemma-4-e4b-it"
    assert "ollama" not in status


def test_service_status_no_key():
    from validator import service_status
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.lm_studio_loaded_state", return_value=_LM_STATE_LOADED), \
         patch("validator.check_gemini", return_value=False), \
         patch("validator.check_9router", return_value=False), \
         patch("validator.GEMINI_API_KEY", ""):
        status = service_status()
    assert status["gemini_key_set"] is False
    assert status["gemini"] is False
    assert status["ninerouter"] is False


def test_service_status_lm_studio_down_skips_loaded_probe():
    """When the server itself is unreachable there's nothing to introspect —
    no v0 call is made and the state is the unknown shape."""
    from validator import service_status
    with patch("validator.check_lm_studio", return_value=False), \
         patch("validator.lm_studio_loaded_state") as loaded, \
         patch("validator.check_gemini", return_value=False), \
         patch("validator.check_9router", return_value=False), \
         patch("validator.GEMINI_API_KEY", ""):
        status = service_status()
    assert status["lm_studio"] is False
    assert status["lm_studio_state"] == {"known": False, "vision_loaded": None, "embed_loaded": None}
    loaded.assert_not_called()


# ── check_9router ─────────────────────────────────────────────────────────────

def test_check_9router_online():
    from validator import check_9router
    with patch("urllib.request.urlopen", return_value=_http_ok()):
        assert check_9router() is True


def test_check_9router_offline():
    from validator import check_9router
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
        assert check_9router() is False


# ── lm_studio_loaded_state ────────────────────────────────────────────────────

def test_lm_studio_loaded_state_reports_loaded_models():
    from validator import lm_studio_loaded_state
    v0 = [
        {"id": "gemma-4-e4b-it", "type": "vlm", "state": "loaded"},
        {"id": "nomic", "type": "embeddings", "state": "loaded"},
        {"id": "other-vlm", "type": "vlm", "state": "not-loaded"},
    ]
    with patch("vision.list_lm_studio_models_v0", return_value=v0):
        s = lm_studio_loaded_state()
    assert s == {"known": True, "vision_loaded": "gemma-4-e4b-it", "embed_loaded": "nomic"}


def test_lm_studio_loaded_state_nothing_loaded():
    """Server up, models listed, nothing resident — the exact situation the
    UI used to misreport as ready."""
    from validator import lm_studio_loaded_state
    v0 = [
        {"id": "gemma-4-e4b-it", "type": "vlm", "state": "not-loaded"},
        {"id": "nomic", "type": "embeddings", "state": "not-loaded"},
    ]
    with patch("vision.list_lm_studio_models_v0", return_value=v0):
        s = lm_studio_loaded_state()
    assert s == {"known": True, "vision_loaded": None, "embed_loaded": None}


def test_lm_studio_loaded_state_v0_unreachable():
    from validator import lm_studio_loaded_state
    with patch("vision.list_lm_studio_models_v0", return_value=[]):
        s = lm_studio_loaded_state()
    assert s == {"known": False, "vision_loaded": None, "embed_loaded": None}
