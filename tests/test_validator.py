import json
import pytest
import urllib.error
from unittest.mock import patch, MagicMock


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


# ── check_ollama ─────────────────────────────────────────────────────────────

def test_check_ollama_online():
    from validator import check_ollama
    with patch("urllib.request.urlopen", return_value=_http_ok()):
        assert check_ollama() is True


def test_check_ollama_offline():
    from validator import check_ollama
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        assert check_ollama() is False


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


# ── validate_environment ──────────────────────────────────────────────────────

def test_validate_all_online():
    from validator import validate_environment
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.check_ollama", return_value=True):
        errors = validate_environment()
    assert errors == []


def test_validate_lm_offline_gemini_fallback():
    from validator import validate_environment
    with patch("validator.check_lm_studio", return_value=False), \
         patch("validator.check_ollama", return_value=True), \
         patch("validator.check_gemini", return_value=True):
        errors = validate_environment()
    assert len(errors) == 1
    assert "Gemini fallback" in errors[0]
    assert "LM Studio" in errors[0]


def test_validate_ollama_offline_gemini_fallback():
    from validator import validate_environment
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.check_ollama", return_value=False), \
         patch("validator.check_gemini", return_value=True):
        errors = validate_environment()
    assert len(errors) == 1
    assert "Gemini fallback" in errors[0]
    assert "Ollama" in errors[0]


def test_validate_both_offline_no_gemini_critical_error():
    from validator import validate_environment
    with patch("validator.check_lm_studio", return_value=False), \
         patch("validator.check_ollama", return_value=False), \
         patch("validator.check_gemini", return_value=False):
        errors = validate_environment()
    assert len(errors) == 2
    assert all("GEMINI_API_KEY" in e or "Gemini unavailable" in e for e in errors)


def test_validate_both_offline_gemini_available():
    from validator import validate_environment
    with patch("validator.check_lm_studio", return_value=False), \
         patch("validator.check_ollama", return_value=False), \
         patch("validator.check_gemini", return_value=True):
        errors = validate_environment()
    assert len(errors) == 2
    assert all("Gemini fallback" in e for e in errors)


# ── service_status ────────────────────────────────────────────────────────────

def test_service_status_all_online():
    from validator import service_status
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.check_ollama", return_value=True), \
         patch("validator.check_gemini", return_value=True), \
         patch("validator.GEMINI_API_KEY", "fake-key"):
        status = service_status()
    assert status["lm_studio"] is True
    assert status["ollama"] is True
    assert status["gemini"] is True
    assert status["gemini_key_set"] is True


def test_service_status_no_key():
    from validator import service_status
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.check_ollama", return_value=True), \
         patch("validator.check_gemini", return_value=False), \
         patch("validator.GEMINI_API_KEY", ""):
        status = service_status()
    assert status["gemini_key_set"] is False
    assert status["gemini"] is False
