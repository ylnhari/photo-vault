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

def test_service_status_all_online():
    from validator import service_status
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.check_gemini", return_value=True), \
         patch("validator.GEMINI_API_KEY", "fake-key"):
        status = service_status()
    assert status["lm_studio"] is True
    assert status["gemini"] is True
    assert status["gemini_key_set"] is True
    assert "ollama" not in status


def test_service_status_no_key():
    from validator import service_status
    with patch("validator.check_lm_studio", return_value=True), \
         patch("validator.check_gemini", return_value=False), \
         patch("validator.GEMINI_API_KEY", ""):
        status = service_status()
    assert status["gemini_key_set"] is False
    assert status["gemini"] is False
