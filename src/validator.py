import urllib.request
import urllib.error
from constants import GEMINI_API_KEY, GEMINI_BASE, LM_STUDIO_URL


def check_lm_studio(url: str | None = None) -> bool:
    """Probe LM Studio's OpenAI-compatible /models endpoint. Defaults to
    constants.LM_STUDIO_URL, which already includes the `/v1` suffix (and is
    itself overridable via the LM_STUDIO_URL env var, e.g. for a Docker host)
    — so the check respects the configurable host instead of hardcoding
    localhost. Pass `url` only if it already includes the `/v1` (or
    equivalent) suffix; it is not appended here."""
    base = url if url is not None else LM_STUDIO_URL
    try:
        urllib.request.urlopen(f"{base}/models", timeout=3)
        return True
    except Exception:
        return False


def check_gemini() -> bool:
    if not GEMINI_API_KEY:
        return False
    # Quick probe: list models endpoint
    url = f"{GEMINI_BASE}/models?key={GEMINI_API_KEY}&pageSize=1"
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False


def service_status() -> dict:
    """Service health for the Services panel: LM Studio (vision + embeddings) + Gemini fallback."""
    return {
        "lm_studio": check_lm_studio(),
        "gemini": check_gemini(),
        "gemini_key_set": bool(GEMINI_API_KEY),
    }
