import urllib.request
import urllib.error
from constants import GEMINI_API_KEY, GEMINI_BASE


def check_lm_studio(url="http://localhost:1234") -> bool:
    try:
        urllib.request.urlopen(f"{url}/v1/models", timeout=3)
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
