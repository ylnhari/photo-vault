import urllib.request
import urllib.error
from constants import GEMINI_API_KEY, GEMINI_BASE


def check_lm_studio(url="http://localhost:1234") -> bool:
    try:
        urllib.request.urlopen(f"{url}/v1/models", timeout=3)
        return True
    except Exception:
        return False


def check_ollama(url="http://localhost:11434") -> bool:
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=3)
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


def validate_environment() -> list[str]:
    """Returns list of warning strings. Empty = all good."""
    lm_ok     = check_lm_studio()
    ollama_ok = check_ollama()
    gemini_ok = check_gemini() if (not lm_ok or not ollama_ok) else None

    errors = []

    if not lm_ok:
        if gemini_ok:
            errors.append("LM Studio offline — vision analysis will use Gemini fallback (images sent to Google API)")
        else:
            errors.append("LM Studio offline and Gemini unavailable — indexing will fail. Start LM Studio or add GEMINI_API_KEY to .env")

    if not ollama_ok:
        if gemini_ok:
            errors.append("Ollama offline — embeddings will use Gemini fallback (note: mixing Ollama+Gemini embeddings reduces search accuracy)")
        else:
            errors.append("Ollama offline and Gemini unavailable — indexing will fail. Start Ollama or add GEMINI_API_KEY to .env")

    return errors


def service_status() -> dict:
    """Full status dict for the Services panel in the UI."""
    lm_ok     = check_lm_studio()
    ollama_ok = check_ollama()
    gemini_ok = check_gemini()
    return {
        "lm_studio": lm_ok,
        "ollama":    ollama_ok,
        "gemini":    gemini_ok,
        "gemini_key_set": bool(GEMINI_API_KEY),
    }
