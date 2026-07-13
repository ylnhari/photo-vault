import concurrent.futures
import time
import urllib.request
import urllib.error
from constants import GEMINI_API_KEY, GEMINI_BASE, LM_STUDIO_URL, NINEROUTER_URL


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


def lm_studio_loaded_state() -> dict:
    """What LM Studio actually has resident, from its native v0 API — the
    server answering /v1/models does NOT mean a model is loaded (it lists
    JIT-loadable models even with nothing in memory, which is exactly how the
    UI used to show 'online' while vision requests were about to fail).
    Returns {"known": bool, "vision_loaded": str|None, "embed_loaded": str|None};
    known=False when the v0 API is unreachable (old LM Studio or offline)."""
    from vision import list_lm_studio_models_v0

    v0 = list_lm_studio_models_v0()
    if not v0:
        return {"known": False, "vision_loaded": None, "embed_loaded": None}
    vision = next((m["id"] for m in v0 if m.get("state") == "loaded" and m.get("type") == "vlm"), None)
    embed = next((m["id"] for m in v0 if m.get("state") == "loaded" and m.get("type") == "embeddings"), None)
    return {"known": True, "vision_loaded": vision, "embed_loaded": embed}


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


def check_9router() -> bool:
    """Probe the local 9Router gateway (OpenAI-compatible /models endpoint)."""
    try:
        urllib.request.urlopen(f"{NINEROUTER_URL}/models", timeout=3)
        return True
    except Exception:
        return False


# The three service probes are independent network round-trips, each blocking on
# its own connection timeout (LM Studio 3s + Gemini 5s + 9Router 3s). Run serially
# that is up to ~11s (measured ~6.4s) on every /api/health — which the SPA calls on
# load, gating the first render. Run them concurrently instead (wall time = the
# slowest probe) and memoize the result briefly so repeat calls are instant. TTL is
# short: bringing a service online should reflect in the UI within a few seconds.
_STATUS_TTL = 8.0
_status_cache: dict = {"at": 0.0, "data": None}


def service_status(force: bool = False) -> dict:
    """Service health for the Services panel: LM Studio (server + what's
    actually loaded), Gemini fallback, and the 9Router gateway.

    Pass force=True to bypass the short-TTL memo — the "Recheck" button needs a
    live probe (the user just started/stopped a model and wants the truth now),
    whereas the automatic on-load fetch is happy with a few-seconds-old snapshot."""
    now = time.time()
    if not force and _status_cache["data"] is not None and now - _status_cache["at"] < _STATUS_TTL:
        return _status_cache["data"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_lm = ex.submit(check_lm_studio)
        f_gemini = ex.submit(check_gemini)
        f_9router = ex.submit(check_9router)
        lm_up = f_lm.result()
        gemini_up = f_gemini.result()
        ninerouter_up = f_9router.result()
    # Only probe LM Studio's loaded-model state when the server answered — that
    # call is a second round-trip and is pointless (and slow) when it's offline.
    lm_state = lm_studio_loaded_state() if lm_up else {"known": False, "vision_loaded": None, "embed_loaded": None}
    data = {
        "lm_studio": lm_up,
        "lm_studio_state": lm_state,
        "gemini": gemini_up,
        "gemini_key_set": bool(GEMINI_API_KEY),
        "ninerouter": ninerouter_up,
    }
    _status_cache.update({"at": time.time(), "data": data})
    return data
