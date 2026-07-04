import json
import os
import re
import threading
import urllib.request
import urllib.error
from datetime import datetime
from constants import (
    LM_STUDIO_URL,
    GEMINI_API_KEY,
    GEMINI_BASE,
    EMBEDDING_REGISTRY_PATH,
)
from vision import list_lm_studio_models_v0

_GEMINI_EMBED_MODEL = "text-embedding-004"
import time as _time

_gemini_embed_cache: tuple[float, list[str]] | None = None


def list_gemini_embed_models(fallback: bool = True) -> list[str]:
    """Fetch Gemini embedding models from the API, cached 5 min.
    When `fallback=False` returns [] instead of hardcoded fallback on failure."""
    global _gemini_embed_cache
    if _gemini_embed_cache and _time.time() - _gemini_embed_cache[0] < 300:
        return _gemini_embed_cache[1]
    if not GEMINI_API_KEY:
        return [_GEMINI_EMBED_MODEL] if fallback else []
    try:
        url = f"{GEMINI_BASE}/models?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        models = [
            m["name"].replace("models/", "")
            for m in data.get("models", [])
            if "embedContent" in m.get("supportedGenerationMethods", [])
        ]
        result = models if models else ([_GEMINI_EMBED_MODEL] if fallback else [])
        _gemini_embed_cache = (_time.time(), result)
        return result
    except Exception as e:
        print(f"[embeddings] Gemini model list failed: {e}")
        return [_GEMINI_EMBED_MODEL] if fallback else []


# ── Collection naming ─────────────────────────────────────────────────────────


def collection_name_for(model_name: str) -> str:
    """Stable, ChromaDB-safe collection name for a given embedding model."""
    safe = re.sub(r"[^a-z0-9]", "_", model_name.lower()).strip("_")
    return f"img_{safe}"[:63]


# ── Registry (persists all models ever used + active selection) ───────────────


def _default_registry() -> dict:
    return {"active_model": None, "models": {}}


def _load_registry() -> dict:
    if os.path.exists(EMBEDDING_REGISTRY_PATH):
        try:
            with open(EMBEDDING_REGISTRY_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[embeddings] registry read failed ({e}); starting from an empty registry")
    return _default_registry()


def _save_registry(reg: dict):
    """Atomic write: a crash/kill mid-write must never leave a truncated
    registry file that _load_registry then fails to parse."""
    d = os.path.dirname(EMBEDDING_REGISTRY_PATH) or "."
    os.makedirs(d, exist_ok=True)
    tmp_path = f"{EMBEDDING_REGISTRY_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp_path, EMBEDDING_REGISTRY_PATH)


# Guards the load-modify-save sequence in register_model/set_active_model — the
# API thread and a background job thread can both hit these around the same
# time, and an unlocked read-modify-write can silently lose one update.
_registry_lock = threading.Lock()


def register_model(source: str, model_name: str, dimension: int):
    """Record a model in the registry. Sets it as active if first model."""
    with _registry_lock:
        reg = _load_registry()
        existing = reg["models"].get(model_name)
        if existing is None:
            reg["models"][model_name] = {
                "source": source,
                "dimension": dimension,
                "collection": collection_name_for(model_name),
                "first_used": datetime.now().isoformat(timespec="seconds"),
            }
            print(
                f"[embeddings] Registered new model: {model_name} ({source}, {dimension}d)"
            )
        elif existing.get("dimension") != dimension:
            # Different dimension under the same model name would silently
            # corrupt the collection (Chroma stores whatever vector it's
            # given, and a mixed-dimension collection breaks similarity
            # search for every vector already in it) — refuse instead of
            # quietly registering/using the mismatched dimension. Callers
            # (get_embedding/get_embeddings_batch) already catch exceptions
            # from register_model and surface them as a normal per-item
            # failure, so this doesn't crash the whole indexing pass.
            raise RuntimeError(
                f"Model '{model_name}' embedding dimension changed "
                f"({existing.get('dimension')} -> {dimension}) — mixing vector "
                "sizes in one collection would corrupt search results; "
                "re-index required (use a fresh model name/collection)."
            )
        reg["models"].setdefault(model_name, {})["last_used"] = datetime.now().isoformat(
            timespec="seconds"
        )
        if reg["active_model"] is None:
            reg["active_model"] = model_name
            print(f"[embeddings] Active model set to: {model_name}")
        _save_registry(reg)


def get_registry() -> dict:
    return _load_registry()


def get_active_model() -> str | None:
    return _load_registry().get("active_model")


def set_active_model(model_name: str):
    with _registry_lock:
        reg = _load_registry()
        if model_name not in reg.get("models", {}):
            raise ValueError(
                f"Model '{model_name}' not in registry — index some photos with it first"
            )
        reg["active_model"] = model_name
        _save_registry(reg)


# ── Connection error detection ────────────────────────────────────────────────


def _is_connection_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        w in msg
        for w in (
            "connection",
            "refused",
            "unreachable",
            "timeout",
            "connect error",
            "cannot connect",
        )
    )


# ── Provider implementations ──────────────────────────────────────────────────


def _lm_embed_model_id() -> str | None:
    """Best-effort id of the currently loaded LM Studio EMBEDDING model, using
    the v0 API's real type/loaded-state info (mirrors vision._lm_model_id()'s
    vision-model lookup). Returns None when the v0 API is unreachable or no
    embeddings model is currently loaded — callers fall back to the plain
    /v1/models heuristic (first entry, no type/state filtering) in that case."""
    try:
        v0 = list_lm_studio_models_v0()
    except Exception:
        v0 = []
    for m in v0:
        if m.get("state") == "loaded" and m.get("type") == "embeddings":
            return m.get("id")
    return None


def _lm_studio_embed(text: str, model: str = None) -> tuple[list, str]:
    """Embed via LM Studio /v1/embeddings. Uses `model` if given, else prefers
    the v0-API-reported loaded embeddings model, else the /v1/models heuristic."""
    model_name = model or _lm_embed_model_id()
    if not model_name:
        model_name = "lm_studio_embed"
        try:
            req = urllib.request.Request(f"{LM_STUDIO_URL}/models")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
                models = data.get("data", [])
                if models:
                    model_name = models[0].get("id", model_name)
        except Exception:
            pass

    payload = json.dumps({"model": model_name, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        result = json.loads(r.read())
    return result["data"][0]["embedding"], model_name


def _lm_studio_embed_batch(texts: list[str], model: str = None) -> tuple[list, str]:
    """Embed many texts in ONE /v1/embeddings call (the API takes a list).
    Returns (vectors in input order, model_name)."""
    model_name = model or _lm_embed_model_id()
    if not model_name:
        model_name = "lm_studio_embed"
        try:
            req = urllib.request.Request(f"{LM_STUDIO_URL}/models")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
                models = data.get("data", [])
                if models:
                    model_name = models[0].get("id", model_name)
        except Exception:
            pass

    payload = json.dumps({"model": model_name, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
    rows = sorted(result["data"], key=lambda d: d.get("index", 0))
    return [row["embedding"] for row in rows], model_name


# Same cooldown-tracking pattern as vision._call_gemini/gemini_cooldowns(): Gemini
# has no "remaining quota" endpoint, so a 429 is the only real signal. Embeddings
# has no fallback chain *within* Gemini (single model), so this can't skip to a
# sibling model like vision does — but it does stop hammering an already-limited
# model immediately, and exposes cooldown state for the UI the same way.
_gemini_embed_cooldown: dict[str, float] = {}
_EMBED_RATE_LIMIT_COOLDOWN_SEC = 90


def _mark_embed_rate_limited(model: str, retry_after: str | None = None):
    delay = _EMBED_RATE_LIMIT_COOLDOWN_SEC
    if retry_after:
        try:
            delay = max(delay, int(retry_after))
        except ValueError:
            pass
    _gemini_embed_cooldown[model] = _time.time() + delay


def gemini_embed_cooldowns() -> dict[str, float]:
    """{model: seconds_remaining} for embedding models currently in a post-429 cooldown."""
    now = _time.time()
    return {
        m: round(until - now, 1) for m, until in _gemini_embed_cooldown.items() if until > now
    }


def _gemini_embed(text: str, model: str = None) -> tuple[list, str]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    model_name = model or _GEMINI_EMBED_MODEL
    if _gemini_embed_cooldown.get(model_name, 0) > _time.time():
        raise RuntimeError(
            f"Gemini embed model {model_name} in post-429 cooldown — skipping retry"
        )
    url = f"{GEMINI_BASE}/models/{model_name}:embedContent?key={GEMINI_API_KEY}"
    payload = json.dumps(
        {
            "model": f"models/{model_name}",
            "content": {"parts": [{"text": text}]},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            result = json.loads(r.read())
        return result["embedding"]["values"], model_name
    except urllib.error.HTTPError as e:
        if e.code == 429:
            _mark_embed_rate_limited(model_name, e.headers.get("Retry-After") if e.headers else None)
        raise RuntimeError(f"Gemini embed {e.code}: {e.read()[:200]}")


# ── Public API ────────────────────────────────────────────────────────────────


def get_embedding(
    text: str, force_provider: str = "auto", model: str = None
) -> tuple[list | None, str, str]:
    """Returns (vector, model_name, source).
    force_provider: "auto" (LM Studio → Gemini), "lm_studio", or "gemini".
    model: explicit embedding model id for the forced provider. In "auto" mode it
    is only passed to LM Studio (a Gemini fallback picks its own default).
    Registers the model on success. Returns (None, '', 'error') on full failure."""
    lm = ("lm_studio", lambda t: _lm_studio_embed(t, model))
    gem = ("gemini", lambda t: _gemini_embed(t, model if force_provider == "gemini" else None))
    if force_provider == "lm_studio":
        chain = [lm]
    elif force_provider == "gemini":
        chain = [gem]
    else:
        chain = [lm, gem]

    for source, fn in chain:
        try:
            vector, model_name = fn(text)
            register_model(source, model_name, len(vector))
            return vector, model_name, source
        except Exception as e:
            if _is_connection_error(e):
                print(f"[embeddings] {source} offline, trying next")
                continue
            print(f"[embeddings] {source} error: {e}")
            continue
    return None, "", "error"


def get_embeddings_batch(
    texts: list[str], force_provider: str = "auto", model: str = None
) -> tuple[list | None, str, str]:
    """Batch variant of get_embedding: returns (vectors in input order,
    model_name, source), or (None, '', 'error') on full failure.
    LM Studio embeds the whole list in ONE request — a failure there fails the
    whole chunk since there's no partial result to salvage from a single HTTP
    call. Gemini has no batch endpoint on the free tier, so it loops one
    request per text; a per-item failure there does NOT discard the rest of
    the chunk — that slot in the returned list is None while every other text
    still gets embedded, so callers must check for (and handle) None entries
    whenever the overall result isn't None."""
    if not texts:
        return [], "", ""

    def _gem_batch(ts):
        m = model if force_provider == "gemini" else None
        vecs, name = [], None
        for t in ts:
            try:
                v, name = _gemini_embed(t, m)
                vecs.append(v)
            except Exception as e:
                print(f"[embeddings] gemini batch item error: {e}")
                vecs.append(None)
        return vecs, name

    lm = ("lm_studio", lambda ts: _lm_studio_embed_batch(ts, model))
    gem = ("gemini", _gem_batch)
    if force_provider == "lm_studio":
        chain = [lm]
    elif force_provider == "gemini":
        chain = [gem]
    else:
        chain = [lm, gem]

    for source, fn in chain:
        try:
            vectors, model_name = fn(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"{source} returned {len(vectors)} vectors for {len(texts)} inputs"
                )
            dims = [len(v) for v in vectors if v is not None]
            if not dims:
                raise RuntimeError(f"{source} failed to embed every item in the batch")
            register_model(source, model_name, dims[0])
            return vectors, model_name, source
        except Exception as e:
            if _is_connection_error(e):
                print(f"[embeddings] {source} offline, trying next")
                continue
            print(f"[embeddings] {source} batch error: {e}")
            continue
    return None, "", "error"
