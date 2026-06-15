import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime
from constants import LM_STUDIO_URL, GEMINI_API_KEY, GEMINI_BASE, EMBEDDING_REGISTRY_PATH

_GEMINI_EMBED_MODEL = "text-embedding-004"


# ── Collection naming ─────────────────────────────────────────────────────────

def collection_name_for(model_name: str) -> str:
    """Stable, ChromaDB-safe collection name for a given embedding model."""
    safe = re.sub(r'[^a-z0-9]', '_', model_name.lower()).strip('_')
    return f"img_{safe}"[:63]


# ── Registry (persists all models ever used + active selection) ───────────────

def _load_registry() -> dict:
    if os.path.exists(EMBEDDING_REGISTRY_PATH):
        with open(EMBEDDING_REGISTRY_PATH) as f:
            return json.load(f)
    return {"active_model": None, "models": {}}


def _save_registry(reg: dict):
    os.makedirs(os.path.dirname(EMBEDDING_REGISTRY_PATH), exist_ok=True)
    with open(EMBEDDING_REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def register_model(source: str, model_name: str, dimension: int):
    """Record a model in the registry. Sets it as active if first model."""
    reg = _load_registry()
    if model_name not in reg["models"]:
        reg["models"][model_name] = {
            "source": source,
            "dimension": dimension,
            "collection": collection_name_for(model_name),
            "first_used": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"[embeddings] Registered new model: {model_name} ({source}, {dimension}d)")
    reg["models"][model_name]["last_used"] = datetime.now().isoformat(timespec="seconds")
    if reg["active_model"] is None:
        reg["active_model"] = model_name
        print(f"[embeddings] Active model set to: {model_name}")
    _save_registry(reg)


def get_registry() -> dict:
    return _load_registry()


def get_active_model() -> str | None:
    return _load_registry().get("active_model")


def set_active_model(model_name: str):
    reg = _load_registry()
    if model_name not in reg.get("models", {}):
        raise ValueError(f"Model '{model_name}' not in registry — index some photos with it first")
    reg["active_model"] = model_name
    _save_registry(reg)


# ── Connection error detection ────────────────────────────────────────────────

def _is_connection_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(w in msg for w in ("connection", "refused", "unreachable", "timeout", "connect error", "cannot connect"))


# ── Provider implementations ──────────────────────────────────────────────────

def _lm_studio_embed(text: str, model: str = None) -> tuple[list, str]:
    """Embed via LM Studio /v1/embeddings. Uses `model` if given, else auto-detects."""
    model_name = model or "lm_studio_embed"
    if not model:
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
        f"{LM_STUDIO_URL}/embeddings", data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        result = json.loads(r.read())
    return result["data"][0]["embedding"], model_name


def _gemini_embed(text: str) -> tuple[list, str]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"{GEMINI_BASE}/models/{_GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = json.dumps({
        "model": f"models/{_GEMINI_EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            result = json.loads(r.read())
        return result["embedding"]["values"], _GEMINI_EMBED_MODEL
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gemini embed {e.code}: {e.read()[:200]}")


# ── Public API ────────────────────────────────────────────────────────────────

def get_embedding(text: str, force_provider: str = "auto", model: str = None) -> tuple[list | None, str, str]:
    """Returns (vector, model_name, source).
    force_provider: "auto" (LM Studio → Gemini), "lm_studio", or "gemini".
    model: explicit LM Studio embedding model id (ignored for Gemini, which is fixed).
    Registers the model on success. Returns (None, '', 'error') on full failure."""
    lm = ("lm_studio", lambda t: _lm_studio_embed(t, model))
    gem = ("gemini", _gemini_embed)
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
