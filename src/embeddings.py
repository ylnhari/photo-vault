import json
import hashlib
import math
import os
import re
import threading
import urllib.request
import urllib.error
from collections.abc import Sequence
from datetime import datetime
from constants import (
    LM_STUDIO_URL,
    NINEROUTER_URL,
    GEMINI_API_KEY,
    GEMINI_BASE,
    EMBEDDING_REGISTRY_PATH,
)
from vision import list_lm_studio_models_v0, _EMBED_NAME_PATTERNS
import ratelimit

_GEMINI_EMBED_MODEL = "gemini-embedding-001"
_PROFILE_SCHEMA_VERSION = 1
_EMBEDDING_TEXT_SCHEMA = "caption-attributes-v1"
_VECTOR_METRIC = "cosine"
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
        # Avoid echoing request URLs or query-string credentials from urllib
        # exceptions into logs.
        status = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__
        print(f"[embeddings] Gemini model list failed ({status})")
        return [_GEMINI_EMBED_MODEL] if fallback else []


# ── Collection naming ─────────────────────────────────────────────────────────


def _legacy_collection_name(model_name: str) -> str:
    safe = re.sub(r"[^a-z0-9]", "_", model_name.lower()).strip("_")
    return f"img_{safe}"[:63]


def _canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _profile_for(source: str, model_name: str) -> dict:
    """Return the immutable retrieval recipe for an unregistered model.

    Existing registry entries without ``profile`` are legacy raw-text profiles;
    callers must preserve their old preprocessing and collection.
    """
    if source == "gemini" and model_name == "gemini-embedding-2":
        strategy = "gemini-embedding-2-query-document-prefix-v1"
    elif source == "gemini" and model_name == "gemini-embedding-001":
        strategy = "gemini-embedding-001-retrieval-task-v1"
    elif source == "lm_studio" and "nomic-embed-text" in model_name.lower():
        strategy = "nomic-search-prefix-v1"
    else:
        strategy = "symmetric-raw-v1"
    return {
        "schema_version": _PROFILE_SCHEMA_VERSION,
        "provider": source,
        "model": model_name,
        "input_schema": _EMBEDDING_TEXT_SCHEMA,
        "task_strategy": strategy,
        "metric": _VECTOR_METRIC,
    }


def _profile_for_registered(source: str, model_name: str) -> tuple[dict, bool]:
    """Return (profile, is_legacy) and reject provider identity drift."""
    entry = _load_registry().get("models", {}).get(model_name)
    if entry:
        old_source = entry.get("source")
        if old_source and old_source != source:
            raise RuntimeError(
                f"Embedding model '{model_name}' is registered for provider "
                f"'{old_source}', not '{source}'; choose a distinct model id "
                "to keep vector spaces separate."
            )
        if not entry.get("profile"):
            return {
                "schema_version": 0,
                "provider": source,
                "model": model_name,
                "input_schema": "legacy-raw-text",
                "task_strategy": "legacy-raw-v0",
                "metric": "legacy-unspecified",
            }, True
        return entry["profile"], False
    return _profile_for(source, model_name), False


def _profile_id(profile: dict, dimension: int) -> str:
    complete = {**profile, "dimension": int(dimension)}
    return hashlib.sha256(_canonical_json(complete).encode("utf-8")).hexdigest()[:20]


def _profile_collection_name(model_name: str, profile: dict, dimension: int) -> str:
    safe = re.sub(r"[^a-z0-9]", "_", model_name.lower()).strip("_")[:28].strip("_") or "model"
    digest = _profile_id(profile, dimension)
    return f"img_{safe}_{digest}"[:63]


def collection_name_for(model_name: str) -> str:
    """Return the recorded collection, preserving legacy names exactly."""
    entry = _load_registry().get("models", {}).get(model_name)
    if entry:
        return entry.get("collection") or _legacy_collection_name(model_name)
    # This fallback is for old read-only callers before a model has been
    # registered. Successful new embeddings register their hashed profile first.
    return _legacy_collection_name(model_name)


# ── Registry (persists all models ever used + active selection) ───────────────


def _default_registry() -> dict:
    return {"active_model": None, "models": {}}


def _load_registry() -> dict:
    if os.path.exists(EMBEDDING_REGISTRY_PATH):
        try:
            with open(EMBEDDING_REGISTRY_PATH) as f:
                registry = json.load(f)
            if (not isinstance(registry, dict)
                    or not isinstance(registry.get("models", {}), dict)
                    or registry.get("active_model") is not None
                    and not isinstance(registry.get("active_model"), str)):
                raise ValueError("registry has an invalid structure")
            return registry
        except (json.JSONDecodeError, OSError, ValueError) as e:
            raise RuntimeError(f"embedding registry is unreadable: {e}") from e
    return _default_registry()


def _save_registry(reg: dict):
    """Atomic write: a crash/kill mid-write must never leave a truncated
    registry file that _load_registry then fails to parse.

    On Windows, os.replace raises PermissionError (WinError 5/32) when another
    process momentarily holds the destination open — the UI polls endpoints
    that READ this file (/api/models, /api/provider-models) while an embed job
    rewrites it, and Windows opens the reader's handle without FILE_SHARE_DELETE,
    so the replace hits a sharing violation. That's transient, not a real
    failure, so retry the replace briefly before giving up (a whole embed job
    was aborted by exactly this race)."""
    d = os.path.dirname(EMBEDDING_REGISTRY_PATH) or "."
    os.makedirs(d, exist_ok=True)
    tmp_path = f"{EMBEDDING_REGISTRY_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(reg, f, indent=2)
    last_err = None
    for attempt in range(12):
        try:
            os.replace(tmp_path, EMBEDDING_REGISTRY_PATH)
            return
        except PermissionError as e:  # transient Windows sharing violation
            last_err = e
            _time.sleep(0.05 * (attempt + 1))  # ~0.05..0.6s, ~3.9s total
    # Exhausted retries — remove the temp so we don't leak .tmp files, then
    # re-raise so the caller counts a real failure instead of silently losing
    # the update.
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    raise last_err


# Guards the load-modify-save sequence in register_model/set_active_model — the
# API thread and a background job thread can both hit these around the same
# time, and an unlocked read-modify-write can silently lose one update.
_registry_lock = threading.Lock()


def register_model(source: str, model_name: str, dimension: int, profile: dict = None):
    """Record or validate a model profile. Sets it active only when first used."""
    with _registry_lock:
        reg = _load_registry()
        existing = reg["models"].get(model_name)
        now_iso = datetime.now().isoformat(timespec="seconds")
        dimension = int(dimension)
        # Only persist to disk when something worth persisting actually changed.
        # register_model runs on EVERY embed, and a per-image rewrite just to
        # advance a seconds-resolution last_used is pure churn — 24k disk writes
        # for a timestamp — and every one of those writes is a chance to lose
        # the Windows os.replace sharing-violation race above. So touch
        # last_used at day granularity only.
        changed = False
        if existing is None:
            profile = profile or _profile_for(source, model_name)
            complete_profile = {**profile, "dimension": dimension}
            profile_id = _profile_id(profile, dimension)
            reg["models"][model_name] = {
                "source": source,
                "dimension": dimension,
                "profile": complete_profile,
                "profile_id": profile_id,
                "collection": _profile_collection_name(model_name, profile, dimension),
                "first_used": now_iso,
                "last_used": now_iso,
            }
            changed = True
            print(
                f"[embeddings] Registered new model: {model_name} ({source}, {dimension}d)"
            )
        elif existing.get("source") != source:
            raise RuntimeError(
                f"Embedding model '{model_name}' is already registered for "
                f"provider '{existing.get('source')}', not '{source}'; refusing "
                "to mix provider vector spaces under one model id."
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
        else:
            if existing.get("profile"):
                expected = profile or _profile_for(source, model_name)
                expected_id = _profile_id(expected, dimension)
                if existing.get("profile_id") != expected_id:
                    raise RuntimeError(
                        f"Embedding profile changed for '{model_name}'; refusing "
                        "to mix vectors. Select a new profile and rebuild its collection."
                    )
            # Existing, same-dimension model: bump last_used only when the
            # calendar day advanced, not on every image.
            prev = existing.get("last_used", "")
            if prev[:10] != now_iso[:10]:
                existing["last_used"] = now_iso
                changed = True
        if reg["active_model"] is None:
            reg["active_model"] = model_name
            changed = True
            print(f"[embeddings] Active model set to: {model_name}")
        if changed:
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
    """Best-effort id of the LM Studio EMBEDDING model to use, via the v0
    API's real type/loaded-state info (mirrors vision._lm_model_id()'s
    vision-model lookup). Prefers a loaded embeddings model; when none is
    loaded but the v0 API answered, falls back to any embeddings-TYPED model
    (LM Studio JIT-loads it on request — naming a correct-type model beats
    the old behavior of letting callers grab /v1/models[0], which could be a
    6GB chat model that then JIT-loads just to fail the embed call).
    Returns None only when the v0 API is unreachable."""
    try:
        v0 = list_lm_studio_models_v0()
    except Exception:
        v0 = []
    for m in v0:
        if m.get("state") == "loaded" and m.get("type") == "embeddings":
            return m.get("id")
    for m in v0:
        if m.get("type") == "embeddings":
            return m.get("id")
    return None


def _lm_v1_embed_fallback_id(models: list[dict]) -> str | None:
    """Pick an embedding model from a raw /v1/models list (old LM Studio, no
    v0 API): first id matching the shared embed name patterns, never an
    arbitrary [0] (which can be a chat/vision model)."""
    for m in models:
        mid = m.get("id", "")
        if any(p in mid.lower() for p in _EMBED_NAME_PATTERNS):
            return mid
    return None


def _check_served_model(requested: str, served: str | None, provider: str,
                        *, allow_prefixless: bool = False):
    if not served:
        return
    served = str(served)
    accepted = {requested}
    if allow_prefixless and "/" in requested:
        accepted.add(requested.rsplit("/", 1)[1])
    if served not in accepted:
        raise RuntimeError(
            f"{provider} substituted embedding model ({requested} -> {served}); "
            "the response was rejected to protect vector-space integrity."
        )


def _ordered_rows(rows: list, expected: int, provider: str) -> list:
    if not isinstance(rows, list) or len(rows) != expected:
        actual = len(rows) if isinstance(rows, list) else "non-list"
        raise RuntimeError(f"{provider} returned {actual} rows for {expected} inputs")
    indexed = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("index"), int):
            raise RuntimeError(f"{provider} batch response has a missing/invalid row index")
        index = row["index"]
        if index < 0 or index >= expected or index in seen:
            raise RuntimeError(f"{provider} batch response has duplicate/out-of-range index {index}")
        seen.add(index)
        indexed.append((index, row))
    if seen != set(range(expected)):
        raise RuntimeError(f"{provider} batch response indexes are incomplete")
    return [row for _, row in sorted(indexed)]


def _validate_vector(vector, provider: str) -> list[float]:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
        raise RuntimeError(f"{provider} returned an empty or invalid embedding vector")
    try:
        result = [float(value) for value in vector]
    except (TypeError, ValueError, OverflowError) as e:
        raise RuntimeError(f"{provider} returned a non-numeric embedding vector") from e
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"{provider} returned a non-finite embedding vector")
    if not any(value != 0.0 for value in result):
        raise RuntimeError(f"{provider} returned an all-zero embedding vector")
    return result


def _validate_vectors(vectors, expected: int, provider: str,
                      *, allow_partial: bool = False) -> list[list[float] | None]:
    if not isinstance(vectors, list) or len(vectors) != expected:
        actual = len(vectors) if isinstance(vectors, list) else "non-list"
        raise RuntimeError(f"{provider} returned {actual} vectors for {expected} inputs")
    validated = []
    dims = set()
    for vector in vectors:
        if vector is None and allow_partial:
            validated.append(None)
            continue
        normalized = _validate_vector(vector, provider)
        dims.add(len(normalized))
        validated.append(normalized)
    if len(dims) > 1:
        raise RuntimeError(f"{provider} returned vectors with inconsistent dimensions")
    return validated


def _profile_text(text: str, source: str, model_name: str, purpose: str) -> tuple[str, str | None]:
    profile, _legacy = _profile_for_registered(source, model_name)
    strategy = profile.get("task_strategy")
    if strategy == "nomic-search-prefix-v1":
        return f"search_{'query' if purpose == 'query' else 'document'}: {text}", None
    if strategy == "gemini-embedding-2-query-document-prefix-v1":
        return f"search_{'query' if purpose == 'query' else 'document'}: {text}", None
    if strategy == "gemini-embedding-001-retrieval-task-v1":
        return text, "RETRIEVAL_QUERY" if purpose == "query" else "RETRIEVAL_DOCUMENT"
    return text, None


def _lm_studio_embed(text: str, model: str = None, purpose: str = "document") -> tuple[list, str]:
    """Embed via LM Studio /v1/embeddings. Uses `model` if given, else prefers
    the v0-API-reported loaded embeddings model, else the /v1/models heuristic."""
    model_name = model or _lm_embed_model_id()
    if not model_name:
        model_name = "lm_studio_embed"
        try:
            req = urllib.request.Request(f"{LM_STUDIO_URL}/models")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
            model_name = _lm_v1_embed_fallback_id(data.get("data", [])) or model_name
        except Exception:
            pass

    ratelimit.acquire("lm_studio")
    text, _ = _profile_text(text, "lm_studio", model_name, purpose)
    payload = json.dumps({"model": model_name, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        result = json.loads(r.read())
    _check_served_model(model_name, result.get("model"), "LM Studio")
    rows = result.get("data")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("LM Studio returned an unexpected single-embedding response")
    return rows[0]["embedding"], model_name


def _lm_studio_embed_batch(texts: list[str], model: str = None,
                           purpose: str = "document") -> tuple[list, str]:
    """Embed many texts in ONE /v1/embeddings call (the API takes a list).
    Returns (vectors in input order, model_name)."""
    model_name = model or _lm_embed_model_id()
    if not model_name:
        model_name = "lm_studio_embed"
        try:
            req = urllib.request.Request(f"{LM_STUDIO_URL}/models")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
            model_name = _lm_v1_embed_fallback_id(data.get("data", [])) or model_name
        except Exception:
            pass

    # One batch POST is ONE request against the provider's quota — acquire a
    # single slot for the whole chunk, matching how providers count.
    ratelimit.acquire("lm_studio")
    texts = [_profile_text(t, "lm_studio", model_name, purpose)[0] for t in texts]
    payload = json.dumps({"model": model_name, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
    _check_served_model(model_name, result.get("model"), "LM Studio")
    rows = _ordered_rows(result.get("data"), len(texts), "LM Studio")
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


def _gemini_embed(text: str, model: str = None,
                  purpose: str = "document") -> tuple[list, str]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    model_name = model or _GEMINI_EMBED_MODEL
    if _gemini_embed_cooldown.get(model_name, 0) > _time.time():
        raise RuntimeError(
            f"Gemini embed model {model_name} in post-429 cooldown — skipping retry"
        )
    ratelimit.acquire("gemini")
    url = f"{GEMINI_BASE}/models/{model_name}:embedContent?key={GEMINI_API_KEY}"
    text, task_type = _profile_text(text, "gemini", model_name, purpose)
    request_body = {
        "model": f"models/{model_name}",
        "content": {"parts": [{"text": text}]},
    }
    if task_type:
        request_body["embedContentConfig"] = {"taskType": task_type}
    payload = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            result = json.loads(r.read())
        _check_served_model(model_name, result.get("model"), "Gemini")
        return result["embedding"]["values"], model_name
    except urllib.error.HTTPError as e:
        body = e.read()  # readable only once — capture before using twice
        if e.code == 429:
            _mark_embed_rate_limited(model_name, e.headers.get("Retry-After") if e.headers else None)
            ratelimit.learn_from_gemini_429(model_name, body)
        raise RuntimeError(f"Gemini embed {e.code}: {body[:200]}")


def _gemini_embed_batch(texts: list[str], model: str = None,
                        purpose: str = "document") -> tuple[list, str]:
    """Synchronous Gemini batchEmbedContents call; responses are input-ordered."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    model_name = model or _GEMINI_EMBED_MODEL
    if _gemini_embed_cooldown.get(model_name, 0) > _time.time():
        raise RuntimeError(
            f"Gemini embed model {model_name} in post-429 cooldown — skipping retry"
        )
    requests = []
    for text in texts:
        prepared, task_type = _profile_text(text, "gemini", model_name, purpose)
        row = {
            "model": f"models/{model_name}",
            "content": {"parts": [{"text": prepared}]},
        }
        if task_type:
            row["embedContentConfig"] = {"taskType": task_type}
        requests.append(row)
    ratelimit.acquire("gemini")
    url = f"{GEMINI_BASE}/models/{model_name}:batchEmbedContents?key={GEMINI_API_KEY}"
    req = urllib.request.Request(
        url, data=json.dumps({"requests": requests}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        _check_served_model(model_name, result.get("model"), "Gemini")
        rows = result.get("embeddings")
        if not isinstance(rows, list) or len(rows) != len(texts):
            actual = len(rows) if isinstance(rows, list) else "non-list"
            raise RuntimeError(
                f"Gemini batch returned {actual} vectors for {len(texts)} inputs"
            )
        return [row["values"] for row in rows], model_name
    except urllib.error.HTTPError as e:
        body = e.read()
        if e.code == 429:
            _mark_embed_rate_limited(
                model_name, e.headers.get("Retry-After") if e.headers else None
            )
            ratelimit.learn_from_gemini_429(model_name, body)
        raise RuntimeError(f"Gemini batch embed {e.code}: {body[:200]}")


# ── 9Router (local multi-provider gateway) ────────────────────────────────────
# 9Router pools multiple API keys/accounts per provider and rotates them on 429
# internally — that's the whole point of routing embeddings through it. Model
# ids are provider-prefixed (e.g. "gemini/gemini-embedding-001"), which keeps
# them distinct from LM Studio/direct-Gemini ids in the registry, so the same
# upstream model reached via different providers can never share a collection.

_9r_embed_models_cache: tuple[float, list[str]] | None = None


def list_9router_embed_models() -> list[str]:
    """Embedding model ids from 9Router's dedicated /v1/models/embedding list,
    cached 5 min. Returns [] when 9Router is unreachable."""
    global _9r_embed_models_cache
    if _9r_embed_models_cache and _time.time() - _9r_embed_models_cache[0] < 300:
        return _9r_embed_models_cache[1]
    try:
        req = urllib.request.Request(f"{NINEROUTER_URL}/models/embedding")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        models = [m["id"] for m in data.get("data", [])]
        _9r_embed_models_cache = (_time.time(), models)
        return models
    except Exception as e:
        print(f"[embeddings] 9Router embed model list failed: {e}")
        return []


# A 429 surfacing through 9Router means every pooled key/account for that
# model is exhausted (the gateway already rotated through them) — cool down.
_9r_embed_cooldown: dict[str, float] = {}


def ninerouter_embed_cooldowns() -> dict[str, float]:
    """{model: seconds_remaining} for 9Router embed models in post-429 cooldown."""
    now = _time.time()
    return {m: round(until - now, 1) for m, until in _9r_embed_cooldown.items() if until > now}


def _9router_embed_request(payload_input, model: str, timeout: int) -> list:
    """Shared single/batch POST to 9Router /v1/embeddings. Returns the raw
    data rows. `model` is REQUIRED (no auto-pick by design)."""
    if not model:
        raise ValueError("9Router requires an explicit embedding model id")
    if _9r_embed_cooldown.get(model, 0) > _time.time():
        raise RuntimeError(f"9Router embed model {model} in post-429 cooldown — skipping retry")
    ratelimit.acquire("9router")
    payload = json.dumps({"model": model, "input": payload_input}).encode("utf-8")
    req = urllib.request.Request(
        f"{NINEROUTER_URL}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer 9router"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            _9r_embed_cooldown[model] = _time.time() + _EMBED_RATE_LIMIT_COOLDOWN_SEC
        raise RuntimeError(f"9Router embed {e.code}: {e.read()[:200]}")
    # Unlike captions, a substituted EMBEDDING model must be rejected: vectors
    # from two models — even at the same dimension (gemini-embedding-2-preview
    # is also 3072-d) — live in different vector spaces, and mixing them in one
    # ChromaDB collection silently corrupts similarity search for everything
    # already stored. The response echoes the serving model (verified live:
    # requested "gemini/gemini-embedding-001" → model "gemini-embedding-001",
    # prefix stripped), so containment == same model.
    served = result.get("model")
    if served:
        try:
            _check_served_model(model, served, "9Router", allow_prefixless=True)
        except RuntimeError:
            global _last_substitution
            _last_substitution = {"requested": model, "served": served}
            raise
    return result.get("data")


def _9router_embed(text: str, model: str) -> tuple[list, str]:
    rows = _9router_embed_request(text, model, timeout=30)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("9Router returned an unexpected single-embedding response")
    return rows[0]["embedding"], model


def _9router_embed_batch(texts: list[str], model: str) -> tuple[list, str]:
    """Batch embed through 9Router in ONE request (verified live: list input
    returns index-tagged rows, same contract as LM Studio's endpoint)."""
    rows = _ordered_rows(_9router_embed_request(texts, model, timeout=120), len(texts), "9Router")
    return [row["embedding"] for row in rows], model


# ── Public API ────────────────────────────────────────────────────────────────


def get_embedding(
    text: str, force_provider: str = "auto", model: str = None,
    purpose: str = "document",
) -> tuple[list | None, str, str]:
    """Returns (vector, model_name, source).
    force_provider: "auto" (LM Studio → Gemini), "lm_studio", "gemini", or
    "9router" (explicit opt-in, never part of the auto chain, model required).
    model: explicit embedding model id for the forced provider. In "auto" mode it
    is only passed to LM Studio (a Gemini fallback picks its own default).
    Registers the model on success. Returns (None, '', 'error') on full failure."""
    global _last_error, _last_substitution
    _last_error = None
    _last_substitution = None
    lm = ("lm_studio", lambda t: _lm_studio_embed(t, model, purpose))
    gem = ("gemini", lambda t: _gemini_embed(
        t, model if force_provider == "gemini" else None, purpose
    ))
    if force_provider == "lm_studio":
        chain = [lm]
    elif force_provider == "gemini":
        chain = [gem]
    elif force_provider == "9router":
        chain = [("9router", lambda t: _9router_embed(
            _profile_text(t, "9router", model, purpose)[0], model
        ))]
    else:
        chain = [lm, gem]

    for source, fn in chain:
        try:
            vector, model_name = fn(text)
            vector = _validate_vector(vector, source)
            register_model(source, model_name, len(vector))
            return vector, model_name, source
        except ratelimit.Cancelled:
            # Stop pressed during a rate-limit wait — not a provider failure;
            # must reach the job manager, not roll over to the next provider.
            raise
        except Exception as e:
            _last_error = f"{source}: {e}"
            if _is_connection_error(e):
                print(f"[embeddings] {source} offline, trying next")
                continue
            print(f"[embeddings] {source} error: {e}")
            continue
    return None, "", "error"


# Why the last get_embedding/get_embeddings_batch call failed entirely, for
# the job log. The (None, "", "error") return can't carry the reason, and the
# reason now matters: "9Router substituted the model — pick a different one"
# needs a restart with a NEW model, while "LM Studio offline" just needs the
# service back. Single worker thread runs jobs, so module-level slots are fine.
_last_error: str | None = None

# Set when the last full failure was specifically a 9Router embed-model
# substitution: {"requested": <id we asked for>, "served": <id that answered>}.
# The job manager surfaces this so the UI can offer a one-click "switch the
# embed model to what 9Router actually serves and re-run".
_last_substitution: dict | None = None


def last_embed_error() -> str | None:
    return _last_error


def last_substitution() -> dict | None:
    return _last_substitution


def resolve_9router_embed_id(served: str, requested: str) -> str:
    """Map a served model name (echoed WITHOUT its provider prefix, e.g.
    'gemini-embedding-2-preview') back to the full selectable 9Router id.
    Prefers a match from the live embedding list; falls back to reusing the
    requested id's provider prefix."""
    for mid in list_9router_embed_models():
        if served in mid:
            return mid
    prefix = requested.split("/", 1)[0] if "/" in requested else ""
    return f"{prefix}/{served}" if prefix else served


def get_embeddings_batch(
    texts: list[str], force_provider: str = "auto", model: str = None,
    purpose: str = "document",
) -> tuple[list | None, str, str]:
    """Batch variant of get_embedding: returns (vectors in input order,
    model_name, source), or (None, '', 'error') on full failure.
    Provider batch responses must have exact cardinality and order. Malformed
    or partial responses fail the chunk instead of risking vector/id misalignment."""
    global _last_error, _last_substitution
    _last_error = None
    _last_substitution = None
    if not texts:
        return [], "", ""

    lm = ("lm_studio", lambda ts: _lm_studio_embed_batch(ts, model, purpose))
    gem = ("gemini", lambda ts: _gemini_embed_batch(
        ts, model if force_provider == "gemini" else None, purpose
    ))
    if force_provider == "lm_studio":
        chain = [lm]
    elif force_provider == "gemini":
        chain = [gem]
    elif force_provider == "9router":
        chain = [("9router", lambda ts: _9router_embed_batch(
            [_profile_text(t, "9router", model, purpose)[0] for t in ts], model
        ))]
    else:
        chain = [lm, gem]

    for source, fn in chain:
        try:
            vectors, model_name = fn(texts)
            vectors = _validate_vectors(vectors, len(texts), source)
            dims = {len(v) for v in vectors}
            if len(dims) != 1:
                raise RuntimeError(f"{source} batch returned no consistent vector dimension")
            register_model(source, model_name, dims.pop())
            return vectors, model_name, source
        except ratelimit.Cancelled:
            raise
        except Exception as e:
            _last_error = f"{source}: {e}"
            if _is_connection_error(e):
                print(f"[embeddings] {source} offline, trying next")
                continue
            print(f"[embeddings] {source} batch error: {e}")
            continue
    return None, "", "error"
