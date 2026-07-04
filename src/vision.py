import os
import base64
import io
import json
import re
import time
import urllib.request
import urllib.error
from openai import OpenAI
from PIL import Image, ImageOps
import imaging  # noqa: F401 — registers HEIF opener + sets the pixel-bomb cap on import
from constants import LM_STUDIO_URL, GEMINI_API_KEY, GEMINI_BASE, GEMINI_VISION_MODELS

_PROMPT = (
    "Analyze this photo and respond ONLY with valid JSON (no markdown, no explanation). "
    "Use these exact keys and allowed values:\n"
    "{\n"
    '  "caption": "2-3 detailed sentences mentioning all salient people, animals, objects, activities, and setting",\n'
    '  "scene": "indoor" or "outdoor",\n'
    '  "location_type": one of [home, beach, restaurant, park, office, travel, street, gym, school, unknown],\n'
    '  "weather": one of [sunny, cloudy, rainy, snowy, indoor, unknown],\n'
    '  "season": one of [spring, summer, autumn, winter, unknown],\n'
    '  "time_of_day": one of [morning, afternoon, evening, night, unknown],\n'
    '  "occasion": one of [birthday, wedding, vacation, everyday, sports, festival, graduation, family, unknown],\n'
    '  "festival_name": specific named holiday/festival if recognizable (e.g. Diwali, Christmas, Halloween, Holi, Eid, New Year), else "",\n'
    '  "group_size": one of [solo, couple, small_group, large_group, no_people],\n'
    '  "person_count": exact integer number of people visible (0 if none, your best count if partially obscured/crowded),\n'
    '  "clothing_style": one of [formal, casual, sports, traditional, swimwear, unknown],\n'
    '  "mood": one of [happy, celebration, relaxed, adventurous, serious, romantic, unknown],\n'
    '  "objects": ["list", "of", "key", "objects"],\n'
    '  "animals": ["list", "of", "animal/pet", "types", "present"],\n'
    '  "vehicles": ["list", "of", "vehicle", "types", "present"],\n'
    '  "food_items": ["list", "of", "visible", "food/drink", "items"],\n'
    '  "activities": ["list", "of", "activities/actions", "happening"],\n'
    '  "photo_type": one of [photo, screenshot, document, meme, selfie, artwork, unknown],\n'
    '  "text_in_image": "short transcription of any prominent visible text/sign, else empty string",\n'
    '  "landmark": "named landmark/monument/building if recognizable, else empty string",\n'
    '  "dominant_colors": ["1-3", "dominant", "colors"],\n'
    '  "people_description": "brief description of people if present, else empty string"\n'
    "}"
)

# List-valued keys: parse_vision_attributes joins these into a comma-separated
# string (Chroma metadata only accepts scalar values) — same treatment as the
# original "objects" field.
_LIST_KEYS = ("objects", "animals", "vehicles", "food_items", "activities", "dominant_colors")

_lm_client = None


def _get_lm_client():
    global _lm_client
    if _lm_client is None:
        _lm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    return _lm_client


def encode_image(image_path, max_size=(1024, 1024)):
    try:
        with Image.open(image_path) as img:
            # Without this, a portrait photo shot with a rotated sensor is
            # handed to the model sideways — it then captions what it sees
            # (e.g. people described as "lying down" who are actually
            # standing in a correctly-oriented portrait photo).
            img = ImageOps.exif_transpose(img)
            img.thumbnail(max_size)
            # JPEG can't encode alpha/palette modes (RGBA screenshots, P-mode
            # PNGs, LA) — convert or every such file fails vision entirely.
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[vision] encode error {image_path}: {e}")
        return None


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


def _strip_markdown(text: str) -> str:
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        # Strip a leading language tag (e.g. "json", "JSON5") left over from the
        # fence — lstrip(chars) strips individual characters, not the word, and
        # would also eat any leading 'j'/'s'/'o'/'n' from real JSON content.
        text = re.sub(r"^\s*json5?\s*\n?", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def _lm_studio_host() -> str:
    """LM_STUDIO_URL is the OpenAI-compat base (…/v1); LM Studio's own native
    API lives at the same host under /api/v0, not /v1."""
    return LM_STUDIO_URL[:-3] if LM_STUDIO_URL.endswith("/v1") else LM_STUDIO_URL


def list_lm_studio_models_v0() -> list[dict]:
    """LM Studio's native REST API (not OpenAI-compat): reports the REAL model
    type ('vlm'/'embeddings'/'llm') and REAL loaded state ('loaded'/'not-loaded'),
    unlike /v1/models which lists every JIT-loadable model with no state info.
    Returns [] if unreachable (older LM Studio, or offline) — callers should
    fall back to the name-heuristic path in that case."""
    try:
        url = f"{_lm_studio_host()}/api/v0/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return data.get("data", [])
    except Exception:
        return []


def _lm_model_id() -> str:
    """Best-effort id of the currently loaded LM Studio vision model. Prefers
    the v0 API's real 'loaded' vlm; falls back to the first /v1/models entry
    (which may not actually be the resident model) if v0 is unavailable."""
    v0 = list_lm_studio_models_v0()
    for m in v0:
        if m.get("state") == "loaded" and m.get("type") == "vlm":
            return f"lm_studio:{m['id']}"
    try:
        client = _get_lm_client()
        models = client.models.list()
        return f"lm_studio:{models.data[0].id}"
    except Exception:
        return "lm_studio"


def list_lm_studio_models() -> list[str]:
    """All model ids LM Studio knows about (loaded + JIT-loadable), vision +
    embedding mixed. Use classify_lm_studio_model()/list_lm_studio_models_v0()
    for type/loaded-state — this list alone doesn't tell you which is resident."""
    try:
        client = _get_lm_client()
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


_VISION_NAME_PATTERNS = (
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
    "pixtral",
    "cogvlm",
    "idefics",
    "-vl",
    "vl-",
    "vision",
    "visual",
    "multimodal",
    "gemma-3-",  # gemma-3/4 chat models are multimodal; trailing dash keeps
    "gemma-4-",  # "embeddinggemma-300m" (contains "gemma-3") out of the match
)
_EMBED_NAME_PATTERNS = (
    "embed",
    "nomic",
    "bge-",
    "e5-",
    "gte-",
    "minilm",
    "mpnet",
    "mxbai",
    "sfr-embedding",
)


_V0_TYPE_MAP = {"vlm": "vision", "embeddings": "embed", "llm": "text-only"}


def classify_lm_studio_model(model_id: str, v0_info: dict = None) -> dict:
    """Type + live-loaded state for an LM Studio model.
    When v0_info (from list_lm_studio_models_v0()) is available, uses LM Studio's
    own reported type/state — authoritative, not a guess. Otherwise falls back to
    a name-pattern heuristic (older LM Studio versions without the v0 API).
    Returns {type: 'vision'|'embed'|'text-only'|'unknown', state: 'loaded'|'not-loaded'|None, warning: str|None}"""
    if v0_info is not None:
        t = _V0_TYPE_MAP.get(v0_info.get("type"), "unknown")
        state = v0_info.get("state")
        warning = None
        if t == "embed":
            warning = "Embedding model — not suitable for image analysis"
        elif t == "text-only":
            warning = "Text-only model — cannot analyze images"
        return {"type": t, "state": state, "warning": warning}

    lower = model_id.lower()
    is_embed = any(p in lower for p in _EMBED_NAME_PATTERNS)
    is_vision = any(p in lower for p in _VISION_NAME_PATTERNS)
    if is_embed and not is_vision:
        return {
            "type": "embed",
            "state": None,
            "warning": "Embedding model — not suitable for image analysis",
        }
    if is_vision and not is_embed:
        return {"type": "vision", "state": None, "warning": None}
    if is_vision and is_embed:
        return {
            "type": "unknown",
            "state": None,
            "warning": "Name matches both vision and embed patterns — verify model type",
        }
    return {
        "type": "unknown",
        "state": None,
        "warning": "Cannot determine model type from name — image output may be unreliable",
    }


_gemini_cache: dict[str, tuple[float, list[str]]] = {}


def _fetch_gemini_models(method: str, fallback: bool = True) -> list[str]:
    """Fetch Gemini models supporting `method`, with 5-min in-memory cache.
    When `fallback=False` returns [] instead of hardcoded list on failure."""
    cached = _gemini_cache.get(method)
    if cached and time.time() - cached[0] < 300:
        return cached[1]
    if not GEMINI_API_KEY:
        return (
            GEMINI_VISION_MODELS if (method == "generateContent" and fallback) else []
        )
    try:
        url = f"{GEMINI_BASE}/models?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        models = [
            m["name"].replace("models/", "")
            for m in data.get("models", [])
            if method in m.get("supportedGenerationMethods", [])
            and m.get("outputTokenLimit", 1) > 0
        ]
        _gemini_cache[method] = (time.time(), models)
        return models
    except Exception as e:
        print(f"[vision] Gemini model list failed: {e}")
        return (
            GEMINI_VISION_MODELS if (method == "generateContent" and fallback) else []
        )


def list_gemini_vision_models(fallback: bool = True) -> list[str]:
    return _fetch_gemini_models("generateContent", fallback=fallback)


_REQUIRED_VISION_KEYS = ("caption", "scene", "occasion", "weather", "group_size")


def validate_vision_output(text: str) -> dict:
    """Check vision output is valid JSON with expected schema.
    Returns {valid: bool, warning: str|None}"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "warning": "Output is not valid JSON — model may not support image input",
        }
    if "error" in data:
        return {"valid": False, "warning": f"Model returned error: {data['error']}"}
    missing = [k for k in _REQUIRED_VISION_KEYS if k not in data]
    if missing:
        return {
            "valid": False,
            "warning": f"Missing expected keys: {missing} — model may be text-only",
        }
    return {"valid": True, "warning": None}


def _call_lm_studio(base64_image: str, model: str = "vision-model") -> str:
    client = _get_lm_client()
    response = client.chat.completions.create(
        model=model or "vision-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1400,
    )
    return _strip_markdown(response.choices[0].message.content.strip())


# Gemini has no public "remaining quota" endpoint — the closest real signal is
# observing 429s ourselves. Track a short cooldown per model so the picker (and
# the dropdown, via gemini_cooldowns()) can skip/flag a model that just got
# rate-limited instead of hammering it again immediately.
_gemini_cooldown: dict[str, float] = {}
_RATE_LIMIT_COOLDOWN_SEC = 90


def _mark_rate_limited(model: str, retry_after: str | None = None):
    delay = _RATE_LIMIT_COOLDOWN_SEC
    if retry_after:
        try:
            delay = max(delay, int(retry_after))
        except ValueError:
            pass
    _gemini_cooldown[model] = time.time() + delay


def gemini_cooldowns() -> dict[str, float]:
    """{model: seconds_remaining} for models currently in a post-429 cooldown."""
    now = time.time()
    return {
        m: round(until - now, 1)
        for m, until in _gemini_cooldown.items()
        if until > now
    }


def _call_gemini(base64_image: str, model: str = None) -> tuple[str, str]:
    """Returns (caption_json, model_used). Tries `model` first if given, then
    falls through the dynamically-known vision model list on 429/404/503 —
    a pinned model no longer means "no fallback if it's rate-limited"."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set — LM Studio is offline and no fallback available"
        )

    payload = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {"text": _PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1400,
                "response_mime_type": "application/json",
            },
        }
    ).encode("utf-8")

    # GEMINI_VISION_MODELS is deliberately ordered by rate-limit friendliness
    # (cheapest/highest-quota first) — keep that order for the fallback pool
    # rather than the API's arbitrary listing order, and don't spend an extra
    # network call on every single image just to re-fetch it.
    if model:
        candidates = [model] + [m for m in GEMINI_VISION_MODELS if m != model]
    else:
        candidates = GEMINI_VISION_MODELS

    now = time.time()
    candidates = [m for m in candidates if _gemini_cooldown.get(m, 0) <= now] or candidates

    last_err = None
    for m in candidates:
        url = f"{GEMINI_BASE}/models/{m}:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read())
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"[vision] Gemini model used: {m}")
            return _strip_markdown(text), m
        except urllib.error.HTTPError as e:
            if e.code in (429, 404, 503):
                if e.code == 429:
                    _mark_rate_limited(m, e.headers.get("Retry-After") if e.headers else None)
                last_err = f"{m}:{e.code}"
                continue
            raise RuntimeError(f"Gemini {e.code} ({m}): {e.read()[:200]}")
    raise RuntimeError(f"All Gemini vision models exhausted. Last: {last_err}")


def get_image_caption(
    image_path: str,
    force_provider: str = "auto",
    with_model: bool = False,
    model: str = None,
):
    """
    force_provider: "auto" (LM Studio → Gemini fallback), "lm_studio", or "gemini"
    model: explicit model id to use (only honored when force_provider is lm_studio/gemini).
    with_model=False → returns caption text (str, backward compatible).
    with_model=True  → returns (caption_text, model_label) identifying which model produced it.
    """

    def _ret(text, label):
        return (text, label) if with_model else text

    base64_image = encode_image(image_path)
    if not base64_image:
        return _ret(json.dumps({"error": "encoding failed"}), "error")

    if force_provider == "gemini":
        try:
            text, used_model = _call_gemini(base64_image, model)
            return _ret(text, f"gemini:{used_model}")
        except Exception as ge:
            return _ret(json.dumps({"error": f"Gemini failed: {ge}"}), "error")

    if force_provider == "lm_studio":
        try:
            label = f"lm_studio:{model}" if model else _lm_model_id()
            text = _call_lm_studio(base64_image, model)
            return _ret(text, label)
        except Exception as e:
            return _ret(json.dumps({"error": f"LM Studio failed: {e}"}), "error")

    try:
        label = f"lm_studio:{model}" if model else _lm_model_id()
        return _ret(_call_lm_studio(base64_image, model), label)
    except Exception as e:
        if _is_connection_error(e):
            print(f"[vision] LM Studio offline, falling back to Gemini")
            try:
                text, used_model = _call_gemini(base64_image)
                return _ret(text, f"gemini:{used_model}")
            except Exception as ge:
                return _ret(
                    json.dumps({"error": f"LM Studio offline; Gemini failed: {ge}"}),
                    "error",
                )
        return _ret(json.dumps({"error": str(e)}), "error")


def parse_vision_attributes(caption_json: str) -> dict:
    defaults = {
        "caption": "",
        "scene": "unknown",
        "location_type": "unknown",
        "weather": "unknown",
        "season": "unknown",
        "time_of_day": "unknown",
        "occasion": "unknown",
        "festival_name": "",
        "group_size": "unknown",
        "person_count": 0,
        "clothing_style": "unknown",
        "mood": "unknown",
        "objects": [],
        "animals": [],
        "vehicles": [],
        "food_items": [],
        "activities": [],
        "photo_type": "unknown",
        "text_in_image": "",
        "landmark": "",
        "dominant_colors": [],
        "people_description": "",
    }
    try:
        data = json.loads(caption_json)
        defaults.update({k: v for k, v in data.items() if k in defaults})
        if not isinstance(defaults["person_count"], int):
            try:
                defaults["person_count"] = int(defaults["person_count"])
            except (TypeError, ValueError):
                defaults["person_count"] = 0
        for key in _LIST_KEYS:
            val = defaults[key]
            if isinstance(val, list):
                defaults[key] = ", ".join(str(v) for v in val)
            elif not isinstance(val, str):
                defaults[key] = ""
        # Chroma metadata only accepts scalars (str/int/float/bool/None). A
        # model can still hand back a list/dict for a field we expect to be
        # scalar (e.g. "caption": ["a", "b"]) — coerce instead of letting that
        # reach build_embed_payload and crash the Chroma add()/upsert() call.
        for key, val in defaults.items():
            if key in _LIST_KEYS or key == "person_count":
                continue
            if val is None or isinstance(val, (str, int, float, bool)):
                continue
            if isinstance(val, list):
                defaults[key] = ", ".join(str(v) for v in val)
            elif isinstance(val, dict):
                defaults[key] = ", ".join(f"{k}: {v}" for k, v in val.items())
            else:
                defaults[key] = str(val)
    except Exception:
        pass
    return defaults


def build_embedding_text(attrs: dict) -> str:
    """Natural-language text for the embedding model. Embedding the raw
    caption_json (braces/keys/quotes) feeds noise tokens to a text-embedding
    model and measurably hurts semantic search — this assembles plain
    sentences from the same parsed attributes instead."""
    parts = []
    if attrs.get("caption"):
        parts.append(attrs["caption"])
    if attrs.get("people_description"):
        parts.append(attrs["people_description"])
    for label, key in (
        ("Animals", "animals"), ("Vehicles", "vehicles"), ("Food", "food_items"),
        ("Activities", "activities"), ("Objects", "objects"),
    ):
        if attrs.get(key):
            parts.append(f"{label}: {attrs[key]}.")
    if attrs.get("text_in_image"):
        parts.append(f"Visible text: {attrs['text_in_image']}.")
    if attrs.get("landmark"):
        parts.append(f"Landmark: {attrs['landmark']}.")
    if attrs.get("festival_name"):
        parts.append(f"Festival: {attrs['festival_name']}.")
    tags = [
        attrs[k] for k in (
            "occasion", "mood", "weather", "season", "time_of_day",
            "location_type", "clothing_style", "group_size", "photo_type",
        )
        if attrs.get(k) and attrs[k] != "unknown"
    ]
    if tags:
        parts.append("Tags: " + ", ".join(tags) + ".")
    if attrs.get("dominant_colors"):
        parts.append(f"Colors: {attrs['dominant_colors']}.")
    return " ".join(parts)
