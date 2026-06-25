import os
import base64
import io
import json
import time
import urllib.request
import urllib.error
from openai import OpenAI
from PIL import Image
import imaging  # noqa: F401 — registers HEIF opener + sets the pixel-bomb cap on import
from constants import LM_STUDIO_URL, GEMINI_API_KEY, GEMINI_BASE, GEMINI_VISION_MODELS

_PROMPT = (
    "Analyze this photo and respond ONLY with valid JSON (no markdown, no explanation). "
    "Use these exact keys and allowed values:\n"
    "{\n"
    '  "caption": "one sentence description",\n'
    '  "scene": "indoor" or "outdoor",\n'
    '  "location_type": one of [home, beach, restaurant, park, office, travel, street, gym, school, unknown],\n'
    '  "weather": one of [sunny, cloudy, rainy, snowy, indoor, unknown],\n'
    '  "season": one of [spring, summer, autumn, winter, unknown],\n'
    '  "time_of_day": one of [morning, afternoon, evening, night, unknown],\n'
    '  "occasion": one of [birthday, wedding, vacation, everyday, sports, festival, graduation, family, unknown],\n'
    '  "group_size": one of [solo, couple, small_group, large_group, no_people],\n'
    '  "clothing_style": one of [formal, casual, sports, traditional, swimwear, unknown],\n'
    '  "mood": one of [happy, celebration, relaxed, adventurous, serious, romantic, unknown],\n'
    '  "objects": ["list", "of", "key", "objects"],\n'
    '  "people_description": "brief description of people if present, else empty string"\n'
    "}"
)

_lm_client = None

def _get_lm_client():
    global _lm_client
    if _lm_client is None:
        _lm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    return _lm_client


def encode_image(image_path, max_size=(1024, 1024)):
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[vision] encode error {image_path}: {e}")
        return None


def _is_connection_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(w in msg for w in ("connection", "refused", "unreachable", "timeout", "connect error", "cannot connect"))


def _strip_markdown(text: str) -> str:
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        text = text.lstrip("json").strip()
    return text.strip()


def _lm_model_id() -> str:
    """Best-effort id of the currently loaded LM Studio model. Falls back to 'lm_studio'."""
    try:
        client = _get_lm_client()
        models = client.models.list()
        return f"lm_studio:{models.data[0].id}"
    except Exception:
        return "lm_studio"


def list_lm_studio_models() -> list[str]:
    """All model ids currently loaded in LM Studio (vision + embedding mixed)."""
    try:
        client = _get_lm_client()
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


_VISION_NAME_PATTERNS = (
    "llava", "bakllava", "moondream", "minicpm-v", "pixtral", "cogvlm", "idefics",
    "-vl", "vl-", "vision", "visual", "multimodal",
)
_EMBED_NAME_PATTERNS = (
    "embed", "nomic", "bge-", "e5-", "gte-", "minilm", "mpnet", "mxbai", "sfr-embedding",
)


def classify_lm_studio_model(model_id: str) -> dict:
    """Heuristic type for an LM Studio model from its name.
    Returns {type: 'vision'|'embed'|'unknown', warning: str|None}"""
    lower = model_id.lower()
    is_embed = any(p in lower for p in _EMBED_NAME_PATTERNS)
    is_vision = any(p in lower for p in _VISION_NAME_PATTERNS)
    if is_embed and not is_vision:
        return {"type": "embed", "warning": "Embedding model — not suitable for image analysis"}
    if is_vision and not is_embed:
        return {"type": "vision", "warning": None}
    if is_vision and is_embed:
        return {"type": "unknown", "warning": "Name matches both vision and embed patterns — verify model type"}
    return {"type": "unknown", "warning": "Cannot determine model type from name — image output may be unreliable"}


_gemini_cache: dict[str, tuple[float, list[str]]] = {}


def _fetch_gemini_models(method: str) -> list[str]:
    """Fetch Gemini models supporting `method`, with 5-min in-memory cache."""
    cached = _gemini_cache.get(method)
    if cached and time.time() - cached[0] < 300:
        return cached[1]
    if not GEMINI_API_KEY:
        return GEMINI_VISION_MODELS if method == "generateContent" else []
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
        return GEMINI_VISION_MODELS if method == "generateContent" else []


def list_gemini_vision_models() -> list[str]:
    return _fetch_gemini_models("generateContent")


_REQUIRED_VISION_KEYS = ("caption", "scene", "occasion", "weather", "group_size")


def validate_vision_output(text: str) -> dict:
    """Check vision output is valid JSON with expected schema.
    Returns {valid: bool, warning: str|None}"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"valid": False, "warning": "Output is not valid JSON — model may not support image input"}
    if "error" in data:
        return {"valid": False, "warning": f"Model returned error: {data['error']}"}
    missing = [k for k in _REQUIRED_VISION_KEYS if k not in data]
    if missing:
        return {"valid": False, "warning": f"Missing expected keys: {missing} — model may be text-only"}
    return {"valid": True, "warning": None}


def _call_lm_studio(base64_image: str, model: str = "vision-model") -> str:
    client = _get_lm_client()
    response = client.chat.completions.create(
        model=model or "vision-model",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ],
        }],
        max_tokens=800,
    )
    return _strip_markdown(response.choices[0].message.content.strip())


def _call_gemini(base64_image: str, model: str = None) -> tuple[str, str]:
    """Returns (caption_json, model_used)."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set — LM Studio is offline and no fallback available")

    payload = json.dumps({
        "contents": [{"parts": [
            {"text": _PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 800},
    }).encode("utf-8")

    candidates = [model] if model else GEMINI_VISION_MODELS
    last_err = None
    for m in candidates:
        url = f"{GEMINI_BASE}/models/{m}:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read())
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"[vision] Gemini model used: {m}")
            return _strip_markdown(text), m
        except urllib.error.HTTPError as e:
            if e.code in (429, 404, 503):
                last_err = f"{m}:{e.code}"
                continue
            raise RuntimeError(f"Gemini {e.code} ({m}): {e.read()[:200]}")
    raise RuntimeError(f"All Gemini vision models exhausted. Last: {last_err}")


def get_image_caption(image_path: str, force_provider: str = "auto",
                      with_model: bool = False, model: str = None):
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
                return _ret(json.dumps({"error": f"LM Studio offline; Gemini failed: {ge}"}), "error")
        return _ret(json.dumps({"error": str(e)}), "error")


def parse_vision_attributes(caption_json: str) -> dict:
    defaults = {
        "caption": "", "scene": "unknown", "location_type": "unknown",
        "weather": "unknown", "season": "unknown", "time_of_day": "unknown",
        "occasion": "unknown", "group_size": "unknown", "clothing_style": "unknown",
        "mood": "unknown", "objects": [], "people_description": ""
    }
    try:
        data = json.loads(caption_json)
        defaults.update({k: v for k, v in data.items() if k in defaults})
        if isinstance(defaults["objects"], list):
            defaults["objects"] = ", ".join(defaults["objects"])
    except Exception:
        pass
    return defaults
