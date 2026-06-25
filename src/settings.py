"""
App-level settings: model provider/model selection and pipeline configuration.
Stored in data/settings.json. These are the defaults used by every indexing job
unless overridden at job-start time.
"""
import json
import os
from constants import SETTINGS_PATH, DATA_DIR

DEFAULTS: dict = {
    # Vision (captioning)
    "vision_provider": "auto",   # "auto" | "lm_studio" | "gemini"
    "vision_model": None,        # null → auto-pick within provider

    # Embedding
    "embed_provider": "auto",
    "embed_model": None,

    # Which vision model's caption to use when creating embeddings.
    # null → use the latest caption (caption_json field) regardless of which model produced it.
    # A model label like "lm_studio:qwen2-vl-7b" or "gemini:gemini-2.5-flash" → only embed
    # images that have a caption entry from that specific model.
    "caption_source_model": None,

    "max_fail": 5,

    # How many vision (captioning) calls to run in parallel. Vision is
    # network-bound (LM Studio / Gemini), so concurrency gives a big speedup.
    "vision_concurrency": 4,

    # Whether the embed stage also runs face detection inline. Turn off to keep
    # face detection a fully separate, user-triggered stage (the "faces" job).
    "faces_during_embed": True,

    # DBSCAN face-clustering parameters (advanced).
    "face_cluster_eps": 0.5,
    "face_cluster_min_samples": 3,
}


def load() -> dict:
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            return {**DEFAULTS, **saved}
        except Exception:
            pass
    return dict(DEFAULTS)


def save(settings: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    merged = {**DEFAULTS, **settings}
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, SETTINGS_PATH)


def update(patch: dict) -> dict:
    current = load()
    current.update(patch)
    save(current)
    return current


def vision_model_label(settings: dict) -> str | None:
    """
    The model label key used in caption_history (e.g. "lm_studio:qwen2-vl-7b").
    Returns None when provider/model is "auto" (label can't be predicted without
    knowing what's loaded in LM Studio at runtime).
    """
    p = settings.get("vision_provider", "auto")
    m = settings.get("vision_model")
    if p == "auto" or m is None:
        return None
    return f"{p}:{m}"
