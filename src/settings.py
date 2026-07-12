"""
App-level settings: model provider/model selection and pipeline configuration.
Stored in data/settings.json. These are the defaults used by every indexing job
unless overridden at job-start time.
"""
import json
import os
import threading
from constants import SETTINGS_PATH, DATA_DIR

# Serializes the read-modify-write cycle in update() so two same-process
# callers (e.g. the API thread and the job worker thread) can't race and
# lose one caller's patch. Same-process only, per this app's
# single-user-local-tool scope.
_lock = threading.Lock()

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

    # Max output tokens per caption request. Only a CEILING — providers bill
    # for tokens actually produced, so a generous value is free for normal
    # images but leaves headroom for "thinking" models (gemini-2.5/3.x-flash)
    # that spend hidden reasoning tokens against this same budget before
    # emitting the JSON. Too low → JSON cut mid-field → the caption fails to
    # parse. A truncated response also auto-escalates this budget per-image up
    # to vision._MAX_TOKENS_CEILING, so raising this is rarely needed; it's the
    # knob the truncation error message points at. 0 → the code default (4096).
    "vision_max_tokens": 4096,

    # How many keyframes to sample per video for captioning + face detection.
    # A video is analyzed by extracting this many frames evenly across the clip
    # (avoiding the black start/end), captioning each with the vision model, and
    # folding them into ONE caption + one search vector. So a video costs
    # ~video_frames vision calls, NOT one-per-actual-frame. Higher = richer
    # coverage of long/varied clips but proportionally more calls/quota; lower =
    # faster & cheaper. Clamped to 1..12 by the API. 4 is a sensible default.
    "video_frames": 4,

    # Whether the embed stage also runs face detection inline. Turn off to keep
    # face detection a fully separate, user-triggered stage (the "faces" job).
    "faces_during_embed": True,

    # Which accelerator InsightFace face detection runs on. "auto" picks the
    # fastest execution provider the installed onnxruntime wheel exposes
    # (GPU/NPU over CPU); an explicit id from faces.available_accelerators()
    # (e.g. "openvino:GPU", "openvino:NPU", "cuda", "dml", "cpu") pins it.
    # CPU is always the fallback if the chosen accelerator is unavailable.
    "face_provider": "auto",

    # DBSCAN face-clustering parameters (advanced).
    "face_cluster_eps": 0.5,
    "face_cluster_min_samples": 3,

    # Where the ingest job copies newly imported media (organized YYYY/MM).
    # null → an Imported/ folder inside the first included scan folder, so
    # new photos are picked up by the normal Scan with zero extra config.
    "ingest_dest": None,

    # Backup mirror destination — a folder on the SD card / external drive,
    # e.g. "E:\\PhotoVaultBackup". null → backup not configured.
    "backup_dest": None,

    # Client-side request ceilings per provider — 0 = unlimited. Providers
    # (Gemini free tier especially) throttle requests per second/minute/day;
    # setting these at or under the published quota makes jobs pace themselves
    # instead of burning quota on 429s and tripping model cooldowns. Enforced
    # by ratelimit.acquire() before every provider inference call; counters
    # are in-memory sliding windows that reset on server restart.
    # LM Studio is local and 9Router rotates pooled accounts internally, so
    # both default to unlimited. Gemini gets a conservative, approximate
    # free-tier default (a low pace that shouldn't 429 on a typical free key) —
    # all-zeros would mean every fresh install hammers the free tier into 429s
    # by default. Real limits vary by account/tier/model and change over time
    # (https://ai.google.dev/gemini-api/docs/rate-limits); the app learns your
    # actual caps from real 429s, and you can set exact values in Settings.
    "rate_limits": {
        "lm_studio": {"rps": 0, "rpm": 0, "rph": 0, "rpd": 0},
        "gemini":    {"rps": 0, "rpm": 10, "rph": 0, "rpd": 500},
        "9router":   {"rps": 0, "rpm": 0, "rph": 0, "rpd": 0},
    },
}


def _merge_rate_limits(saved_rl: dict | None) -> dict:
    """rate_limits is nested one level deeper than everything else — a plain
    {**DEFAULTS, **saved} would let a partial saved dict (e.g. only "gemini")
    silently drop the other providers' entries. Merge per provider instead."""
    merged = {p: dict(lims) for p, lims in DEFAULTS["rate_limits"].items()}
    for prov, lims in (saved_rl or {}).items():
        if isinstance(lims, dict):
            merged[prov] = {**merged.get(prov, {}), **lims}
    return merged


def load() -> dict:
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            merged = {**DEFAULTS, **saved}
            merged["rate_limits"] = _merge_rate_limits(saved.get("rate_limits"))
            return merged
        except Exception:
            pass
    # deep-copy the nested rate_limits so callers mutating the returned dict
    # can't corrupt DEFAULTS
    fresh = dict(DEFAULTS)
    fresh["rate_limits"] = _merge_rate_limits(None)
    return fresh


def save(settings: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    merged = {**DEFAULTS, **settings}
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, SETTINGS_PATH)


def update(patch: dict) -> dict:
    with _lock:
        current = load()
        current.update(patch)
        save(current)
        return current


def vision_model_label(settings: dict) -> str | None:
    """
    The model label key used in caption_history (e.g. "lm_studio:qwen2-vl-7b").
    Returns None when the label can't be predicted before the call:
      - "auto": depends on what's loaded in LM Studio at runtime
      - "9router": the gateway may substitute the serving model, and captions
        are stored under the model that ACTUALLY produced them — so a 9Router
        run targets photos with no caption at all (coverage), not per-label
        completeness, which would loop forever on substituted labels.
    """
    p = settings.get("vision_provider", "auto")
    m = settings.get("vision_model")
    if p in ("auto", "9router") or m is None:
        return None
    return f"{p}:{m}"
