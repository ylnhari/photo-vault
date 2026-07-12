"""Per-provider request throttling with user-configurable RPS/RPM/RPH/RPD.

Providers enforce their own request quotas (Gemini free tier most visibly:
per-minute and per-day caps that vary by model and account). Hitting them
mid-job burns quota on 429s and trips the post-429 model cooldowns. This
module lets the user set client-side ceilings per provider in Settings
(data/settings.json → "rate_limits"), and every provider inference call in
vision.py / embeddings.py calls acquire() first, sleeping until a slot frees.
0 (or a missing key) means unlimited for that window.

Windows are sliding and in-memory only: counters reset when the server
restarts (the provider's own enforcement is the backstop for that gap).
acquire() honors a cancel event registered by the job manager, so Stop
interrupts a rate-limit sleep within a tick instead of hanging the worker
for up to a full window (rpd could otherwise mean hours).

Limits are re-read from settings on every acquire — a settings change takes
effect mid-job, no restart or job restart needed. One acquire == one HTTP
inference request (an embed batch is one request; each Gemini vision model
attempt in the fallback loop is its own request), which matches how the
providers count.
"""
import json
import threading
import time
from collections import deque

WINDOWS = {"rps": 1.0, "rpm": 60.0, "rph": 3600.0, "rpd": 86400.0}
PROVIDERS = ("lm_studio", "gemini", "9router")

_TICK = 0.2  # cancel-check granularity while sleeping

_lock = threading.Lock()
_history: dict[str, deque] = {}   # provider -> timestamps of granted slots
_waiting: dict[str, dict] = {}    # provider -> {"until": ts, "limit": key} while sleeping
_cancel: threading.Event | None = None


class Cancelled(Exception):
    """A rate-limit wait was interrupted by the registered stop event."""


def set_cancel_event(evt: threading.Event):
    """Register the job manager's stop event so Stop can abort a wait."""
    global _cancel
    _cancel = evt


def _limits_for(provider: str) -> dict[str, int]:
    import settings as settings_mod
    raw = (settings_mod.load().get("rate_limits") or {}).get(provider) or {}
    out = {}
    for key in WINDOWS:
        try:
            val = int(raw.get(key) or 0)
        except (TypeError, ValueError):
            val = 0
        out[key] = max(0, val)
    return out


def _try_take(provider: str, limits: dict) -> tuple[float, str | None]:
    """Under _lock: take a slot (returns (0, None)) or report the wait needed
    before the most-constrained window frees one (returns (seconds, window))."""
    now = time.time()
    hist = _history.setdefault(provider, deque())
    while hist and hist[0] <= now - WINDOWS["rpd"]:
        hist.popleft()
    wait, which = 0.0, None
    for key, span in WINDOWS.items():
        cap = limits[key]
        if not cap:
            continue
        in_window = [t for t in hist if t > now - span]
        if len(in_window) >= cap:
            # a slot frees when the oldest in-window request ages out
            need = in_window[0] + span - now
            if need > wait:
                wait, which = need, key
    if wait <= 0:
        hist.append(now)
        return 0.0, None
    return wait, which


def acquire(provider: str):
    """Block until a request slot is free for `provider`, then consume it.
    Raises Cancelled if the registered stop event fires during the wait."""
    while True:
        limits = _limits_for(provider)
        with _lock:
            wait, which = _try_take(provider, limits)
            if not wait:
                _waiting.pop(provider, None)
                return
            _waiting[provider] = {"until": time.time() + wait, "limit": which}
        end = time.time() + wait
        while True:
            if _cancel is not None and _cancel.is_set():
                with _lock:
                    _waiting.pop(provider, None)
                raise Cancelled(
                    f"stopped while waiting on {provider} rate limit ({which})"
                )
            remaining = end - time.time()
            if remaining <= 0:
                break
            time.sleep(min(_TICK, remaining))
        # loop back and re-contend — another thread may have taken the slot


def waiting() -> dict:
    """{provider: {"seconds": s, "limit": "rpm"}} for calls currently blocked
    in acquire(). Polled by jobs.status() so the UI can show WHY progress
    paused instead of a silently frozen bar."""
    now = time.time()
    with _lock:
        return {
            p: {"seconds": round(w["until"] - now, 1), "limit": w["limit"]}
            for p, w in _waiting.items()
            if w["until"] > now
        }


def reset():
    """Clear all request history and wait markers (tests / debugging)."""
    with _lock:
        _history.clear()
        _waiting.clear()


# ── Suggested limits ─────────────────────────────────────────────────────────
# There is NO "what are my limits" endpoint for an AI Studio API key (quota
# introspection needs the Cloud Quotas API with OAuth on a GCP project), so
# suggestions come from two sources, best first:
#   1. LEARNED (account-accurate): Gemini 429 bodies carry google.rpc.QuotaFailure
#      metadata stating the violated quota and its exact per-account value —
#      parsed by learn_from_gemini_429() at both 429 sites and kept here
#      (in-memory). This is the ONLY source that reflects your real limits.
#   2. PUBLISHED (approximate): the static table below — conservative, ballpark
#      free-tier starting points. Real free-tier limits vary by account, model,
#      tier and region and change over time, so these are a safe default only,
#      NOT a promise for any given account. See Google's canonical page:
#      https://ai.google.dev/gemini-api/docs/rate-limits — and set your own in
#      Settings → rate_limits (the "Suggest" button prefers learned values).
# LM Studio is local (no meaningful ceiling) and 9Router rotates pooled
# accounts/keys on 429 internally — its whole job — so neither gets a
# suggestion; suggest() returns None for them.

# Substring match, specific → generic (first hit wins). Values are conservative
# approximations that the app self-corrects from real 429s — see the note above.
SUGGESTED_GEMINI = (
    ("embedding",  {"rpm": 100, "rpd": 1000}),  # text-embedding free tier (approx)
    ("flash-lite", {"rpm": 15,  "rpd": 500}),   # lite models: largest allowance (approx)
    ("",           {"rpm": 10,  "rpd": 250}),   # any other Gemini model (conservative)
)

_learned: dict[tuple[str, str], dict[str, int]] = {}  # (provider, model) -> {window: value}

_QUOTA_ID_WINDOWS = (
    ("PerSecond", "rps"), ("PerMinute", "rpm"), ("PerHour", "rph"), ("PerDay", "rpd"),
)


def learn_from_gemini_429(model: str, body) -> None:
    """Harvest real per-account quota values from a Gemini 429 response body.
    Best-effort: any parse failure is silently ignored — the 429 itself is
    already handled by the caller's cooldown logic."""
    try:
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        data = json.loads(body)
        for detail in data.get("error", {}).get("details", []):
            if not detail.get("@type", "").endswith("QuotaFailure"):
                continue
            for v in detail.get("violations", []):
                qid = v.get("quotaId", "")
                try:
                    val = int(v.get("quotaValue", 0))
                except (TypeError, ValueError):
                    continue
                m = v.get("quotaDimensions", {}).get("model") or model
                for token, window in _QUOTA_ID_WINDOWS:
                    if token in qid and val > 0:
                        with _lock:
                            _learned.setdefault(("gemini", m), {})[window] = val
    except Exception:
        pass


def suggest(provider: str, model: str | None = None) -> dict | None:
    """Suggested {rps, rpm, rph, rpd, sources} for a provider (+model), or
    None when throttling isn't meaningful for it (local LM Studio; 9Router,
    which rotates pooled accounts internally). sources maps each non-zero
    window to "learned" (from a real 429) or "published" (static table)."""
    if provider != "gemini":
        return None
    if not model:
        from constants import GEMINI_VISION_MODELS
        model = GEMINI_VISION_MODELS[0]
    limits = {k: 0 for k in WINDOWS}
    sources: dict[str, str] = {}
    for pattern, vals in SUGGESTED_GEMINI:
        if pattern in model:
            for k, v in vals.items():
                limits[k] = v
                sources[k] = "published"
            break
    with _lock:
        learned = dict(_learned.get(("gemini", model), {}))
    for k, v in learned.items():
        limits[k] = v
        sources[k] = "learned"
    return {**limits, "model": model, "sources": sources}
