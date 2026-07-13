"""Photo Vault HTTP API (FastAPI).

Thin JSON layer over the existing, UI-agnostic backend (indexer / search /
embeddings / vision / faces / tagger). Serves the built Svelte SPA from web/dist
in production. Run:  uv run uvicorn api:app --app-dir src --port <port>
"""

import json
import mimetypes
import os
import threading
from contextlib import contextmanager
from datetime import datetime as _dt
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import ImageOps
from pydantic import BaseModel, Field

import imaging  # noqa: F401 — registers HEIF opener + pixel-bomb cap on import
from imaging import (
    safe_open,
    derivative_path,
    legacy_derivative_path,
    ensure_derivative,
    THUMB_PX as _THUMB_PX,
    MEDIUM_PX as _MEDIUM_PX,
)

from constants import (
    THUMB_DIR,
    PROJECT_ROOT,
    SERVER_PORT,
)
import db
import security
from indexer import (
    Indexer,
    catalog_path_for,
    load_catalog_cached,
    resolve_photo_date as indexer_resolve_photo_date,
)

# id is a content hash, so a given id's bytes never change → cache forever.
_IMMUTABLE_CACHE = {"Cache-Control": "private, max-age=31536000, immutable"}
from search import search_images, get_available_filter_values, SearchUnavailableError
from vision import (
    list_lm_studio_models,
    list_lm_studio_models_v0,
    classify_lm_studio_model,
    list_gemini_vision_models,
    list_9router_vision_models,
    validate_vision_output,
    gemini_cooldowns,
    ninerouter_cooldowns,
)
from embeddings import (
    get_registry,
    get_active_model,
    set_active_model,
    collection_name_for,
    list_gemini_embed_models,
    list_9router_embed_models,
    ninerouter_embed_cooldowns,
)
from tagger import (
    add_person_reference,
    add_person_embedding,
    get_all_persons,
    rename_person,
    delete_person,
    set_relation,
    get_people_detailed,
)
import dupes as dupes_mod
import trash as trash_mod
from faces import load_face_data, face_index_count, rebuild_face_index
from validator import service_status
from jobs import manager, JOB_TYPES
import clustering
from clustering import ClusterMembersStaleError
import albums as albums_mgr
import folders as folder_mgr
import ratelimit
import settings as settings_mgr

os.makedirs(THUMB_DIR, exist_ok=True)

app = FastAPI(title="Photo Vault", version="1.0")

# Mutual exclusion between a (synchronous) scan and the background index job:
# both write to the catalog DB, so they must never run concurrently.
_scan_active = threading.Event()

# Held for the full duration of a multi-id destructive op (batch delete,
# orphaned cleanup, trash purge) so a scan can't start mid-loop and resurrect
# a row the op is in the middle of removing (see _reject_if_writer_active).
_writer_active = threading.Event()

# Guards atomic check-and-set across _scan_active / _writer_active so two
# concurrent requests can never both observe "not active" and both proceed.
_activity_lock = threading.Lock()

# Reject requests whose Host header isn't a loopback name. This is the primary
# defense against DNS-rebinding: a malicious page can point its domain at
# 127.0.0.1, but the browser still sends that domain in the Host header.
# PV_ALLOWED_HOSTS (comma-separated) lets a deployment add its own hostname/IP
# (e.g. a LAN address when self-hosting beyond localhost).
_BASE_HOSTS = ["localhost", "127.0.0.1", "[::1]", "testserver"]
_EXTRA_HOSTS = [
    h.strip() for h in os.environ.get("PV_ALLOWED_HOSTS", "").split(",") if h.strip()
]
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_BASE_HOSTS + _EXTRA_HOSTS,
)

# Restrict CORS to the loopback origins the SPA actually runs on (prod port +
# Vite dev port). No wildcard — a random website can no longer read API
# responses cross-origin.
_ALLOWED_ORIGINS = [
    f"http://localhost:{SERVER_PORT}",
    f"http://127.0.0.1:{SERVER_PORT}",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress JSON / JS / CSS responses. (Already-compressed JPEGs barely change,
# but the API's JSON payloads — search results, status — shrink substantially.)
app.add_middleware(GZipMiddleware, minimum_size=1024)


# FastAPI's auto docs (/docs, /redoc, /openapi.json) don't go through /api/* and
# would otherwise bypass the bearer-token check entirely — route them through
# the same auth gate as the API itself when auth is required.
_DOC_PATHS = {"/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def _require_token(request: Request, call_next):
    """Enforce the bearer token on /api/* (and the auto-docs routes) when
    PV_REQUIRE_AUTH=1 (set by serve.py)."""
    if security.auth_enabled():
        path = request.url.path
        needs_auth = (
            path.startswith("/api/") and path not in security.EXEMPT_API_PATHS
        ) or path in _DOC_PATHS
        if needs_auth:
            authorized = security.request_authorized(
                request.headers.get("authorization"),
                request.query_params.get("_t"),
            )
            if not authorized:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


# ── models ──────────────────────────────────────────────────────────────────
class ScanReq(BaseModel):
    dirs: list[str] = []  # empty → use folder registry


class JobRef(BaseModel):
    # Optional target for stop/reset — a specific job id from the progress
    # `jobs` list. Omitted → act on all jobs (back-compat single-job behavior).
    job_id: str | None = None


class IndexReq(BaseModel):
    type: str
    vision_provider: str = "auto"
    vision_model: str | None = None
    embed_provider: str = "auto"
    embed_model: str | None = None
    caption_source_model: str | None = None
    max_fail: int = 5
    source_path: str | None = None  # ingest only: the staging folder to import
    ingest_media: str = "both"      # ingest only: "both" | "photos" | "videos"
    ingest_video_dest: str | None = None  # ingest only: override video destination


class PersonReq(BaseModel):
    name: str
    ref_dir: str


class ActiveModelReq(BaseModel):
    model: str


class FolderReq(BaseModel):
    path: str


class OrphanedCleanupReq(BaseModel):
    ids: list[str] = []  # empty → remove all orphaned


class BatchDeleteReq(BaseModel):
    ids: list[str]
    delete_file: bool = False


class SettingsReq(BaseModel):
    vision_provider: str | None = None
    vision_model: str | None = None
    embed_provider: str | None = None
    embed_model: str | None = None
    caption_source_model: str | None = None
    max_fail: int | None = None
    vision_concurrency: int | None = None
    vision_max_tokens: int | None = None
    video_frames: int | None = None
    faces_during_embed: bool | None = None
    # Accelerator for face detection — an id from /api/face-providers
    # ("auto" | "openvino:GPU" | "openvino:NPU" | "cuda" | "dml" | "cpu" | …).
    face_provider: str | None = None
    face_cluster_eps: float | None = None
    face_cluster_min_samples: int | None = None
    # {provider: {rps|rpm|rph|rpd: int}} — client-side request ceilings,
    # 0 = unlimited. See settings.DEFAULTS["rate_limits"] / ratelimit.py.
    rate_limits: dict | None = None
    # Where ingest copies new media (null → <first scan folder>\Imported).
    ingest_dest: str | None = None
    # Backup mirror destination (the SD card folder), e.g. E:\PhotoVaultBackup.
    backup_dest: str | None = None


class ClusterReq(BaseModel):
    eps: float | None = None
    min_samples: int | None = None


class NameClusterReq(BaseModel):
    cluster_id: int
    name: str


class IgnoreClusterReq(BaseModel):
    cluster_id: int


class AlbumCreateReq(BaseModel):
    name: str


class AlbumRenameReq(BaseModel):
    name: str


class AlbumItemsReq(BaseModel):
    ids: list[str]


# ── helpers ───────────────────────────────────────────────────────────────────
def _reject_if_writer_active():
    """Deletes/purges write to the catalog DB from a private Indexer copy. A
    concurrent vision/embed job's incremental saves can't resurrect a deleted
    row (they only touch ids they themselves changed), but a concurrent SCAN
    still can: scan checkpoints do a full sync (upsert its whole in-memory
    catalog + delete anything missing from it), so a scan holding a stale
    in-memory copy of a just-deleted id would write it right back. Refuse
    instead of risking that. Also refuses while another multi-id destructive
    op is already in flight (see _writer_guard) — this is a point-in-time
    check, so callers doing more than one write must hold _writer_guard for
    the duration rather than relying on this alone."""
    if _scan_active.is_set() or _writer_active.is_set() or manager.status().get("active"):
        raise HTTPException(
            409, "a scan or indexing job is running; stop it before deleting"
        )


@contextmanager
def _writer_guard():
    """Reserve exclusive writer access for the full duration of a multi-id
    destructive operation (batch delete, orphaned cleanup, trash purge), not
    just at entry. Holding _writer_active for the whole loop — atomically
    checked-and-set alongside _scan_active — closes the race where a scan
    starts partway through the loop and resurrects a row already removed by
    an earlier iteration (see _reject_if_writer_active's docstring)."""
    with _activity_lock:
        _reject_if_writer_active()
        _writer_active.set()
    try:
        yield
    finally:
        _writer_active.clear()


def _chroma_meta(img_id: str) -> dict:
    """Best-effort metadata lookup. db.collection() always get-or-creates the
    collection and Chroma's .get() on an unknown id just returns an empty
    list rather than raising, so there's no distinct "not found" exception to
    special-case here — any exception reaching this point is a genuine,
    unexpected failure (corruption, IO error, etc). Log it so it's
    diagnosable server-side instead of silently and indistinguishably
    collapsing into "not indexed" for the caller."""
    try:
        col = db.collection()
        res = col.get(ids=[img_id], include=["metadatas"])
        if res["ids"]:
            return res["metadatas"][0]
    except Exception as e:
        print(f"[api] chroma metadata lookup failed for {img_id}: {e}")
    return {}


# ── status / health ───────────────────────────────────────────────────────────
@app.get("/api/health")
def health(fresh: bool = False):
    # fresh=1 bypasses the short-TTL memo — used by the Services "Recheck" button
    # so a just-started model shows up immediately instead of within the TTL.
    return service_status(force=fresh)


@app.get("/api/status")
def status():
    s = settings_mgr.load()
    idx = Indexer(use_cache=True)  # read-only; share the cached catalog snapshot
    stage = idx.get_stage_stats()

    # Model-aware counts based on current settings
    vm_label = settings_mgr.vision_model_label(
        s
    )  # e.g. "lm_studio:qwen2-vl-7b" or None
    csm = s.get("caption_source_model")  # caption source for embed
    em = s.get("embed_model")  # embed model (raw name, no provider prefix)

    total = stage.get("total_scanned", 0)
    # Vision progress is a photo-only ratio (videos are captioned by their own
    # keyframe job), so use the photo count as the denominator here.
    photo_total = stage.get("photo_total", total)

    # Legacy pending counts (used by backward-compat code paths)
    legacy_vision_pending = len(idx.get_vision_pending())
    legacy_embed_pending = len(idx.get_embed_pending())

    # Model-specific vision counts
    if vm_label:
        model_vision_pending_list = idx.get_vision_pending_for_model(vm_label)
        vision_done = photo_total - len(model_vision_pending_list)
        vision_pending = len(model_vision_pending_list)
    else:
        vision_done = stage.get("vision_done", 0)
        vision_pending = legacy_vision_pending

    # Embed counts
    eligible_ids = idx.get_embed_eligible_ids(csm)
    eligible = len(eligible_ids)
    if em:
        embed_done = stage.get("models", {}).get(em, {}).get("indexed_count", 0)
    else:
        embed_done = stage.get("active_model_embedded", 0)
    embed_pending = max(0, eligible - embed_done)

    faces = idx.get_faces_stats()
    video_faces = idx.get_video_faces_stats()

    return {
        # Legacy fields kept for backward compat
        "stage": stage,
        "vision_pending": vision_pending,
        "embed_pending": embed_pending if csm or em else legacy_embed_pending,
        "missing_attrs": len(idx.get_missing_attributes(use_cache=True)),
        "missing_full": len(idx.get_missing()),
        "missing_files": len(idx.get_missing_files(use_cache=True)),
        # New model-aware fields
        "model_status": {
            "vision": {
                "selected_label": vm_label,
                "done": vision_done,
                "pending": vision_pending,
                "any_done": stage.get("vision_done", 0),
                "model_summary": idx.get_vision_model_summary(),
            },
            "embed": {
                "selected_model": em or stage.get("active_model"),
                "caption_source": csm,
                "eligible": eligible,
                "done": embed_done,
                "pending": embed_pending,
            },
            "faces": faces,  # {total, detected, pending}
            "video": {
                "total": stage.get("video_total", 0),
                "vision_done": stage.get("video_vision_done", 0),
                "vision_pending": stage.get("video_vision_pending", 0),
                "faces": video_faces,
            },
        },
        "faces_pending": faces["pending"],
        "faces_done": faces["detected"],
        "video_total": stage.get("video_total", 0),
        "video_vision_pending": stage.get("video_vision_pending", 0),
        "video_faces_pending": video_faces["pending"],
        "thumbs_pending": idx.count_thumbs_missing(),
        "dhash_pending": sum(
            1 for d in idx.image_catalog.get("images", {}).values()
            if not d.get("dhash") and d.get("media_type") != "video"
        ),
        "trash_count": len(trash_mod.list_items()),
        "settings": s,
    }


@app.get("/api/settings")
def get_settings():
    return settings_mgr.load()


@app.put("/api/settings")
def put_settings(req: SettingsReq):
    # exclude_unset (not "v is not None") distinguishes "field omitted" from
    # "field explicitly set to null": an explicit null clears that setting
    # back to its default/auto behavior, while an omitted field leaves the
    # existing stored value untouched.
    patch = req.model_dump(exclude_unset=True)
    # Keyframes-per-video: clamp to the same sane bounds the job uses, so a bad
    # value can't be persisted (the job re-clamps too, as defense in depth).
    if "video_frames" in patch and patch["video_frames"] is not None:
        from jobs import _clamp_video_frames
        patch["video_frames"] = _clamp_video_frames(patch["video_frames"])
    # Destination settings go through the same validators as the pre-flight
    # UI, so a rule violation is refused with the same friendly explanation.
    if patch.get("ingest_dest"):
        import ingest as ingest_mod
        v = ingest_mod.validate_dest(patch["ingest_dest"])
        if not v["ok"]:
            raise HTTPException(422, v["reason"])
    if patch.get("backup_dest"):
        import backup as backup_mod
        v = backup_mod.validate_dest(patch["backup_dest"])
        if not v["ok"]:
            raise HTTPException(422, v["reason"])
    updated = settings_mgr.update(patch)
    # Changing the face accelerator must drop the cached FaceAnalysis session so
    # the next faces run rebuilds on the newly chosen device (it's keyed by
    # choice, but reset eagerly so a later same-process read can't serve stale).
    if "face_provider" in patch:
        import faces
        faces.reset_face_app()
    return updated


@app.get("/api/face-providers")
def face_providers():
    """Accelerators available for face detection on this machine — auto-detected
    from the installed onnxruntime build (never hardcoded), so the UI can offer
    only what will actually run. `selected` is the saved setting; `active` is
    what that choice resolves to after CPU-fallback (an unavailable pick shows
    CPU)."""
    import faces
    s = settings_mgr.load()
    selected = s.get("face_provider", "auto")
    return {
        "options": faces.available_accelerators(),
        "selected": selected,
        "active": faces.resolved_provider_label(selected),
    }


@app.delete("/api/settings")
def reset_settings():
    """Reset to factory defaults."""
    settings_mgr.save(settings_mgr.DEFAULTS)
    return settings_mgr.load()


@app.get("/api/rate-limits/suggest")
def rate_limit_suggest(provider: str, model: str | None = None):
    """Suggested request ceilings for a provider (+model). Sources, best
    first: values LEARNED from real Gemini 429 QuotaFailure metadata (exact
    per-account numbers), else the PUBLISHED free-tier table in ratelimit.py.
    There is no query-my-quota endpoint for an AI Studio API key, so these
    two are the only honest signals. available=False for providers where a
    client-side ceiling isn't meaningful."""
    s = ratelimit.suggest(provider, model)
    if s is None:
        reason = (
            "local server — no provider quota to respect"
            if provider == "lm_studio"
            else "9Router rotates pooled accounts/keys on 429 internally — "
                 "throttle only if you want to reserve quota for other apps"
            if provider == "9router"
            else "unknown provider"
        )
        return {"available": False, "reason": reason}
    return {"available": True, **s}


# ── scanning ────────────────────────────────────────────────────────────────
@app.post("/api/scan")
def scan(req: ScanReq):
    """
    Explicit dirs → synchronous scan (API/tests). No dirs → the folder-registry
    scan runs as a background job ("scan" type) so a 26k-file walk doesn't hold
    an HTTP request open for minutes; poll /api/index/progress like any job.
    """
    if manager.status().get("active"):
        raise HTTPException(409, "a job is running; stop it before scanning")

    # Atomic check-and-set: without the lock, two concurrent requests could
    # both observe _scan_active clear and both proceed past the guard.
    with _activity_lock:
        if _writer_active.is_set():
            raise HTTPException(
                409, "a delete/cleanup operation is in progress; try again shortly"
            )
        if _scan_active.is_set():
            raise HTTPException(409, "a scan is already in progress")
        _scan_active.set()

    try:
        dirs = [d.strip() for d in req.dirs if d.strip()]

        if not dirs:
            cfg = folder_mgr.ensure_defaults()
            if not cfg.get("included"):
                raise HTTPException(400, "No folders configured. Add a folder first.")
            try:
                return manager.start("scan")
            except RuntimeError as e:
                raise HTTPException(409, str(e))

        # Explicit dirs: validate they exist on disk, scan synchronously
        invalid = [d for d in dirs if not os.path.isdir(d)]
        if invalid:
            raise HTTPException(400, f"directories not found: {', '.join(invalid)}")
        idx = Indexer(target_directories=dirs)
        summary = idx.scan_only()
        st = _status_dict(idx)
        return {"summary": summary, **st}
    finally:
        _scan_active.clear()


def _status_dict(idx: Indexer = None) -> dict:
    if idx is None:
        idx = Indexer(use_cache=True)
    stage = idx.get_stage_stats()
    return {
        "stage": stage,
        "vision_pending": len(idx.get_vision_pending()),
        "embed_pending": len(idx.get_embed_pending()),
        "missing_attrs": len(idx.get_missing_attributes()),
        "missing_full": len(idx.get_missing()),
        "missing_files": len(idx.get_missing_files()),
    }


# ── folder management ─────────────────────────────────────────────────────────
@app.get("/api/folders")
def get_folders():
    """Return full folder config (included + excluded) seeded with OS defaults if empty."""
    return folder_mgr.ensure_defaults()


@app.get("/api/folders/defaults")
def get_folder_defaults():
    """Suggest OS-appropriate default photo directories that exist on this machine."""
    return {"defaults": folder_mgr.get_defaults()}


@app.post("/api/folders/include")
def add_folder(req: FolderReq):
    path = req.path.strip()
    if not path:
        raise HTTPException(400, "path is required")
    if not os.path.isdir(path):
        raise HTTPException(400, f"directory not found: {path}")
    # Scanning the backup mirror would index a duplicate of every photo —
    # same overlap rule as the backup_dest check in put_settings, enforced
    # from this direction too (folders can be added after backup is set up).
    backup_dest = settings_mgr.load().get("backup_dest")
    if backup_dest:
        p = os.path.normcase(str(Path(path).resolve()))
        b = os.path.normcase(str(Path(backup_dest).resolve()))
        if p == b or p.startswith(b + os.sep) or b.startswith(p + os.sep):
            raise HTTPException(
                422, "that folder overlaps the backup destination — scanning your "
                     "own backup would duplicate every photo in the catalog")
    result = folder_mgr.add_included(path)
    if result["status"] == "not_found":
        raise HTTPException(400, f"directory not found: {path}")
    return result


@app.delete("/api/folders/include")
def remove_folder(path: str = Query(...), purge: bool = False):
    """
    Remove a folder from the included list.
    When purge=true, also deletes all indexed data (captions/embeddings/faces)
    for images under that path. Defaults to false — removing a folder from the
    scan list is not implicitly destructive; pass purge=true to also wipe data.
    Returns {images_purged, config}.
    """
    path = path.strip()
    if not path:
        raise HTTPException(400, "path is required")

    images_purged = 0
    if purge:
        _reject_if_writer_active()
        idx = Indexer()
        images_purged = idx.purge_folder(path)

    result = folder_mgr.remove_included(path)
    return {"images_purged": images_purged, **result}


@app.get("/api/folders/include/count")
def count_folder_images(path: str = Query(...)):
    """Return how many indexed images are under the given folder path (for purge warnings)."""
    idx = Indexer(use_cache=True)
    return {"path": path, "count": idx.count_images_under(path)}


@app.post("/api/folders/exclude")
def add_exclusion(req: FolderReq):
    path = req.path.strip()
    if not path:
        raise HTTPException(400, "path is required")
    if not os.path.isdir(path):
        raise HTTPException(400, f"directory not found: {path}")
    result = folder_mgr.add_excluded(path)
    if result.get("status") == "not_found":
        raise HTTPException(400, f"directory not found: {path}")
    return result


@app.delete("/api/folders/exclude")
def remove_exclusion(path: str = Query(...)):
    """Remove an exclusion (the folder will be scanned again on next scan)."""
    path = path.strip()
    if not path:
        raise HTTPException(400, "path is required")
    return folder_mgr.remove_excluded(path)


# ── orphaned image management ─────────────────────────────────────────────────
@app.get("/api/orphaned")
def get_orphaned():
    """List catalog entries whose file no longer exists on disk."""
    idx = Indexer(use_cache=True)
    missing = idx.get_missing_files()
    return {
        "orphaned": [
            {
                "id": img_id,
                "path": data.get("path", ""),
                "filename": data.get(
                    "filename", os.path.basename(data.get("path", ""))
                ),
            }
            for img_id, data in missing
        ],
        "total": len(missing),
    }


@app.post("/api/orphaned/cleanup")
def cleanup_orphaned(req: OrphanedCleanupReq = None):
    """
    Remove orphaned images from the library. If req.ids is empty, removes all orphaned.
    Returns {removed: count}.

    POST (not DELETE) because this takes a scoped id list in the body — some
    HTTP clients/proxies strip bodies from DELETE requests, which would
    silently turn a scoped cleanup into an all-orphaned wipe.
    """
    with _writer_guard():
        idx = Indexer()
        if req and req.ids:
            target_ids = set(req.ids)
            to_delete = [
                img_id for img_id, _ in idx.get_missing_files() if img_id in target_ids
            ]
        else:
            to_delete = [img_id for img_id, _ in idx.get_missing_files()]

        for img_id in to_delete:
            idx.delete_image(img_id)

    return {"removed": len(to_delete)}


# ── indexing jobs ─────────────────────────────────────────────────────────────
@app.post("/api/index/start")
def index_start(req: IndexReq):
    if req.type not in JOB_TYPES:
        raise HTTPException(400, f"unknown job type: {req.type}")
    if _scan_active.is_set():
        raise HTTPException(409, "a scan is in progress; wait for it to finish")
    if _writer_active.is_set():
        raise HTTPException(
            409, "a delete/cleanup operation is in progress; wait for it to finish"
        )
    # 9Router has no LM-Studio-style auto-detect and the design rule is "only
    # the user-chosen model, never a silent auto-pick" — refuse to start a job
    # that would need one. Only checked for the stages this job type actually
    # runs, so e.g. a faces job isn't blocked by an incomplete embed setting.
    uses_vision = req.type in ("vision", "full", "reanalyze")
    uses_embed = req.type in ("embed", "full", "reanalyze")
    if uses_vision and req.vision_provider == "9router" and not req.vision_model:
        raise HTTPException(422, "9Router requires an explicit vision model — pick one in Run configuration")
    if uses_embed and req.embed_provider == "9router" and not req.embed_model:
        raise HTTPException(422, "9Router requires an explicit embedding model — pick one in Run configuration")
    # Compute the vision model label used in caption_history (for model-aware
    # pending queries). None for auto AND 9router — see settings.vision_model_label,
    # which jobs.start recomputes as the trusted value anyway.
    vml = None
    if req.vision_provider not in ("auto", "9router", None) and req.vision_model:
        vml = f"{req.vision_provider}:{req.vision_model}"
    if req.type == "ingest":
        # Same validators the pre-flight UI uses — one source of truth for
        # the rules and for the friendly explanations.
        import ingest as ingest_mod
        v = ingest_mod.validate_source(req.source_path)
        if not v["ok"]:
            raise HTTPException(422, v["reason"])
        media = req.ingest_media if req.ingest_media in ("both", "photos", "videos") else "both"
        # Validate only the destinations this run will actually write to: the
        # photo dest when importing photos, the video dest when importing videos.
        if media in ("both", "photos"):
            dv = ingest_mod.validate_dest(ingest_mod.default_dest() or "")
            if not dv["ok"]:
                raise HTTPException(422, dv["reason"])
        if media in ("both", "videos"):
            vdest = req.ingest_video_dest or ingest_mod.default_video_dest() or ""
            dvv = ingest_mod.validate_dest(vdest)
            if not dvv["ok"]:
                raise HTTPException(422, f"Video destination: {dvv['reason']}")
    if req.type == "backup":
        import backup as backup_mod
        bs = backup_mod.status()
        if not bs["configured"]:
            raise HTTPException(422, "set a backup destination first")
        if not bs["available"]:
            raise HTTPException(422, f"backup drive not connected ({bs['dest']})")
    try:
        return manager.start(
            req.type,
            vision_provider=req.vision_provider,
            max_fail=req.max_fail,
            vision_model=req.vision_model,
            embed_provider=req.embed_provider,
            embed_model=req.embed_model,
            caption_source_model=req.caption_source_model,
            vision_model_label=vml,
            source_path=req.source_path,
            ingest_media=req.ingest_media,
            ingest_video_dest=req.ingest_video_dest,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/backup/status")
def backup_status():
    import backup as backup_mod
    return backup_mod.status()


@app.get("/api/fs/list")
def fs_list(path: str | None = None):
    """Server-side folder browser for the folder-picker UI (browsers can't
    hand a real filesystem path to a web page). No path → the filesystem
    roots (drive letters on Windows, / + mounted volumes on macOS/Linux);
    otherwise the folder's subdirectories. Local single-user app behind the
    bearer token — directory names are not a secret from the app's own user."""
    import platformfs
    if not path:
        return {"path": None, "parent": None, "dirs": platformfs.list_roots(),
                "sep": os.sep}
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(404, f"folder not found: {path}")
    p = p.resolve()
    dirs = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            name = child.name
            if name.startswith(("$", ".")) or platformfs.is_system_name(name):
                continue
            dirs.append(name)
    except PermissionError:
        raise HTTPException(403, f"no permission to list: {path}")
    parent = None if p == p.parent else str(p.parent)
    return {"path": str(p), "parent": parent, "dirs": dirs, "sep": os.sep}


@app.get("/api/ingest/validate")
def ingest_validate(source: str):
    """Pre-flight for the Import UI: is this source importable, and what
    would the import look at (media count/size, ignored files)? The nice
    refusal sentences come straight from ingest.validate_source."""
    import ingest as ingest_mod
    v = ingest_mod.validate_source(source)
    out = {"ok": v["ok"], "reason": v["reason"],
           "dest": ingest_mod.default_dest()}
    if v["ok"]:
        out.update(ingest_mod.source_stats(source))
        dv = ingest_mod.validate_dest(out["dest"] or "")
        if not dv["ok"]:
            out["ok"] = False
            out["reason"] = dv["reason"]
    return out


@app.get("/api/backup/validate")
def backup_validate(dest: str):
    import backup as backup_mod
    return backup_mod.validate_dest(dest)


@app.get("/api/dedupe/pending")
def dedupe_pending():
    """How many byte-identical extra copies the last scans recorded — drives
    the 'Remove N duplicate copies' button."""
    return {"count": len(Indexer().get_redundant_copies())}


@app.get("/api/provider-models")
def provider_models():
    """Models available per provider for the run-config dropdowns.
    Gemini lists only include verified models — no hardcoded fallback.
    LM Studio type/loaded-state comes from its native v0 API when reachable
    (authoritative), falling back to a name-pattern guess otherwise.
    9Router lists come from the live gateway ([] when it's offline)."""
    lm_models = list_lm_studio_models()
    v0_by_id = {m["id"]: m for m in list_lm_studio_models_v0()}
    return {
        "lm_studio": lm_models,
        "lm_studio_types": {
            m: classify_lm_studio_model(m, v0_by_id.get(m)) for m in lm_models
        },
        "gemini_vision": list_gemini_vision_models(fallback=False),
        "gemini_embed": list_gemini_embed_models(fallback=False),
        "gemini_cooldowns": gemini_cooldowns(),
        "ninerouter_vision": list_9router_vision_models(),
        "ninerouter_embed": list_9router_embed_models(),
        "ninerouter_cooldowns": {
            **ninerouter_cooldowns(),
            **ninerouter_embed_cooldowns(),
        },
    }


@app.post("/api/index/stop")
def index_stop(req: JobRef | None = None):
    # With a job_id → stop just that job (jobs can run concurrently now);
    # without → stop every active job (back-compat with the old single-job UI).
    manager.stop(req.job_id if req else None)
    return manager.status()


@app.get("/api/index/progress")
def index_progress():
    return manager.status()


@app.post("/api/index/reset")
def index_reset(req: JobRef | None = None):
    manager.reset(req.job_id if req else None)
    return manager.status()


# ── search / filters ────────────────────────────────────────────────────────
@app.get("/api/filters")
def filters():
    return get_available_filter_values()


# Shared upper bound for any "how many results" query param — generous for a
# personal-library-scale app while keeping a single malicious/typo'd value
# from forcing a huge chroma fetch.
_MAX_RESULT_LIMIT = 500


def _search_response(res: dict | None) -> dict:
    metas = res.get("metadatas", [[]])[0] if res else []
    ids = res.get("ids", [[]])[0] if res else []
    missing = _missing_ids_cached()
    out = {"results": [_card(i, m, missing) for i, m in zip(ids, metas)]}
    if res:
        if res.get("person_not_found"):
            out["person_not_found"] = True
        if res.get("filter_error"):
            out["filter_error"] = True
    return out


@app.get("/api/search")
def search(
    q: str = Query(""),
    person: str | None = None,
    filters: str | None = None,
    top_k: int = Query(200, ge=1, le=_MAX_RESULT_LIMIT),
):
    # `filters` (GET only): optional JSON-encoded object, e.g.
    # ?filters={"year":"2024"} — the POST route already accepted a filters
    # body; GET previously had no way to filter at all.
    filters_in = {}
    if filters:
        try:
            parsed = json.loads(filters)
            if isinstance(parsed, dict):
                filters_in = {k: v for k, v in parsed.items() if v and v != "All"}
        except json.JSONDecodeError:
            raise HTTPException(400, "filters must be a JSON object")
    # Pass q through as-is: an empty q with a person set is the person-only
    # browse path (all their photos), which coercing q to "photo" would disable.
    try:
        res = search_images(q, top_k=top_k, filters=filters_in, person=person or None)
    except SearchUnavailableError as e:
        raise HTTPException(503, f"search is temporarily unavailable: {e}")
    return _search_response(res)


class SearchReq(BaseModel):
    q: str | None = ""
    person: str | None = None
    filters: dict = {}
    top_k: int = Field(200, ge=1, le=_MAX_RESULT_LIMIT)


@app.post("/api/search")
def search_post(body: SearchReq):
    q = body.q or ""
    person = body.person or None
    filters_in = {
        k: v for k, v in (body.filters or {}).items() if v and v != "All"
    }
    try:
        res = search_images(q, top_k=body.top_k, filters=filters_in, person=person)
    except SearchUnavailableError as e:
        raise HTTPException(503, f"search is temporarily unavailable: {e}")
    return _search_response(res)


def _missing_ids_cached() -> set | None:
    """Set of catalog ids whose file is gone, from the indexer's 30s-cached
    missing-files scan. A broad filter-browse can return thousands of cards, and
    doing os.path.exists() per card meant thousands of filesystem stats per
    request (and disk contention under concurrent load). One shared cached set +
    a membership test replaces that. Returns None if the scan fails, so callers
    fall back to a live per-file stat rather than mislabel everything present."""
    try:
        return {img_id for img_id, _ in Indexer(use_cache=True).get_missing_files(use_cache=True)}
    except Exception:
        return None


def _exists(path: str, img_id: str, missing: set | None) -> bool:
    return (img_id not in missing) if missing is not None else os.path.exists(path)


def _card(img_id: str, meta: dict, missing: set | None = None) -> dict:
    return {
        "id": img_id,
        "filename": os.path.basename(meta.get("path", img_id)),
        "caption": meta.get("caption", ""),
        "year": meta.get("year", ""),
        "occasion": meta.get("occasion", ""),
        "exists": _exists(meta.get("path", ""), img_id, missing),
        "media_type": meta.get("media_type", "image"),
        "duration_s": meta.get("duration_s", 0),
    }


def _cards_for_ids(ids: list[str]) -> list[dict]:
    """Build grid cards for an explicit id list (e.g. an album), pulling captions
    from the active embedding collection and falling back to the catalog."""
    if not ids:
        return []
    meta_by_id = {}
    try:
        active = get_active_model()
        if active:
            res = db.collection(active).get(ids=ids, include=["metadatas"])
            meta_by_id = dict(zip(res["ids"], res["metadatas"]))
    except Exception:
        pass
    catalog = load_catalog_cached().get("images", {})
    missing = _missing_ids_cached()
    cards = []
    for iid in ids:
        m = meta_by_id.get(iid, {})
        c = catalog.get(iid, {})
        path = m.get("path") or c.get("path", "")
        cards.append(
            {
                "id": iid,
                "filename": m.get("filename")
                or c.get("filename")
                or os.path.basename(path),
                "caption": m.get("caption", ""),
                "year": m.get("year", ""),
                "occasion": m.get("occasion", ""),
                "exists": _exists(path, iid, missing),
                "media_type": m.get("media_type") or c.get("media_type", "image"),
                "duration_s": m.get("duration_s") or c.get("duration_s", 0),
            }
        )
    return cards


# ── albums ────────────────────────────────────────────────────────────────────
@app.get("/api/albums")
def albums_list():
    return {"albums": albums_mgr.list_albums()}


@app.post("/api/albums")
def albums_create(req: AlbumCreateReq):
    try:
        return albums_mgr.create_album(req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/albums/{album_id}")
def albums_get(album_id: str):
    a = albums_mgr.get_album(album_id)
    if not a:
        raise HTTPException(404, "album not found")
    return {
        "id": album_id,
        "name": a["name"],
        "photos": _cards_for_ids(a.get("image_ids", [])),
    }


@app.put("/api/albums/{album_id}")
def albums_rename(album_id: str, req: AlbumRenameReq):
    try:
        albums_mgr.rename_album(album_id, req.name)
    except KeyError:
        raise HTTPException(404, "album not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.delete("/api/albums/{album_id}")
def albums_delete(album_id: str):
    albums_mgr.delete_album(album_id)
    return {"ok": True}


@app.post("/api/albums/{album_id}/add")
def albums_add(album_id: str, req: AlbumItemsReq):
    try:
        count = albums_mgr.add_to_album(album_id, req.ids)
    except KeyError:
        raise HTTPException(404, "album not found")
    return {"count": count}


@app.post("/api/albums/{album_id}/remove")
def albums_remove(album_id: str, req: AlbumItemsReq):
    try:
        count = albums_mgr.remove_from_album(album_id, req.ids)
    except KeyError:
        raise HTTPException(404, "album not found")
    return {"count": count}


# ── recent / timeline ─────────────────────────────────────────────────────────
@app.get("/api/recent")
def recent(limit: int = Query(60, ge=1, le=_MAX_RESULT_LIMIT)):
    active = get_active_model()
    if not active:
        return {"results": []}
    # True recency from the (cached, in-memory) catalog by created_at, then fetch
    # metadata for just those ids — no full-collection scan. Over-fetch a little
    # because some recent photos may not be embedded into the active model yet.
    cat = load_catalog_cached().get("images", {})
    ordered = sorted(
        cat.items(), key=lambda kv: kv[1].get("created_at", 0), reverse=True
    )
    candidate_ids = [img_id for img_id, _ in ordered[: max(limit * 3, limit)]]
    if not candidate_ids:
        return {"results": []}
    res = db.collection(active).get(ids=candidate_ids, include=["metadatas"])
    meta_by_id = dict(zip(res["ids"], res["metadatas"]))
    out = []
    for img_id, _ in ordered:
        m = meta_by_id.get(img_id)
        if m is not None:
            out.append(_card(img_id, m))
            if len(out) >= limit:
                break
    if not out:
        # Early in indexing the newest-by-date candidates may not be embedded
        # yet — show whatever IS in the collection instead of an empty grid.
        got = db.collection(active).get(limit=limit, include=["metadatas"])
        out = [_card(i, m) for i, m in zip(got["ids"], got["metadatas"])]
    return {"results": out}


@app.get("/api/map")
def map_photos():
    """Geotagged photos (from EXIF GPS) for the Map tab."""
    catalog = load_catalog_cached().get("images", {})
    points = []
    for img_id, data in list(catalog.items()):
        meta = data.get("metadata", {})
        lat, lon = meta.get("gps_lat"), meta.get("gps_lon")
        if lat is not None and lon is not None:
            points.append(
                {
                    "id": img_id,
                    "lat": lat,
                    "lon": lon,
                    "filename": data.get("filename", ""),
                    "exists": os.path.exists(data.get("path", "")),
                }
            )
    return {"points": points}


def _resolve_photo_date(data: dict) -> str:
    """EXIF date -> a date parsed from the filename -> file/import timestamp.
    Single source of truth shared with the embed payload (indexer), so the
    Timeline and the Search 'Year' filter place a photo under the same year."""
    return indexer_resolve_photo_date(data)


@app.get("/api/timeline/summary")
def timeline_summary():
    """Cheap year -> month -> count map for quick-jump navigation. No
    per-photo os.path.exists check (that's what makes /api/timeline itself
    expensive at scale) — just a date tally, safe to call eagerly."""
    catalog = load_catalog_cached().get("images", {})
    summary: dict[str, dict[str, int]] = {}
    for data in catalog.values():
        date = _resolve_photo_date(data)
        y = date[:4] if date and len(date) >= 4 else "Unknown"
        m = date[5:7] if y != "Unknown" and len(date) >= 7 else "00"
        year_bucket = summary.setdefault(y, {})
        year_bucket[m] = year_bucket.get(m, 0) + 1
    return {"summary": summary}


@app.get("/api/timeline")
def timeline(year: str | None = None, offset: int = 0, limit: int = 60):
    """
    Paged timeline. Without `year`: every year with its count and the first
    `limit` photos (newest first). With `year`: one page of that year's photos.
    Returning the whole catalog in one response was 2.4 MB / 4.4 s at 25k
    photos (an os.path.exists per photo) — pages keep both bounded.
    """
    if year is None and offset:
        # The all-years response is grouped per year (each capped at `limit`
        # newest photos) — an offset into a flattened cross-year list isn't a
        # meaningful operation on that shape, and the frontend never sends one
        # without `year`. Reject rather than silently ignoring it, which used
        # to make a caller believe pagination was honored when it wasn't.
        raise HTTPException(400, "offset requires year to be specified")

    catalog = load_catalog_cached().get("images", {})
    by_year: dict[str, list] = {}
    for img_id, data in list(catalog.items()):
        date = _resolve_photo_date(data)
        y = date[:4] if date and len(date) >= 4 else "Unknown"
        by_year.setdefault(y, []).append((date, img_id, data))

    def _tcard(img_id, data, date):
        return {
            "id": img_id,
            "filename": data.get("filename", ""),
            "date": date,  # "YYYY:MM:DD hh:mm:ss" — the UI groups by month
            "exists": os.path.exists(data.get("path", "")),
            # Catalog-driven, so videos show in the timeline right after Scan —
            # no embedding needed for the browse-and-play surface.
            "media_type": data.get("media_type", "image"),
            "duration_s": data.get("duration_s", 0),
        }

    limit = max(1, min(limit, 500))
    if year is not None:
        rows = sorted(by_year.get(year, []), key=lambda t: t[0], reverse=True)
        page = rows[max(0, offset) : max(0, offset) + limit]
        return {
            "year": year,
            "count": len(rows),
            "photos": [_tcard(i, d, dt) for dt, i, d in page],
        }

    # Newest year first; "Unknown" (sorts above digits) belongs at the end.
    year_order = sorted((y for y in by_year if y != "Unknown"), reverse=True)
    if "Unknown" in by_year:
        year_order.append("Unknown")
    years_out = []
    for y in year_order:
        rows = sorted(by_year[y], key=lambda t: t[0], reverse=True)
        years_out.append(
            {
                "year": y,
                "count": len(rows),
                "photos": [_tcard(i, d, dt) for dt, i, d in rows[:limit]],
            }
        )
    return {"years": years_out}


# ── people ────────────────────────────────────────────────────────────────────
@app.get("/api/people")
def people():
    # `people` stays a flat name list (search's person filter + older callers);
    # `detailed` adds each person's relation/family metadata for the People UI.
    return {"people": get_all_persons(), "detailed": get_people_detailed()}


@app.post("/api/people")
def add_person(req: PersonReq):
    name = req.name.strip()
    ref_dir = req.ref_dir.strip()
    if not name:
        raise HTTPException(400, "name required")
    if not os.path.isdir(ref_dir):
        raise HTTPException(400, "reference folder not found")
    result = add_person_reference(name, ref_dir)
    if not result.get("registered"):
        raise HTTPException(
            400, "no faces detected in the reference folder — person not registered"
        )
    out = {"ok": True, "name": name, "faces": result.get("faces_used", 0)}
    if result.get("skipped_multi_face"):
        out["skipped_multi_face"] = result["skipped_multi_face"]
    return out


class PersonRenameReq(BaseModel):
    new_name: str


@app.put("/api/people/{name}")
def person_rename(name: str, req: PersonRenameReq):
    new = req.new_name.strip()
    if not new:
        raise HTTPException(400, "new name required")
    try:
        rename_person(name, new)
    except KeyError:
        raise HTTPException(404, "person not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "name": new}


@app.delete("/api/people/{name}")
def person_delete(name: str):
    if not delete_person(name):
        raise HTTPException(404, "person not found")
    return {"ok": True}


class RelationReq(BaseModel):
    relation: str | None = None      # e.g. "daughter"; "" clears it
    is_family: bool | None = None    # explicit override; None → derive from relation


@app.put("/api/people/{name}/relation")
def person_set_relation(name: str, req: RelationReq):
    """Set a person's relationship/family metadata — identity (name) is
    unchanged. Relation is a separate, structured field so 'name' stays the
    unique identifier while 'daughter'/'spouse'/… describes the relationship."""
    try:
        ok = set_relation(name, relation=req.relation, is_family=req.is_family)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "person not found")
    return {"ok": True, "name": name, "relation": (req.relation or "").strip().lower()}


# ── face clustering / tagging ─────────────────────────────────────────────────
@app.get("/api/faces/status")
def faces_status():
    """Setup info for the Faces UI: detection progress + current clusters."""
    idx = Indexer(use_cache=True)
    stats = idx.get_faces_stats()
    clusters = clustering.load_clusters()
    cl = clusters.get("clusters", [])
    return {
        **stats,
        "clusters_count": len(cl),
        "clustered_at_faces": clusters.get("total_faces", 0),
        "named": sum(1 for c in cl if c.get("status") == "named"),
        "params": clusters.get("params", {}),
        "ann_index_count": face_index_count(),
    }


@app.post("/api/faces/reindex")
def faces_reindex():
    """Rebuild the ANN face index from on-disk face JSON (for libraries indexed
    before the index existed, or to repair it)."""
    if _scan_active.is_set() or manager.status().get("active"):
        raise HTTPException(
            409, "a scan or indexing job is running; wait for it to finish"
        )
    return {"indexed": rebuild_face_index()}


@app.post("/api/faces/cluster")
def faces_cluster(req: ClusterReq):
    """Run DBSCAN clustering over all detected faces. User-triggered."""
    if _scan_active.is_set() or manager.status().get("active"):
        raise HTTPException(
            409, "a scan or indexing job is running; wait for it to finish"
        )
    s = settings_mgr.load()
    eps = req.eps if req.eps is not None else s.get("face_cluster_eps", 0.5)
    min_samples = (
        req.min_samples
        if req.min_samples is not None
        else s.get("face_cluster_min_samples", 3)
    )
    summary = clustering.cluster_faces(eps=eps, min_samples=min_samples)
    return summary


@app.get("/api/faces/clusters")
def faces_clusters(samples: int = Query(6, ge=1, le=20)):
    """List clusters (excluding ignored) with a few sample faces each for review."""
    data = clustering.load_clusters()
    out = []
    for c in data.get("clusters", []):
        if c.get("status") == "ignored":
            continue
        out.append(
            {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "status": c.get("status", "new"),
                "name": c.get("name"),
                "samples": c["members"][:samples],
            }
        )
    return {"clusters": out}


@app.post("/api/faces/name")
def faces_name(req: NameClusterReq):
    """Name a cluster → register it as a person from its mean face embedding."""
    if not req.name.strip():
        raise HTTPException(400, "name required")
    try:
        emb = clustering.cluster_mean_embedding(req.cluster_id)
    except ClusterMembersStaleError:
        raise HTTPException(
            409,
            "this cluster's photos have changed since it was grouped (re-detect "
            "faces or re-cluster) — none of its members are valid anymore",
        )
    if emb is None:
        raise HTTPException(404, "cluster not found or has no faces")
    add_person_embedding(req.name.strip(), emb)
    clustering.set_cluster_status(req.cluster_id, "named", name=req.name.strip())
    return {"ok": True, "name": req.name.strip()}


@app.post("/api/faces/ignore")
def faces_ignore(req: IgnoreClusterReq):
    clustering.set_cluster_status(req.cluster_id, "ignored")
    return {"ok": True}


@app.get("/api/faces/crop")
def face_crop(image_id: str = Query(...), face_index: int = 0):
    """Serve a cropped, cached JPEG of one detected face for cluster review."""
    out = derivative_path(f"{image_id}:{face_index}", "_face")
    legacy = legacy_derivative_path(f"{image_id}:{face_index}", "_face")
    if os.path.exists(legacy) and not os.path.exists(out):
        return FileResponse(legacy, media_type="image/jpeg", headers=_IMMUTABLE_CACHE)
    if not os.path.exists(out):
        path = _resolve_indexed_path(image_id)
        if not path:
            raise HTTPException(404, "source image not found")
        faces = load_face_data(image_id)
        if face_index < 0 or face_index >= len(faces):
            raise HTTPException(404, "face index out of range")
        bbox = faces[face_index].get("bbox") or []
        try:
            with safe_open(path) as im:
                # Apply EXIF orientation BEFORE computing/using the bbox — the
                # bbox coordinates come from face detection, and must be read
                # against the same (rotated) coordinate space the detector
                # used, or a 90/270-degree-rotated photo crops the wrong
                # region (width/height are swapped between the two spaces).
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                w, h = im.size
                x1, y1, x2, y2 = bbox if len(bbox) == 4 else (0, 0, w, h)
                # pad ~30% around the face box, clamped to image bounds
                pw, ph = (x2 - x1) * 0.3, (y2 - y1) * 0.3
                box = (
                    max(0, int(x1 - pw)),
                    max(0, int(y1 - ph)),
                    min(w, int(x2 + pw)),
                    min(h, int(y2 + ph)),
                )
                crop = im.crop(box)
                crop.thumbnail((200, 200))
                crop.save(out, "WEBP", quality=80)
        except Exception as e:
            print(f"[api] face crop failed {image_id}:{face_index}: {e}")
            return _placeholder_thumb()
    return FileResponse(out, media_type="image/webp", headers=_IMMUTABLE_CACHE)


# ── models ──────────────────────────────────────────────────────────────────
@app.get("/api/models")
def models():
    reg = get_registry()
    models_out = {
        name: {**info, "indexed_count": db.collection(name).count()}
        for name, info in reg.get("models", {}).items()
    }
    return {"active": reg.get("active_model"), "models": models_out}


@app.post("/api/models/active")
def set_model(req: ActiveModelReq):
    try:
        set_active_model(req.model)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"active": req.model}


# ── image / thumbnail / delete ─────────────────────────────────────────────────
def _resolve_indexed_path(img_id: str) -> str | None:
    """
    Map a client-supplied id to an on-disk path ONLY via the catalog / vector
    metadata. The id is never used as a path itself — this is what prevents
    arbitrary-file-read (e.g. id=C:\\Windows\\win.ini). Returns a confined,
    canonical path to an existing file, or None.
    """
    path = _chroma_meta(img_id).get("path") or catalog_path_for(img_id)
    if not path:
        return None
    return security.is_safe_real_path(path)


def _serve_derivative(img_id: str, suffix: str, max_px: int, src_path: str):
    """Serve the WebP derivative, falling back to a pre-existing legacy JPEG,
    generating the WebP on demand. Returns a FileResponse or None on failure."""
    out = derivative_path(img_id, suffix)
    if os.path.exists(out):
        return FileResponse(out, media_type="image/webp", headers=_IMMUTABLE_CACHE)
    legacy = legacy_derivative_path(img_id, suffix)
    if os.path.exists(legacy):
        return FileResponse(legacy, media_type="image/jpeg", headers=_IMMUTABLE_CACHE)
    if not ensure_derivative(src_path, out, max_px):
        return None
    return FileResponse(out, media_type="image/webp", headers=_IMMUTABLE_CACHE)


_EXT_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    # Videos — browsers are picky about the exact type for <video> playback.
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".3gp": "video/3gpp",
    ".wmv": "video/x-ms-wmv",
    ".mts": "video/mp2t",
    ".m2ts": "video/mp2t",
}


def _guess_media_type(path: str) -> str:
    """mimetypes.guess_type() can't resolve some extensions (notably .webp) on
    every system, which used to leave FileResponse defaulting to text/plain
    and breaking inline rendering. Check a small known-extension map first,
    then fall back to mimetypes, then a generic binary type."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _EXT_MIME:
        return _EXT_MIME[ext]
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _placeholder_thumb() -> Response:
    """A neutral gray JPEG returned when a real thumbnail can't be produced
    (corrupt / unsupported / oversized file), so the grid doesn't show a broken
    image and we don't leak a 500."""
    ph = os.path.join(THUMB_DIR, "_placeholder.jpg")
    if not os.path.exists(ph):
        try:
            from PIL import Image as _Img

            _Img.new("RGB", (_THUMB_PX, _THUMB_PX), (40, 44, 52)).save(
                ph, "JPEG", quality=70
            )
        except Exception:
            return Response(status_code=204)
    return FileResponse(ph, media_type="image/jpeg")


@app.get("/api/image")
def image(id: str = Query(...), thumb: bool = False, size: str = "full"):
    """
    size: "thumb" (400px grid), "medium" (1600px lightbox), or "full" (original).
    The legacy `thumb=true` flag maps to size=thumb. All tiers are immutable
    (id is a content hash) so they cache forever in the browser.
    """
    if thumb:
        size = "thumb"
    path = _resolve_indexed_path(id)
    if not path:
        raise HTTPException(404, "not an indexed photo, or file missing on disk")

    if size == "full":
        return FileResponse(
            path, media_type=_guess_media_type(path), headers=_IMMUTABLE_CACHE
        )

    if size == "medium":
        resp = _serve_derivative(id, "_m", _MEDIUM_PX, path)
        return resp or FileResponse(
            path, media_type=_guess_media_type(path), headers=_IMMUTABLE_CACHE
        )  # original fallback

    # default: thumb
    resp = _serve_derivative(id, "", _THUMB_PX, path)
    return resp or _placeholder_thumb()


# Cap one 206 response so an open-ended `bytes=0-` can't pull a multi-GB video
# into memory — the browser simply requests the next slice. 4 MiB is plenty for
# smooth seeking while keeping memory bounded and Content-Length exact.
_VIDEO_MAX_SLICE = 4 * 1024 * 1024


def _parse_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single-range 'bytes=start-end' header into inclusive (start, end)
    byte offsets, or None if unparseable/unsatisfiable. Only the single-range
    form is supported — all a <video> element ever sends for seeking."""
    if not range_header or not range_header.strip().lower().startswith("bytes="):
        return None
    spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":              # suffix range: bytes=-N (last N bytes)
            n = int(end_s)
            if n <= 0:
                return None
            start = max(0, file_size - n)
            end = file_size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return None
    return start, end


@app.get("/api/video")
def video_stream(id: str = Query(...), request: Request = None):
    """Stream a catalogued video with HTTP range support so the <video> element
    can seek (206 Partial Content). Path is resolved only via the catalog/vector
    metadata — the id is never used as a filesystem path (same arbitrary-read
    guard as /api/image)."""
    path = _resolve_indexed_path(id)
    if not path:
        raise HTTPException(404, "not an indexed video, or file missing on disk")
    file_size = os.path.getsize(path)
    media_type = _guess_media_type(path)
    range_header = request.headers.get("range") if request is not None else None
    rng = _parse_range(range_header, file_size) if range_header else None

    if rng is None:
        # No/!unsatisfiable range → whole file, but advertise range support so
        # the browser knows it can seek on the next request.
        return FileResponse(
            path, media_type=media_type,
            headers={**_IMMUTABLE_CACHE, "Accept-Ranges": "bytes"},
        )

    start, end = rng
    end = min(end, start + _VIDEO_MAX_SLICE - 1)  # bound this slice
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)

    headers = {
        **_IMMUTABLE_CACHE,
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(len(data)),
    }
    return Response(content=data, status_code=206, media_type=media_type,
                    headers=headers)


@app.get("/api/similar")
def similar(id: str = Query(...), top_k: int = Query(24, ge=1, le=_MAX_RESULT_LIMIT)):
    """Photos most similar to the given one, by its stored caption embedding
    in the active collection ('More like this')."""
    active = get_active_model()
    if not active:
        return {"results": []}
    col = db.collection(active)
    got = col.get(ids=[id], include=["embeddings"])
    if not len(got["ids"]):
        raise HTTPException(404, "photo is not in the active search index yet")
    emb = got["embeddings"][0]
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    n = min(max(1, top_k) + 1, col.count())
    res = col.query(query_embeddings=[emb], n_results=n)
    out = [
        _card(i, m)
        for i, m in zip(res["ids"][0], res["metadatas"][0])
        if i != id
    ]
    return {"results": out[:top_k]}


@app.get("/api/meta")
def meta(id: str = Query(...)):
    m = _chroma_meta(id)
    cat = load_catalog_cached().get("images", {}).get(id)
    if not m and not cat:
        raise HTTPException(404, "no metadata (not indexed)")
    # Merge in catalog media info so a video that hasn't been captioned/embedded
    # yet still reports its type, duration and dimensions (P0 browse surface).
    m = dict(m or {})
    if cat:
        m.setdefault("path", cat.get("path", ""))
        m.setdefault("filename", cat.get("filename", ""))
        m["media_type"] = m.get("media_type") or cat.get("media_type", "image")
        for k in ("duration_s", "width", "height", "codec"):
            if cat.get(k) is not None:
                m.setdefault(k, cat.get(k))
    return m


def _delete_file_recoverably(path: str) -> bool:
    """Windows Recycle Bin first, plain delete elsewhere. True when gone."""
    if trash_mod.delete_file_to_recycle_bin(path):
        return True
    try:
        os.remove(path)
        return True
    except Exception as e:
        print(f"[api] file delete failed for {path}: {e}")
        return False


@app.delete("/api/image")
def delete_image(id: str = Query(...), delete_file: bool = False):
    """Soft-delete: the photo moves to the app trash (restorable). With
    delete_file the file itself goes to the OS Recycle Bin."""
    _reject_if_writer_active()
    idx = Indexer()
    path = idx.image_catalog.get("images", {}).get(id, {}).get("path")
    removed_file = False
    if delete_file and path and os.path.exists(path):
        removed_file = _delete_file_recoverably(path)
        if not removed_file:
            raise HTTPException(500, "file delete failed; photo left in the index")
    idx.delete_image(id, to_trash=True, file_deleted=removed_file)
    return {"ok": True, "removed_file": removed_file}


@app.post("/api/images/delete")
def batch_delete(req: BatchDeleteReq):
    """Soft-delete multiple images (files optionally to the Recycle Bin).

    Unlike the single-image DELETE (which aborts with a 500 when the on-disk
    delete fails), a batch keeps soft-deleting the catalog entry even when the
    file delete failed for one item — aborting the whole batch over one bad
    file would be worse. `files_failed` surfaces those ids prominently at the
    top level so a failed disk-delete is never buried in per-item results.
    """
    if not req.ids:
        return {"removed": 0, "files_removed": 0, "files_failed": []}
    with _writer_guard():
        idx = Indexer()
        removed = files_removed = 0
        files_failed = []
        for img_id in req.ids:
            path = idx.image_catalog.get("images", {}).get(img_id, {}).get("path")
            file_deleted = False
            if req.delete_file and path and os.path.exists(path):
                file_deleted = _delete_file_recoverably(path)
                if file_deleted:
                    files_removed += 1
                else:
                    files_failed.append(img_id)
            idx.delete_image(img_id, to_trash=True, file_deleted=file_deleted)
            removed += 1
    return {
        "removed": removed,
        "files_removed": files_removed,
        "files_failed": files_failed,
    }


# ── trash ─────────────────────────────────────────────────────────────────────
class TrashIdsReq(BaseModel):
    ids: list[str] = []  # empty → all


@app.get("/api/trash")
def trash_list():
    items = trash_mod.list_items()
    out = []
    for iid, item in items.items():
        e = item.get("entry", {})
        out.append(
            {
                "id": iid,
                "filename": e.get("filename", ""),
                "path": e.get("path", ""),
                "deleted_at": item.get("deleted_at"),
                "file_deleted": item.get("file_deleted", False),
            }
        )
    out.sort(key=lambda x: x.get("deleted_at") or 0, reverse=True)
    return {"items": out, "total": len(out)}


@app.post("/api/trash/restore")
def trash_restore(req: TrashIdsReq):
    """Restore trashed photos to the catalog. Their caption survives, so they
    reappear as embed-pending (run C to make them searchable again)."""
    _reject_if_writer_active()
    ids = req.ids or list(trash_mod.list_items().keys())
    idx = Indexer()
    return {"restored": idx.restore_images(ids)}


@app.post("/api/trash/purge")
def trash_purge(req: TrashIdsReq = None):
    """Permanently drop trashed entries (all when ids empty).

    POST (not DELETE) because this takes a scoped id list in the body — some
    HTTP clients/proxies strip bodies from DELETE requests, which would
    silently turn a scoped purge into an all-trash wipe.
    """
    with _writer_guard():
        idx = Indexer(use_cache=True)  # purge doesn't touch the live catalog
        purged = idx.purge_trash(req.ids if req and req.ids else None)
    return {"purged": purged}


# ── duplicates ────────────────────────────────────────────────────────────────
@app.get("/api/duplicates")
def duplicates(threshold: int = Query(dupes_mod.DEFAULT_THRESHOLD, ge=0, le=16),
               limit: int = Query(100, ge=1, le=_MAX_RESULT_LIMIT)):
    """Near-duplicate groups from the stored perceptual hashes (run the
    'dhash' job first). Photos in each group are largest-file-first."""
    catalog = load_catalog_cached().get("images", {})
    groups = dupes_mod.group_duplicates(catalog, threshold=threshold)
    out = []
    for g in groups[:limit]:
        photos = []
        for iid in g:
            d = catalog.get(iid, {})
            photos.append(
                {
                    "id": iid,
                    "filename": d.get("filename", ""),
                    "path": d.get("path", ""),
                    "size_bytes": d.get("size_bytes", 0),
                    "exists": os.path.exists(d.get("path", "")),
                }
            )
        photos.sort(key=lambda p: -(p["size_bytes"] or 0))
        out.append({"photos": photos, "count": len(photos)})
    hashed = sum(1 for d in catalog.values() if d.get("dhash"))
    return {
        "groups": out,
        "total_groups": len(groups),
        "hashed": hashed,
        "total": len(catalog),
    }


@app.get("/api/explore")
def explore(id: str = Query(...)):
    """Full details for one photo: caption history, embedding model memberships."""
    catalog = load_catalog_cached().get("images", {})
    img_data = catalog.get(id)
    if not img_data:
        raise HTTPException(404, "photo not in catalog")

    history = img_data.get("caption_history", [])
    if not history and img_data.get("caption_json"):
        history = [
            {
                "model": img_data.get("caption_model", "unknown"),
                "caption_json": img_data["caption_json"],
            }
        ]
    history_out = []
    for h in history:
        cj = h.get("caption_json", "")
        validation = (
            validate_vision_output(cj)
            if cj
            else {"valid": False, "warning": "No output"}
        )
        history_out.append({**h, "validation": validation})

    reg = get_registry()
    client = db.client()
    embed_info = []
    for model_name, info in reg.get("models", {}).items():
        try:
            col = client.get_or_create_collection(name=collection_name_for(model_name))
            res = col.get(ids=[id], include=["metadatas"])
            if res["ids"]:
                embed_info.append(
                    {
                        "model": model_name,
                        "source": info.get("source"),
                        "dimension": info.get("dimension"),
                        "is_active": model_name == reg.get("active_model"),
                    }
                )
        except Exception:
            pass

    return {
        "id": id,
        "path": img_data.get("path", ""),
        "filename": img_data.get("filename", ""),
        "exists": os.path.exists(img_data.get("path", "")),
        "caption_history": history_out,
        "embeddings": embed_info,
        "exif": img_data.get("metadata", {}),
        "size_bytes": img_data.get("size_bytes"),
    }


@app.post("/api/cleanup-missing")
def cleanup_missing():
    """Legacy endpoint: remove all orphaned entries at once."""
    _reject_if_writer_active()
    idx = Indexer()
    missing = idx.get_missing_files()
    for img_id, _ in missing:
        idx.delete_image(img_id)
    return {"removed": len(missing)}


# ── auth bootstrap ────────────────────────────────────────────────────────────
@app.get("/api/token")
def token(request: Request):
    """
    Hand the bearer token to the same-origin SPA. Exempt from the token check so
    the dev SPA (served by Vite, with no HTML injection) can bootstrap.

    That exemption is exactly what makes this endpoint dangerous once the app
    is reachable beyond loopback (PV_ALLOWED_HOSTS): CORS and the trusted-host
    check both key off the Host *header*, which any direct (non-browser)
    client can simply set to an allowed value — they don't stop a raw `curl`
    from a remote box on the tailnet/LAN from hitting this route and reading
    the token, defeating auth entirely. So when auth is required, only a
    client whose actual connection originates from loopback gets the real
    token; everyone else is refused outright rather than handed a token over
    the same unauthenticated channel it's meant to protect. A remote client
    must be provisioned the token some other way (copied in manually, or
    baked in at build time).
    """
    if not security.auth_enabled():
        return {"token": None}
    client_host = request.client.host if request.client else None
    if not security.is_loopback_client(client_host):
        raise HTTPException(
            403, "token bootstrap is only available to loopback clients"
        )
    return {"token": security.get_token()}


# ── static SPA (production build) ─────────────────────────────────────────────
_DIST = os.path.join(PROJECT_ROOT, "web", "dist")


def _index_html_with_token(request: Request) -> str:
    """Read the built index.html and inject the per-install token so the SPA can
    authenticate same-origin without an extra round-trip. Same loopback
    restriction as GET /api/token, and for the same reason: a non-loopback
    request gets the page without the token embedded (the SPA then simply
    can't bootstrap over /api/token either — see above)."""
    with open(os.path.join(_DIST, "index.html"), encoding="utf-8") as f:
        html = f.read()
    if security.auth_enabled():
        client_host = request.client.host if request.client else None
        if security.is_loopback_client(client_host):
            inject = f"<script>window.__PV_TOKEN__={security.get_token()!r};</script>"
            html = html.replace("</head>", inject + "</head>", 1)
    return html


if os.path.isdir(_DIST):

    @app.get("/", response_class=HTMLResponse)
    def _spa_index(request: Request):
        return HTMLResponse(_index_html_with_token(request))

    @app.get("/index.html", response_class=HTMLResponse)
    def _spa_index_html(request: Request):
        return HTMLResponse(_index_html_with_token(request))

    # All other static assets (hashed JS/CSS) served verbatim.
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")
else:

    @app.get("/")
    def _no_build():
        return JSONResponse(
            {
                "message": "SPA not built yet. Run `cd web && npm install && npm run build`, "
                "or use the Vite dev server (`npm run dev`).",
                "api_port": SERVER_PORT,
            },
        )
