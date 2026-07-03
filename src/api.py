"""Photo Vault HTTP API (FastAPI).

Thin JSON layer over the existing, UI-agnostic backend (indexer / search /
embeddings / vision / faces / tagger). Serves the built Svelte SPA from web/dist
in production. Run:  uv run uvicorn api:app --app-dir src --port <port>
"""

import os
import threading
from datetime import datetime as _dt

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
from indexer import Indexer, catalog_path_for, load_catalog_cached

# id is a content hash, so a given id's bytes never change → cache forever.
_IMMUTABLE_CACHE = {"Cache-Control": "private, max-age=31536000, immutable"}
from search import search_images, get_available_filter_values
from vision import (
    list_lm_studio_models,
    classify_lm_studio_model,
    list_gemini_vision_models,
    validate_vision_output,
)
from embeddings import (
    get_registry,
    get_active_model,
    set_active_model,
    collection_name_for,
    list_gemini_embed_models,
)
from tagger import (
    add_person_reference,
    add_person_embedding,
    get_all_persons,
    rename_person,
    delete_person,
)
import dupes as dupes_mod
import trash as trash_mod
from faces import load_face_data, face_index_count, rebuild_face_index
from validator import service_status
from jobs import manager, JOB_TYPES
import clustering
import albums as albums_mgr
import folders as folder_mgr
import settings as settings_mgr

os.makedirs(THUMB_DIR, exist_ok=True)

app = FastAPI(title="Photo Vault", version="1.0")

# Mutual exclusion between a (synchronous) scan and the background index job:
# both rewrite the whole images.json, so they must never run concurrently.
_scan_active = threading.Event()

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


@app.middleware("http")
async def _require_token(request: Request, call_next):
    """Enforce the bearer token on /api/* when PV_REQUIRE_AUTH=1 (set by serve.py)."""
    if security.auth_enabled():
        path = request.url.path
        if path.startswith("/api/") and path not in security.EXEMPT_API_PATHS:
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


class IndexReq(BaseModel):
    type: str
    vision_provider: str = "auto"
    vision_model: str | None = None
    embed_provider: str = "auto"
    embed_model: str | None = None
    caption_source_model: str | None = None
    max_fail: int = 5


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
    faces_during_embed: bool | None = None
    face_cluster_eps: float | None = None
    face_cluster_min_samples: int | None = None


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
    """Deletes/purges rewrite images.json from a private Indexer copy. If a scan
    or index job is mid-flight (each holds its own copy), whichever saves last
    silently resurrects what the other removed — so refuse instead."""
    if _scan_active.is_set() or manager.status().get("active"):
        raise HTTPException(
            409, "a scan or indexing job is running; stop it before deleting"
        )


def _chroma_meta(img_id: str) -> dict:
    try:
        col = db.collection()
        res = col.get(ids=[img_id], include=["metadatas"])
        if res["ids"]:
            return res["metadatas"][0]
    except Exception:
        pass
    return {}


# ── status / health ───────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return service_status()


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

    # Legacy pending counts (used by backward-compat code paths)
    legacy_vision_pending = len(idx.get_vision_pending())
    legacy_embed_pending = len(idx.get_embed_pending())

    # Model-specific vision counts
    if vm_label:
        model_vision_pending_list = idx.get_vision_pending_for_model(vm_label)
        vision_done = total - len(model_vision_pending_list)
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

    return {
        # Legacy fields kept for backward compat
        "stage": stage,
        "vision_pending": vision_pending,
        "embed_pending": embed_pending if csm or em else legacy_embed_pending,
        "missing_attrs": len(idx.get_missing_attributes()),
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
        },
        "faces_pending": faces["pending"],
        "faces_done": faces["detected"],
        "thumbs_pending": idx.count_thumbs_missing(),
        "dhash_pending": sum(
            1 for d in idx.image_catalog.get("images", {}).values()
            if not d.get("dhash")
        ),
        "trash_count": len(trash_mod.list_items()),
        "settings": s,
    }


@app.get("/api/settings")
def get_settings():
    return settings_mgr.load()


@app.put("/api/settings")
def put_settings(req: SettingsReq):
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    return settings_mgr.update(patch)


@app.delete("/api/settings")
def reset_settings():
    """Reset to factory defaults."""
    settings_mgr.save(settings_mgr.DEFAULTS)
    return settings_mgr.load()


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
    if _scan_active.is_set():
        raise HTTPException(409, "a scan is already in progress")

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

    _scan_active.set()
    try:
        summary = idx.scan_only()
        st = _status_dict(idx)
    finally:
        _scan_active.clear()
    return {"summary": summary, **st}


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
    result = folder_mgr.add_included(path)
    if result["status"] == "not_found":
        raise HTTPException(400, f"directory not found: {path}")
    return result


@app.delete("/api/folders/include")
def remove_folder(path: str = Query(...), purge: bool = True):
    """
    Remove a folder from the included list.
    When purge=true (default), also deletes all indexed data for images under that path.
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


@app.delete("/api/orphaned")
def cleanup_orphaned(req: OrphanedCleanupReq = None):
    """
    Remove orphaned images from the library. If req.ids is empty, removes all orphaned.
    Returns {removed: count}.
    """
    _reject_if_writer_active()
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
    # Compute the vision model label used in caption_history (for model-aware pending queries)
    vml = None
    if req.vision_provider not in ("auto", None) and req.vision_model:
        vml = f"{req.vision_provider}:{req.vision_model}"
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
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/provider-models")
def provider_models():
    """Models available per provider for the run-config dropdowns.
    Gemini lists only include verified models — no hardcoded fallback."""
    lm_models = list_lm_studio_models()
    return {
        "lm_studio": lm_models,
        "lm_studio_types": {m: classify_lm_studio_model(m) for m in lm_models},
        "gemini_vision": list_gemini_vision_models(fallback=False),
        "gemini_embed": list_gemini_embed_models(fallback=False),
    }


@app.post("/api/index/stop")
def index_stop():
    manager.stop()
    return manager.status()


@app.get("/api/index/progress")
def index_progress():
    return manager.status()


@app.post("/api/index/reset")
def index_reset():
    manager.reset()
    return manager.status()


# ── search / filters ────────────────────────────────────────────────────────
@app.get("/api/filters")
def filters():
    return get_available_filter_values()


@app.get("/api/search")
def search(
    q: str = Query(""),
    person: str | None = None,
    top_k: int = 200,
):
    # Pass q through as-is: an empty q with a person set is the person-only
    # browse path (all their photos), which coercing q to "photo" would disable.
    res = search_images(q, top_k=top_k, person=person or None)
    metas = res.get("metadatas", [[]])[0] if res else []
    ids = res.get("ids", [[]])[0] if res else []
    return {"results": [_card(i, m) for i, m in zip(ids, metas)]}


@app.post("/api/search")
def search_post(body: dict):
    q = body.get("q") or ""
    person = body.get("person") or None
    filters_in = {
        k: v for k, v in (body.get("filters") or {}).items() if v and v != "All"
    }
    res = search_images(
        q, top_k=body.get("top_k", 200), filters=filters_in, person=person
    )
    metas = res.get("metadatas", [[]])[0] if res else []
    ids = res.get("ids", [[]])[0] if res else []
    return {"results": [_card(i, m) for i, m in zip(ids, metas)]}


def _card(img_id: str, meta: dict) -> dict:
    return {
        "id": img_id,
        "filename": os.path.basename(meta.get("path", img_id)),
        "caption": meta.get("caption", ""),
        "year": meta.get("year", ""),
        "occasion": meta.get("occasion", ""),
        "exists": os.path.exists(meta.get("path", "")),
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
                "exists": os.path.exists(path),
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
def recent(limit: int = 60):
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


@app.get("/api/timeline")
def timeline(year: str | None = None, offset: int = 0, limit: int = 60):
    """
    Paged timeline. Without `year`: every year with its count and the first
    `limit` photos (newest first). With `year`: one page of that year's photos.
    Returning the whole catalog in one response was 2.4 MB / 4.4 s at 25k
    photos (an os.path.exists per photo) — pages keep both bounded.
    """
    catalog = load_catalog_cached().get("images", {})
    by_year: dict[str, list] = {}
    for img_id, data in list(catalog.items()):
        date = data.get("metadata", {}).get("date", "")
        if not date or len(date) < 4:
            # No EXIF date (WhatsApp strips it, screenshots never had it) —
            # fall back to the file timestamp so most of the library lands in
            # a real year instead of one giant "Unknown" bucket.
            ts = data.get("created_at")
            if ts:
                try:
                    date = _dt.fromtimestamp(ts).strftime("%Y:%m:%d %H:%M:%S")
                except (OSError, OverflowError, ValueError):
                    date = ""
        y = date[:4] if date and len(date) >= 4 else "Unknown"
        by_year.setdefault(y, []).append((date, img_id, data))

    def _tcard(img_id, data, date):
        return {
            "id": img_id,
            "filename": data.get("filename", ""),
            "date": date,  # "YYYY:MM:DD hh:mm:ss" — the UI groups by month
            "exists": os.path.exists(data.get("path", "")),
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
    return {"people": get_all_persons()}


@app.post("/api/people")
def add_person(req: PersonReq):
    if not req.name:
        raise HTTPException(400, "name required")
    if not os.path.isdir(req.ref_dir):
        raise HTTPException(400, "reference folder not found")
    faces_found = add_person_reference(req.name, req.ref_dir)
    if not faces_found:
        raise HTTPException(
            400, "no faces detected in the reference folder — person not registered"
        )
    return {"ok": True, "name": req.name, "faces": faces_found}


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
def faces_clusters(samples: int = 6):
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
    emb = clustering.cluster_mean_embedding(req.cluster_id)
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
    return {"active": reg.get("active_model"), "models": reg.get("models", {})}


@app.post("/api/models/active")
def set_model(req: ActiveModelReq):
    set_active_model(req.model)
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
        return FileResponse(path, headers=_IMMUTABLE_CACHE)

    if size == "medium":
        resp = _serve_derivative(id, "_m", _MEDIUM_PX, path)
        return resp or FileResponse(path, headers=_IMMUTABLE_CACHE)  # original fallback

    # default: thumb
    resp = _serve_derivative(id, "", _THUMB_PX, path)
    return resp or _placeholder_thumb()


@app.get("/api/similar")
def similar(id: str = Query(...), top_k: int = 24):
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
    if not m:
        raise HTTPException(404, "no metadata (not indexed)")
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
    """Soft-delete multiple images (files optionally to the Recycle Bin)."""
    if not req.ids:
        return {"removed": 0, "files_removed": 0}
    _reject_if_writer_active()
    idx = Indexer()
    removed = files_removed = 0
    for img_id in req.ids:
        path = idx.image_catalog.get("images", {}).get(img_id, {}).get("path")
        file_deleted = False
        if req.delete_file and path and os.path.exists(path):
            file_deleted = _delete_file_recoverably(path)
            if file_deleted:
                files_removed += 1
        idx.delete_image(img_id, to_trash=True, file_deleted=file_deleted)
        removed += 1
    return {"removed": removed, "files_removed": files_removed}


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


@app.delete("/api/trash")
def trash_purge(req: TrashIdsReq = None):
    """Permanently drop trashed entries (all when ids empty)."""
    _reject_if_writer_active()
    idx = Indexer(use_cache=True)  # purge doesn't touch the live catalog
    return {"purged": idx.purge_trash(req.ids if req and req.ids else None)}


# ── duplicates ────────────────────────────────────────────────────────────────
@app.get("/api/duplicates")
def duplicates(threshold: int = Query(dupes_mod.DEFAULT_THRESHOLD, ge=0, le=16),
               limit: int = 100):
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
def token():
    """
    Hand the bearer token to the same-origin SPA. Exempt from the token check so
    the dev SPA (served by Vite, with no HTML injection) can bootstrap. Reading
    this cross-origin is blocked by the CORS allowlist, and DNS-rebinding by the
    trusted-host check, so only the loopback SPA can obtain it.
    """
    return {"token": security.get_token() if security.auth_enabled() else None}


# ── static SPA (production build) ─────────────────────────────────────────────
_DIST = os.path.join(PROJECT_ROOT, "web", "dist")


def _index_html_with_token() -> str:
    """Read the built index.html and inject the per-install token so the SPA can
    authenticate same-origin without an extra round-trip."""
    with open(os.path.join(_DIST, "index.html"), encoding="utf-8") as f:
        html = f.read()
    if security.auth_enabled():
        inject = f"<script>window.__PV_TOKEN__={security.get_token()!r};</script>"
        html = html.replace("</head>", inject + "</head>", 1)
    return html


if os.path.isdir(_DIST):

    @app.get("/", response_class=HTMLResponse)
    def _spa_index():
        return HTMLResponse(_index_html_with_token())

    @app.get("/index.html", response_class=HTMLResponse)
    def _spa_index_html():
        return HTMLResponse(_index_html_with_token())

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
