"""Photo Vault HTTP API (FastAPI).

Thin JSON layer over the existing, UI-agnostic backend (indexer / search /
embeddings / vision / faces / tagger). Serves the built Svelte SPA from web/dist
in production. Run:  uv run uvicorn api:app --app-dir src --port <port>
"""
import hashlib
import io
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from constants import (
    CHROMA_DB_PATH, THUMB_DIR, PROJECT_ROOT, SERVER_PORT,
)
from indexer import Indexer
from search import search_images, get_available_filter_values
from embeddings import (
    get_registry, get_active_model, set_active_model, collection_name_for,
)
from tagger import add_person_reference, get_all_persons
from validator import service_status
from jobs import manager, JOB_TYPES
import chromadb

os.makedirs(THUMB_DIR, exist_ok=True)

app = FastAPI(title="Photo Vault", version="1.0")

# Dev convenience: Vite dev server runs on a different port. In production the
# SPA is served same-origin from web/dist so CORS is a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── models ──────────────────────────────────────────────────────────────────
class ScanReq(BaseModel):
    dirs: list[str]


class IndexReq(BaseModel):
    type: str
    force_provider: str = "auto"
    max_fail: int = 5


class PersonReq(BaseModel):
    name: str
    ref_dir: str


class ActiveModelReq(BaseModel):
    model: str


# ── helpers ───────────────────────────────────────────────────────────────────
def _thumb_path(img_id: str) -> str:
    h = hashlib.sha1(img_id.encode("utf-8")).hexdigest()
    return os.path.join(THUMB_DIR, f"{h}.jpg")


def _chroma_meta(img_id: str) -> dict:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        active = get_active_model()
        col = client.get_or_create_collection(collection_name_for(active) if active else "images")
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
    idx = Indexer()
    stage = idx.get_stage_stats()
    return {
        "stage": stage,
        "vision_pending": len(idx.get_vision_pending()),
        "embed_pending": len(idx.get_embed_pending()),
        "missing_attrs": len(idx.get_missing_attributes()),
        "missing_full": len(idx.get_missing()),
        "missing_files": len(idx.get_missing_files()),
    }


# ── scanning ────────────────────────────────────────────────────────────────
@app.post("/api/scan")
def scan(req: ScanReq):
    invalid = [d for d in req.dirs if not os.path.isdir(d)]
    if invalid:
        raise HTTPException(400, f"paths not found: {', '.join(invalid)}")
    if not req.dirs:
        raise HTTPException(400, "no directories given")
    Indexer(target_directories=req.dirs).scan_only()
    return status()


# ── indexing jobs ─────────────────────────────────────────────────────────────
@app.post("/api/index/start")
def index_start(req: IndexReq):
    if req.type not in JOB_TYPES:
        raise HTTPException(400, f"unknown job type: {req.type}")
    try:
        return manager.start(req.type, req.force_provider, req.max_fail)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


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
    q: str = Query("photo"),
    person: str | None = None,
    top_k: int = 200,
):
    res = search_images(q or "photo", top_k=top_k, person=person or None)
    metas = res.get("metadatas", [[]])[0] if res else []
    ids = res.get("ids", [[]])[0] if res else []
    return {"results": [_card(i, m) for i, m in zip(ids, metas)]}


@app.post("/api/search")
def search_post(body: dict):
    q = body.get("q") or "photo"
    person = body.get("person") or None
    filters_in = {k: v for k, v in (body.get("filters") or {}).items() if v and v != "All"}
    res = search_images(q, top_k=body.get("top_k", 200), filters=filters_in, person=person)
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


# ── recent / timeline ─────────────────────────────────────────────────────────
@app.get("/api/recent")
def recent(limit: int = 60):
    active = get_active_model()
    if not active:
        return {"results": []}
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_or_create_collection(collection_name_for(active))
    data = col.get(include=["metadatas"])
    items = list(zip(data["ids"], data["metadatas"]))
    return {"results": [_card(i, m) for i, m in reversed(items[-limit:])]}


@app.get("/api/timeline")
def timeline():
    idx = Indexer()
    by_year: dict[str, list] = {}
    for img_id, data in idx.image_catalog.get("images", {}).items():
        date = data.get("metadata", {}).get("date", "")
        year = date[:4] if date and len(date) >= 4 else "Unknown"
        by_year.setdefault(year, []).append({
            "id": img_id,
            "filename": data.get("filename", ""),
            "exists": os.path.exists(data.get("path", "")),
        })
    return {"years": [
        {"year": y, "count": len(by_year[y]),
         "photos": by_year[y]}
        for y in sorted(by_year, reverse=True)
    ]}


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
    add_person_reference(req.name, req.ref_dir)
    return {"ok": True, "name": req.name}


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
@app.get("/api/image")
def image(id: str = Query(...), thumb: bool = False):
    meta = _chroma_meta(id)
    path = meta.get("path") or id
    if not os.path.exists(path):
        raise HTTPException(404, "file not found on disk")
    if not thumb:
        return FileResponse(path)
    tp = _thumb_path(id)
    if not os.path.exists(tp):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((400, 400))
                im.save(tp, "JPEG", quality=80)
        except Exception:
            raise HTTPException(500, "thumbnail generation failed")
    return FileResponse(tp, media_type="image/jpeg")


@app.get("/api/meta")
def meta(id: str = Query(...)):
    m = _chroma_meta(id)
    if not m:
        raise HTTPException(404, "no metadata (not indexed)")
    return m


@app.delete("/api/image")
def delete_image(id: str = Query(...), delete_file: bool = False):
    idx = Indexer()
    path = idx.delete_image(id)
    removed_file = False
    if delete_file and path and os.path.exists(path):
        try:
            os.remove(path)
            removed_file = True
        except Exception as e:
            raise HTTPException(500, f"index removed, file delete failed: {e}")
    tp = _thumb_path(id)
    if os.path.exists(tp):
        try:
            os.remove(tp)
        except Exception:
            pass
    return {"ok": True, "removed_file": removed_file}


@app.post("/api/cleanup-missing")
def cleanup_missing():
    idx = Indexer()
    missing = idx.get_missing_files()
    for img_id, _ in missing:
        idx.delete_image(img_id)
    return {"removed": len(missing)}


# ── static SPA (production build) ─────────────────────────────────────────────
_DIST = os.path.join(PROJECT_ROOT, "web", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")
else:
    @app.get("/")
    def _no_build():
        return JSONResponse(
            {"message": "SPA not built yet. Run `cd web && npm install && npm run build`, "
                        "or use the Vite dev server (`npm run dev`).",
             "api_port": SERVER_PORT},
        )
