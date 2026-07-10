# Claude Instructions — photo-vault

## Role
Maintain a Google Photos-like local app: index personal photos, search by face/occasion/weather/year,
browse by timeline and people. UI is a Svelte SPA talking to a FastAPI backend over JSON — no CLI scripts.

## Context
All local, no cloud, no paid APIs. Primary: LM Studio (vision + embeddings) + InsightFace (faces).
Fallback: Gemini free tier for both vision and embeddings when LM Studio is offline.
Opt-in third provider: **9Router** (local OpenAI-compatible gateway, `constants.NINEROUTER_URL`
from `../ports.json → registry.9router.port`) — pools multiple Gemini CLI accounts/API keys and
rotates them on 429 internally. Never part of the "auto" chain; always needs an explicit model id
(no LM-Studio-style auto-detect). The gateway may substitute the serving model: vision keeps the
caption but labels it with the model that ACTUALLY produced it (so 9Router vision runs target
any-caption-missing photos, and the job panel tallies "Models used"); embeddings REJECT
substitution (one vector space per collection). See `FEATURE-REQUEST-9router-integration.md` +
`../9ROUTER.md`.
GEMINI_API_KEY loaded from `.env` via `constants._load_env()` — never hardcoded.
Multi-model embeddings: each embedding model gets its own ChromaDB collection; active model
selected by user in UI; registry stored in `data/embedding_registry.json`.
Port read from `../ports.json` (photo-vault → 8768) via `constants.SERVER_PORT`; never hardcode.

## Architecture
Backend (Python, UI-agnostic) + frontend (Svelte SPA). The backend modules are the durable core;
the SPA is the only UI. (The old Streamlit `app.py` was removed in the FastAPI migration.)
```
src/
  api.py          ← FastAPI: JSON endpoints + serves web/dist SPA same-origin
  serve.py        ← launcher; reads SERVER_PORT, runs uvicorn
  jobs.py         ← background indexing job manager (worker thread, RLock, stop flag,
                    consecutive-failure abort); API polls status(), calls stop()
  vision.py       ← LM Studio first → Gemini fallback; get_image_caption(with_model=True)
                    returns (text, model_label) so the vision model is recorded per caption
  embeddings.py   ← LM Studio /v1/embeddings first → Gemini text-embedding-004 fallback;
                    multi-model registry; one ChromaDB collection per model
  indexer.py      ← scan + vision + embed + ChromaDB; caption_json + caption_model +
                    caption_history per image; single-item ops (vision_one/embed_one/
                    index_one_full) for the job manager; get_missing_files() for health check
  search.py       ← semantic search + attribute filters + person face filter;
                    get_available_filter_values() drives data-driven UI filters
  faces.py        ← InsightFace face detection/embedding (CPU + CUDA auto-fallback)
  tagger.py       ← register person from reference images → person_map.json
  clustering.py   ← DBSCAN face clustering
  scanner.py / metadata.py ← recursive image discovery + EXIF
  ingest.py       ← import & consolidate: staging folder → library (content-hash
                    dedupe incl. videos via media_hashes.json, YYYY/MM layout)
  backup.py       ← opportunistic robocopy /MIR of scan folders + data/ to the
                    SD card (backup_dest setting); status() = drive + staleness
  validator.py    ← LM Studio / Gemini health checks
  ratelimit.py    ← per-provider RPS/RPM/RPH/RPD throttle (user-set in Settings →
                    "rate_limits", 0 = unlimited); acquire() before every provider
                    inference call; sliding in-memory windows; Stop-aware (Cancelled)
  constants.py    ← paths + endpoints + GEMINI_VISION_MODELS + SERVER_PORT + _load_env()
web/
  src/App.svelte  ← tabs: Search, Timeline, People, Index & Manage
  src/lib/        ← api.js (fetch wrapper), PhotoGrid, Lightbox, *Tab.svelte
  dist/           ← built SPA (gitignored), served by FastAPI
data/             ← gitignored: images.json, chroma_db/, faces/, thumbs/, person_map.json
tests/            ← pytest; all external calls mocked; test_jobs + test_api included
```

## API endpoints (all under /api)
health, status, scan, index/start|stop|progress|reset, search (GET+POST), recent, filters,
timeline, people (GET+POST), models (+models/active), image (GET thumb/full, DELETE),
meta, cleanup-missing. Indexing runs as a background job — never blocks a request.

## Deps (with-deps project)
```bash
make install      # uv sync --extra dev  +  cd web && npm install
make run          # build SPA + serve → http://127.0.0.1:8768
make serve        # backend only (no rebuild)
make web          # Vite dev server with hot reload (proxies /api → 8768)
make test         # uv run python -m pytest tests/ -q
```

**Prerequisites (optional — fallback via Gemini):**
- LM Studio at localhost:1234 with a vision model loaded (for image analysis)
- LM Studio with an embedding model loaded (for text embeddings) — same server, different model
- `GEMINI_API_KEY` in `.env` for fallback (free key at https://aistudio.google.com/)
- LM Studio embedding model auto-detected via `/v1/models` endpoint

## Gemini fallback
- Vision: tries `GEMINI_VISION_MODELS` in order (lite → heavier); skips 429/404/503
- Embeddings: `text-embedding-004` (single model, no fallback chain within Gemini)
- Both fall back on connection errors AND non-connection errors — always tries next provider
- `embedding_source` ("lm_studio" or "gemini") + `embedding_model` stored in ChromaDB per image
- Different models produce different vector spaces — each gets its own collection, no mixing

## Attributes extracted per photo (single LM Studio/Gemini call at index time)
caption, scene, location_type, weather, season, time_of_day,
occasion, group_size, clothing_style, mood, objects, people_description

## Rules
1. No paid APIs. All AI runs locally or via Gemini free tier.
2. All user interaction through the Svelte SPA via the FastAPI JSON API. No standalone CLI scripts.
3. constants.py is the single source of truth for all paths/port/API config — never hardcode.
4. data/ is gitignored. Never commit it. web/dist and node_modules also gitignored.
5. No auto-commit.
6. GPU optional — onnxruntime falls back to CPU automatically.
7. Indexing runs in jobs.py worker thread; the API never blocks. Stop is honored between photos.
   jobs._lock is an RLock (start() calls status() while holding it).
8. Backend functions return failed-id lists / raise per-image; callers surface, never crash a pass.
9. After editing web/src, rebuild (`make build`) or use `make web` — FastAPI serves web/dist.
