# Photo Vault

Local Google Photos — index your photos, search by face, occasion, weather, year, and more.
Everything runs on your machine. No cloud subscription. No data leaves your device.

## What it does

- **Search**: "birthday party 2023", "mom in red saree", "beach sunset" → instant results
- **Timeline**: Browse all photos organized by year
- **People**: Register faces, search all photos of a specific person
- **Smart attributes**: Every photo gets tagged with scene, weather, occasion, group size,
  clothing style, mood, location type — all searchable via dropdowns
- **Multi-model embeddings**: Index with LM Studio's embed model or Gemini — both stored,
  switch the active search model anytime from the UI without re-indexing

## Before you start — choose your services

### Vision analysis (what's in each photo)

**Option A — LM Studio (fully local, no internet needed):**
1. Download [LM Studio](https://lmstudio.ai/) and install a **multimodal / vision** model
   (Qwen2-VL, LLaVA, Moondream, Gemma-3, or similar)
2. Go to "Local Server" tab in LM Studio → load the model → Start Server
3. Server must be running at `http://localhost:1234` when you index photos

**Option B — Gemini free tier (works even without LM Studio):**
1. Get a free API key at https://aistudio.google.com/
2. Create `photo-vault/.env` with: `GEMINI_API_KEY=your_key_here`
3. Gemini activates automatically whenever LM Studio is offline

You can have **both** — LM Studio is used first, Gemini is the fallback.

### Embeddings (what powers the search)

**LM Studio** (recommended): Load any **text embedding** model in LM Studio
(e.g. `text-embedding-nomic-embed-text-v1.5`, `all-minilm-l6-v2`, or any model tagged "embedding").
The app auto-detects which model is loaded.

**Gemini** (fallback): Same `GEMINI_API_KEY` used above — `text-embedding-004` is called automatically
when LM Studio is offline.

> **Important**: Each embedding model gets its own search index. If you switch models mid-library,
> existing photos stay findable under the old model. Use the **Active model** selector in
> Index & Manage to choose which index Search uses.

### Face recognition

InsightFace runs on CPU automatically. No setup needed. CUDA is used if available.

## Architecture

FastAPI backend (Python) + Svelte single-page app (Vite). The backend modules do all the work —
indexing, vision, embeddings, faces, search — and expose a JSON API. The SPA is the only UI.
Indexing runs as a **background job**: progress streams to the UI, a Stop button works mid-run,
and the page never freezes. Everything is local; your photos and index never leave your machine.

## Requirements

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- Node 18+ and npm (to build the SPA)

## Setup

```powershell
# Windows (PowerShell)
cd photo-vault
copy .env.example .env          # then edit .env — add GEMINI_API_KEY if using Gemini

uv sync --extra dev             # backend deps + pytest
cd web; npm install; npm run build; cd ..   # build the SPA
uv run python src/serve.py      # → http://127.0.0.1:8768

# or just double-click: start.bat   (builds the SPA on first run)
```

```bash
# macOS / Linux
cd photo-vault
cp .env.example .env
make install && make run        # builds SPA + serves → http://127.0.0.1:8768
```

**Frontend hot-reload during development:**
```bash
make serve     # terminal 1 — backend on :8768
make web       # terminal 2 — Vite dev server on :5173 (proxies /api → :8768)
```

## First use — step by step

1. **Start your services** (before opening the app):
   - LM Studio → Local Server → load a vision model + an embedding model → Start Server
   - Or ensure `GEMINI_API_KEY` is in `.env` for Gemini-only mode

2. **Open** http://127.0.0.1:8768

3. Go to **Index & Manage** tab → **Services** — confirm LM Studio and/or Gemini show online

4. **A — Scan**: enter your Photos folder path (e.g. `C:\Users\you\Pictures`) → **Scan folders**

5. **B — Vision** then **C — Embed** (or **D — Full index** to do both). Watch the live progress
   bar; **Stop** anytime — it halts between photos. Each photo logs its provider (✅ local / ☁️ Gemini)

6. After indexing: use **Search**, browse **Timeline**, register people in **People**

## Fallback table

| LM Studio | GEMINI_API_KEY | Vision | Embeddings |
|-----------|----------------|--------|------------|
| ✓ running | any            | LM Studio | LM Studio embed model |
| ✗ offline | set            | Gemini (free) | Gemini text-embedding-004 |
| ✗ offline | not set        | ✗ fails | ✗ fails |

Both LM Studio and Gemini can be active simultaneously — LM Studio is always tried first.

## Running tests

```bash
make test                            # or: uv run python -m pytest tests/ -q
```

137 tests covering: embeddings, vision, search, indexer, validator, tagger, the background
job manager, and the FastAPI endpoints. All mocked — no services needed to run tests.

## File structure

```
src/
  api.py          ← FastAPI: JSON endpoints + serves the built SPA
  serve.py        ← launcher (reads port from ../ports.json → 8768)
  jobs.py         ← background indexing job manager (worker thread, stop, fail-abort)
  vision.py       ← LM Studio vision → 12 attributes JSON (Gemini fallback); records model used
  embeddings.py   ← LM Studio embed model (Gemini fallback); multi-model registry
  indexer.py      ← scan + index + gap detection + caption history; per-model ChromaDB collections
  search.py       ← semantic search + attribute filters + person filter
  faces.py        ← InsightFace face detection/embedding (CPU + CUDA auto)
  tagger.py       ← person registration from reference images
  clustering.py   ← DBSCAN face clustering
  scanner.py / metadata.py ← recursive image discovery + EXIF
  validator.py    ← LM Studio / Gemini health checks
  constants.py    ← paths + endpoints + SERVER_PORT + .env loader
web/              ← Svelte SPA (Vite). src/lib: api.js, PhotoGrid, Lightbox, *Tab.svelte
                     dist/ is the built UI served by FastAPI (gitignored)
tests/            ← pytest suite (137 tests, no external services needed)
data/             ← gitignored: images.json, chroma_db/, faces/, thumbs/, person_map.json,
                     embedding_registry.json
```

## Attributes extracted per photo

`caption`, `scene`, `location_type`, `weather`, `season`, `time_of_day`,
`occasion`, `group_size`, `clothing_style`, `mood`, `objects`, `people_description`

All stored in ChromaDB — searchable via dropdowns in the Search tab.

## License

MIT — Hari Yelesetty
