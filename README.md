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
- **Videos too**: Videos are first-class — browsed in the timeline with a ▶/duration badge,
  played inline (seekable), and made searchable by captioning a few sampled keyframes.
  Photos and videos can live in one folder or separate folders — your choice, never forced.

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

InsightFace runs on the **CPU by default** — no setup needed, works on every machine. For big
libraries you can offload it to a GPU/NPU; see **GPU/NPU acceleration** below. Either way,
**Index & Manage → Face detection → "Face detection runs on"** shows exactly which accelerators
your install exposes and lets you pick one (the list is auto-detected, never hardcoded).

## GPU/NPU acceleration (optional)

Face detection is the one CPU-heavy stage. It runs on CPU out of the box; to accelerate it,
install **one** onnxruntime accelerator wheel and restart. They're mutually exclusive (all
provide the same `onnxruntime` module), so use the matching `make accel-*` target — it swaps
cleanly, and the run targets use `uv run --no-sync` so your choice isn't reverted:

| Command | Hardware | Notes |
|---|---|---|
| `make accel-cpu` | any | the default; always available |
| `make accel-openvino` | Intel Arc iGPU + **AI Boost NPU** (Core Ultra) | NPU measured **~14× faster** than CPU here; pins `openvino==2025.4.1` (the onnxruntime bridge needs an exact match) |
| `make accel-nvidia` | NVIDIA GPU | needs a matching CUDA/cuDNN runtime |
| `make accel-directml` | any DirectX 12 GPU on Windows (Intel/AMD/NVIDIA) | fully self-contained — no external runtime |

Then restart the server and pick the device in **Settings → "Face detection runs on"**
(e.g. *Intel NPU (OpenVINO)*). `make accel-show` prints the active wheel + providers.
`make install` / `uv sync` resets to the CPU wheel — re-run an accel target afterward.

> On Windows there's no `make`; run the two lines from the target directly, e.g.
> `uv pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml` then
> `uv pip install --reinstall "onnxruntime-openvino==1.24.1" "openvino==2025.4.1"`,
> and start the server with `uv run --no-sync python src/serve.py`.

## Architecture

FastAPI backend (Python) + Svelte single-page app (Vite). The backend modules do all the work —
indexing, vision, embeddings, faces, search — and expose a JSON API. The SPA is the only UI.
Indexing runs as a **background job**: progress streams to the UI, a Stop button works mid-run,
and the page never freezes. **Independent jobs run concurrently** — e.g. GPU/NPU face detection
alongside LM-Studio embedding — while jobs that would collide (two that write the same catalog
records, or two that drive the same model) are automatically serialized. Everything is local;
your photos and index never leave your machine.

## Platform support

Windows, macOS, and Linux. All OS-specific filesystem behavior lives in one module
(`src/platformfs.py`) and adapts automatically:

| Capability | Windows | macOS | Linux |
|---|---|---|---|
| Folder picker roots | Drive letters (`C:\`, `D:\`, …) | `/` + `/Volumes/*` | `/` + `/media/*`, `/run/media/<user>/*`, `/mnt/*` |
| Backup copy engine | robocopy (built in, incremental) | stdlib incremental mirror | stdlib incremental mirror |
| Recoverable delete | Recycle Bin | Finder Trash | freedesktop Trash (`~/.local/share/Trash`) |
| Cloud-placeholder skip (OneDrive etc.) | ✓ | n/a | n/a |
| Default photo folder | `~/Pictures`, OneDrive Pictures | `~/Pictures` | XDG pictures dir (localized), `~/Pictures` |

Both engines use the same incremental compare (size + mtime within 2 s — FAT/exFAT
timestamp granularity), so backing up to an exFAT SD card doesn't re-copy the world each run.
System folders (`$RECYCLE.BIN`, `System Volume Information`, `.Trashes`, `.Trash-1000`,
`lost+found`, Spotlight/fseventsd) are never scanned, imported from, or mirrored on any OS.

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

4. **A — Scan**: add your Photos folder (Browse opens a visual folder picker) → **Scan folders**

5. **B — Vision** then **C — Embed** (or **D — Full index** to do both). Watch the live progress
   bar; **Stop** anytime — it halts between photos. Each photo logs its provider (✅ local / ☁️ Gemini)

6. After indexing: use **Search**, browse **Timeline**, register people in **People**

## Videos

Videos are catalogued alongside photos and need no separate setup:

- **Browse & play**: they appear in the Timeline and grids by capture date with a ▶ badge and
  duration pill; the lightbox plays them inline via a range-streaming endpoint (seekable).
- **Searchable**: **Index & Manage → Video analysis** understands each video the production way:
  **shot-based keyframes** (adaptive scene detection, not blind uniform sampling) are sent
  **together, in one multimodal call** to your chosen Vision provider so the model reasons across
  the clip temporally (motion, how the scene evolves) — not per-frame. Faces are detected across
  those keyframes and **grouped into the distinct people** in the clip (sparse-keyframe face
  tracking), so People counts and matches are accurate. All of this uses the **same provider you
  pick** for photos — never forced local or cloud.
- **Speech search (optional)**: enable local **Whisper ASR** with `make video-ai` and spoken words
  in videos are transcribed, folded into the caption, and made searchable — all on-device, no API.
  Also installs `ffprobe` (via static-ffmpeg) for robust metadata (fps/rotation/audio). Both are
  optional: without them, captioning still works (metadata falls back to an ffmpeg parse; no
  transcript).
- **Your folders, your way**: keep everything in one folder (scan/import treat a mixed folder
  as-is) *or* keep videos in a separate folder (add it to Scan; Import can file videos into
  their own tree). Audio and other non-media files are simply ignored — a mixed folder never
  breaks a scan. All frame extraction uses a bundled `imageio-ffmpeg` (no user install).

## Import & consolidate

**Index & Manage → Import** merges any staging folder — Google Takeout extract, pen-drive
dump, SD card, phone download — into your library. Every file is identified by SHA-1 of its
bytes, so content the library has *ever* seen is skipped no matter how many times it was
copied or renamed; only genuinely new files are copied in, organized `YYYY/MM/` by EXIF date.
The staging source is read-only — originals are never touched, so a botched run costs nothing.

Pick folders visually (no typing paths). The pre-flight check explains, in plain sentences,
anything that can't proceed:
- importing **from inside your scanned library** (that would re-import your own photos),
- importing **into** a folder outside the scanned library (imports would sit invisible,
  never scanned or captioned),
- a source or destination that overlaps an excluded folder.

## Backup

**Index & Manage → Backup** mirrors every scanned folder plus photo-vault's own `data/`
(captions, faces, embeddings, settings) to any connected destination — external drive,
SD card, USB stick, another internal disk, a mounted network share. A restore therefore
brings back your *index*, not just pixels.

- **Incremental**: only new/changed files copy; a re-run after captioning refreshes just
  the metadata folder.
- **Opportunistic**: the destination doesn't need to be always plugged in — the Backup card
  shows "last backup N days ago" and whether the drive is currently connected.
- **Safe by rule**: the destination may not overlap the scanned library in either direction
  (inside it → the mirror would back itself up recursively; containing it → the next scan
  would index your own backup and double every photo). The UI explains any refusal.
- **Videos are left alone**: photo-root mirroring ignores video files entirely, and never
  deletes anything extra it finds at the destination. Only the `photo-vault-data` folder is
  a strict mirror.

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

500+ tests covering: embeddings, vision, video (ffmpeg mocked), faces + execution-provider
selection, search, indexer, validator, tagger, ingest, backup, rate limiting, the
cross-platform filesystem layer, the concurrent background job manager, and the FastAPI
endpoints. All external calls mocked — no services (and no real
ffmpeg) needed to run tests, and the full OS matrix (Windows/macOS/Linux behavior) is
exercised on whatever OS runs them.

## File structure

```
src/
  api.py          ← FastAPI: JSON endpoints + serves the built SPA
  serve.py        ← launcher (reads port from ../ports.json → 8768)
  jobs.py         ← background indexing job manager (worker thread, stop, fail-abort)
  vision.py       ← LM Studio vision → 12 attributes JSON (Gemini fallback); records model used
  video.py        ← all ffmpeg use (bundled imageio-ffmpeg): probe + poster + keyframe frames
  embeddings.py   ← LM Studio embed model (Gemini fallback); multi-model registry
  indexer.py      ← scan + index + gap detection + caption history; per-model ChromaDB collections
  search.py       ← semantic search + attribute filters + person filter
  faces.py        ← InsightFace face detection/embedding; auto-detects execution providers
                    (CPU / OpenVINO GPU+NPU / CUDA / DirectML), user-selectable in Settings
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
