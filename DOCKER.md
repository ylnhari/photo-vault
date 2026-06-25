# Running Photo Vault with Docker

One command, no Python/Node/uv setup required.

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) or Docker Engine + Compose (Linux).

## Start it
From the project folder:

```bash
# Point PHOTOS_DIR at the folder of photos you want indexed.
# macOS / Linux:
PHOTOS_DIR=/Users/you/Pictures docker compose up --build

# Windows (PowerShell):
$env:PHOTOS_DIR="C:\Users\you\Pictures"; docker compose up --build
```

Then open **http://localhost:8768**.

In the app: **Index & Manage → Add folder →** type `/photos`, then **Scan**, then run
**Vision**, **Embed**, and **Face detection**.

## Optional: AI providers
- **Gemini (easiest):** put `GEMINI_API_KEY=...` in a `.env` file beside `docker-compose.yml`
  (get a free key at https://aistudio.google.com/). It's passed in automatically.
- **LM Studio (local, private):** run LM Studio on your host with a vision model and an
  embedding model loaded. The container reaches it at `host.docker.internal:1234`
  (already configured). On Linux this works via the `host-gateway` mapping.

## What persists
- `./data` — the index, thumbnails, faces, and settings (a bind mount you can back up).
- A named volume caches the InsightFace face model so it isn't re-downloaded each run.

## Notes
- Your photos are mounted **read-only** (`:ro`), so the app can never modify or delete
  your originals. The "delete file from disk" action is therefore disabled in this mode;
  "remove from index" still works.
- The local API requires a per-install token; the web UI receives it automatically, so
  there's nothing to copy. Access is restricted to `localhost` by default — set
  `PV_ALLOWED_HOSTS=your-host` if you intend to reach it from another machine.
