# syntax=docker/dockerfile:1

# ---- Stage 1: build the Svelte SPA ----
FROM node:20-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm install
COPY web/ ./
RUN npm run build

# ---- Stage 2: Python runtime ----
FROM python:3.11-slim AS app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Native libs needed by onnxruntime (libgomp) and opencv-headless (libglib);
# build-essential + cmake cover any source builds (e.g. insightface).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from pyproject (single source of truth — no drift).
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

# App code + the built SPA from stage 1.
COPY src/ ./src/
COPY --from=web /web/dist ./web/dist

# Hardened by default; container binds all interfaces (port-mapped + token auth).
ENV PV_REQUIRE_AUTH=1 \
    PV_BIND_HOST=0.0.0.0 \
    PHOTO_VAULT_PORT=8768

EXPOSE 8768
CMD ["python", "src/serve.py"]
