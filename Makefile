.PHONY: install build run serve web test clean \
        accel-cpu accel-openvino accel-nvidia accel-directml accel-show \
        docker-build docker-up docker-down

install:
	uv sync --extra dev
	cd web && npm install

build:
	cd web && npm run build

# Production: build the SPA, then serve API + SPA same-origin (port from ports.json).
# --no-sync so a chosen face-detection accelerator (see accel-* below) isn't
# reverted back to the CPU wheel on every run.
run: build
	uv run --no-sync python src/serve.py

# Backend only (no SPA build) — pair with `make web` in another terminal for hot reload
serve:
	uv run --no-sync python src/serve.py

# Frontend dev server with hot reload (proxies /api to the backend)
web:
	cd web && npm run dev

test:
	uv run --no-sync python -m pytest tests/ -q

clean:
	rm -rf web/dist web/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Face-detection accelerator (optional) ────────────────────────────────────
# InsightFace face detection runs on the CPU by default (works everywhere). To
# use a GPU/NPU, swap the onnxruntime wheel with ONE of the targets below, then
# restart the server. The in-app Settings → "Face detection runs on" picker
# auto-detects whatever you install and lets you choose the device. These are
# mutually exclusive (all provide the same `onnxruntime` module), so each target
# removes the others first. `make install`/`uv sync` resets to the CPU wheel —
# re-run an accel target afterward. See README "GPU/NPU acceleration".
accel-cpu:
	-uv pip uninstall onnxruntime-openvino onnxruntime-gpu onnxruntime-directml openvino openvino-telemetry
	uv pip install --reinstall "onnxruntime>=1.16.0"

# Intel Arc iGPU + AI Boost NPU (Core Ultra). The NPU is typically the fastest
# for face detection (measured ~14× faster than CPU); pick it in Settings.
# Versions are pinned because the onnxruntime OpenVINO bridge needs an exact
# OpenVINO runtime match (1.24.1 ↔ openvino 2025.4.1).
accel-openvino:
	-uv pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml
	uv pip install --reinstall "onnxruntime-openvino==1.24.1" "openvino==2025.4.1"

# NVIDIA GPU (CUDA). Needs a matching CUDA/cuDNN runtime on the machine.
accel-nvidia:
	-uv pip uninstall onnxruntime onnxruntime-openvino onnxruntime-directml openvino openvino-telemetry
	uv pip install --reinstall "onnxruntime-gpu>=1.16.0"

# Any DirectX 12 GPU on Windows (incl. Intel/AMD iGPUs) via DirectML. Fully
# self-contained — no external runtime to install.
accel-directml:
	-uv pip uninstall onnxruntime onnxruntime-openvino onnxruntime-gpu openvino openvino-telemetry
	uv pip install --reinstall "onnxruntime-directml>=1.16.0"

# Show which onnxruntime wheel + execution providers are currently active.
accel-show:
	uv run --no-sync python -c "import onnxruntime as o; print('onnxruntime', o.__version__); print('providers', o.get_available_providers())"

# ── Video AI enhancements (optional) ─────────────────────────────────────────
# Videos are captioned + face-scanned out of the box. This adds two optional,
# fully-local upgrades toward production quality:
#   * ffprobe (via static-ffmpeg): robust JSON metadata (fps/rotation/audio),
#     replacing the ffmpeg-stderr regex parse.
#   * Whisper ASR (faster-whisper): transcribe speech in videos so spoken words
#     are folded into captions and become searchable.
# faster-whisper normally pulls plain `onnxruntime` (for its VAD), which would
# clobber an accelerator wheel (accel-openvino etc.) — its VAD is satisfied by
# whatever onnxruntime is already installed, so we install it WITHOUT deps and
# add only the safe ones. Both are optional: without them, captioning still
# works (probe falls back to regex; no transcript).
video-ai:
	uv pip install av ctranslate2 static-ffmpeg
	uv pip install --no-deps faster-whisper
	uv run --no-sync python -c "import transcribe, video; print('ASR available:', transcribe.available()); print('ffprobe:', bool(video.ffprobe_exe()))"

# Docker (see DOCKER.md). Set PHOTOS_DIR to the folder you want indexed.
docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down
