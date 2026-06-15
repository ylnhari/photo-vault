.PHONY: install build run serve web test clean

install:
	uv sync --extra dev
	cd web && npm install

build:
	cd web && npm run build

# Production: build the SPA, then serve API + SPA same-origin (port from ports.json)
run: build
	uv run python src/serve.py

# Backend only (no SPA build) — pair with `make web` in another terminal for hot reload
serve:
	uv run python src/serve.py

# Frontend dev server with hot reload (proxies /api to the backend)
web:
	cd web && npm run dev

test:
	uv run python -m pytest tests/ -q

clean:
	rm -rf web/dist web/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
