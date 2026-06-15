#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d web/dist ]; then
  echo "Building SPA..."
  (cd web && npm install && npm run build)
fi
uv run python src/serve.py "$@"
