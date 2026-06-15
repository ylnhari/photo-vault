@echo off
cd /d "%~dp0"
if not exist web\dist (
  echo Building SPA...
  pushd web
  call npm install
  call npm run build
  popd
)
uv run python src/serve.py %*
