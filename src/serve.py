"""Photo Vault launcher — reads port from the ports.json registry (constants.SERVER_PORT)
and serves the FastAPI app + built SPA. Run:  uv run python src/serve.py
"""
import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import SERVER_PORT, PROJECT_ROOT  # noqa: E402

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8 if possible.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    dist = os.path.join(PROJECT_ROOT, "web", "dist")
    if not os.path.isdir(dist):
        print("[!] web/dist not found - the SPA isn't built.")
        print("    Run:  cd web && npm install && npm run build")
        print("    (API will still serve; only the UI is missing.)\n")
    print(f"Photo Vault -> http://127.0.0.1:{SERVER_PORT}   (Ctrl+C to stop)")
    uvicorn.run("api:app", host="127.0.0.1", port=SERVER_PORT, reload=False)


if __name__ == "__main__":
    main()
