"""Photo Vault launcher — reads port from the ports.json registry (constants.SERVER_PORT)
and serves the FastAPI app + built SPA. Run:  uv run python src/serve.py
"""
import argparse
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
    ap = argparse.ArgumentParser(description="Photo Vault server")
    ap.add_argument("--port", type=int, default=SERVER_PORT,
                    help="Port to bind (default: SERVER_PORT from ports.json)")
    args = ap.parse_args()
    port = args.port

    dist = os.path.join(PROJECT_ROOT, "web", "dist")
    if not os.path.isdir(dist):
        print("[!] web/dist not found - the SPA isn't built.")
        print("    Run:  cd web && npm install && npm run build")
        print("    (API will still serve; only the UI is missing.)\n")

    # Harden the local API by default: require the per-install bearer token.
    # The SPA receives it automatically (injected into index.html / via /api/token),
    # so this is transparent to the user but blocks other browser origins.
    os.environ.setdefault("PV_REQUIRE_AUTH", "1")
    import security  # noqa: E402
    security.get_token()  # generate + persist on first run

    # Loopback by default. Docker sets PV_BIND_HOST=0.0.0.0 (safe there because the
    # token auth above is enabled and the container network is isolated + port-mapped).
    host = os.environ.get("PV_BIND_HOST", "127.0.0.1")
    print(f"Photo Vault -> http://127.0.0.1:{port}   (Ctrl+C to stop)")
    uvicorn.run("api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
