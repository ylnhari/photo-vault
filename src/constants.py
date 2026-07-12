import os
import json

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_CATALOG_PATH = os.path.join(DATA_DIR, "catalog.db")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
FACE_DIR = os.path.join(DATA_DIR, "faces")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
PERSON_MAP_PATH = os.path.join(DATA_DIR, "person_map.json")


def _load_port(default: int = 8768) -> int:
    """Port resolution order: PHOTO_VAULT_PORT env → sibling ports.json registry
    (Hari's machine) → default. The env var is what Docker/other hosts set."""
    env_port = os.environ.get("PHOTO_VAULT_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            print(f"[constants] PHOTO_VAULT_PORT={env_port!r} is not a valid "
                  f"integer; falling back to ports.json/default.")
    registry = os.path.join(os.path.dirname(PROJECT_ROOT), "ports.json")
    try:
        with open(registry) as f:
            return int(json.load(f)["registry"]["photo-vault"]["port"])
    except FileNotFoundError:
        print(f"[constants] ports.json not found at {registry}; using default port {default}.")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"[constants] ports.json at {registry} is malformed ({e!r}); using default port {default}.")
    except Exception as e:
        print(f"[constants] could not read port from {registry} ({e!r}); using default port {default}.")
    return default


SERVER_PORT = _load_port()

# API Endpoints — LM Studio host is overridable so a container can reach the host
# (e.g. http://host.docker.internal:1234/v1).
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")


def _load_9router_url(default_port: int = 20128) -> str:
    """9Router (local OpenAI-compatible LLM gateway) base URL. Resolution order
    mirrors _load_port: NINEROUTER_URL env → sibling ports.json registry
    (registry.9router.port) → default. Loopback only, per house rule."""
    env_url = os.environ.get("NINEROUTER_URL")
    if env_url:
        return env_url.rstrip("/")
    port = default_port
    registry = os.path.join(os.path.dirname(PROJECT_ROOT), "ports.json")
    try:
        with open(registry) as f:
            port = int(json.load(f)["registry"]["9router"]["port"])
    except Exception:
        pass  # ports.json missing/malformed → default port; same spirit as _load_port
    return f"http://127.0.0.1:{port}/v1"


NINEROUTER_URL = _load_9router_url()

# Gemini fallback — loaded from .env in project root
def _load_env():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Standard .env convention: KEY="value" / KEY='value' —
                    # strip one matching layer of surrounding quotes so the
                    # literal quote characters don't end up embedded in the
                    # value. Only strip when both ends match, so values that
                    # legitimately contain a stray quote aren't mangled.
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                        v = v[1:-1]
                    os.environ.setdefault(k, v)

_load_env()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Tried in order; escalate to the next only on 429/404/503. Ordered lightest-
# first: the "flash-lite" tier generally carries the largest free-tier request
# allowance and gives clean JSON-mode output, so it leads, with heavier flash
# models as fallbacks. Exact free-tier quotas vary by account, tier and region
# and change over time (https://ai.google.dev/gemini-api/docs/rate-limits), so
# this is an ordering heuristic, not a guarantee for any specific account.
GEMINI_VISION_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
]

EMBEDDING_REGISTRY_PATH = os.path.join(DATA_DIR, "embedding_registry.json")
FOLDERS_CONFIG_PATH = os.path.join(DATA_DIR, "folders.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
ALBUMS_PATH = os.path.join(DATA_DIR, "albums.json")

# Defaults
DEFAULT_TARGET_DIR = os.path.join(os.path.expanduser("~"), "Pictures")
SIMILARITY_THRESHOLD = 0.6
