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
            pass
    try:
        registry = os.path.join(os.path.dirname(PROJECT_ROOT), "ports.json")
        with open(registry) as f:
            return int(json.load(f)["registry"]["photo-vault"]["port"])
    except Exception:
        return default


SERVER_PORT = _load_port()

# API Endpoints — LM Studio host is overridable so a container can reach the host
# (e.g. http://host.docker.internal:1234/v1).
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

# Gemini fallback — loaded from .env in project root
def _load_env():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Prefer highest-rate-limit models first; escalate only if 429/404/503
# Ordered by verified free-tier daily quota on this account (AI Studio ->
# Rate Limit, 2026-07-04) — gemini-2.0-flash* variants showed 0/0 (unavailable
# on this project/tier) and were dropped; gemini-3.1-flash-lite has by far the
# best quota (RPD 500 vs 20 for the rest) plus the cleanest JSON-mode output
# in testing, so it goes first.
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
