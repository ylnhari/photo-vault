import os
import json

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_CATALOG_PATH = os.path.join(DATA_DIR, "images.json")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
FACE_DIR = os.path.join(DATA_DIR, "faces")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
PERSON_MAP_PATH = os.path.join(DATA_DIR, "person_map.json")


def _load_port(default: int = 8768) -> int:
    """Read this project's port from the sibling ports.json registry (Hari's machine);
    fall back to default for contributors who don't have the registry."""
    try:
        registry = os.path.join(os.path.dirname(PROJECT_ROOT), "ports.json")
        with open(registry) as f:
            return int(json.load(f)["registry"]["photo-vault"]["port"])
    except Exception:
        return default


SERVER_PORT = _load_port()

# API Endpoints
LM_STUDIO_URL = "http://localhost:1234/v1"
OLLAMA_URL = "http://localhost:11434"

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
GEMINI_VISION_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
]

EMBEDDING_REGISTRY_PATH = os.path.join(DATA_DIR, "embedding_registry.json")

# Defaults
DEFAULT_TARGET_DIR = os.path.join(os.path.expanduser("~"), "Pictures")
SIMILARITY_THRESHOLD = 0.6
