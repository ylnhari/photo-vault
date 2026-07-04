"""
Singleton ChromaDB access. Previously every request/op constructed a fresh
PersistentClient(path=...), which reopens the on-disk store each time. This
module hands back one process-wide client and a collection helper.
"""
import threading

import chromadb

from constants import CHROMA_DB_PATH
from embeddings import collection_name_for, get_active_model

_client = None
_client_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        with _client_lock:
            # Double-checked locking: re-test inside the lock so two threads
            # racing on first call don't each construct a separate
            # PersistentClient against the same on-disk path.
            if _client is None:
                _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def collection(model_name: str | None = None, *, allow_default: bool = False):
    """Get-or-create the collection for a model (active model when
    model_name is None).

    Raises ValueError when no model_name is given and no active model is
    configured yet, instead of silently falling back to a hardcoded
    "images" collection name that bypasses the versioned per-model naming
    scheme. Pass allow_default=True to opt into that old safe-fallback
    behavior for a caller that genuinely needs a collection handle before
    any model has ever been selected (e.g. an empty-state read that just
    wants an empty collection rather than an error)."""
    if model_name is None:
        model_name = get_active_model()
        if not model_name:
            if allow_default:
                return client().get_or_create_collection(name="images")
            raise ValueError(
                "no active embedding model configured — set one before embedding"
            )
    name = collection_name_for(model_name)
    return client().get_or_create_collection(name=name)


FACES_COLLECTION = "faces"


def faces_collection():
    """Dedicated ANN index of individual face embeddings (cosine space)."""
    return client().get_or_create_collection(
        name=FACES_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
