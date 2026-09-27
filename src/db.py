"""
Singleton ChromaDB access. Previously every request/op constructed a fresh
PersistentClient(path=...), which reopens the on-disk store each time. This
module hands back one process-wide client and a collection helper.
"""
import threading
import time

import chromadb
from chromadb.config import Settings

from constants import CHROMA_DB_PATH
from embeddings import collection_name_for, get_active_model, get_registry

_client = None
_client_lock = threading.Lock()

# Shared short-TTL snapshot of a collection's full metadata set. Pulling every
# row's metadata (`collection.get(include=["metadatas"])`) is a ~3–4s scan at
# ~26k rows, and TWO hot read paths need it on the initial dashboard load — the
# Search filter-values scan and the status "all-attributes-unknown" scan — so
# without sharing they each pay it (and contend on the GIL). Cache one snapshot,
# keyed on (collection name, row count), behind a lock so a burst of concurrent
# callers triggers a single fetch. TTL is short so re-captioned/edited metadata
# surfaces in the UI within a few seconds.
_ALL_META_TTL = 5.0
_all_meta_cache: dict = {"key": None, "at": 0.0, "data": None}
_all_meta_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        with _client_lock:
            # Double-checked locking: re-test inside the lock so two threads
            # racing on first call don't each construct a separate
            # PersistentClient against the same on-disk path.
            if _client is None:
                _client = chromadb.PersistentClient(
                    path=CHROMA_DB_PATH,
                    settings=Settings(anonymized_telemetry=False),
                )
    return _client


def collection(model_name: str | None = None, *, allow_default: bool = False,
               model_info: dict | None = None):
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
    if model_info is None:
        model_info = get_registry().get("models", {}).get(model_name)
    name = (model_info or {}).get("collection") or collection_name_for(model_name)
    profile = (model_info or {}).get("profile")
    if profile:
        profile_id = (model_info or {}).get("profile_id")
        if not profile_id:
            raise ValueError(f"embedding profile for {model_name!r} has no profile id")
        col = client().get_or_create_collection(
            name=name,
            metadata={"hnsw:space": profile.get("metric", "cosine"),
                      "embedding_profile_id": profile_id},
        )
        metadata = col.metadata or {}
        if metadata.get("embedding_profile_id") != profile_id:
            raise ValueError(f"collection {name!r} belongs to a different embedding profile")
        if metadata.get("hnsw:space") != profile.get("metric", "cosine"):
            raise ValueError(f"collection {name!r} uses a different distance metric")
        return col
    # Legacy collections retain their existing name and Chroma distance metric.
    return client().get_or_create_collection(name=name)


def all_metadatas(col=None, ttl: float = _ALL_META_TTL) -> dict:
    """Cached full-collection metadata pull, shared across the read paths that
    each need every row's metadata on the same dashboard load (see the cache
    note above). Returns the raw ChromaDB shape {"ids": [...], "metadatas": [...]}.
    Falls back to the active collection when `col` is None."""
    c = col if col is not None else collection(allow_default=True)
    key = (c.name, c.count())
    now = time.time()
    cached = _all_meta_cache
    if cached["key"] == key and cached["data"] is not None and now - cached["at"] < ttl:
        return cached["data"]
    with _all_meta_lock:
        # Re-check under the lock so concurrent first-callers pull only once.
        if (
            _all_meta_cache["key"] == key
            and _all_meta_cache["data"] is not None
            and time.time() - _all_meta_cache["at"] < ttl
        ):
            return _all_meta_cache["data"]
        data = c.get(include=["metadatas"])
        _all_meta_cache.update({"key": key, "at": time.time(), "data": data})
        return data


FACES_COLLECTION = "faces"


def faces_collection():
    """Dedicated ANN index of individual face embeddings (cosine space)."""
    return client().get_or_create_collection(
        name=FACES_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
