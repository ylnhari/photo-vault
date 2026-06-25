"""
Singleton ChromaDB access. Previously every request/op constructed a fresh
PersistentClient(path=...), which reopens the on-disk store each time. This
module hands back one process-wide client and a collection helper.
"""
import chromadb

from constants import CHROMA_DB_PATH
from embeddings import collection_name_for, get_active_model

_client = None


def client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def collection(model_name: str | None = None):
    """Get-or-create the collection for a model (active model when None)."""
    if model_name is None:
        model_name = get_active_model()
    name = collection_name_for(model_name) if model_name else "images"
    return client().get_or_create_collection(name=name)


FACES_COLLECTION = "faces"


def faces_collection():
    """Dedicated ANN index of individual face embeddings (cosine space)."""
    return client().get_or_create_collection(
        name=FACES_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
