import chromadb
import numpy as np
import json
import os
from embeddings import get_embedding, get_active_model, collection_name_for
from tagger import get_person_embedding
from faces import load_face_data
from constants import CHROMA_DB_PATH, SIMILARITY_THRESHOLD

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_where_clause(filters: dict) -> dict | None:
    clauses = []
    for key, value in filters.items():
        if value and value != "All":
            clauses.append({key: {"$eq": value}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def _active_collection(client):
    active = get_active_model()
    col_name = collection_name_for(active) if active else "images"
    return client.get_or_create_collection(name=col_name)

def search_images(query: str, top_k: int = 50, filters: dict = None, person: str = None):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = _active_collection(client)

    if collection.count() == 0:
        return {"ids": [[]], "metadatas": [[]]}

    query_embedding, _, _ = get_embedding(query)
    if query_embedding is None:
        return None

    where_clause = build_where_clause(filters or {})

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where_clause,
        )
    except Exception:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

    if not person:
        return results

    target_embedding = get_person_embedding(person)
    if not target_embedding:
        return results

    filtered = {"ids": [[]], "metadatas": [[]]}
    for i, img_id in enumerate(results['ids'][0]):
        for face in load_face_data(img_id):
            if cosine_similarity(target_embedding, face['embedding']) > SIMILARITY_THRESHOLD:
                filtered['ids'][0].append(img_id)
                filtered['metadatas'][0].append(results['metadatas'][0][i])
                break
    return filtered

def get_available_filter_values() -> dict:
    """Returns dict of attribute → sorted unique values found in the active collection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = _active_collection(client)
        if collection.count() == 0:
            return {}
        result = collection.get(include=["metadatas"])
        attrs = ["weather", "occasion", "scene", "group_size", "clothing_style",
                 "mood", "location_type", "season", "time_of_day", "year"]
        values = {}
        for attr in attrs:
            seen = set()
            for meta in result["metadatas"]:
                val = meta.get(attr, "")
                if val and val not in ("unknown", "", None):
                    seen.add(str(val))
            if seen:
                values[attr] = sorted(seen)
        return values
    except Exception:
        return {}

def get_all_images(filters: dict = None) -> dict:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = _active_collection(client)
    if collection.count() == 0:
        return {"ids": [], "metadatas": []}
    where_clause = build_where_clause(filters or {})
    try:
        return collection.get(where=where_clause, include=["metadatas"])
    except Exception:
        return collection.get(include=["metadatas"])
