import db
from embeddings import get_embedding, get_active_model, collection_name_for
from tagger import get_person_embedding
from faces import query_person_faces

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

def _active_collection(client=None):
    return db.collection()

def search_images(query: str, top_k: int = 50, filters: dict = None, person: str = None):
    collection = _active_collection()
    if collection.count() == 0:
        return {"ids": [[]], "metadatas": [[]]}

    # Resolve the person to a set of matching image ids via the face ANN index.
    person_ids = None
    if person:
        target = get_person_embedding(person)
        person_ids = query_person_faces(target) if target else set()

    q = (query or "").strip()
    where_clause = build_where_clause(filters or {})

    # Person-only browse (no text query, no attribute filters): return ALL of the
    # person's photos straight from the face index — not capped to a semantic top_k.
    if person is not None and not q and not where_clause:
        if not person_ids:
            return {"ids": [[]], "metadatas": [[]]}
        idlist = list(person_ids)[:top_k]
        got = collection.get(ids=idlist, include=["metadatas"])
        return {"ids": [got["ids"]], "metadatas": [got["metadatas"]]}

    if not q:
        q = "photo"  # default for pure-filter / empty searches
    query_embedding, _, _ = get_embedding(q)
    if query_embedding is None:
        return None

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

    if person_ids is None:
        return results

    # Intersect semantic/filter results with the person's matched images.
    filtered = {"ids": [[]], "metadatas": [[]]}
    for i, img_id in enumerate(results["ids"][0]):
        if img_id in person_ids:
            filtered["ids"][0].append(img_id)
            filtered["metadatas"][0].append(results["metadatas"][0][i])
    return filtered

def get_available_filter_values() -> dict:
    """Returns dict of attribute → sorted unique values found in the active collection."""
    try:
        collection = _active_collection()
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

