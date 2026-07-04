import time

import db
from embeddings import get_embedding, get_active_model, get_registry  # noqa: F401 — get_active_model is a test patch point
from tagger import get_person_embedding
from faces import query_person_faces


def _embed_query(text: str):
    """Embed a search query IN THE ACTIVE MODEL'S vector space. Letting the
    auto provider chain pick (whatever LM Studio has loaded, or Gemini) can
    produce a vector from a different model than the collection being queried —
    wrong dimension or meaningless distances."""
    reg = get_registry()
    active = reg.get("active_model")
    info = reg.get("models", {}).get(active)
    if active and info:
        vec, _, _ = get_embedding(
            text, force_provider=info.get("source", "auto"), model=active
        )
        return vec
    vec, _, _ = get_embedding(text)
    return vec


def build_where_clause(filters: dict) -> dict | None:
    clauses = []
    for key, value in filters.items():
        if value and value != "All":
            # person_count is stored as an int in Chroma metadata; the UI
            # sends filter values as strings, so $eq needs the same type.
            if key == "person_count":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
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
        got = collection.get(ids=list(person_ids), include=["metadatas"])
        return {"ids": [got["ids"]], "metadatas": [got["metadatas"]]}

    if not q:
        q = "photo"  # default for pure-filter / empty searches
    query_embedding = _embed_query(q)
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

# Filter values require a full metadata scan of the collection; cache briefly
# keyed on (active model, count) so Search-tab loads don't rescan every time.
_filter_values_cache = {"key": None, "at": 0.0, "data": {}}


def get_available_filter_values() -> dict:
    """Returns dict of attribute → sorted unique values found in the active collection."""
    try:
        collection = _active_collection()
        n = collection.count()
        if n == 0:
            return {}
        key = (get_active_model(), n)
        if (
            _filter_values_cache["key"] == key
            and time.time() - _filter_values_cache["at"] < 60
        ):
            return _filter_values_cache["data"]
        result = collection.get(include=["metadatas"])
        attrs = ["weather", "occasion", "festival_name", "scene", "group_size", "person_count",
                 "clothing_style", "mood", "location_type", "season", "time_of_day",
                 "photo_type", "year", "month", "place"]
        values = {}
        for attr in attrs:
            seen = set()
            for meta in result["metadatas"]:
                val = meta.get(attr, "")
                if val not in ("unknown", "", None):
                    seen.add(str(val))
            if seen:
                values[attr] = (
                    sorted(seen, key=int) if attr == "person_count" else sorted(seen)
                )
        _filter_values_cache.update({"key": key, "at": time.time(), "data": values})
        return values
    except Exception:
        return {}

