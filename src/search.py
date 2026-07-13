import time

import chromadb.errors

import db
from embeddings import get_embedding, get_active_model, get_registry  # noqa: F401 — get_active_model is a test patch point
from tagger import get_person_embedding
from faces import query_person_faces

# ChromaDB's own signal for "the where clause itself is malformed/unsupported"
# (bad operator, wrong value type, etc). Anything else (connection drops,
# internal errors) is a real failure and must NOT be silently downgraded to
# an unfiltered query — the caller would get results that look right but
# quietly ignore the filter the user asked for.
_MALFORMED_FILTER_ERRORS = (ValueError, chromadb.errors.InvalidArgumentError)

# Hard cap on the exhaustive metadata-only "browse by filter" fetch (#4).
# ChromaDB's collection.get(where=...) is a direct metadata scan, not a
# vector search, so it isn't naturally rank-limited the way a similarity
# query is — this is just a sane ceiling for a personal-library-scale
# collection so one enormous filter match can't blow up memory/response size.
FILTER_BROWSE_LIMIT = 10000


class SearchUnavailableError(Exception):
    """Raised by _embed_query (and therefore search_images) when the
    embedding backend (LM Studio + Gemini fallback chain) cannot be reached
    or fails for a non-connection reason while embedding a search query.
    This means the search backend itself is unavailable — distinct from a
    query that legitimately returns zero results. Callers such as api.py's
    /api/search handlers should catch this specifically and return a clear
    "search is temporarily unavailable" response rather than a raw 500 or a
    misleading empty result set."""


def _embed_query(text: str):
    """Embed a search query IN THE ACTIVE MODEL'S vector space. Letting the
    auto provider chain pick (whatever LM Studio has loaded, or Gemini) can
    produce a vector from a different model than the collection being queried —
    wrong dimension or meaningless distances."""
    try:
        reg = get_registry()
        active = reg.get("active_model")
        info = reg.get("models", {}).get(active)
        if active and info:
            vec, _, _ = get_embedding(
                text, force_provider=info.get("source", "auto"), model=active
            )
        else:
            vec, _, _ = get_embedding(text)
    except Exception as e:
        raise SearchUnavailableError(
            f"Could not embed search query {text!r}: {e}"
        ) from e
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
    # allow_default: search must degrade to "no results yet" for a fresh
    # install with no model selected, not raise — db.collection() otherwise
    # raises ValueError here specifically to stop embedding/indexing code
    # from silently writing into an ungoverned fallback collection.
    return db.collection(allow_default=True)


def _empty_result() -> dict:
    return {"ids": [[]], "metadatas": [[]]}


def _intersect_with_person(result: dict, person_ids: set) -> dict:
    """Narrow a {"ids": [[...]], "metadatas": [[...]]} result down to the
    entries whose id is in person_ids."""
    filtered = _empty_result()
    for i, img_id in enumerate(result["ids"][0]):
        if img_id in person_ids:
            filtered["ids"][0].append(img_id)
            filtered["metadatas"][0].append(result["metadatas"][0][i])
    return filtered


def search_images(query: str, top_k: int = 50, filters: dict = None, person: str = None):
    collection = _active_collection()
    if collection.count() == 0:
        return _empty_result()

    # Resolve the person to a set of matching image ids via the face ANN index.
    # A name that isn't in person_map at all (typo/unregistered) and a real,
    # registered person whose face just doesn't match any indexed photo yet
    # both used to collapse into the same "empty result, no explanation" —
    # person_not_found distinguishes the two for the caller.
    person_ids = None
    person_not_found = False
    if person:
        target = get_person_embedding(person)
        if target is None:
            person_not_found = True
            person_ids = set()
        else:
            person_ids = query_person_faces(target)

    q = (query or "").strip()
    where_clause = build_where_clause(filters or {})

    def _finish(result: dict) -> dict:
        if person_not_found:
            result["person_not_found"] = True
        return result

    # Person-only browse (no text query, no attribute filters): return ALL of the
    # person's photos straight from the face index — not capped to a semantic top_k.
    if person is not None and not q and not where_clause:
        if not person_ids:
            return _finish(_empty_result())
        got = collection.get(ids=list(person_ids), include=["metadatas"])
        return _finish({"ids": [got["ids"]], "metadatas": [got["metadatas"]]})

    # Filter-only browse (no text query, but attribute filters ARE present,
    # optionally combined with a person filter too): do an EXHAUSTIVE
    # metadata-filtered fetch via collection.get(where=...) rather than a
    # semantic-similarity collection.query() against a synthetic "photo"
    # embedding capped at top_k. The old query() approach silently dropped
    # any real match that didn't happen to rank in the top_k nearest to that
    # arbitrary literal-word vector — this is the most important fix here.
    if not q and where_clause:
        got = collection.get(
            where=where_clause, include=["metadatas"], limit=FILTER_BROWSE_LIMIT
        )
        result = {"ids": [got["ids"]], "metadatas": [got["metadatas"]]}
        if person_ids is not None:
            result = _intersect_with_person(result, person_ids)
        return _finish(result)

    if not q:
        q = "photo"  # default for pure-browse (no query, no filters, no person)
    query_embedding = _embed_query(q)
    if query_embedding is None:
        return None

    filter_error = False
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where_clause,
        )
    except _MALFORMED_FILTER_ERRORS as e:
        if where_clause is None:
            # Nothing filter-related to blame this on — a genuine query
            # failure, not a malformed-filter fallback case. Let it propagate.
            raise
        # A malformed/unsupported filter must not silently turn into an
        # unfiltered query that looks like it honored the filter. Fall back,
        # but mark the result so callers can surface that the filter did NOT
        # apply, and log server-side for diagnosis.
        print(f"[search] where clause {where_clause!r} rejected by the index "
              f"({e}); falling back to an UNFILTERED query")
        filter_error = True
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

    if person_ids is not None:
        results = _intersect_with_person(results, person_ids)
    if filter_error:
        results["filter_error"] = True
    return _finish(results)

# Filter values require a full metadata scan of the collection; cache briefly
# keyed on (active model, count) so Search-tab loads don't rescan every time.
# TTL is short (not e.g. 60s) because count alone doesn't change when an
# existing photo's metadata is edited/re-captioned — a longer TTL would let
# stale filter values (e.g. a renamed occasion) linger in the UI.
_FILTER_VALUES_CACHE_TTL = 5
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
            and time.time() - _filter_values_cache["at"] < _FILTER_VALUES_CACHE_TTL
        ):
            return _filter_values_cache["data"]
        result = db.all_metadatas(collection)
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
