# Indexing freshness findings

## Finding

The embed queues treated Chroma ID presence as proof that a vector matched the current catalog. A successful vision rerun updates `caption_json` and `caption_history`, but does not change that ID, so the prior vector remained invisible to the queue. Model-specific selection had the same gap and could queue blank or malformed caption records when history existed.

## Changes

- Added a deterministic SHA-256 fingerprint over the exact embedding text, selected caption source, versioned payload/preprocessing schema, and raw relevant catalog metadata. Metadata is canonicalized with sorted keys. The fingerprint calculation does not call geocoding or an embedding provider.
- Store the fingerprint in Chroma metadata from `build_embed_payload`; legacy rows without a fingerprint are pending once, based only on missing/mismatched current state.
- Make active and model-specific embed selectors compare the expected fingerprint with the selected collection. Respect `caption_source_model`, reject malformed JSON and explicit blank captions, and preserve legacy schema-light captions that omit the `caption` key.
- Route index, reconcile, count, purge, and delete collection access through `db.collection(model_name, model_info=...)` so model profile collection configuration is honored.
- Import the canonical `resolve_photo_date` and `_date_from_filename` from `photo_date.py`; the indexer no longer duplicates capture-date behavior.

## Verification

`.venv/Scripts/python.exe -m pytest tests/test_indexer.py -q` passed: 50 tests. Added synthetic coverage for deterministic metadata hashing, source-specific caption selection, blank/invalid caption exclusion, and re-caption → pending → embed → no longer pending. Provider calls are mocked; no personal media or runtime data was read.

## Integration and limits
