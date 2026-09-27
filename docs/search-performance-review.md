# Independent review: Browse search performance optimization

Read-only review of the v3 projection optimization in `src/catalog_db.py` and
the synthetic regression suite. I did not read real catalog data, media, or
`.env`; I did not edit implementation code or run real-library benchmarks.

## Review result

I found no correctness blocker in the current implementation.

- FTS5 is only a coarse candidate selector for positive queries. Candidate
  terms are ASCII alphanumeric tokens extracted from the already-normalized
  positive terms ([src/catalog_db.py:518](../src/catalog_db.py)); the original
  field-scoped `instr` predicates still decide every result
  ([src/catalog_db.py:613](../src/catalog_db.py),
  [src/catalog_db.py:713](../src/catalog_db.py)). Thus FTS false positives are
  removed by exact checks. Unicode-only, punctuation-only, and exclusion-only
  queries have no candidate expression and retain the full exact path. A
  candidate query can omit Unicode/punctuation constraints only by widening
  the candidate set; it does not replace them.
- The FTS table is external-content over the projection. Migration drops old
  triggers/table, rebuilds projection rows, creates insert/update/delete
  triggers, rebuilds FTS, and commits the projection version in one SQLite
  transaction ([src/catalog_db.py:434](../src/catalog_db.py),
  [src/catalog_db.py:490](../src/catalog_db.py)). Normal projection upserts and
  deletes therefore update FTS in the same transaction. Fresh catalogs on a
  SQLite build without FTS5 skip optional index creation and use exact SQL;
  failed migration rolls back the version marker and projection rebuild.
- The new expression indexes align with capture-date newest/oldest order and
  keep unknown dates last; the summary index includes the values consumed by
  the summary aggregate ([src/catalog_db.py:473](../src/catalog_db.py),
  [src/catalog_db.py:481](../src/catalog_db.py)).

The backend worker reports 10 focused synthetic tests passed. The suite covers
sort query plans, v2 migration/rebuild, cross-field AND and exclusion behavior,
exact semantics after FTS narrowing, Unicode/punctuation fallback, upsert/delete
synchronization, absent-index fallback, and migration rollback/retry
([tests/test_library_query_indexes.py](../tests/test_library_query_indexes.py)).
I reviewed these test sources but did not run them independently.

## Portability limit

Migrating a catalog that already contains `library_search_fts` requires an
SQLite build with FTS5. A build without the module can use the exact SQL
fallback for a fresh catalog, but dropping a persisted FTS virtual table from
another build may fail before fallback is reached. The primary accepted this
as a non-blocking local portability limit: the current SQLite has FTS5 and
Photo Vault's `data/` is ignored per clone. No workaround should compromise
transaction rollback or catalog preservation.
