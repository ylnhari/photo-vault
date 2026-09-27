# Independent backend and reliability review

Static review of the backend/reliability working diff, including `src/indexer.py`,
the profile-aware collection calls in `src/db.py`, and the current retrieval
changes in `src/search.py`, `src/embeddings.py`, and `src/vision.py`. No tests,
indexing, provider inference, dependency installation, or personal-data reads
were performed. The two findings below were rechecked against the subsequent
fixes; both are resolved in the current diff.

## Findings

### Resolved P2 — A backup could report success after its roots changed mid-run

`JobManager._pending_ids()` snapshots the source roots at job start
([src/jobs.py:232](../src/jobs.py)), but `backup_one()` resolves the source-to-
destination map again for every root ([src/backup.py:170](../src/backup.py)).
The API can change `backup_dest` while the job is active through
`PUT /api/settings`; that route validates and saves the setting without checking
for an active backup ([src/api.py:457](../src/api.py)). Adding an included scan
folder during a run can likewise change the root set. A run can therefore copy
different roots to different destinations, omit a newly configured root, and
still satisfy the completion counter and call `record_success()`
([src/jobs.py:809](../src/jobs.py)).

Resolution verified statically: each backup job freezes its full root and
destination pairs and passes the frozen destination to each copy
([src/jobs.py:233](../src/jobs.py), [src/jobs.py:815](../src/jobs.py),
[src/backup.py:189](../src/backup.py)). The settings endpoint rejects
destination changes during an active backup ([src/api.py:487](../src/api.py)),
and completion verifies that the frozen roots still match current settings
before recording success ([src/backup.py:100](../src/backup.py),
[src/jobs.py:873](../src/jobs.py)).
The persisted state now includes a fingerprint of the exact normalized root
map, and `status()` compares it against the current map before presenting a
fresh timestamp ([src/backup.py:101](../src/backup.py),
[src/backup.py:139](../src/backup.py)). This also closes the check/write race
for folder changes: a timestamp written for an old map is not treated as fresh
against a changed map. A missing source root now raises and counts as a failure,
so a partial backup cannot reach the success path ([src/backup.py:219](../src/backup.py)).

### Resolved P2 — Catalog-backed `/api/meta` dropped fields still rendered by the Lightbox

When a catalog row exists, `/api/meta` skips Chroma metadata and constructs the
response from caption JSON and EXIF metadata. The current diff derives year and
month from the shared date resolver and includes the catalog caption model
([src/api.py:1693](../src/api.py)). `embedding_source` remains in the on-demand
details path, consistent with the decision to keep the initial Lightbox request
catalog-only ([web/src/lib/Lightbox.svelte:78](../web/src/lib/Lightbox.svelte)).
The normal metadata response therefore preserves the fields needed for initial
photo details while avoiding vector-store access. This was verified statically;
the API tests were not run in this review.

## Additional review

`src/serve.py:63-65` disables Uvicorn access logging because media URLs include
the bearer token in their query string. This prevents the token from appearing
in routine request logs; application error logging remains enabled.

The new serialized-import path also passed static review. `runtime_import.py`
uses a process-wide `RLock`, so optional imports made through API, jobs, face,
and clustering wrappers cannot initialize NumPy concurrently, and nested
imports from the importing thread do not deadlock. The former direct API imports
of `faces` in settings/provider routes now use the helper
([src/api.py:498](../src/api.py), [src/api.py:510](../src/api.py)); OpenVINO
loads in `faces.py` also go through it. Clustering defers NumPy and sklearn
imports until actual clustering work, so the read-only clusters route can stay
on the cold path. The new fresh-process regression test checks simultaneous
`/api/people`, `/api/faces/status`, and `/api/faces/clusters` requests and
asserts that the clustering stack remains absent for the cluster-list route
([tests/test_runtime_import.py](../tests/test_runtime_import.py)). I did not run
it; the primary reports that its live three-route check returned 200 for all
routes. Chroma may still load SciPy on the face-status path; that is outside the
clusters route's deferred sklearn/SciPy stack.

## Checks and limits

- The catalog projection migration is transactional: projection rows and the
  version marker commit together, and migration exceptions roll back. Separate
  processes can both observe a missing marker and rebuild serially, which costs
  extra work but does not expose a partial projection.
- The embedding fingerprint selectors agree with the payload builder for both
  default/latest captions and an explicitly selected caption source: both use
  the same resolved caption, source label, embedding text, and catalog fields.
  Profile-specific model collections keep those hashes within their model
  space.
- Thumbnail generation writes a same-directory temporary file and replaces the
  target only after encoding completes. The keyed lock prevents duplicate
  decoding for concurrent requests in one process.
- No tracked test files were deleted. The existing `tests/test_indexer.py`
  inventory remains present; one prior pending-count regression was updated to
  assert fingerprint staleness, and the new fingerprint cases were appended.
  Static review only; tests were not run per the assignment.
