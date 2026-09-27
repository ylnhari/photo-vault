# Reliability audit

## Findings

1. **P1 — captions can diverge from retrieval vectors.** `record_caption()` updates `caption_json` and `caption_history`, while `get_embed_pending()` treats any existing id in the active collection as current. A fresh vision run can therefore leave retrieval using the old vector and old Chroma metadata. The indexer fingerprint work now in progress should make these rows pending for repair; the batched embed path must store the same fingerprint and selected caption source as the single-item path.
2. **P1 — backups could race active writers and overstate completion.** The original backup resource set did not intersect catalog, vector, face, or thumbnail writers, despite `data/` being mirrored with strict purge semantics. Also, each successful root updated `last_backup_at`, even if a later root failed. This worktree now gives backup all mutable resource locks and records success only after every root finishes with no failures, skips, or stop. Backup tests use mocked copy calls or temporary trees only.
3. **P1 — scan persistence failures looked successful.** `save_data()` caught and printed SQLite errors, allowing the job to finish after its final catalog write failed. It now propagates the error so the job can report failure; a synthetic regression test covers it.
4. **P2 — Stop can wait for an active unit.** Vision waits on futures in submission order and exits its executor only after submitted calls return, so Stop waits for the current concurrency batch. Sequential ingest cannot interrupt a `copy2()` or its initial library-video hash refresh; backup checks Stop between roots, not during a root mirror. These paths have bounded work units but no mid-unit cancellation.
5. **P2 — interrupted ingest copies can leave partial files.** `IngestSession.ingest_one()` copies directly to the final library name and only updates its hash cache after the copy. A disk or process failure can leave a partial file that a later scan may catalog. Use a same-directory temporary file and atomic replacement, then cache the completed copy.
6. **P2 — malformed video ranges can fall through to a full response.** `/api/video` returns a whole-file `FileResponse` whenever range parsing yields `None`, including unsatisfiable ranges. Valid partial reads are capped at 4 MiB and buffered per request. Consider returning `416` for unsatisfiable ranges and explicitly deciding how malformed or multi-range headers should behave.

## Changes in this worktree

- `jobs.py` lazily imports indexer and vision callables so status/progress imports do not load the inference stack, while retaining the existing module-level patch points. Batched embedding now routes through `db.collection(model_name)` and passes caption-source identity and the exact embedding text into the fingerprint payload.
- `scanner.py` propagates failed catalog saves.
- `backup.py` no longer records success per root; `jobs.py` records one success only after a complete backup.
- `imaging.py` downsamples JPEG drafts before full decode when possible, serializes same-output generation with a reference-counted keyed lock, and writes WebP derivatives to a same-directory temporary file before atomic replacement. Added synthetic orientation, concurrent-request, and write-failure cases.
- Test collection now points default app storage at a temporary directory and disables `.env` loading. Explicit `.env` parser tests opt into their own temporary file. Teardown closes test-owned Chroma and SQLite clients before removing the temporary directory.

## Coverage and verification

Existing tests exercise job stop/resource behavior with fake indexers, scanner behavior with temporary folders and a temporary catalog, ingest with temporary staging/library trees, video parsing with mocked ffmpeg, backup with mocked robocopy or synthetic trees, and API video ranges with temporary files. Provider HTTP calls are mocked in their unit tests. No personal `data/`, settings, `.env`, or media was opened; no live scan, indexing, backup, inference, or provider request was run.

The isolated backend suite last completed with **511 passed, 2 failed, 1 skipped**. The failures are in concurrent API work outside this audit: `test_search_unavailable_returns_503` and `test_faces_name_stale_cluster_returns_409`. An earlier isolated run passed 513 tests but failed during Windows temporary-directory cleanup because open Chroma/SQLite clients held files; teardown now closes those clients. The primary agent requested pausing further suite runs while integration proceeds, so the latest source and test edits still need the shared serialized validation pass.

## Remaining bounded work

Make ingest copies atomic and define the expected Stop behavior for a long per-file copy or first-run video hash refresh. Return standards-appropriate status for unsatisfiable video ranges. After the shared fingerprint changes settle, verify stale-vector repair across active embedding models and compare isolated cold import/browse behavior against the pre-revamp baseline. This audit did not measure live-library performance.
