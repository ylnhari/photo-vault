# Backend responsiveness audit

## Findings

1. **The API import eagerly loaded the optional inference stack.** `api.py`
   imported `indexer`, `search`, `vision`, `embeddings`, `tagger`, `faces`,
   `validator`, `jobs`, and `clustering`. `indexer` imported vision, embedding,
   face, and scanner modules; `faces` imported InsightFace, ONNX Runtime, and
   OpenCV; `clustering` imported scikit-learn and `faces`; `db` imported
   ChromaDB and embeddings. This made a browse-only launch depend on packages
   that the initial gallery does not use. The primary measured isolated API
   import at **17.272 s before** and **1.949 s after** the lazy boundary change.
   A separate import probe found none of `insightface`, `onnxruntime`, `cv2`,
   `sklearn`, `chromadb`, `openvino`, `vision`, `embeddings`, `faces`,
   `clustering`, `jobs`, `indexer`, or `search` loaded by `import api`.

2. **Browse requests hydrated the entire JSON catalog into Python.**
   `load_catalog_cached()` read every row and parsed every JSON payload. The
   first cache fill feeds `/api/recent`, `/api/map`, `/api/timeline/summary`,
   `/api/timeline`, `/api/meta`, and several management views. Timeline then
   groups and sorts the whole catalog for a page; the old recent path sorted all
   rows before fetching a small set of vector metadata.

3. **Several first-screen checks did unbounded work.** `/api/status` builds an
   `Indexer`, queries embedding/face state, and computes missing and pending
   counts. `/api/search` computes a cached missing-file set by statting the
   catalog paths. Those are useful management/search signals, but they should
   not gate a catalog gallery.

4. **Serving one image could open ChromaDB first.** `/api/image` tried a vector
   metadata lookup before its catalog fallback. The catalog is already the
   authority for scanned media paths, so this point lookup need not initialize
   the vector database.

5. **The catalog DB had no browse projection.** `images` had only an indexed
   primary key and stored each record as opaque JSON. Pagination, year/media
   filtering, and timeline ordering therefore happened after loading all rows.
   Its in-memory write version also only observes writes in the current
   process; the app's normal single-server process is covered, but separate
   processes can leave the old `indexer` snapshot stale.

## Changes in this work

- Kept API route contracts and installed lazy module/function facades for
  inference, vector search, jobs, face, and clustering operations. Browse-only
  imports and image-path lookup do not initialize those dependencies. `/api/meta`
  now uses catalog metadata and saved caption JSON first, with a vector-store
  fallback only when the catalog row is absent.
- Added a shared, dependency-light capture-date resolver in
  `src/photo_date.py` for EXIF date, filename date, and created-time fallback.
- Added a versioned SQLite `library_projection` with indexed id, media type,
  year, date, recency, and GPS fields. Catalog writes update the JSON source and
  projection together; full scans sync stale rows; deletes remove both. The
  one-time legacy projection rebuild is in one SQLite transaction and rolls
  back completely on malformed data or interruption. Rows are prepared before
  writes, and failed mutations roll back instead of leaving a partial batch.
- Added model-independent `GET /api/library` and
  `GET /api/library/summary`. The page is bounded, supports literal lexical
  matching across filenames and saved captions/attributes, plus media/year
  filters, and returns PhotoGrid-compatible cards. It marks file presence
  optimistically to avoid blocking on disconnected drives; explicit missing
  file cleanup/status retains its existing check.
- Switched media path resolution to the SQLite primary-key point lookup before
  any compatibility fallback to ChromaDB.
- Moved timeline year/month summaries, grouped timeline pages, and map points to
  SQL projection queries. These routes preserve their response shape while
  avoiding full-catalog hydration and per-file existence checks.
- Status now counts selected-model pending embeddings by current fingerprints,
  so stale vectors remain visible as pending even when raw collection count
  equals eligible caption count. Index job starts also request background
  preparation, with job registration serialized against backup destination
  changes; the destination cannot change while a backup is active.
- Catalog-first lightbox metadata includes the resolved year/month and caption
  model without opening ChromaDB.

## Verification and limits

- Synthetic catalog/API and targeted legacy regressions:
  `tests/test_catalog_db.py`, `tests/test_library_api.py`, and the search,
  cluster-name, image-serving, timeline, map, status, settings, and background
  preparation API cases — **30 passed** with the project `.venv` runtime.
- Synthetic 25,000-row benchmark on this Windows host: API import **0.531 s**;
  seed **0.890 s**; whole-catalog load + sort **0.140 s**; first SQL page of 60
  **0.023 s**; warm page **0.009 s**; lexical browse **0.016 s**; summary
  **0.015 s**. These are local observations. The benchmark seeds the projection
  before timing the first page, so it does not measure one-time legacy backfill.
- The subprocess import test verifies that API import omits the optional ML and
  vector modules while data and dotenv paths point to isolated temporary state.
- The first request against a pre-projection catalog performs a synchronous,
  atomic one-time JSON backfill. Steady-state page/summary reads are SQL-bounded;
  the migration's first-use latency still needs measurement on a synthetic
  catalog at the target library size.
- The benchmark reached all timing checks but its process exited nonzero while
  Windows removed the temporary directory: the test client left the cached
  SQLite connection open. The harness also needs `src` on `PYTHONPATH` for its
  early catalog import. The primary agent owns that harness cleanup/invocation
  adjustment.
- `/api/status`, semantic search, and filters still perform their established
  broader work. The library, timeline, map, and catalog-first lightbox paths
  avoid the inference/vector stack. Lexical text matching uses bounded parameterized
  substring checks and can scan matching catalog rows; it is not a semantic
  search replacement. Gallery `exists` is intentionally optimistic to avoid
  filesystem stalls.

## Final integration update

The synthetic benchmark harness was subsequently repaired to close cached SQLite
connections and completed successfully, including one-time legacy projection
migration. See `revamp-benchmarks.md` for those final results and `REVAMP.md` for
the live original-library measurements. These supersede the earlier harness
limitations recorded above. The integrated backend suite finished with 567 passed
and one unsupported-symlink skip. The final production app is running locally.
