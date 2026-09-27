# Responsiveness revamp benchmark

`tests/benchmark_revamp.py` is a development-only measurement harness. Run it
from the repository root with:

```powershell
.venv/Scripts/python.exe tests/benchmark_revamp.py
```

It creates 25,000 deterministic synthetic catalog records with captions, date
and year fields, EXIF-like metadata, and a 20% video mix. It stores them under a
temporary `PHOTO_VAULT_DATA_DIR`, sets `PHOTO_VAULT_ENV_FILE=-`, and bypasses
the sibling port registry. Thumbnail timings use a small locally generated
geometric image. The harness makes no provider or network calls and does not
read the real catalog, `.env`, or personal media. Temporary data is removed
when it exits.

The output measures subprocess cold import of `api`, synthetic catalog seeding,
whole-catalog load and date sorting as a baseline, first-time projection
migration from a legacy SQLite catalog, warm direct-store and bounded HTTP
pages, lexical caption search, library summary, and generated thumbnail
creation/cache-hit time. API timings are ordinary local observations, not
assertions, service-level targets, or performance promises. The script stops
with a clear error if the library API is unavailable or returns an unexpected
response shape.

## Available baseline evidence

Before the library endpoint changes, the primary agent measured cold `import
api` at **17.272 seconds**. The built frontend assets measured **431,759 bytes
of JavaScript** and **48,530 bytes of CSS**. Those asset sizes are context only;
this harness does not rebuild or measure the frontend. The import measurements
are historical observations rather than a controlled before/after comparison.
Do not compare timings from different machines or environments as a guaranteed
speedup.

## Results

One post-change run completed on 2026-09-26 with the repository virtual
environment (`.venv/Scripts/python.exe`). The command exited 0 after closing
its temporary SQLite connection. Starlette emitted a non-fatal deprecation
warning about its `httpx` test-client integration. Timings are seconds:

```json
{
  "catalog_rows": 25000,
  "catalog_seed_seconds": 0.8458,
  "cold_import_api_seconds": 0.5154,
  "cold_subprocess_wall_seconds": 0.6389,
  "legacy_catalog_seed_seconds": 0.1796,
  "legacy_first_page_with_projection_migration_seconds": 0.8104,
  "legacy_migration_catalog_total": 25000,
  "legacy_migration_page_rows": 60,
  "legacy_warm_direct_store_page_seconds": 0.000155,
  "library_first_page_rows": 60,
  "library_first_page_seconds": 0.0145,
  "library_lexical_search_seconds": 0.0135,
  "library_lexical_search_total": 25000,
  "library_summary_seconds": 0.0131,
  "library_summary_total": 25000,
  "library_warm_page_seconds": 0.0050,
  "synthetic_thumbnail_cache_hit_seconds": 0.000061,
  "synthetic_thumbnail_generate_seconds": 0.0999,
  "whole_catalog_load_date_sort_seconds": 0.1258
}
```

The first page returned exactly 60 rows. The generated catalog had a 20%
video mix and synthetic captions and date/EXIF-like metadata. The earlier
17.272-second import measurement above is retained as historical context; it is
not treated as a controlled before/after comparison. Keep generated row count
and query parameters identical for future comparisons. Actual token spend is
unavailable and is not estimated here.

The migration timing uses a separate SQLite file seeded with only the legacy
`images(id, data_json)` table; no projection is prebuilt. Its first direct
`catalog_db.library_page` call builds the projection and returns 60 of 25,000
rows. The warm direct-store time measures a subsequent page query on that
projection. These migration timings are separate from the HTTP page timings.
