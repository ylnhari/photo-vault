# Search expansion audit

Date: 2026-09-26. Read-only audit of the current catalog, search API, and Search
screen. No private catalog, media, `.env`, host account config, or provider was
accessed. This report recommends a bounded local-first feature batch; it does
not authorize implementation.

## Current searchable information

The SQLite `library_projection` stores `path`, `filename`, `created_at`,
`capture_date`, `year`, `month`, `media_type`, `duration_s`, GPS coordinates,
caption, occasion, and one concatenated `search_text` field
(`src/catalog_db.py:56-67,179-227`). The text includes filename, the current
caption and occasion, top-level `place` if one exists, EXIF `place`/`location`,
camera make/model, and current vision fields for festival, OCR text, landmark,
objects, animals, vehicles, food, activities, scene, weather, season, time of
day, group size, clothing, mood, location type, people description, and colors.
It also appends selected fields from every `caption_history` record. The
projection does not include all EXIF fields such as lens, ISO, exposure,
aperture, focal length, dimensions, or GPS as searchable terms.

The Chroma payload in `src/indexer.py:1139-1188` carries the same generated
caption attributes plus `year`, `month`, `place`, embedding identity, media
type, and duration. Its place is an offline reverse-geocoded city/region derived
from valid GPS (`src/indexer.py:1150`, `src/geocode.py`). That derived place is
stored in Chroma metadata, but the catalog projection does not compute or
persist it; `library_page()` therefore cannot reliably find GPS-only items by
their derived city. Caption JSON contains the current model output, while
`caption_history` retains older model outputs. Captions and their categorical
fields are model-generated, so an object, person count, occasion, or weather
tag is a useful hint rather than verified ground truth.

## Current limitations and correctness risks

1. **The browse search language is very limited.** `/api/library` accepts one
   free-text `q`, media type, and exact year (`src/api.py:947-961`). The SQL
   implementation splits `q` on whitespace, ANDs substring checks over the
   whole combined `search_text`, and silently keeps only the first eight terms
   (`src/catalog_db.py:287-311`). It cannot express an exact phrase, a field
   scope, OR, or exclusions. A quoted phrase is split into separate words, and
   `-beach` is treated as a required literal substring. Because fields are
   concatenated, a multiword query can match across different fields. The
   broad “Search names, captions, or places…” placeholder
   (`web/src/lib/SearchTab.svelte:326`) overstates what “place” means for
   GPS-only records.
2. **Date exists in the catalog, but the browse filters do not use its full
   shape.** The projection already has `capture_date`, `year`, and `month`, plus
   an index on `capture_date` and grouping for year/month counts
   (`src/catalog_db.py:57-67,331-342`). Browse exposes only media type and year
   (`SearchTab.svelte:345-353`; `api.library()` at `src/api.py:947-961`). Month
   is available only as a Smart Search Chroma filter, and there is no start/end
   date range. `resolve_photo_date()` uses EXIF first, then recognized filename
   dates, then the local import/created timestamp (`src/photo_date.py:11-42`).
   The UI does not reveal that fallback source, so a date can look like a
   capture date when it is only an import-time estimate.
3. **Facet availability is coupled to the active vector index.** The Search
   screen loads `/api/filters` only when Smart Search is opened
   (`SearchTab.svelte:154-171,366-384`). `get_available_filter_values()` scans
   all metadata in the active Chroma collection and returns unique values for
   a fixed list (`src/search.py:245-274`). This misses uncatalogued/unembedded
   items and can lag changed captions until vectors are repaired. It returns no
   counts or page-size bound. It omits several already stored useful fields:
   `objects`, `animals`, `vehicles`, `food_items`, `activities`,
   `text_in_image`, `landmark`, `dominant_colors`, and `people_description`.
   The comma-joined Chroma representation for list fields is a poor exact facet:
   a whole combination such as `cup, plate` varies by caption, and equality
   cannot select just `cup`.
4. **Browse and Smart Search are different contracts.** Browse query and
   filters run in SQLite, use stable offset pagination and an exact total, and
   do not need an embedding provider. Smart Search uses Chroma metadata filters
   and a semantic top-k capped at 250 in `src/search.py:25,206-231`; the API
   does not return a total or continuation cursor for those results. A feature
   that mixes date/facets into Smart Search must apply them before ranking and
   define how pagination/counts work; otherwise users can see inconsistent
   results between modes.
5. **The combined browse text is not a proper full-text index.** Each
   `instr(search_text, ?)` predicate is safe from SQL injection because its
   value is bound, but it scans a concatenated text column and cannot use the
   existing date/media indexes for text matching. Caption history is included
   without an explicit “previous caption” scope, so an old caption can retrieve
   an item whose displayed current caption no longer supports that match.

## Recommended coherent first batch

Build the next search increment around the existing SQL catalog. This keeps
filtering available without Chroma or inference, returns exact pagination
totals, and avoids blending two ranking contracts prematurely.

### 1. Calendar search: month and date range

Add optional `date_from`, `date_to`, and `month` parameters to `/api/library`
and its client wrapper; retain `year` for the current dropdown/backward
compatibility. Compose all active constraints in one parameterized SQL `WHERE`
clause before both `COUNT(*)` and page selection. Use inclusive calendar-day
semantics for the UI and a normalized sortable date key in the projection, with
an index that supports the common date/media query. A month selector should
show month counts and preserve the selected year/context; date-range controls
are more useful for memories such as “the week around the wedding.” Keep
`Unknown`/undated separate from a false match, and make date provenance
discoverable: EXIF, parsed filename, or import-time fallback. Validate strict
calendar dates and define local-time behavior for unzoned EXIF/timestamps.

### 2. Explicit text scopes, exact phrases, and exclusions

Replace silent whitespace truncation with a small documented query grammar for
the local browse path. Useful scopes can map to current-only `caption:`,
`filename:`, `ocr:`, `place:`, `object:`, `animal:`, `occasion:`, and
`camera:` fields. Support `"quoted phrases"` and `-excluded-term`; define
positive terms as AND, negative terms as NOT, and whether a scope applies to a
following phrase or one token. Parse and validate scopes in the app, quote
literal values for FTS, and never pass raw user syntax into SQL or an FTS query
without escaping. Reject malformed/overlong syntax with a visible validation
message instead of silently dropping extra terms. Default caption scope should
search the current caption. Previous captions can remain an optional explicit
scope if users need historical model results.

The existing `search_text` column is enough for a small-library prototype but
not for field scopes, phrase accuracy, or scalable text lookup. A versioned
SQLite FTS5 projection with separate columns for filename, current caption,
OCR, caption tags, place, and camera text is the natural implementation. Keep
the original `images.data_json` records authoritative; rebuild and upsert the
FTS projection transactionally, and update it on caption edits, deletes, and
renames. If FTS5 is unavailable, return an explicit degraded-mode indication
or use a carefully bounded literal fallback; do not pretend phrase/scope
operators worked.

### 3. Catalog-backed facets with counts

Move filter-value discovery from a full Chroma scan to the SQL projection.
Start with low-cardinality caption fields already in the current catalog:
occasion, weather, season, time of day, scene, photo type, clothing style,
location type, group size, and festival. Add normalized multi-value facets for
objects, animals, activities, food, vehicles, visible text, and landmarks. Use
the exact individual list elements from caption JSON rather than comma-joined
strings. Return bounded `(value, count)` suggestions, with search/autocomplete
for high-cardinality fields such as landmarks and OCR; do not return the full
OCR corpus as a dropdown. Facet counts and result totals should use the same
date/media/text constraints. Prefer counts that account for the other active
filters while omitting that facet's own filter (so selecting one weather value
does not hide the alternatives); document if a simpler global count ships
first. Include unknown as an explicit opt-in and visually label generated
caption facets as detected/estimated.

Support `place` only after the offline reverse-geocoded city/region is
materialized in the catalog projection from GPS at indexing time. Today it is
computed for Chroma only. Avoid importing/building the geocoder during app
startup or a browse request. Camera make/model are already in local
`search_text`; add indexed facets if their use is common. Preserve source
provenance for those values (EXIF versus generated caption versus geocoder) so
the UI can distinguish reliable file metadata from model estimates.

### 4. Saved searches, after the query contract settles

Add local saved searches as a follow-up, not as the first dependency. Persist a
named versioned search specification (parsed query, media/date constraints,
facet values, and mode), not just a raw string or cached result IDs. Restore it
through the same validated API contract so future grammar changes can migrate
or explain an unsupported saved query. Store it in the local catalog database
or another existing private local settings store; no account, cloud sync, or
new external service is needed. Keep saved search distinct from Albums, which
are manually curated static membership.

## Failure modes and testable boundaries

- Date boundaries, leap days, unknown dates, filename-derived dates, and
  import-time fallback dates must have explicit tests. Bad EXIF strings should
  not create a misleading year/month.
- Parser tests should cover escaped quotes, scoped phrases, multiple scopes,
  exclusions, punctuation, Unicode, max terms, empty scopes, and malformed
  syntax. Terms must never disappear silently.
- FTS migrations should prove rollback leaves the prior projection usable;
  upsert/delete/re-caption should keep normalized facets and FTS text in sync.
- Count, `limit`, and `offset` must use identical text/date/media/facet
  predicates, including `EXISTS` for multi-value facets to avoid duplicate
  rows. Parameters, not string-built SQL, must carry user values.
- Generated facet values can be wrong or stale; keep `unknown`, allow clear
  filters, and source facets from current catalog caption data. Do not combine
  old caption-history values with current-caption facets unless history is
  explicitly selected.
- Semantic Smart Search should remain available. If common filters are shared
  with it, constrain Chroma candidates before top-k and surface unsupported
  filters as errors rather than retrying an unfiltered search silently.

## Suggested file scope

- Backend: `src/catalog_db.py` (projection, date/text/facet predicates and
  transactional schema migration), `src/api.py` (validated library params and
  facet endpoint contract), `src/photo_date.py` (strict normalized date and
  source metadata), and optionally a focused `src/search_query.py` parser.
- Interface: `src/api.py` plus `web/src/lib/api.js` and
  `web/src/lib/SearchTab.svelte` for shared date controls, scoped-query hints,
  count-bearing facets, and saved-search UI later.
- Tests: `tests/test_catalog_db.py`, `tests/test_library_api.py`, and focused
  parser/API tests using only synthetic catalog rows. No provider, Chroma, or
  media fixture is required for these local search features.

## Recommendation

Ship calendar search, a small escaped local query grammar, and catalog-backed
bounded facets together as one local Search upgrade. Keep semantic ranking
separate but allow its future candidate set to be filtered by the same
validated catalog constraints. Defer saved searches until the query grammar
and canonical filter specification are stable. This adds recall and precision
without depending on model availability and lets one SQL query determine both
results and totals.
