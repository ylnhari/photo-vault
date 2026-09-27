# Search expansion

## Scope

Extend the current Photo Vault search experience with useful ways to find items
from existing catalog metadata. Preserve the prior uncommitted revamp. Do not
reset data, run personal inference, use paid services, commit, or push. All
delegated work is source-only with synthetic fixtures and blocked provider calls.
Do not read `.env`, `data/`, personal media, or host-local account configuration.

The primary owns feature selection, contracts, integration, migration and live
checks. Existing native `gpt-6-luna` workers are the currently verified smaller
model route; their relevant source context can be reused. External runner
catalog entries are historical and not used for private application data.

## Work queue

| Unit | Authority | Status |
| --- | --- | --- |
| Search audit | Read source; isolated adversarial tests and audit report | Complete |
| Backend/filter implementation | Catalog projection, library API, tests | Complete; integrated presence corrections |
| Interface implementation | Browse controls, saved searches, frontend tests | Complete; integrated reactive and phone fixes |
| Independent review | Read final diff and synthetic tests | Complete; main findings closed |
| Primary | Decisions, docs, integration, isolated browser and live checks | Complete |

## Acceptance

New local search/filter features must work without providers or Chroma. Keep
queries parameterized, validate filter values, bound result pages and facet
responses, and keep pagination totals consistent with filtering. Search must
not silently drop terms or constraints. Migrate the projection transactionally
without modifying original catalog records. Preserve legacy semantic search.

Use the project `.venv/Scripts/python.exe`, existing isolated test fixtures, and
mocked inference. Build the SPA and verify synthetic desktop/mobile interaction
before restarting the user's app. Record final features, checks, measurements,
and limitations here. After two failed recovery attempts, report a precise
blocker instead of improvising around the scope.

## Selected feature contract

Extend `GET /api/library` without changing existing defaults. New parameters:

- `text_scope`: `all` (default), `filename`, `caption`, `details`, `text`,
  `place`, or `camera`. Details include recognized objects, activities, food,
  animals, colors, and the remaining current visual attributes.
- `q`: up to 500 characters; whitespace-separated complete normalized words are
  ANDed, quoted phrases match adjacent words, and a leading minus excludes a
  word or phrase. Excluding `cat` must not exclude `vacation`.
  Reject malformed quotes or more than 20 terms instead of dropping constraints.
  Unicode and literal punctuation stay safe through bound SQL parameters.
- `date_from` and `date_to`: inclusive ISO dates; reject reversed ranges.
- `month`: 1–12; `day`: 1–31 and requires month. This supports an “On this day”
  shortcut across years. Reject impossible month/day combinations.
- `photo_type`, `scene`, `weather`, `occasion`, `time_of_day`, `camera`: exact
  catalog facet values; use the latest stored caption attributes only. Camera
  labels combine make/model without repeating an already-prefixed make.
- `has_text`, `has_location`, `has_caption`: `yes` or `no`; no empty flags sent.
  These check nonplaceholder recognized text, a valid stored GPS coordinate
  pair, and the current nonplaceholder caption respectively. `no` means the
  corresponding data is absent, not a negative observation about the photo.
- `sort`: `newest` (default), `oldest`, `added`, `filename`. Keep unknown capture
  dates last in both capture-date directions, and include stable ID tie-breaks.

Add `GET /api/library/facets`: global data-driven value/count options for the
six exact facets above, bounded to 100 values per facet, excluding blank and
unknown placeholders. Return `{facets: {field: [{value, count}, ...]}}` and
optional explicit truncation metadata. Fetch only when Browse filters open.

Store field-specific normalized searchable strings in the SQLite projection;
`all` includes filename, latest caption plus current visual attributes, OCR,
place/landmark, and camera metadata. Historical captions must not create stale
matches. Empty or absent metadata must not imply a known negative observation.
Use “Has recognized text” and “Has caption” labels; missing means no stored data,
not a claim about the actual photo. No new inference or geocoding.

Add `match_fields` to a text-query card, a short list from `filename`, `caption`,
`details`, `text`, `place`, `camera` that matched positive query tokens. Display
readable reasons (for example “Text in photo”) without exposing paths or old
caption histories. Exclusions alone have no positive match reason.

The interface keeps Browse and Smart search explicit. These new filters apply
to Browse only; never silently carry them into Smart search where unsupported.
Show active removable chips and a clear-all action. Offer quick shortcuts for
recognized text, screenshots, missing GPS location, missing captions, and the
same month/day across all years. Save named Browse query/filter/sort combinations explicitly
in this browser's local storage (max 12, validated bounded data, no automatic
query history); users can restore or remove them. Explain their local scope.
The existing Smart person field gets registered-name suggestions, loaded only
when Smart search is opened. Merely opening a screen does not run inference.

Source ownership: backend owns `src/catalog_db.py`, `/api/library*` in
`src/api.py`, backend tests, and `docs/search-backend.md`. Interface owns
`web/src/lib/SearchTab.svelte`, `PhotoGrid.svelte`, `api.js`, new pure helper and
frontend tests, and `docs/search-interface.md`. Root owns this document, other
docs, integration, runtime and browser checks. Audit/review are read-only.

## Delivered and verified on 2026-09-26

The final production SPA was built and verified with the existing local library
on its registered loopback port. No media was reset, deleted or sent
through another inference pass. A pre-search SQLite backup passed its integrity
check. The final migration rebuilds only derived projections and search indexes.

- Full backend suite: **597 passed, 1 skipped**. The skip is an unsupported
  Windows symlink case. Four existing dependency deprecation warnings remain.
- Frontend tests: **8 passed**. Production build succeeded; existing unused
  Manage CSS warnings remain. Initial JavaScript chunks total approximately
  103 KB uncompressed with the expanded controls; secondary screens stay lazy.
- Fifteen independent synthetic cases passed through the live API, including
  scoped OCR/place/camera search, history isolation, exact exclusions, date
  ranges, anniversaries, screenshots and stored-data presence.
- Rendered desktop (1280 × 900) and phone (390 × 844) checks passed with no
  horizontal overflow. Verified phrase and exclusion results, loaded facet
  counts, match explanations, active chips and clearing, date range reset by
  On this day, save/restore/remove, malformed-query errors and the synthetic
  registered-person suggestion. No browser console errors were captured.
  The temporary saved search was removed and the preview tab closed.
- Live checks caught and fixed missing reactive dependencies in chips/facet
  choices, an overlapping phone filter counter, and presence-filter mismatches.
  Independent source reviews cover the final contract and index optimization.

The larger search projection initially regressed Browse to 205–240 ms and the
tested phrase query to 670–711 ms on the existing local catalog. Exact-sort indexes,
covering summary reads and an optional FTS5 candidate index corrected this.
Final observed HTTP times across three calls were **9–35 ms for Browse**,
**37–59 ms for the phrase query**, **93–95 ms for global facets**, and
**28–33 ms for summary**. All returned 200. The final one-time projection/index
migration plus first page took **19.23 seconds**. These are observations on this
machine, not a guarantee for every query or dataset. A separate 25,000-record
synthetic check measured warm Browse around 0.6 ms and phrase search around
2.8–3.4 ms inside Python (without HTTP).

## Limits and local artifacts

Recognition quality depends on existing captions and metadata; this change does
not evaluate or claim improved model recognition accuracy. Exact Browse uses
complete normalized tokens rather than stemming, synonyms or substring search.
Smart search remains the route for semantic descriptions. Unicode-only,
punctuation-only and exclusion-only queries retain exact scans and can be slower
than queries using the ASCII candidate index. Saved searches stay in one browser.

The shared filename-date parser currently recognizes 2000–2029; other valid
metadata dates are supported. A manually crafted saved year of `0000` can return
an empty result; that year is not offered in the normal UI. Fresh SQLite builds
without FTS5 can use scan-based search, but moving an existing FTS-backed catalog
requires an FTS-capable runtime. See the independent review for details.

Shell policy blocked removal of the two temporary worker build directories
`web/.build-check-search-ux/` and `web/.build-check-search-ux-final/`. They remain
untracked and are not served by the app. Synthetic media stayed in an external
temporary directory; runtime evidence stays in ignored `data/revamp-runtime/`.
No commit or push was performed.
