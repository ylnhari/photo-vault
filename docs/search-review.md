# Independent search implementation review

Final source review of the Browse search expansion. I reviewed the selected
contract, backend projection/API, saved-search helper, and Browse UI. No
personal data, media, `.env`, browser state, or provider calls were used. The
only authored file is this report.

## Findings closed

- **Stale caption history:** scoped search fields now use the current caption
  and attributes only. Historical captions are not projected into Browse
  search or `match_fields` ([src/catalog_db.py:343](../src/catalog_db.py),
  [src/catalog_db.py:380](../src/catalog_db.py)).
- **Presence semantics:** “no” is tied to stored recognized text, a valid
  stored GPS pair, or a current caption. The OCR presence flag now uses the
  same effective OCR value as text search ([src/catalog_db.py:365](../src/catalog_db.py),
  [src/catalog_db.py:393](../src/catalog_db.py)). UI labels describe saved
  information rather than claims about image contents
  ([web/src/lib/SearchTab.svelte:582](../web/src/lib/SearchTab.svelte)).
- **Query lexer and phrase/exclusion behavior:** backend and frontend both
  reject empty/unmatched phrases, inline quotes, missing exclusion terms, and
  over 20 terms. The backend tokenizes NFC/casefold text and uses bound `instr`
  predicates, so SQL wildcard punctuation is literal. Positive terms are ANDed
  across scoped fields; exclusions are applied to every scoped field. Match
  reasons come from those same scoped fields and contain labels only
  ([src/catalog_db.py:492](../src/catalog_db.py),
  [web/src/lib/browseSearch.js:27](../web/src/lib/browseSearch.js)).
- **Date validity, fallback, and sort:** the projection tries valid EXIF, then
  a valid filename date, then the imported timestamp; malformed candidates do
  not set the projected year. Range and month/day filters use normalized
  calendar dates, while capture sort preserves time precision and places
  unknown dates last for newest and oldest. Stable ID tie-breaks are present
  ([src/catalog_db.py:279](../src/catalog_db.py),
  [src/catalog_db.py:304](../src/catalog_db.py),
  [src/catalog_db.py:654](../src/catalog_db.py)).
- **Saved-search validation and UI state:** saved entries reject unknown
  fields, invalid query/date/month-day values, oversized payloads, duplicate
  names, and more than 12 entries. Local-storage access is guarded. Browse
  chips and facet options now receive explicit reactive dependencies; template
  call sites use those derived values ([web/src/lib/browseSearch.js:63](../web/src/lib/browseSearch.js),
  [web/src/lib/browseSearch.js:92](../web/src/lib/browseSearch.js),
  [web/src/lib/SearchTab.svelte:292](../web/src/lib/SearchTab.svelte),
  [web/src/lib/SearchTab.svelte:559](../web/src/lib/SearchTab.svelte)).
- **Facet isolation:** facet values/counts are computed from the SQL projection,
  use normalized latest catalog values, are independent of vector-model state,
  and have a deterministic 100-value limit with truncation flags
  ([src/catalog_db.py:689](../src/catalog_db.py)).

## Residual limitations

- **Low priority — future filename years:** the shared filename-date parser
  only recognizes 2000–2029 ([src/photo_date.py:8](../src/photo_date.py)). A
  valid `2030...` filename date therefore falls back to import time. The
  primary chose to leave this unchanged in this batch because widening the
  shared resolver may affect legacy date/fingerprint behavior. Extend it with
  separate compatibility review before that date range becomes relevant.
- The saved-search validator accepts the string `"0000"` as a year, and the
  library API currently accepts any year string up to seven characters. Such a
  crafted saved entry returns no matches rather than being rejected. This is
  non-blocking because `0000` is not selectable in the UI and does not create
  false matches or unsafe SQL. If strict year validation is desired, constrain
  both to `Unknown` or a real four-digit year.

## Verification

I ran synthetic focused tests before the final OCR-presence and Browse
reactivity corrections: backend browse/library tests passed **24/24**, and
frontend Node tests passed **8/8**. The primary reported a further focused
backend run of **25 passed** after the final fixes. I did not launch the app;
the primary owns rendered browser verification. No tests exercised personal
library contents or real inference.
