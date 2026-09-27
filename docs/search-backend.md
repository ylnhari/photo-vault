# Browse search backend

`GET /api/library` continues to use the SQLite catalog projection for cards and
now evaluates all Browse filters and text queries in SQL. The projection is
rebuildable: the JSON catalog row remains authoritative, and migration never
rewrites that payload. Projection version 3 includes normalized per-field search
tokens, latest-caption facet values, GPS location presence, caption/text
presence, and valid capture-date columns. A single `BEGIN IMMEDIATE` migration
adds missing columns, rebuilds all projection rows, creates indexes, and writes
the version marker. Any failure rolls back the schema/data change so a later
request can retry.

The searchable groups are `filename`, current `caption`, current visual
`details`, recognized `text`, `place`, and `camera`. The default `all` scope
searches each group. It never indexes `caption_history`. Caption attributes may
be stored as a JSON string or a dictionary. Text is normalized to Unicode NFC
and case-folded, then stored as space-delimited Unicode word/punctuation
tokens. Unquoted query terms match complete token sequences, so `cat` does not
match `vacation`; a quoted phrase matches adjacent tokens. Punctuation remains
literal as tokens (`DMC-GX8` searches for that punctuation sequence). Positive
terms are ANDed across the selected scope; each `-term` or `-"phrase"`
excludes a matching item. Queries reject unmatched/misplaced quotes, empty
phrases, a bare exclusion marker, more than 20 terms, or more than 500
characters. All terms and filter values are bound SQL parameters.

`date_from` and `date_to` accept inclusive `YYYY-MM-DD` values. The projection
validates real calendar dates, tries EXIF, filename, then import timestamp, and
stores both a calendar date and a timestamp for correct within-day ordering.
Date shortcuts use the captured calendar month/day across years. Malformed date
values fall through to the next valid source. The legacy displayed `date`
retains the chosen valid source string (EXIF or ISO); shared `photo_date.resolve_photo_date` and its
embedding-fingerprint behavior are unchanged. Unknown capture dates remain
last in both capture-date sorts. `added` and `filename` sorts have stable ID
tie-breaks.

The exact facets are `photo_type`, `scene`, `weather`, `occasion`,
`time_of_day`, and `camera`. Caption facets come from the current caption only;
camera is assembled from EXIF make/model without repeating an already-prefixed
make. Facet values collapse whitespace and case-fold to avoid equivalent
duplicates. `/api/library/facets` returns global counts and at most 100 values
per facet with a truncation flag. Filters compare against the same canonical
value and are not silently approximated.

Presence flags describe stored metadata. `has_text` uses the same effective
non-placeholder recognized text as text search: current `text_in_image`, then
the stored metadata OCR fields if needed. `has_caption` uses the current non-placeholder caption; and
`has_location` means a valid stored GPS coordinate pair. “No” means the
corresponding usable stored value is absent. Partial or invalid GPS coordinates
and textual place labels without coordinates match missing GPS. These flags make no claim about what
the original image contains. GPS values outside legal latitude/longitude
ranges are projected as null, so map results cannot expose malformed points.

Text-query cards receive `match_fields` containing only the names of searchable
groups that matched positive terms. Exclusion-only queries return an empty
list. Cards never expose paths or caption histories. Pages remain bounded to
500 results, totals use the same predicates as the page, and browse/facets do
not initialize Chroma, AI models, or external providers.

The final optimization adds expression indexes matching the exact capture-date
sort order, and a covering index for summary counts. Empty-text Browse reads do
not select the large normalized search fields. Facet values are already
canonical, so grouping does not repeat string cleanup on every row.

When SQLite supports FTS5, an external-content inverted index narrows candidate
rows for positive ASCII alphanumeric query tokens. It indexes the same normalized
fields, and every existing exact phrase, punctuation, exclusion and scope
predicate still applies afterward. Unicode-only, punctuation-only and exclusion-
only queries use exact scans. Projection insert/update/delete triggers keep the
inverted index consistent inside the same transaction. A failed migration rolls
back its schema, rows, index and version together.

Fresh catalogs on SQLite builds without FTS5 use the exact scan fallback.
Moving an existing FTS-backed catalog to a SQLite build without that module is
not supported; retain an FTS-capable Python/SQLite runtime for those catalogs.
The code does not modify SQLite's internal schema to bypass an unavailable module.
