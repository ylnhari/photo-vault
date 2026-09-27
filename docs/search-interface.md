# Local search interface

## Browse search

Browse remains the initial, provider-independent library view. Its query supports up to 500 characters, 20 whole-word terms, quoted adjacent phrases, and leading-minus exclusions; the interface catches malformed quotes, punctuation placement, impossible dates, reversed ranges, and overlong queries before sending a request. Search scope choices are Everything, File names, Captions, Objects & activities, Text in photos, Places, and Camera info. A separate collapsible filter panel contains inclusive date bounds, month/day matching across years, six catalog facets, stored-presence flags, and sort order. Facet values are fetched only when the panel is opened, with selected values retained in the control even when a bounded option list omits them.

Quick find actions cover recognized text, screenshots, missing GPS, missing captions, and On this day. Browse filters stay separate from Smart search. Active constraints can be removed individually or cleared together. Cards show a compact readable match-field label and expose the complete reason to keyboard and pointer users.

Named Browse searches are written only after an explicit Save action to this browser's local storage. The versioned record is limited to 12 entries and 64,000 serialized characters; names, queries, dates, enums, facet lengths, and allowed keys are validated when reading and writing. Invalid/corrupt/unavailable storage is handled without breaking the Library. Duplicate case-insensitive names are collapsed on load. No query history is recorded.

Smart search remains explicit. Registered-person suggestions are fetched only after entering Smart search and remain available if that optional request fails. Search request cancellation and latest-request guards are retained for Browse and Smart modes.

## Verification

- `cd web && npm test`: **8 passed** using Node's built-in runner and synthetic values. Coverage includes API query/signal forwarding, Browse request parameters, query grammar, Gregorian date validation, saved-search schema, corruption/size limits, and blocked local storage.
- `npm run build -- --outDir .build-check-search-ux-final`: **passed** into a temporary output directory. The production `web/dist` was not rebuilt because the primary agent is coordinating backend integration and will perform the final build/restart.
- The build reports pre-existing unused CSS selectors in `IndexTab.svelte`; no new accessibility compiler warnings remain in the changed Library and PhotoGrid components.
- Temporary emitted assets report shell 30.57 kB JS, Library 55.56 kB JS, PhotoGrid 16.48 kB JS, request gate 0.22 kB JS; their first-view CSS is 7.67 + 17.00 + 3.17 kB. Map and Manage remain lazy chunks. These are emitted-file sizes, not browser timing measurements.
- No personal library, media, provider, indexing, runtime browser or mobile session was used by this worker. Rendered desktop/mobile integration checks remain with the primary agent.
