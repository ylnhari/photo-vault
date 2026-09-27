# Photo Vault responsiveness and retrieval revamp

## Scope and authority

The user authorized a comprehensive app audit and improvements to architecture,
design, indexing, embeddings, and usability. Generated indexes may be rebuilt if
necessary; original photos, videos, backups, and personal settings are preserved.
No commits, publication, paid inference, or external transmission of personal media
are authorized by this work. Workers inspect source and synthetic fixtures only.

## Execution plan

The primary agent owns architecture decisions, integration, data operations, and
live verification. Bounded source audits and implementation use the available
`gpt-6-luna` worker model with fresh context to reduce cost. The shared runner
catalog was checked: its free external runners have historical verification only;
native workers provide an explicit current model contract, constrained source-only
tasks, and inspectable results without separate credential or gateway handling.
Actual token spend is unavailable; no cost savings percentage is claimed.

| Work unit | Scope | Delivery status |
| --- | --- | --- |
| Interface | `web/src/`, frontend tests, package test script | Complete |
| Retrieval | `embeddings.py`, `search.py`, `db.py`, `vision.py`, corresponding tests | Complete |
| Backend | `api.py`, `catalog_db.py`, `photo_date.py`, read-path tests | Complete |
| Reliability | Jobs, scanner, imaging, backup, corresponding tests and test isolation | Complete |
| Indexing | `indexer.py`, embedding freshness tests | Complete |
| Benchmarks | `tests/benchmark_revamp.py`, synthetic measurements | Complete |
| Primary | Runtime isolation/configuration, docs, integration and live verification | Complete; running and verified |

Exclusive source scopes above supersede the initial read-only audit instruction
for their assigned workers. Cross-file contracts are coordinated by the primary;
workers do not edit another unit's files.

## Architecture decisions

1. Keep SQLite catalog JSON as the durable source and add an indexed, incremental
   read projection. Ordinary Library browsing, counts, and thumbnail path lookup
   must not load Chroma, face recognition, or providers.
2. Load the shell and first Library page first. Other screens and heavy JavaScript
   load on demand. Detailed indexing status and provider health are management
   operations, not startup dependencies.
3. Keep current vector collections compatible. Newly registered embedding models
   use explicit retrieval profiles and cosine distance. Legacy profiles retain
   their original preprocessing and collection; no silent vector-space mixing.
4. Persist deterministic embedding-content fingerprints, so re-captioned items
   become pending again instead of remaining silently stale. Constrain semantic
   retrieval by the selected person before ranking the limited result set.
5. Preserve original media and personal configuration. Automated tests and
   benchmarks use temporary stores and synthetic media. Provider mocks are
   backed by a socket-denial fixture and blank inherited Gemini credentials.

## Worker instructions

Read this file and repository `AGENTS.md` first. Do not read `.env`, `data/`,
personal media, browser profiles, or host-local configuration. Use synthetic data
and mocked providers. Do not run real indexing or external inference. Never print
secrets or personal paths. Read-only audit is the initial authority; the primary
agent assigns exclusive write scopes before implementation. Do not change another
worker's files, install dependencies, change branches, commit, or push.

Leave a concise report in your assigned `docs/revamp-*.md` file with evidence,
prioritized findings, proposed fixes, tests, and limitations. Report blockers
honestly; after two failed recoveries, ask the primary agent for direction.

## Acceptance criteria

- Measure cold startup and representative browse/search behavior with synthetic
  fixtures; record comparable before/after results where practical.
- The initial screen and catalog browsing must not wait for AI availability.
- Bound expensive work, request payloads, and gallery rendering; cancel or ignore
  stale requests and preserve useful UI state.
- Keep embedding model spaces isolated, validate provider output, and expose
  degraded search behavior honestly rather than silently returning wrong results.
- Run focused regression tests, the full backend suite, and the production web
  build. Independently review changes and verify rendered behavior on loopback.
- Record verified delivery and remaining limitations here before completion.

## Verification on 2026-09-26

- Final integrated backend suite: **567 passed, 1 skipped** in 16.73 seconds.
  The skip was an unsupported symlink case on this Windows environment. Provider
  traffic is blocked by the shared test fixture; fresh-process cold-route tests
  also deny HTTP provider calls explicitly.
- Frontend Node tests: 3 passed. Production build succeeded. Existing unused
  CSS selector warnings remain in Manage; there were no build errors.
- A separate worker independently reviewed backend, retrieval, fingerprint,
  thumbnail, lazy-model, and backup changes. The two reported findings were
  corrected and rechecked; the root-map freshness race was also closed.
- Synthetic browser checks covered Library query/year filtering, Load more,
  virtualized tiles, retained filters after Timeline navigation, timeline date
  grouping, Albums/People loading, Lightbox metadata, arrow navigation, Escape,
  and restored focus. A subsequent fresh live People check reproduced a
  concurrent NumPy first-import failure; shared reentrant import initialization
  fixed it. All three simultaneous People routes now return 200, and People
  rendered its complete empty state without console errors. Rendered phone (390px) and desktop (1280px) views had no
  horizontal overflow; the phone Library rendered 22 tiles with no broken
  images. A runtime gallery-state scope error was caught live and fixed.
- First-view JavaScript is approximately 72.3 KB uncompressed instead of the
  previous single 431.8 KB bundle. Map, Manage, and other secondary views load
  separately. See `revamp-interface.md` and `revamp-benchmarks.md` for measurements.
- The 25,000-record synthetic benchmark measured bounded first-page reads around
  15 ms, warm pages around 5 ms, and first-run projection migration around
  810 ms. Times are observations, not guarantees for every disk or library.

## Test isolation incident and remaining limits

One missing batch-function mock caused one unintended Gemini embedding request
using synthetic test strings. No personal photo or private content was sent.
The mock was fixed, inherited Gemini credentials are now cleared before imports,
and the suite blocks provider network connections (including local services).
Windows asyncio's standard-library control socketpair is permitted narrowly.

Recognition accuracy has not been measured against a labeled personal corpus.
The work corrects retrieval/profile/freshness mechanics, but does not claim a
measured relevance improvement or run personal media through inference. Existing
legacy embedding profiles remain compatible; using new task-aware profiles for
an already registered model requires a deliberate rebuild. Real video playback
and each optional hardware inference backend were not exercised in the browser.
Original media, backups, and personal settings were not removed. No commit or push
was performed.

## Live delivery

The production build was verified against the existing local catalog on the
configured loopback port. Its listener was verified as loopback-only, the server
log had no errors, and no indexing job was running. Opening the app in the user
panel was queued; no personal media was captured into the agent context.

Before first use of the new projection, a SQLite backup of the original catalog
was created under ignored `data/revamp-safety/` and passed `PRAGMA integrity_check`.
The existing local catalog's one-time projection migration and first 60-item page
took **11.205 seconds**. The following 60-item page took **20.0 ms**, library
summary **62.2 ms**, timeline summary **80.1 ms**, and idle job progress **25.7 ms**;
all returned HTTP 200. The real migration was slower than the synthetic benchmark,
which has simpler captions and history. Startup and model-load times remain
hardware/data dependent; there is no claim of instantaneous inference.

Runtime process identifiers and log locations are recorded only in ignored
`data/revamp-runtime/launch.json`. The app was not reset or sent through a new
inference pass. Legacy vectors remain searchable; fingerprint-less or stale
entries are eligible for refresh during the next user-started embedding job.

A live photo thumbnail and its metadata were checked through HTTP without exposing
image bytes, captions, filenames, or identifiers to logs or the agent context.
Results: {"thumbnail": {"status": 200, "milliseconds": 24.1, "nonempty": true}, "metadata": {"status": 200, "milliseconds": 14.9, "nonempty": true}}.

The subsequent search expansion adds exact field scopes, phrase/exclusion syntax,
catalog facets, saved searches and an inverted text index. Its current delivery
checks, bundle growth and performance observations are recorded in
[search improvements](SEARCH-IMPROVEMENTS.md); the measurements above describe
the earlier revamp checkpoint.

## Publication scope — 2026-09-27

After the source, privacy, and path audit, Hari authorized committing, pushing,
and syncing the reviewed source, test, and documentation changes. Earlier
no-commit and no-push statements above record the scope at their original
checkpoints; they do not prohibit this authorized release. Personal media,
catalog data, and host-local configuration remain outside the publication.
