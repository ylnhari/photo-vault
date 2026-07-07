# Session TODO — 2026-07-04 (overnight, autonomous)

Hari is away until morning and will clear Claude's context. **This file is the
resume point.** Read this file first, check "Status" on each item, and continue
from the first non-done item. Do not re-derive anything already marked done —
verify quickly (re-read the diff / re-run the relevant test) then move on.

## Ground rules for this session
- No git commit without Hari's OK (global policy: commit only when asked). Leave
  changes as uncommitted working-tree edits; list them at the bottom for his review.
- No destructive/external actions without asking: no reverse-geocoding (sends GPS
  coords to a third party — privacy call, not mine to make), no re-running vision
  captioning on already-done photos (would derail the multi-day job in flight).
- Vision captioning job (24507 photos, LM Studio gemma-4-e4b-it) is running in the
  background the whole time. Any backend .py change requires restarting
  `python server.py` to take effect, which interrupts that job's in-process thread —
  it's resumable (catalog persists per-batch), so this is safe, just do it once
  after finishing a batch of backend edits, not after every single edit.
- Run `uv run python -m pytest tests/ -q` after backend changes, before restarting
  the server.

## Task list

### 1. [DONE] Diagnose why LM Studio dropdown shows 3 models when 1 is "live"
Root cause: `list_lm_studio_models()` (vision.py) calls the OpenAI-compat
`/v1/models`, which lists every model LM Studio *knows about* (JIT-loadable),
not what's resident in memory. Classification (`classify_lm_studio_model`) is a
name-pattern guess, not a real capability check.
Fix available: LM Studio's own native endpoint `GET /api/v0/models` (same host,
NOT under `/v1`) returns real `type` ("vlm"/"embeddings"/"llm") and real `state`
("loaded"/"not-loaded") per model. Verified live: gemma-4-e4b-it and the embed
model show `state: loaded`; the other 3 show `not-loaded`. This is the correct
source of truth — see task 5.

### 2. [DONE — not a bug] Gemini dropdown "only 3 models"
Gemini vision list is already fully dynamic (`list_gemini_vision_models`, real API
call, 5-min cache, hardcoded 5-model list is fallback-only when key missing/fetch
fails). Verified live: 39 vision models, 3 embedding models returned right now.
The "only three" Hari saw is the true, current count of Gemini models supporting
`embedContent` — not a limitation in our code. No fix needed.

### 3. [DONE] Gemini quota/rate-limit "greater than 0" check
Hari asked for a quick call to check remaining quota per Gemini model. **Not
literally possible** — Google's Generative Language API has no public
quota-remaining endpoint (quota is Cloud-Console-only). Practical substitute
instead: track 429s per model with a short in-memory cooldown (e.g. skip a model
for N minutes after it 429s), surfaced in `/api/provider-models` so the UI can
grey out / label a model "rate-limited, retry after Ns" instead of just failing
silently. Not yet implemented — do this after task 5/6.

### 4. [DONE] Wire schema validation into the real indexing job loop
`validate_vision_output()` (vision.py) checks the JSON has the 5 required keys
(caption/scene/occasion/weather/group_size) — but it's currently only called from
the `/api/explore` diagnostics endpoint, NOT from the actual vision job path
(`indexer.py: compute_caption` only checks for an explicit `"error"` key via
`_caption_has_error`, so a model that returns valid-but-wrong/empty-schema JSON
silently "succeeds"). Verified safe to enable: pulled 5 real captions already
produced by the currently-running job (gemma-4-e4b-it) via `/api/explore?id=...`
— all 5 pass `validate_vision_output` cleanly, so wiring this in won't cause
false-positive failures against the model in current use.
Change: in `indexer.py: compute_caption`, after the existing `_caption_has_error`
check, also call `validate_vision_output(text)` and raise if `not valid`.

### 5. [DONE] Switch LM Studio model info to the native v0 API (real type + loaded state)
- `vision.py`: add `list_lm_studio_models_v0()` hitting `{host}/api/v0/models`
  (strip `/v1` from `LM_STUDIO_URL` for the host). Returns list of dicts with
  `id`, `type` (vlm/embeddings/llm), `state` (loaded/not-loaded).
- Keep the old name-pattern `classify_lm_studio_model` as a fallback ONLY if the
  v0 endpoint is unreachable (older LM Studio versions might lack it).
- Also fix `_lm_model_id()` (picks `models.data[0].id` blindly to label captions
  when no model is pinned) — should prefer the model the v0 API reports as
  actually `state: loaded` with `type: vlm`, not just index 0.
- `api.py: /api/provider-models` — include state per LM Studio model.
- `web/src/lib/IndexTab.svelte` — show a "loaded" badge, sort loaded-first, in
  the LM Studio dropdowns (currently `lmVisionModels`/`lmEmbedModels` derived
  filters around line 79-86).

### 6. [DONE] Gemini: escalate through fallback models on 429/404/503 even when a specific model is pinned
`vision.py: _call_gemini` — currently `candidates = [model] if model else
GEMINI_VISION_MODELS`, so a pinned model that 429s just fails outright (no
escalation). Change to try the pinned model first, then fall through the
dynamically-fetched vision model list (`list_gemini_vision_models(fallback=True)`)
on 429/404/503, same resilience "auto" mode already has.

### 7. [DONE] Frontend: idle background polling (root cause of tonight's "button still enabled" / "job in queue" confusion)
`IndexTab.svelte: onMount` only starts polling `/api/index/progress` if a job was
ALREADY active when the page loaded (`if (job.active) startPolling()`). If a job
starts from a different tab/session (as happened tonight — Claude started the
vision job from an automation tab while Hari's own tab was open from before),
Hari's tab never learns about it — buttons/counts look stale until manual refresh.
Fix: poll status periodically even when idle (e.g. every 5-10s, lighter weight
than the 1s active-job poll), so cross-tab job state propagates automatically.
This is the actual root cause behind tonight's "why is Embed still clickable"
"why does retry say job is in queue" confusion — verified live the disable logic
itself (`{:else if running}` hint branches) is already correct; it's a staleness
problem, not a broken-disabled-state problem.

### 8. [DONE] Timeline: quick-jump nav (year/month) — "can't scroll through 40k photos"
`TimelineTab.svelte` renders every year fully expanded, sequentially, with a
sticky year header but no jump control — for 40k+ photos across many years this
means scrolling the entire rendered list to reach an old year. No week/day
grouping exists at all (only year+month; week/day would need backend changes to
the `/api/timeline` grouping, bigger scope — flagging as a stretch goal, not doing
tonight). Plan: add a compact sticky picker (year list, expandable to months
once that year's data is loaded) that does `scrollIntoView()` to the matching
section — mirrors the existing `gotoSection()` pattern already used in
IndexTab.svelte for jumping to index sections.

### 9. [DONE] Search: month/date-range filtering ("December trip" style queries)
EXIF `date` is already extracted at scan time (confirmed via `/api/explore` output:
`"exif":{"date":"2025:11:27 14:39:57", ...}`). `get_available_filter_values()`
(search.py) already exposes `year` as a filter but not `month`. Low-cost, no new
dependency: add `month` as a stored/filterable ChromaDB metadata field alongside
`year`, and a Month filter dropdown next to the existing Year dropdown in the
Search tab sidebar. This directly answers Hari's "rainy photos from 2022" /
"December trip" search question for the month half — year already works today
via the filter (just wasn't obvious it was there).

### 10. [NEEDS HARI'S OK — do not do autonomously] Reverse geocoding for place names ("Goa" search)
GPS lat/lon is extracted (`scanner.py`) but never reverse-geocoded to a place
name — the Map tab just plots pins, no text is searchable. Real fix needs an
external call (e.g. OpenStreetMap Nominatim, free, no key, rate-limited to
1 req/sec, cache by rounded lat/lon so it's a few dozen calls total not one per
photo) — but that means sending home/family GPS coordinates to a third-party
service. That's a privacy tradeoff for a personal photo vault, not something to
decide unilaterally overnight. **Left undone — ask Hari in the morning**: OK to
use Nominatim for this, or prefer an offline reverse-geocoding dataset instead
(no network calls, larger download, coarser resolution)?

### 11. [NEEDS HARI'S OK — do not do autonomously] Exact person-count attribute
Hari asked whether the vision LLM could also report exact person count (not just
the current categorical `group_size`: solo/couple/small_group/large_group).
Doing this properly means changing the vision prompt schema and re-running vision
on ALL photos (including the ~500 already captioned tonight) — i.e. doubling the
cost of the multi-day job currently in flight. Not doing this without Hari
explicitly signing off on redoing the in-progress captioning pass.
**Left undone — ask Hari in the morning.**

### 12. [INFO ONLY — not a bug, no fix needed] Progress bar placement / duplicate JobPanel theory
Investigated Hari's complaint that the vision progress bar shows "at the top
instead of near the Caption button." Verified live in browser + read
`IndexTab.svelte`: each section's `JobPanel` is gated by `{#if jobIs(job, "<type>")}`
— it only renders in the section matching the actually-running job type (confirmed:
Vision section shows the full panel with Stop button, replacing the Caption
button in-place; Embed/Thumbnails/other sections show only a one-line
"Another job is running" hint, no panel, no button). The always-visible small
pill in the top-right header is a supplementary global indicator (like a
background-task chip) — intentional, not a duplicate. No code change here.

### 13. [INFO ONLY] "Is this how Google would design it" — fundamental UX take
Given #7 is the real bug, the honest answer: the section-scoped panel + hint-text
pattern is reasonable design for a local single-user tool. The actual gap versus
a polished app (Google Photos et al.) is exactly #7 (state doesn't propagate
across tabs/sessions without polling) and #8 (no quick date navigation for a
large library) — both addressed above, not a "everything is wrong" situation.

### 14. [TODO] Full visual/UX redesign of Index & Manage — "out of this world," Google/Claude-design-critique level
Hari's explicit ask (verbatim priorities): the progress indicator must visually
stay attached to/near the button that started it, never feel like it's "floating
somewhere on the page." Beyond placement, do a full design pass, not just a bug
fix:
- **Button design system**: clear, distinct visual states for enabled / disabled
  (and WHY it's disabled should be legible at a glance, not a separate hint line
  below it) / active-loading / just-succeeded. Rethink shape, size, label
  wording ("▶ Caption 24507 photos" is functional but plain — consider icon
  treatment, weight, sizing hierarchy vs. secondary actions).
- **Progress bar / JobPanel redesign**: today it's a generic bordered box with a
  thin bar (`JobPanel.svelte`). Redesign so it reads as a direct continuation of
  the button that spawned it — e.g. the button morphs in place into the progress
  UI (shared position/width, a transition, not a layout jump), stronger visual
  hierarchy for %/done/fail counts, better use of color/motion for the "running"
  state vs. "done" vs. "aborted."
- **Section/card layer**: spacing, icons per section (Scan/Vision/Embed/Faces/etc.),
  visual hierarchy so the page doesn't read as a flat stack of identical grey
  boxes — think Material Design 3 / Google Photos polish level, not a bare
  admin-panel look.
- Reference `web/src/lib/IndexTab.svelte`, `JobPanel.svelte`, and whatever
  shared CSS variables/theme file exists (check `web/src` for a `theme.css` or
  `:root` variables) — keep the dark theme, elevate the execution.
- This is subjective ("visually stunning") — do the best design pass grounded in
  real UX heuristics (state legibility, motion with purpose, hierarchy,
  consistency), take screenshots via claude-in-chrome to sanity-check the
  result myself before calling it done, but flag for Hari's own visual review
  in the morning since taste is inherently his call, not something to declare
  "finished" unilaterally.
- Scope: Index & Manage tab first (where all tonight's discussion focused). Note
  in the morning check-in whether he wants the same treatment carried to
  Search/Timeline/People tabs too.

## Progress log (append as you go)
- 2026-07-04 ~23:50: File created. Starting task 4 (schema validation) and
  task 5 (LM Studio v0 API) next — both backend, will batch into one server
  restart when both are done and tests pass.
- 2026-07-04 ~00:15: Task 4 done (indexer.py wires validate_vision_output into
  compute_caption). Task 5 backend half done (vision.py: list_lm_studio_models_v0,
  classify_lm_studio_model now accepts real v0 info, _lm_model_id prefers the
  real loaded vlm; api.py: provider-models endpoint passes v0 info through).
  Frontend half of task 5 (badges/sort in IndexTab.svelte) still to do.
  Hari added task 14 (full visual/UX redesign) before heading out for the night —
  recorded above, will get to it after the backend correctness tasks (3,5,6) and
  the idle-polling fix (7), since those are more mechanical/lower-risk than a
  full design pass.
- 2026-07-04 ~00:35: Tasks 3 and 6 done together in `vision.py: _call_gemini`
  (Gemini 429 cooldown tracking via `gemini_cooldowns()`, pinned-model fallback
  through the static rate-limit-ordered `GEMINI_VISION_MODELS` list — kept it
  static rather than a live re-fetch, see note below). `api.py: /api/provider-models`
  now also returns `gemini_cooldowns`.
  IMPORTANT correction: my first draft of task 6 made the "no model pinned"
  path fetch the model list dynamically too — reverted that. GEMINI_VISION_MODELS
  is deliberately ordered by rate-limit friendliness (comment in constants.py);
  the API's own listing order is NOT rate-limit-ordered (verified live: pro
  models mixed in near the front), so using it would have been a silent
  regression. Dynamic fetch is only used for the *rest* of a pinned-model's
  fallback pool implicitly via the same static list — no extra network call
  added to the per-image hot path.
  Ran `uv run python -m pytest tests/ -q`: 4 failures, all caused by my own
  changes (not pre-existing) — fixed all 4, now 182 passed:
    - `test_indexer.py::test_vision_one_stores_and_returns_model` — the test's
      mock caption `{"caption":"ok"}` was genuinely missing 4/5 required keys;
      my new schema-validation (task 4) correctly rejected it. Fixed the test
      fixture to use a complete caption, not the validation logic.
    - `test_vision.py::test_call_gemini_skips_429_tries_next` /
      `test_call_gemini_all_429_raises` — caused by the dynamic-fetch mistake
      above; fixed by reverting to the static list (see correction above).
    - `test_vision.py::test_get_image_caption_with_model_returns_tuple` — my new
      `_lm_model_id()` calls the real `list_lm_studio_models_v0()` (hits
      `localhost:1234/api/v0/models`), which isn't mocked by this test and was
      silently picking up MY actual real LM Studio state instead of the test's
      mock. Added `patch("vision.list_lm_studio_models_v0", return_value=[])` to
      this test and 3 others in test_vision.py that call `get_image_caption`
      without pinning a model (same latent issue, didn't fail today only
      because LM Studio happens to be running on this dev machine — fixed for
      hygiene per the project's "all external calls mocked" test convention).
  All backend changes for tasks 3/4/5(backend half)/6 are now test-clean.
  NOT restarting the server yet — still need task 5's frontend half (loaded
  badge in IndexTab.svelte) before doing the one planned restart, to batch it
  with the idle-polling fix (7) so the vision job only gets interrupted once.
  Moving to task 7 (frontend idle polling) next, then finishing task 5's UI half,
  then task 8/9, then restart + verify live, then task 14 (design pass).
- 2026-07-04 ~01:00-02:25: Tasks 5 (frontend badge/sort), 7 (frontend idle
  polling fix), 8 (Timeline year/month quick-jump), 9 (Search Month filter)
  all implemented:
    - Task 7: found the exact bug via code (App.svelte polls `jobStatus` every
      4s unconditionally → global header pill always fresh; IndexTab.svelte's
      OWN `job` var only starts polling if a job was active at ITS OWN mount
      time → button-disable logic could go stale independently of the header
      pill). Fixed via a plain `jobStatus.subscribe()` in IndexTab.svelte that
      catches up and starts local polling when the global store reports a job
      active that this tab isn't tracking yet. (First attempt used a `$:`
      reactive block instead of `.subscribe()` — Svelte's compiler rejected it
      as a cyclical dependency since the block both read and, via nested
      assignment, wrote `job`/`running`. Switched to a plain store subscription,
      which isn't part of Svelte's static reactive graph.)
    - Task 5 UI: `IndexTab.svelte` LM Studio dropdowns now sort loaded-first
      and show real "● loaded"/"○ not loaded" from the v0 API (falls back to
      no state tag when only the name-heuristic is available). Embed dropdown
      previously showed no type info at all — fixed to use the same
      `modelTypeLabel()` helper as the vision dropdown. Also wired Gemini
      cooldown display (task 3) into the same helper — a rate-limited Gemini
      model now shows "⏳ rate-limited (Ns)" in its option label.
    - Task 8: added `GET /api/timeline/summary` (api.py) — cheap year→month→
      count tally, no per-photo os.path.exists check (that's what makes the
      main /api/timeline expensive at 25k+ photos), factored the shared
      date-resolution logic into `_resolve_photo_date()` so it's not duplicated.
      `TimelineTab.svelte` now has a sticky "Jump to" bar (Year select → Month
      select, both populated from the summary, with counts) — since ALL years
      already render their header on initial load (just with only their first
      page of photos), jumping to a year is an instant `scrollIntoView`; jumping
      to a specific month pages that year forward (existing `loadMore`, which
      was already idempotent) until the month appears, then scrolls to it. Week/
      day granularity intentionally NOT done — would need new backend grouping,
      flagged as a stretch goal, not tonight's scope.
    - Task 9: added `month` to the ChromaDB embed payload (`indexer.py:
      build_embed_payload`, alongside the existing `year` field) and to
      `search.py: get_available_filter_values()`'s attrs list. `SearchTab.svelte`
      FILTER_ORDER now includes Month right after Year, with month numbers
      displayed as names (January, February, ...) via a small formatter.
      IMPORTANT caveat for tomorrow: `month` is only stamped at EMBED time —
      the ~300+ photos already embedded before this change won't have a
      `month` field until they're re-embedded (same as any new attribute added
      to an existing pipeline; not a bug, just means the Month filter will only
      show real options once new embeddings accumulate, or if a `reanalyze`/
      `full` pass is later re-run over everything).
  Ran `npm run build` (web/) after each meaningful UI change — all builds
  clean (only pre-existing a11y/unused-CSS warnings, unrelated to tonight's
  edits). Ran `uv run python -m pytest tests/ -q` after the whole batch — 182
  passed, no regressions.
  RESTARTED THE SERVER (the one planned restart, batching all of tonight's
  backend changes): stopped the vision job gracefully via POST /api/index/stop
  first (waited for active:false — done was 308/24507 at that point, 0
  failures), killed the live process, relaunched.
  IMPORTANT gotcha for future restarts: the process listening on :8768 was
  running via `C:\Users\ylnha\.pyenv\pyenv-win\versions\3.12.1\python.exe
  src/serve.py` (per `Get-CimInstance Win32_Process`) — I assumed that was the
  "real" launch command and reused it verbatim, but that pyenv interpreter does
  NOT have chromadb installed (`ModuleNotFoundError: No module named
  'chromadb'` on relaunch) even though the ORIGINAL process under that same
  path had apparently been running fine for hours. Never resolved why; gave up
  investigating and just used the project's documented launch command instead
  (`uv run python src/serve.py`, per CLAUDE.md and Makefile `make serve`),
  which worked immediately. **Use `uv run python src/serve.py` to (re)start
  this server, not a raw interpreter path**, regardless of what a running
  process's command line appears to say.
  Also noted (did NOT touch): a second, separate `src/serve.py` process (PID
  46252 tonight, via `.venv\Scripts\python.exe`, started at the exact same
  timestamp as the live one, 0 CPU / ~3.9MB memory the whole time — looks
  like a failed duplicate launch from whenever photo-vault was last started,
  not bound to any port, inert). Didn't kill it since it predates tonight's
  session and isn't mine to clean up without asking — mention to Hari, he may
  want `taskkill` on it (check `Get-CimInstance Win32_Process -Filter
  "Name='python.exe'"` for its current PID before assuming 46252 still applies).
  After restart: verified live — `/api/provider-models` now returns real
  `state: loaded/not-loaded` per LM Studio model (confirmed gemma-4-e4b-it
  shows `loaded`, the other 3 vision/embed models show `not-loaded`) and
  `/api/timeline/summary` returns real year→month counts. Resumed the vision
  job (`POST /api/index/start` with the same lm_studio/gemma-4-e4b-it config)
  — confirmed progressing again (4 new captions done, 0 failures) before
  moving on. Total pending is now 24199 (308 already done pre-restart
  correctly excluded).
  NEXT: task 14, the visual/UX redesign — starting now.
- 2026-07-04 ~02:30-03:00: Task 14 (visual/UX redesign of Index & Manage) done,
  first pass:
    - `app.css`: new tokens (`--accent-2`, `--radius-lg`, `--shadow-1/2/accent`),
      buttons get real elevation/hover-lift/gradient-on-primary instead of a
      flat filter-brightness change, added a reusable `.tile` (colored icon
      badge) class.
    - New `SectionHead.svelte` component: icon-in-colored-tile + title, used
      to give every pipeline section (Scan=📂 blue, Vision=👁 indigo,
      Embed=🧬 violet, Faces=🙂 pink, Thumbnails=🖼 amber, Duplicates=🔍 cyan,
      Full=⚡ green, Reanalyze=🔄 orange) a distinct visual identity instead of
      identical grey `.section-label` text — directly addresses "flat stack of
      grey boxes."
    - New "blocked" state: replaced every plain `<p class="hint">Another job is
      running — stop it first.</p>` (8 occurrences) with a single styled
      `.blocked-row` pill that names the ACTUAL running job by its real title
      (e.g. "⏸ Blocked — Vision analysis is running, stop it first") instead of
      generic text — the "why is this disabled" reason is now impossible to miss,
      not a barely-visible line under a button that still looked clickable.
      (The buttons themselves were ALREADY correctly hidden in this state, per
      tonight's earlier investigation — task 14 fixes the copy/prominence, not
      new logic.)
    - `JobPanel.svelte` fully rewritten: per-job-type theme color + icon (mirrors
      SectionHead's palette so a running job's panel visually matches its
      section), gradient progress fill with a moving shimmer (so a
      barely-advancing bar on a slow local model still reads as "alive"),
      indeterminate striped animation for scan/dhash-style jobs with no known
      total, larger/bolder % readout in the theme color, chip-styled ok/fail/
      aborted/stopped stats, nicer log rows (monospace filename, alternating
      row tint), and a `fly`-in entrance transition (from `svelte/transition`)
      so the panel visibly grows into the button's old position instead of an
      abrupt layout jump — directly answers "must stay near button."
  Verified live in the browser (server already running the redesigned build):
    - Vision section: icon tile + themed JobPanel render correctly, matches the
      intended design (screenshot-checked).
    - Every other section (Embed/Faces/Thumbnails/Duplicates/Full) correctly
      shows the new "⏸ Blocked — Vision analysis is running, stop it first"
      pill while the vision job is active — consistent, clear, not a bug like
      the old plain-text hint could be missed as.
    - LM Studio dropdown: confirmed "gemma-4-e4b-it ● loaded" shown as the
      resolved/default option (real state from the v0 API, task 5's payoff).
    - Timeline quick-jump: tested for real — picked 2016 → November from the
      Jump-to bar, landed exactly on that month's section instantly. Task 8
      confirmed working end-to-end, not just building cleanly.
    - Search Month filter: confirmed NOT showing yet, exactly as documented —
      no embedded photo carries a `month` value yet since only vision (not
      embed) has run since the restart. Not a bug; will appear once embedding
      resumes and accumulates new-schema photos.
  Spotted (NOT fixed, out of scope for tonight, flagging for Hari): Index &
  Manage section "F · Active search model" renders "... lm_studio · 768d ·
  **undefined** photos" — an existing display bug unrelated to anything
  touched tonight (never opened that section's code). Small fix, someone
  should just check what field IndexTab.svelte expects there vs. what
  /api/models actually returns.
  Ran `npm run build` + `uv run python -m pytest tests/ -q` one more time after
  the whole design pass — clean build (same pre-existing warnings only), 182
  passed.

## Session wrap-up (what to check first tomorrow morning)
1. Read this file top to bottom — every numbered task above is either [DONE],
   [INFO ONLY] (investigated, no code change needed), or explicitly flagged
   [NEEDS HARI'S OK] (items 10, 11 — reverse geocoding + exact person count —
   deliberately left undone, see those sections for why).
2. Vision captioning job should still be running in the background (LM Studio
   gemma-4-e4b-it, ~24k photos pending as of session end). Check
   `GET /api/index/progress` — if it stopped/aborted overnight, it's resumable,
   just hit "Caption N photos" again in Index & Manage section B, or use MCP/curl.
3. **Nothing has been committed.** All of tonight's changes are uncommitted
   working-tree edits across: `src/vision.py`, `src/indexer.py`, `src/api.py`,
   `src/search.py`, `tests/test_vision.py`, `tests/test_indexer.py`,
   `web/src/app.css`, `web/src/lib/IndexTab.svelte`, `web/src/lib/JobPanel.svelte`
   (rewritten), `web/src/lib/SectionHead.svelte` (new),
   `web/src/lib/TimelineTab.svelte`, `web/src/lib/SearchTab.svelte`,
   `web/src/lib/api.js`. Run `git status`/`git diff` to review before deciding
   whether to commit — per your global policy Claude never commits without
   being asked.
4. There's a leftover orphaned `src/serve.py` process from before tonight's
   session (not started by me, didn't touch it) — see task 5/restart notes
   above for how to identify it if it's still around.
5. Ask about items 10 and 11 whenever convenient — they're real, well-scoped
   features, just ones with a genuine cost/privacy tradeoff that felt like your
   call, not mine, to make unilaterally overnight.
