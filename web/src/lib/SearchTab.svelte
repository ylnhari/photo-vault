<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount } from "svelte";
  import { lastDeleted, status } from "./stores.js";
  export let indexedCount = 0;
  const dispatch = createEventDispatcher();

  let q = "";
  let person = "";
  let filterVals = {};
  let selected = {};
  let results = null;
  let recent = [];
  let loading = false;
  let initialLoading = false;
  let err = "";
  let personNotFound = false;
  let filterError = false;

  // Multi-select / batch state
  let selectMode = false;
  let selectedIds = [];
  let resetToken = 0;
  let deleteFiles = false;
  let batchBusy = false;

  function onSelectionChange(e) { selectedIds = e.detail; }
  function clearSelection() { resetToken += 1; selectedIds = []; }
  function toggleSelectMode() { selectMode = !selectMode; if (!selectMode) clearSelection(); }

  // Add-to-album
  let albums = [];
  let albumMsg = "";
  onMount(async () => { try { albums = (await api.albums()).albums; } catch {} });

  async function addToAlbum(e) {
    const val = e.target.value;
    e.target.value = "";
    if (!val || !selectedIds.length) return;
    albumMsg = "";
    try {
      let albumId = val;
      if (val === "__new__") {
        const name = prompt("New album name");
        if (!name || !name.trim()) return;
        albumId = (await api.createAlbum(name.trim())).id;
        albums = (await api.albums()).albums;
      }
      const r = await api.albumAdd(albumId, selectedIds);
      const a = albums.find((x) => x.id === albumId);
      albumMsg = `Added ${selectedIds.length} to “${a ? a.name : "album"}” (${r.count} total).`;
      clearSelection();
    } catch (e) { albumMsg = e.message; }
  }

  async function deleteSelected() {
    if (!selectedIds.length) return;
    batchBusy = true; err = "";
    try {
      const idset = new Set(selectedIds);
      const res = await api.batchDelete(selectedIds, deleteFiles);
      if (results) results = results.filter((p) => !idset.has(p.id));
      recent = recent.filter((p) => !idset.has(p.id));
      // Extend the same "hide everywhere" guarantee single delete gets via
      // the Lightbox: write every deleted id into the shared store so any
      // other mounted PhotoGrid (Timeline, Albums, People, Map) drops them too.
      lastDeleted.set([...selectedIds]);
      // Defensive: the API-layer agent is adding a files_failed (or similarly
      // named) field to this response surfacing per-file disk-delete failures.
      // Surface it if present; silently no-op if the backend hasn't shipped
      // it yet so this doesn't block on that parallel work.
      const failedCount = res?.files_failed?.length ?? res?.failed_files?.length ?? 0;
      if (failedCount > 0) {
        err = `${failedCount} file${failedCount === 1 ? "" : "s"} could not be deleted from disk, removed from index only.`;
      }
      clearSelection();
      selectMode = false;
      deleteFiles = false;
      dispatch("deleted");
    } catch (e) { err = e.message; }
    batchBusy = false;
  }

  const FILTER_ORDER = [
    ["year", "Year"], ["month", "Month"], ["place", "Place"], ["weather", "Weather"],
    ["occasion", "Occasion"], ["festival_name", "Festival"], ["scene", "Scene"],
    ["group_size", "Group"], ["person_count", "People"], ["clothing_style", "Clothing"],
    ["mood", "Mood"], ["location_type", "Location"], ["season", "Season"],
    ["photo_type", "Type"],
  ];

  const MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"];
  function filterOptionLabel(key, v) {
    if (key === "month") {
      const n = parseInt(v, 10);
      return n >= 1 && n <= 12 ? MONTH_NAMES[n] : v;
    }
    return v;
  }

  // React to indexedCount instead of checking it once in onMount: at first
  // page load this component mounts before /api/status resolves (indexedCount
  // is still 0), so an onMount-only check left Recently-indexed empty forever.
  let bootstrapped = false;
  $: if (indexedCount > 0 && !bootstrapped) {
    bootstrapped = true;
    loadInitial();
  }
  async function loadInitial() {
    initialLoading = true;
    try {
      [filterVals, { results: recent }] = await Promise.all([api.filters(), api.recent(60)]);
    } catch (e) { err = e.message; }
    initialLoading = false;
  }

  async function doSearch() {
    loading = true; err = ""; personNotFound = false; filterError = false;
    try {
      const filters = {};
      for (const k in selected) if (selected[k] && selected[k] !== "All") filters[k] = selected[k];
      const res = await api.search(q, filters, person);
      results = res.results;
      personNotFound = !!res.person_not_found;
      filterError = !!res.filter_error;
    } catch (e) { err = e.message; }
    loading = false;
  }
  function clear() {
    results = null; q = ""; person = "";
    // Reset each key to the literal "All" rather than an empty object.
    // bind:value={selected[key]} against a genuinely missing key resolves
    // to undefined, and re-assigning a <select> already showing a real
    // option back to undefined leaves it unmatched (blank) instead of
    // falling back to the "All" option — reproduced by Clear after any
    // filter had been touched.
    selected = Object.fromEntries(FILTER_ORDER.map(([k]) => [k, "All"]));
    personNotFound = false; filterError = false;
  }
</script>

{#if !$status.loaded}
  <!-- Status hasn't loaded yet: show a loader, NOT the onboarding. indexedCount
       is 0 on first paint before /api/status resolves, which used to flash the
       "Set up your library" screen at users who already have a full library. -->
  <div class="card onboard">
    <p class="muted" style="text-align:center; margin:0">Loading your library…</p>
  </div>
{:else if indexedCount === 0}
  <div class="card onboard">
    <h2>Set up your library</h2>
    <ol>
      <li><b>Scan</b> — pick your photo folders and let Photo Vault find every image
        <span class="muted">(Index &amp; Manage → A)</span></li>
      <li><b>Caption</b> — a local vision model describes each photo
        <span class="muted">(B — needs LM Studio or a Gemini key)</span></li>
      <li><b>Embed</b> — captions become search vectors; search, timeline and people light up
        <span class="muted">(C)</span></li>
    </ol>
    <button class="primary" on:click={() => dispatch("goto-index")}>Open Index &amp; Manage →</button>
  </div>
{:else}
  <div class="layout">
    <aside class="card col">
      <div class="section-label">Search</div>
      <input bind:value={q} placeholder="birthday party, beach sunset, mom…"
             on:keydown={(e) => e.key === "Enter" && doSearch()} />
      <input bind:value={person} placeholder="Person name (optional)" />

      {#if Object.keys(filterVals).length}
        <div class="section-label">Filters</div>
        {#each FILTER_ORDER as [key, label]}
          {#if filterVals[key]}
            <label class="col" style="gap:2px">
              <span class="muted" style="font-size:12px">{label}</span>
              <select bind:value={selected[key]}>
                <option value="All">All</option>
                {#each filterVals[key] as v}<option value={v}>{filterOptionLabel(key, v)}</option>{/each}
              </select>
            </label>
          {/if}
        {/each}
      {/if}

      <div class="row" style="margin-top:8px">
        <button class="primary" on:click={doSearch} disabled={loading} style="flex:1">
          {loading ? "Searching…" : "Search"}
        </button>
        <button class="ghost" on:click={clear}>Clear</button>
      </div>
    </aside>

    <section style="flex:1; min-width:0">
      {#if err}<p style="color:var(--danger)">{err}</p>{/if}

      <!-- selection toolbar -->
      <div class="toolbar">
        <button class="ghost sm" class:active={selectMode} on:click={toggleSelectMode}>
          {selectMode ? "✓ Selecting" : "Select"}
        </button>
        {#if selectedIds.length}
          <span class="muted" style="font-size:13px">{selectedIds.length} selected</span>
          <select class="sm" on:change={addToAlbum} title="Add selection to album">
            <option value="">Add to album…</option>
            {#each albums as a}<option value={a.id}>{a.name}</option>{/each}
            <option value="__new__">＋ New album…</option>
          </select>
          <label class="row" style="gap:4px; font-size:12px">
            <input type="checkbox" bind:checked={deleteFiles} style="width:auto" /> delete files too
          </label>
          <button class="danger sm" on:click={deleteSelected} disabled={batchBusy}>
            {batchBusy ? "Removing…" : `Remove ${selectedIds.length}`}
          </button>
          <button class="ghost sm" on:click={clearSelection}>Clear</button>
        {:else if selectMode}
          <span class="muted" style="font-size:12px">Click photos to select · Shift-click for a range</span>
        {/if}
      </div>
      {#if albumMsg}<p class="muted" style="font-size:12px; margin:-4px 0 8px">{albumMsg}</p>{/if}

      {#if results !== null}
        <div class="section-label">{results.length} results</div>
        {#if personNotFound}
          <p class="warn-text" style="font-size:13px; margin:4px 0 10px">
            No one named "{person}" is registered yet — register them in the People tab, or check the spelling.
          </p>
        {/if}
        {#if filterError}
          <p class="warn-text" style="font-size:13px; margin:4px 0 10px">
            One of your filters couldn't be applied and was ignored for this search — the results below are unfiltered.
          </p>
        {/if}
        <PhotoGrid photos={results} {selectMode} {resetToken}
                   on:select={(e) => dispatch("select", e.detail)}
                   on:selectionchange={onSelectionChange} />
      {:else if initialLoading}
        <p class="muted">Loading…</p>
      {:else}
        <div class="section-label">Recently indexed</div>
        <PhotoGrid photos={recent} {selectMode} {resetToken}
                   on:select={(e) => dispatch("select", e.detail)}
                   on:selectionchange={onSelectionChange} />
      {/if}
    </section>
  </div>
{/if}

<style>
  .layout { display: flex; gap: 16px; align-items: flex-start; }
  aside { width: 240px; flex-shrink: 0; position: sticky; top: 12px; }
  @media (max-width: 800px) { .layout { flex-direction: column; } aside { width: 100%; position: static; } }
  .toolbar { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; min-height: 30px; flex-wrap: wrap; }
  .onboard { max-width: 560px; margin: 40px auto; padding: 28px; }
  .onboard h2 { margin-bottom: 14px; font-size: 20px; }
  .onboard ol { margin: 0 0 18px 20px; display: flex; flex-direction: column; gap: 10px; font-size: 14px; }
  .onboard .muted { color: var(--muted); font-size: 12px; }
  .sm { padding: 5px 10px; font-size: 13px; }
  .ghost.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .danger { background: var(--danger); color: #fff; border-color: var(--danger); }
  .warn-text { color: var(--warn); }
</style>
