<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount } from "svelte";
  export let indexedCount = 0;
  const dispatch = createEventDispatcher();

  let q = "";
  let person = "";
  let filterVals = {};
  let selected = {};
  let results = null;
  let recent = [];
  let loading = false;
  let err = "";

  const FILTER_ORDER = [
    ["year", "Year"], ["weather", "Weather"], ["occasion", "Occasion"],
    ["scene", "Scene"], ["group_size", "Group"], ["clothing_style", "Clothing"],
    ["mood", "Mood"], ["location_type", "Location"], ["season", "Season"],
  ];

  onMount(async () => {
    if (indexedCount > 0) {
      try {
        filterVals = await api.filters();
        recent = (await api.recent(60)).results;
      } catch (e) { err = e.message; }
    }
  });

  async function doSearch() {
    loading = true; err = "";
    try {
      const filters = {};
      for (const k in selected) if (selected[k] && selected[k] !== "All") filters[k] = selected[k];
      results = (await api.search(q, filters, person)).results;
    } catch (e) { err = e.message; }
    loading = false;
  }
  function clear() { results = null; q = ""; person = ""; selected = {}; }
</script>

{#if indexedCount === 0}
  <div class="card">
    <p>No photos indexed yet. Go to <b>Index &amp; Manage</b> to scan and index your photos.</p>
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
                {#each filterVals[key] as v}<option>{v}</option>{/each}
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
      {#if results !== null}
        <div class="section-label">{results.length} results</div>
        <PhotoGrid photos={results} on:select={(e) => dispatch("select", e.detail)} />
      {:else}
        <div class="section-label">Recently indexed</div>
        <PhotoGrid photos={recent} on:select={(e) => dispatch("select", e.detail)} />
      {/if}
    </section>
  </div>
{/if}

<style>
  .layout { display: flex; gap: 16px; align-items: flex-start; }
  aside { width: 240px; flex-shrink: 0; position: sticky; top: 12px; }
  @media (max-width: 800px) { .layout { flex-direction: column; } aside { width: 100%; position: static; } }
</style>
