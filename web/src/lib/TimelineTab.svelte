<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();

  let loaded = false;
  let years = [];
  let loading = false;
  let err = "";
  let shown = {}; // year -> count shown

  async function load() {
    loading = true; err = "";
    try {
      years = (await api.timeline()).years;
      for (const y of years) shown[y.year] = 60;
    } catch (e) { err = e.message; }
    loading = false; loaded = true;
  }
</script>

{#if !loaded}
  <div class="card col">
    <p>Browse your photos organized by year.</p>
    <button class="primary" on:click={load} disabled={loading} style="width:max-content">
      {loading ? "Loading…" : "📅 Load Timeline"}
    </button>
  </div>
{:else if err}
  <p style="color:var(--danger)">{err}</p>
{:else if years.length === 0}
  <p class="muted">No photos yet. Scan a folder in Index &amp; Manage.</p>
{:else}
  {#each years as y (y.year)}
    {@const existing = y.photos.filter((p) => p.exists !== false)}
    {@const missing = y.count - existing.length}
    <div class="card" style="margin-bottom:14px">
      <div class="row" style="justify-content:space-between">
        <b>📅 {y.year} — {existing.length} photos
          {#if missing}<span class="pill" style="color:var(--danger)">{missing} missing</span>{/if}
        </b>
      </div>
      <div style="margin-top:10px">
        <PhotoGrid photos={existing.slice(0, shown[y.year])} cols={6}
                   on:select={(e) => dispatch("select", e.detail)} />
      </div>
      {#if existing.length > shown[y.year]}
        <button class="ghost" style="margin-top:10px"
                on:click={() => (shown[y.year] += 60)}>
          Load more ({existing.length - shown[y.year]} remaining)
        </button>
      {/if}
    </div>
  {/each}
  <button class="ghost" on:click={load}>🔄 Reload</button>
{/if}
