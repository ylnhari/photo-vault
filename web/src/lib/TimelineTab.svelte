<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount } from "svelte";
  const dispatch = createEventDispatcher();

  let years = [];
  let loading = false;
  let err = "";
  let moreBusy = {}; // year -> loading more

  async function load() {
    loading = true; err = "";
    try {
      years = (await api.timeline()).years;
    } catch (e) { err = e.message; }
    loading = false;
  }

  async function loadMore(y) {
    moreBusy = { ...moreBusy, [y.year]: true };
    try {
      const page = await api.timelineYear(y.year, y.photos.length, 120);
      y.photos = [...y.photos, ...page.photos];
      y.count = page.count;
      years = years; // trigger reactivity
    } catch (e) { err = e.message; }
    moreBusy = { ...moreBusy, [y.year]: false };
  }

  onMount(load);
</script>

{#if loading}
  <p class="muted">Loading timeline…</p>
{:else if err}
  <p style="color:var(--danger)">{err}</p>
{:else if years.length === 0}
  <p class="muted">No photos yet. Scan a folder in Index &amp; Manage.</p>
{:else}
  {#each years as y (y.year)}
    {@const existing = y.photos.filter((p) => p.exists !== false)}
    {@const missingLoaded = y.photos.length - existing.length}
    <div class="card" style="margin-bottom:14px">
      <div class="row" style="justify-content:space-between">
        <b>📅 {y.year} — {y.count} photos
          {#if missingLoaded}<span class="pill" style="color:var(--danger)">{missingLoaded} missing</span>{/if}
        </b>
      </div>
      <div style="margin-top:10px">
        <PhotoGrid photos={existing} cols={6}
                   on:select={(e) => dispatch("select", e.detail)} />
      </div>
      {#if y.photos.length < y.count}
        <button class="ghost" style="margin-top:10px" disabled={moreBusy[y.year]}
                on:click={() => loadMore(y)}>
          {moreBusy[y.year] ? "Loading…" : `Load more (${y.count - y.photos.length} remaining)`}
        </button>
      {/if}
    </div>
  {/each}
  <button class="ghost" on:click={load} disabled={loading}>Reload</button>
{/if}
