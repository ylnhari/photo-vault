<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount } from "svelte";
  const dispatch = createEventDispatcher();

  let years = [];
  let loading = false;
  let err = "";
  let moreBusy = {}; // year -> loading more

  const MONTHS = ["", "January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"];

  // Group one year's loaded photos into month sections (dates are
  // "YYYY:MM:DD hh:mm:ss"; anything unparsable lands in a catch-all).
  function monthGroups(photos) {
    const groups = [];
    let current = null;
    for (const p of photos) {
      const m = parseInt((p.date || "").slice(5, 7), 10);
      const label = m >= 1 && m <= 12 ? MONTHS[m] : "Undated";
      if (!current || current.label !== label) {
        current = { label, photos: [] };
        groups.push(current);
      }
      current.photos.push(p);
    }
    return groups;
  }

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
    <section class="yearblock">
      <div class="yearhead">
        <b>📅 {y.year}</b>
        <span class="muted" style="font-size:13px">{y.count} photos</span>
        {#if missingLoaded}<span class="pill" style="color:var(--danger)">{missingLoaded} missing</span>{/if}
      </div>
      {#each monthGroups(existing) as mg (y.year + mg.label)}
        <div class="monthlabel">{mg.label}</div>
        <PhotoGrid photos={mg.photos} cols={6}
                   on:select={(e) => dispatch("select", e.detail)} />
      {/each}
      {#if y.photos.length < y.count}
        <button class="ghost" style="margin-top:10px" disabled={moreBusy[y.year]}
                on:click={() => loadMore(y)}>
          {moreBusy[y.year] ? "Loading…" : `Load more (${y.count - y.photos.length} remaining)`}
        </button>
      {/if}
    </section>
  {/each}
  <button class="ghost" on:click={load} disabled={loading}>Reload</button>
{/if}

<style>
  .yearblock { margin-bottom: 26px; }
  .yearhead {
    display: flex; align-items: center; gap: 10px;
    position: sticky; top: 62px; z-index: 5;
    background: color-mix(in srgb, var(--bg) 92%, transparent);
    backdrop-filter: blur(4px);
    padding: 8px 4px; margin: 0 -4px 4px;
    border-bottom: 1px solid var(--border);
  }
  .yearhead b { font-size: 17px; }
  .monthlabel {
    color: var(--muted); font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em;
    margin: 14px 0 8px;
  }
</style>
