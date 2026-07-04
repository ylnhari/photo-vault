<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount, tick } from "svelte";
  const dispatch = createEventDispatcher();

  let years = [];
  let loading = false;
  let err = "";
  let moreBusy = {}; // year -> loading more
  let summary = {}; // year -> { "01".."12"/"00" -> count }, from /api/timeline/summary
  let jumpYear = "";
  let jumpMonth = "";
  let jumping = false;

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
      const key = m >= 1 && m <= 12 ? String(m).padStart(2, "0") : "00";
      if (!current || current.label !== label) {
        current = { label, key, photos: [] };
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
      summary = (await api.timelineSummary()).summary;
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

  // Quick-jump: years/months are already known upfront (from the cheap
  // /api/timeline/summary tally), even before their photos are loaded — no
  // need to scroll a 40k-photo list sequentially to reach an old year/month.
  // Photos load newest-first per year, so reaching an older month may mean
  // paging that year forward first; loadMore() is idempotent once exhausted.
  $: jumpYears = Object.keys(summary).sort((a, b) =>
    a === "Unknown" ? 1 : b === "Unknown" ? -1 : b.localeCompare(a));
  $: jumpMonths = jumpYear && summary[jumpYear]
    ? Object.keys(summary[jumpYear]).sort()
    : [];

  async function jumpTo() {
    if (!jumpYear) return;
    jumping = true;
    try {
      const y = years.find((yy) => yy.year === jumpYear);
      if (!y) return;
      if (jumpMonth) {
        while (y.photos.length < y.count) {
          const existing = y.photos.filter((p) => p.exists !== false);
          if (monthGroups(existing).some((g) => g.key === jumpMonth)) break;
          await loadMore(y);
        }
      }
      await tick();
      const id = jumpMonth ? `m-${jumpYear}-${jumpMonth}` : `y-${jumpYear}`;
      document.getElementById(id)?.scrollIntoView({ block: "start" });
    } finally {
      jumping = false;
    }
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
  <div class="jumpbar">
    <span class="jumplabel">Jump to</span>
    <select bind:value={jumpYear} on:change={() => { jumpMonth = ""; }}>
      <option value="">Year…</option>
      {#each jumpYears as y}
        <option value={y}>{y === "Unknown" ? "Unknown" : y}
          ({Object.values(summary[y]).reduce((a, b) => a + b, 0)})</option>
      {/each}
    </select>
    <select bind:value={jumpMonth} disabled={!jumpYear}>
      <option value="">Whole year</option>
      {#each jumpMonths as m}
        <option value={m}>{m === "00" ? "Undated" : MONTHS[parseInt(m, 10)]} ({summary[jumpYear][m]})</option>
      {/each}
    </select>
    <button class="sm primary" on:click={jumpTo} disabled={!jumpYear || jumping}>
      {jumping ? "Jumping…" : "Go"}
    </button>
  </div>

  {#each years as y (y.year)}
    {@const existing = y.photos.filter((p) => p.exists !== false)}
    {@const missingLoaded = y.photos.length - existing.length}
    <section class="yearblock" id="y-{y.year}">
      <div class="yearhead">
        <b>📅 {y.year}</b>
        <span class="muted" style="font-size:13px">{y.count} photos</span>
        {#if missingLoaded}<span class="pill" style="color:var(--danger)">{missingLoaded} missing</span>{/if}
      </div>
      {#each monthGroups(existing) as mg (y.year + mg.label)}
        <div class="monthlabel" id="m-{y.year}-{mg.key}">{mg.label}</div>
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
  .jumpbar {
    display: flex; align-items: center; gap: 8px;
    position: sticky; top: 62px; z-index: 6;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 12px; margin-bottom: 16px;
  }
  .jumplabel { font-size: 13px; font-weight: 600; color: var(--muted); margin-right: 2px; }
  .jumpbar select { max-width: 160px; }
  .yearblock { margin-bottom: 26px; scroll-margin-top: 116px; }
  .yearhead {
    display: flex; align-items: center; gap: 10px;
    position: sticky; top: 108px; z-index: 5;
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
    scroll-margin-top: 150px;
  }
</style>
