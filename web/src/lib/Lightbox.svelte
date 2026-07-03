<script>
  import { api } from "./api.js";
  import { createEventDispatcher, onMount, onDestroy } from "svelte";

  // Either a single id, or an ordered list + starting index for navigation.
  export let id = null;
  export let ids = null;       // array of photo ids in the current grid
  export let index = 0;        // starting position within ids

  const dispatch = createEventDispatcher();

  // Normalize to a list we can navigate.
  let list = ids && ids.length ? ids : (id != null ? [id] : []);
  let pos = ids && ids.length ? Math.max(0, Math.min(index, ids.length - 1)) : 0;
  $: currentId = list[pos];

  let meta = null;
  let detail = null;
  let err = "";
  let confirmDelete = false;
  let busy = false;
  let showDetail = false;

  const ATTRS = [
    ["caption", "Caption"], ["scene", "Scene"], ["weather", "Weather"],
    ["occasion", "Occasion"], ["mood", "Mood"], ["location_type", "Location"],
    ["group_size", "Group"], ["clothing_style", "Clothing"], ["season", "Season"],
    ["time_of_day", "Time"], ["year", "Year"], ["embedding_source", "Embedding"],
  ];

  // "More like this" (vector similarity from the photo's own embedding)
  let similar = null;
  let simBusy = false;
  async function loadSimilar() {
    if (similar) { similar = null; return; }  // toggle off
    simBusy = true; err = "";
    try { similar = (await api.similar(currentId)).results; }
    catch (e) { err = e.message; }
    simBusy = false;
  }
  function openSimilar(s) {
    // Navigate the lightbox through the similar set.
    list = similar.map((x) => x.id);
    pos = list.indexOf(s.id);
  }

  // Reload metadata whenever the current photo changes.
  let lastLoaded = null;
  $: if (currentId && currentId !== lastLoaded) {
    lastLoaded = currentId;
    meta = null; detail = null; showDetail = false; confirmDelete = false; err = "";
    similar = null;
    load(currentId);
    preloadNeighbors();
  }

  async function load(targetId) {
    try { meta = await api.meta(targetId); } catch (e) { err = e.message; }
  }
  async function loadDetail() {
    if (detail) { showDetail = true; return; }
    try { detail = await api.explore(currentId); showDetail = true; } catch (e) { err = e.message; }
  }

  // Preload the medium derivative of the adjacent photos for instant prev/next.
  function preloadNeighbors() {
    for (const d of [1, -1]) {
      const n = pos + d;
      if (n >= 0 && n < list.length) {
        const img = new Image();
        img.src = api.mediumUrl(list[n]);
      }
    }
  }

  function next() { if (pos < list.length - 1) pos += 1; }
  function prev() { if (pos > 0) pos -= 1; }
  function close() { dispatch("close"); }

  function onKey(e) {
    if (e.key === "Escape") { close(); }
    else if (e.key === "ArrowRight") { next(); }
    else if (e.key === "ArrowLeft") { prev(); }
  }
  onMount(() => window.addEventListener("keydown", onKey));
  onDestroy(() => window.removeEventListener("keydown", onKey));

  async function remove(deleteFile) {
    busy = true;
    try {
      await api.deleteImage(currentId, deleteFile);
      // Drop from the local list and advance, or close if it was the last one.
      list = list.filter((x) => x !== currentId);
      dispatch("deleted", currentId);
      if (list.length === 0) { close(); return; }
      if (pos >= list.length) pos = list.length - 1;
      lastLoaded = null;  // force reload of the now-current photo
      busy = false;
    } catch (e) { err = e.message; busy = false; }
  }

  $: gps = meta && meta.gps_lat != null && meta.gps_lon != null
    ? { lat: meta.gps_lat, lon: meta.gps_lon } : null;
</script>

<div class="overlay" on:click|self={close} role="presentation">
  {#if list.length > 1 && pos > 0}
    <button class="nav prev" on:click|stopPropagation={prev} aria-label="Previous">‹</button>
  {/if}
  {#if list.length > 1 && pos < list.length - 1}
    <button class="nav next" on:click|stopPropagation={next} aria-label="Next">›</button>
  {/if}

  <div class="box">
    <button class="ghost close" on:click={close} aria-label="Close">✕</button>
    {#if list.length > 1}
      <div class="counter">{pos + 1} / {list.length}</div>
    {/if}
    <div class="content">
      <div class="imgwrap">
        {#key currentId}
          <img src={api.mediumUrl(currentId)} alt="" decoding="async" />
        {/key}
      </div>
      <div class="side col">
        {#if err}<p style="color:var(--danger)">{err}</p>{/if}
        {#if meta}
          {#each ATTRS as [k, label]}
            {#if meta[k] && meta[k] !== "unknown"}
              <div><span class="muted">{label}:</span> {meta[k]}</div>
            {/if}
          {/each}
          {#if gps}
            <div style="margin-top:6px">
              <span class="muted">Location:</span>
              <a href={`https://www.openstreetmap.org/?mlat=${gps.lat}&mlon=${gps.lon}#map=15/${gps.lat}/${gps.lon}`}
                 target="_blank" rel="noopener noreferrer">
                {gps.lat}, {gps.lon} ↗
              </a>
            </div>
          {/if}
        {:else if !err}
          <p class="muted">Loading…</p>
        {/if}

        <div style="margin-top:16px" class="row">
          <button class="ghost sm" on:click={() => showDetail ? showDetail = false : loadDetail()}>
            Analysis details {showDetail ? "▴" : "▾"}
          </button>
          <button class="ghost sm" on:click={loadSimilar} disabled={simBusy}>
            {simBusy ? "Finding…" : similar ? "More like this ▴" : "More like this ▾"}
          </button>
        </div>

        {#if similar}
          {#if similar.length === 0}
            <p class="muted" style="font-size:12px">No similar photos in the index yet.</p>
          {:else}
            <div class="simgrid">
              {#each similar as s (s.id)}
                <img class="simthumb" src={api.thumbUrl(s.id)} alt={s.filename}
                     title={s.caption || s.filename} loading="lazy" decoding="async"
                     on:click={() => openSimilar(s)} />
              {/each}
            </div>
          {/if}
        {/if}

        {#if showDetail && detail}
          {#if detail.caption_history?.length}
            <div class="section-label" style="margin-top:14px">Caption history</div>
            {#each detail.caption_history as h}
              <div class="history-entry">
                <div class="history-model">{h.model}</div>
                {#if h.validation && !h.validation.valid}
                  <div class="warn-text" style="font-size:12px">⚠ {h.validation.warning}</div>
                {/if}
                {#if h.caption_json}
                  {@const parsed = (() => { try { return JSON.parse(h.caption_json); } catch { return null; } })()}
                  {#if parsed?.caption}<div class="history-caption">{parsed.caption}</div>{/if}
                {/if}
              </div>
            {/each}
          {/if}

          {#if detail.embeddings?.length}
            <div class="section-label" style="margin-top:14px">Embedding models</div>
            {#each detail.embeddings as e}
              <div class="row" style="justify-content:space-between; font-size:13px; padding:3px 0">
                <span>{#if e.is_active}<span class="ok-text">✓</span>{/if} {e.model}</span>
                <span class="muted">{e.source} · {e.dimension}d</span>
              </div>
            {/each}
          {/if}

          {#if detail.exif && Object.keys(detail.exif).length}
            <div class="section-label" style="margin-top:14px">EXIF</div>
            {#each Object.entries(detail.exif) as [k, v]}
              <div style="font-size:12px"><span class="muted">{k}:</span> {v}</div>
            {/each}
          {/if}
        {/if}

        <div class="section-label" style="margin-top:16px">Manage</div>
        <button on:click={() => remove(false)} disabled={busy}>Remove from index</button>
        <label class="row" style="font-size:13px">
          <input type="checkbox" bind:checked={confirmDelete} style="width:auto" />
          Confirm permanent delete
        </label>
        <button class="danger" on:click={() => remove(true)} disabled={!confirmDelete || busy}>
          Delete file from disk
        </button>
      </div>
    </div>
  </div>
</div>

<style>
  .overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,.85);
    display: flex; align-items: center; justify-content: center; z-index: 100; padding: 24px;
  }
  .box { position: relative; background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; max-width: 1100px; width: 100%; max-height: 90vh; overflow: hidden; }
  .close { position: absolute; top: 10px; right: 10px; z-index: 2; }
  .counter { position: absolute; top: 14px; left: 16px; z-index: 2; font-size: 12px;
    color: var(--muted); background: var(--surface2); padding: 2px 8px; border-radius: 10px; }
  .content { display: grid; grid-template-columns: 1.6fr 1fr; gap: 0; max-height: 90vh; }
  .imgwrap { background: #000; display: flex; align-items: center; justify-content: center; }
  .imgwrap img { max-width: 100%; max-height: 90vh; object-fit: contain; }
  /* Extra top padding keeps the first metadata line clear of the ✕ button. */
  .side { padding: 44px 24px 24px; overflow-y: auto; }
  @media (max-width: 700px) { .content { grid-template-columns: 1fr; } }
  .sm { padding: 5px 10px; font-size: 13px; }
  .muted { color: var(--muted); }
  .ok-text { color: var(--success); }
  .warn-text { color: var(--warn); }
  .simgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 10px; }
  .simthumb { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px;
    cursor: pointer; border: 1px solid var(--border); }
  .simthumb:hover { outline: 2px solid var(--accent); }
  .history-entry { border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
  .history-model { font-size: 11px; color: var(--muted); font-family: monospace; margin-bottom: 4px; }
  .history-caption { font-size: 13px; line-height: 1.5; }

  /* prev / next arrows */
  .nav {
    position: fixed; top: 50%; transform: translateY(-50%); z-index: 101;
    width: 48px; height: 64px; font-size: 34px; line-height: 1;
    background: rgba(0,0,0,.4); color: #fff; border: none; cursor: pointer; border-radius: 8px;
  }
  .nav:hover { background: rgba(0,0,0,.7); }
  .nav.prev { left: 16px; }
  .nav.next { right: 16px; }
</style>
