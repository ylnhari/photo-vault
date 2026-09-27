<script>
  import { api, fmtDuration } from "./api.js";
  import { createEventDispatcher, onMount, onDestroy } from "svelte";
  import { createRequestGate } from "./requestGate.js";

  // Either a single id, or an ordered list + starting index for navigation.
  export let id = null;
  export let ids = null;       // array of photo ids in the current grid
  export let cards = [];
  export let index = 0;        // starting position within ids

  const dispatch = createEventDispatcher();
  const metaGate = createRequestGate();
  const detailGate = createRequestGate();
  const similarGate = createRequestGate();

  // Normalize to a list we can navigate.
  let list = ids && ids.length ? ids : (id != null ? [id] : []);
  let pos = ids && ids.length ? Math.max(0, Math.min(index, ids.length - 1)) : 0;
  $: currentId = list[pos];
  $: currentCard = cards.find((card) => card.id === currentId) || {};
  $: imageAlt = meta?.caption || currentCard.caption || meta?.filename || currentCard.filename || "Photo";

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
    const targetId = currentId;
    const request = similarGate.begin();
    simBusy = true; err = "";
    try {
      const response = await api.similar(targetId, 12, { signal: request.signal });
      if (request.isCurrent() && currentId === targetId) similar = response.results;
    } catch (e) { if (request.isCurrent() && currentId === targetId) err = e.message; }
    finally { if (request.isCurrent()) simBusy = false; }
  }
  function openSimilar(s) {
    // Navigate the lightbox through the similar set.
    list = similar.map((x) => x.id);
    cards = similar;
    pos = list.indexOf(s.id);
  }

  // Reload metadata whenever the current photo changes.
  let lastLoaded = null;
  $: if (currentId && currentId !== lastLoaded) {
    lastLoaded = currentId;
    meta = null; detail = null; showDetail = false; confirmDelete = false; err = "";
    similar = null;
    detailGate.cancel();
    similarGate.cancel();
    load(currentId);
    preloadNeighbors();
  }

  async function load(targetId) {
    const request = metaGate.begin();
    try {
      const response = await api.meta(targetId, { signal: request.signal });
      if (request.isCurrent() && currentId === targetId) meta = response;
    } catch (e) { if (request.isCurrent() && currentId === targetId) err = e.message; }
  }
  async function loadDetail() {
    if (detail) { showDetail = true; return; }
    const targetId = currentId;
    const request = detailGate.begin();
    try {
      const response = await api.explore(targetId, { signal: request.signal });
      if (request.isCurrent() && currentId === targetId) { detail = response; showDetail = true; }
    } catch (e) { if (request.isCurrent() && currentId === targetId) err = e.message; }
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

  function next() { if (!busy && pos < list.length - 1) pos += 1; }
  function prev() { if (!busy && pos > 0) pos -= 1; }
  function close() { dispatch("close"); }

  // ── focus trap ──────────────────────────────────────────────────────────
  // Keeps Tab/Shift+Tab cycling within the dialog instead of leaking focus
  // into the tab nav behind the dark overlay.
  let boxEl;
  function focusableEls() {
    if (!boxEl) return [];
    return [...boxEl.querySelectorAll(
      'button, a[href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )].filter((el) => !el.disabled && el.getClientRects().length > 0);
  }

  function onKey(e) {
    const target = e.target;
    const editing = target?.matches?.("input, textarea, select, [contenteditable='true']") || target?.isContentEditable;
    if (e.key === "Escape") { close(); }
    else if (!editing && e.key === "ArrowRight") { next(); }
    else if (!editing && e.key === "ArrowLeft") { prev(); }
    else if (e.key === "Tab") {
      const els = focusableEls();
      if (!els.length) return;
      const first = els[0], last = els[els.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    }
  }
  let returnFocusTarget;
  onMount(() => {
    returnFocusTarget = document.activeElement;
    window.addEventListener("keydown", onKey);
    // Move focus into the dialog so it doesn't stay on whatever triggered it
    // (a grid cell behind the overlay).
    focusableEls()[0]?.focus();
  });
  onDestroy(() => {
    window.removeEventListener("keydown", onKey);
    metaGate.cancel(); detailGate.cancel(); similarGate.cancel();
    returnFocusTarget?.focus?.();
  });

  async function remove(deleteFile) {
    if (busy || !currentId) return;
    const targetId = currentId;
    busy = true;
    try {
      await api.deleteImage(targetId, deleteFile);
      // Drop from the local list and advance, or close if it was the last one.
      list = list.filter((x) => x !== targetId);
      dispatch("deleted", targetId);
      if (list.length === 0) { close(); return; }
      if (pos >= list.length) pos = list.length - 1;
      lastLoaded = null;  // force reload of the now-current photo
      busy = false;
    } catch (e) { err = e.message; busy = false; }
  }

  $: gps = meta && meta.gps_lat != null && meta.gps_lon != null
    ? { lat: meta.gps_lat, lon: meta.gps_lon } : null;
  $: isVideo = !!meta && meta.media_type === "video";
</script>

<div class="overlay" on:click|self={close} role="presentation">
  {#if list.length > 1 && pos > 0}
    <button class="nav prev" on:click|stopPropagation={prev} aria-label="Previous">‹</button>
  {/if}
  {#if list.length > 1 && pos < list.length - 1}
    <button class="nav next" on:click|stopPropagation={next} aria-label="Next">›</button>
  {/if}

  <div class="box" bind:this={boxEl} role="dialog" aria-modal="true" aria-labelledby="lightbox-title" tabindex="-1">
    <button class="ghost close" on:click={close} aria-label="Close">✕</button>
    {#if list.length > 1}
      <div class="counter">{pos + 1} / {list.length}</div>
    {/if}
    <div class="content">
      <div class="imgwrap">
        <h2 id="lightbox-title" class="sr-only">{imageAlt}</h2>
        {#key currentId}
          {#if isVideo}
            <!-- Streamed via /api/video (HTTP range → seekable). Poster is the
                 same frame the grid shows so it doesn't flash black before play. -->
            <video src={api.videoUrl(currentId)} poster={api.thumbUrl(currentId)} aria-label={imageAlt}
                   controls autoplay playsinline preload="metadata">
              <track kind="captions" />
            </video>
          {:else}
            <img src={api.mediumUrl(currentId)} alt={imageAlt} decoding="async" />
          {/if}
        {/key}
      </div>
      <section class="side col" aria-label="Photo details and actions">
        {#if err}<p style="color:var(--danger)">{err}</p>{/if}
        {#if meta}
          {#if isVideo}
            <div><span class="muted">Type:</span> Video</div>
            {#if meta.duration_s}<div><span class="muted">Duration:</span> {fmtDuration(meta.duration_s)}</div>{/if}
            {#if meta.width && meta.height}<div><span class="muted">Resolution:</span> {meta.width}×{meta.height}</div>{/if}
          {/if}
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
                  <button class="simthumb-wrap"
                     title={s.caption || s.filename}
                     on:click={() => openSimilar(s)}
                     aria-label={`Open similar photo: ${s.caption || s.filename || "memory"}`}>
                  <img class="simthumb" src={api.thumbUrl(s.id)} alt={s.filename}
                       decoding="async" />
                  </button>
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
      </section>
    </div>
  </div>
</div>

<style>
  .overlay {
    position: fixed; inset: 0; background: rgba(25,35,31,.76); backdrop-filter: blur(5px);
    display: flex; align-items: center; justify-content: center; z-index: 100; padding: 24px;
  }
  .box { position: relative; background: var(--surface); border: 1px solid rgba(255,255,255,.36);
    border-radius: 15px; max-width: 1420px; width: 100%; max-height: 92vh; overflow: hidden; box-shadow: 0 26px 90px rgba(0,0,0,.32); }
  .close { position: absolute; top: 10px; right: 10px; z-index: 2; width: 35px; height: 35px; padding: 0;
    color: #31403a; background: rgba(255,255,255,.91); border: 0; border-radius: 50%; }
  .counter { position: absolute; top: 14px; left: 16px; z-index: 2; font-size: 12px;
    color: #eaf1ed; background: rgba(29,43,37,.7); padding: 4px 9px; border-radius: 12px; }
  .content { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(290px, .78fr); max-height: 92vh; }
  .imgwrap { position: relative; min-width: 0; min-height: 280px; background: #18221e; display: flex; align-items: center; justify-content: center; }
  .imgwrap img, .imgwrap video { display: block; max-width: 100%; max-height: 92vh; object-fit: contain; }
  .side { padding: 54px 24px 24px; overflow-y: auto; max-height: 92vh; background: #fff; color: #39463f; }
  .side .muted { color: var(--muted); }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
  @media (max-width: 760px) {
    .overlay { align-items: flex-end; padding: 0; }
    .box { width: 100%; max-height: 100dvh; border-radius: 15px 15px 0 0; border-bottom: 0; }
    .content { display: flex; flex-direction: column; max-height: 100dvh; }
    .imgwrap { min-height: 0; height: min(53dvh, 530px); flex: 0 0 auto; }
    .imgwrap img, .imgwrap video { max-width: 100%; max-height: 53dvh; }
    .side { max-height: min(43dvh, 390px); padding: 20px 18px calc(22px + env(safe-area-inset-bottom)); }
    .close { top: 10px; right: 10px; }
    .counter { top: 17px; left: 14px; }
    .nav { position: absolute; top: 27dvh; width: 38px; height: 48px; }
    .nav.prev { left: 8px; } .nav.next { right: 8px; }
  }
  .sm { padding: 5px 10px; font-size: 13px; }
  .muted { color: var(--muted); }
  .ok-text { color: var(--success); }
  .warn-text { color: var(--warn); }
  .simgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 10px; }
  .simthumb-wrap { display: block; width: 100%; aspect-ratio: 1; padding: 0; border-radius: 6px; overflow: hidden;
    cursor: pointer; background: var(--surface2); border: 1px solid var(--border); }
  .simthumb-wrap:hover, .simthumb-wrap:focus-visible { outline: 2px solid var(--accent); }
  .simthumb { width: 100%; height: 100%; object-fit: cover; display: block; }
  .history-entry { border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
  .history-model { font-size: 11px; color: var(--muted); font-family: monospace; margin-bottom: 4px; }
  .history-caption { font-size: 13px; line-height: 1.5; }

  /* prev / next arrows */
  .nav {
    position: fixed; top: 50%; transform: translateY(-50%); z-index: 101;
    width: 48px; height: 64px; font-size: 34px; line-height: 1;
    background: rgba(23,35,29,.82); color: #fff; border: none; cursor: pointer; border-radius: 8px;
  }
  .nav:hover { background: rgba(0,0,0,.7); }
  .nav.prev { left: 16px; }
  .nav.next { right: 16px; }
</style>
