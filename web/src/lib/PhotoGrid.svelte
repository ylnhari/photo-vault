<script>
  import { api, fmtDuration } from "./api.js";
  import { lastDeleted } from "./stores.js";
  import { createEventDispatcher, onDestroy } from "svelte";
  export let photos = [];
  export let cols = 5;
  export let selectMode = false;
  export let resetToken = 0;       // bump from parent to clear the selection
  const dispatch = createEventDispatcher();

  let selected = new Set();
  let lastIndex = -1;

  // Hide photos deleted elsewhere (e.g. from the Lightbox, or a batch delete
  // in another tab) without requiring every parent tab to prune its own list.
  // lastDeleted is always an array — one id for a single delete, many for a
  // batch delete — so every id from the most recent delete op gets hidden.
  let hidden = new Set();
  $: if ($lastDeleted && $lastDeleted.length) {
    const toAdd = $lastDeleted.filter((id) => !hidden.has(id));
    if (toAdd.length) hidden = new Set([...hidden, ...toAdd]);
  }
  $: visible = photos.filter((p) => !hidden.has(p.id));

  $: ids = visible.map((p) => p.id);

  // Clear selection when the parent bumps resetToken (e.g. after a batch delete).
  let _lastReset = resetToken;
  $: if (resetToken !== _lastReset) { _lastReset = resetToken; selected = new Set(); emit(); }

  function emit() { dispatch("selectionchange", Array.from(selected)); }

  // Aspect ratios learned as thumbs load → justified rows (Google-Photos-style)
  // instead of square crops. Cells render square until their thumb arrives.
  let ratios = {};
  function onImgLoad(e, p) {
    const img = e.target;
    if (img.naturalWidth && img.naturalHeight) {
      const ar = Math.max(0.55, Math.min(2.4, img.naturalWidth / img.naturalHeight));
      if (ratios[p.id] !== ar) { ratios[p.id] = ar; ratios = ratios; }
    }
  }

  // Live cell elements, in visible order. Queried on demand — a cached
  // bind:this array goes stale when the keyed {#each} reuses nodes.
  function cellNodes() {
    return wrapEl ? [...wrapEl.querySelectorAll(".cell")] : [];
  }

  function toggleSelect(p, i) {
    if (selected.has(p.id)) selected.delete(p.id);
    else selected.add(p.id);
    lastIndex = i;
    selected = selected;  // trigger reactivity
    emit();
  }

  function onCellClick(e, p, i) {
    if (suppressClick) { suppressClick = false; return; }
    if (p.exists === false) return;
    const multi = selectMode || e.ctrlKey || e.metaKey || e.shiftKey;
    if (multi) {
      if (e.shiftKey && lastIndex >= 0) {
        const [a, b] = [Math.min(lastIndex, i), Math.max(lastIndex, i)];
        for (let k = a; k <= b; k++) if (visible[k].exists !== false) selected.add(visible[k].id);
        selected = selected;
        emit();
      } else {
        toggleSelect(p, i);
      }
    } else {
      dispatch("select", { id: p.id, ids });
    }
  }

  // Keyboard: arrows move focus through the grid, Enter opens, Space selects.
  function onCellKey(e, p, i) {
    if (e.key === "Enter") {
      if (p.exists !== false) dispatch("select", { id: p.id, ids });
    } else if (e.key === " ") {
      e.preventDefault();
      if (p.exists !== false) toggleSelect(p, i);
    } else if (e.key.startsWith("Arrow")) {
      const delta = { ArrowLeft: -1, ArrowRight: 1,
                      ArrowUp: -cols, ArrowDown: cols }[e.key];
      if (delta === undefined) return;
      e.preventDefault();
      const n = i + delta;
      const cells = cellNodes();
      if (n >= 0 && n < cells.length) cells[n].focus();
    }
  }

  // ── marquee (rubber-band) selection, only in select mode ────────────────────
  let wrapEl;
  let dragging = false, dragMoved = false, suppressClick = false;
  let baseSel = null;
  let sx = 0, sy = 0, cx = 0, cy = 0;

  function rel(e) {
    const r = wrapEl.getBoundingClientRect();
    return [e.clientX - r.left + wrapEl.scrollLeft, e.clientY - r.top + wrapEl.scrollTop];
  }
  function box() {
    return { left: Math.min(sx, cx), top: Math.min(sy, cy), right: Math.max(sx, cx), bottom: Math.max(sy, cy) };
  }
  function intersects(el, b) {
    const cr = wrapEl.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    const left = r.left - cr.left + wrapEl.scrollLeft;
    const top = r.top - cr.top + wrapEl.scrollTop;
    return !(left + r.width < b.left || left > b.right || top + r.height < b.top || top > b.bottom);
  }
  function onDown(e) {
    if (!selectMode || e.button !== 0) return;
    if (e.target.closest("button, a, input, select")) return;
    [sx, sy] = rel(e); cx = sx; cy = sy;
    dragging = true; dragMoved = false;
    baseSel = new Set(selected);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    e.preventDefault();  // don't start a text/image drag
  }
  function onMove(e) {
    if (!dragging) return;
    [cx, cy] = rel(e);
    if (!dragMoved && (Math.abs(cx - sx) > 4 || Math.abs(cy - sy) > 4)) dragMoved = true;
    if (!dragMoved) return;
    const b = box();
    const next = new Set(baseSel);
    cellNodes().forEach((el, i) => {
      const p = visible[i];
      if (el && p && p.exists !== false && intersects(el, b)) next.add(p.id);
    });
    selected = next; emit();
  }
  function onUp() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    if (dragMoved) suppressClick = true;  // swallow the click that follows a drag
    dragging = false; dragMoved = false;
  }

  // If the component unmounts mid-drag (e.g. switching tabs while marquee-
  // selecting), the window listeners added in onDown would otherwise leak.
  onDestroy(() => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  });
</script>

{#if visible.length === 0}
  <p class="muted">No photos to show.</p>
{:else}
  <div class="gridwrap" class:dragging bind:this={wrapEl} on:mousedown={onDown} role="presentation">
    <div class="grid" style="--rowh:{cols >= 6 ? 180 : 220}px">
      {#each visible as p, i (p.id)}
        <div class="cell" class:missing={p.exists === false} class:selected={selected.has(p.id)}
             style="--ar:{ratios[p.id] || 1}"
             on:click={(e) => onCellClick(e, p, i)}
             on:keydown={(e) => onCellKey(e, p, i)}
             role="button" tabindex="0">
          {#if p.exists === false}
            <div class="gone">🚫<br /><small>{p.filename}</small></div>
          {:else}
            <!-- Not loading="lazy": native lazy-loading unreliably stops
                 fetching images entirely after enough client-side tab
                 switches (Svelte destroys/remounts this grid on every {#if
                 tab === ...} navigation) — reproduced live, recoverable only
                 by a full page reload. Grids are capped (~200 items) and
                 thumbnails are small cached WebP, so eager loading is cheap. -->
            <img src={api.thumbUrl(p.id)} alt={p.filename} decoding="async"
                 draggable="false" on:load={(e) => onImgLoad(e, p)} />
            {#if p.media_type === "video"}
              <!-- Poster thumb is served by /api/image; badge marks it playable -->
              <div class="play" aria-hidden="true">▶</div>
              {#if fmtDuration(p.duration_s)}
                <div class="dur">{fmtDuration(p.duration_s)}</div>
              {/if}
            {/if}
            {#if p.caption}<div class="cap">{p.caption}</div>{/if}
            {#if selectMode || selected.has(p.id)}
              <div class="check" class:on={selected.has(p.id)}>{selected.has(p.id) ? "✓" : ""}</div>
            {/if}
          {/if}
        </div>
      {/each}
      <span class="rowfill"></span>
    </div>
    {#if dragging && dragMoved}
      <div class="marquee"
           style="left:{box().left}px; top:{box().top}px; width:{box().right - box().left}px; height:{box().bottom - box().top}px"></div>
    {/if}
  </div>
{/if}

<style>
  .gridwrap { position: relative; }
  .gridwrap.dragging { user-select: none; }
  /* Justified rows: each cell's flex share is its aspect ratio, so photos
     keep their shape and every row lines up flush (Google-Photos style). */
  .grid {
    display: flex; flex-wrap: wrap; gap: 8px;
  }
  .cell {
    position: relative; border-radius: 8px; overflow: hidden;
    background: var(--surface); cursor: pointer;
    border: 1px solid var(--border);
    height: var(--rowh);
    flex-grow: calc(var(--ar) * 100);
    flex-basis: calc(var(--ar) * var(--rowh));
    max-width: calc(var(--ar) * var(--rowh) * 1.6);
    /* Skip rendering/layout of off-screen cells so a 10k-photo grid stays
       smooth. The intrinsic size keeps the scrollbar accurate. */
    content-visibility: auto;
    contain-intrinsic-size: calc(var(--ar) * var(--rowh)) var(--rowh);
  }
  /* Absorb leftover space so a short last row doesn't stretch its photos. */
  .rowfill { flex-grow: 1000000; }
  .cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .cell:hover img { filter: brightness(1.1); }
  .cell:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .cell.selected { outline: 3px solid var(--accent); outline-offset: -3px; }
  .cap {
    position: absolute; bottom: 0; left: 0; right: 0;
    font-size: 10px; padding: 6px 4px 4px;
    background: linear-gradient(transparent, rgba(0,0,0,.82));
    color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    opacity: 0; transition: opacity .15s;
  }
  .cell:hover .cap, .cell:focus-visible .cap { opacity: 1; }
  /* Video affordances: centered play glyph + a bottom-right duration pill. */
  .play {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    width: 44px; height: 44px; border-radius: 50%;
    background: rgba(0,0,0,.5); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; padding-left: 3px; pointer-events: none;
    transition: background .15s;
  }
  .cell:hover .play { background: var(--accent); }
  .dur {
    position: absolute; bottom: 6px; right: 6px;
    background: rgba(0,0,0,.72); color: #fff;
    font-size: 11px; line-height: 1; padding: 3px 5px; border-radius: 4px;
    font-variant-numeric: tabular-nums; pointer-events: none;
  }
  .check {
    position: absolute; top: 6px; left: 6px; width: 20px; height: 20px;
    border-radius: 50%; border: 2px solid #fff; background: rgba(0,0,0,.35);
    color: #fff; font-size: 12px; line-height: 18px; text-align: center;
  }
  .check.on { background: var(--accent); border-color: var(--accent); }
  .cell.missing { cursor: default; border-style: dashed; border-color: var(--danger); }
  .gone { display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; color: var(--danger); font-size: 11px; text-align: center; }
  .marquee {
    position: absolute; z-index: 5; pointer-events: none;
    background: color-mix(in srgb, var(--accent) 22%, transparent);
    border: 1px solid var(--accent); border-radius: 2px;
  }
</style>
