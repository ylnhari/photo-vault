<script>
  import { createEventDispatcher, onDestroy, onMount, tick } from "svelte";
  import { api, fmtDuration } from "./api.js";
  import { lastDeleted } from "./stores.js";
  import { readableMatchFields } from "./browseSearch.js";

  export let photos = [];
  export let cols = 5; // retained for the existing Timeline and People layouts
  export let selectMode = false;
  export let resetToken = 0;
  const dispatch = createEventDispatcher();

  const GAP = 7;
  const MIN_CARD = 142;
  const OVERSCAN_ROWS = 2;
  let selected = new Set();
  let lastIndex = -1;
  let hidden = new Set();
  let imageErrors = new Set();
  let wrapEl;
  let columns = cols;
  let rowPitch = 170;
  let windowStart = 0;
  let windowEnd = 60;
  let forcedFocusIndex = null;
  let resizeObserver;
  let scrollFrame = 0;

  $: if ($lastDeleted?.length) {
    const additions = $lastDeleted.filter((id) => !hidden.has(id));
    if (additions.length) hidden = new Set([...hidden, ...additions]);
  }
  $: visible = photos.filter((photo) => !hidden.has(photo.id));
  $: ids = visible.map((photo) => photo.id);
  $: windowStart = Math.max(0, Math.min(windowStart, Math.max(0, visible.length - 1)));
  $: windowEnd = Math.min(visible.length, Math.max(windowStart, windowEnd));
  $: windowPhotos = visible.slice(windowStart, windowEnd);
  $: topSpace = Math.floor(windowStart / columns) * rowPitch;
  $: renderedRows = Math.ceil(windowPhotos.length / columns);
  $: bottomRows = Math.max(0, Math.ceil(visible.length / columns) - Math.ceil(windowStart / columns) - renderedRows);
  $: bottomSpace = bottomRows * rowPitch;
  $: gridStyle = `--columns:${columns}; --card-ratio:${Math.min(2.4,Math.max(.55,wrapEl?.clientWidth ? wrapEl.clientWidth / columns / Math.max(1, rowPitch - GAP) : 1.12))}`;

  let _lastReset = resetToken;
  $: if (resetToken !== _lastReset) {
    _lastReset = resetToken;
    selected = new Set();
    windowStart = 0;
    windowEnd = Math.min(60, visible.length);
    emit();
  }

  function emit() { dispatch("selectionchange", Array.from(selected)); }

  function measureAndWindow() {
    if (!wrapEl) return;
    const width = wrapEl.clientWidth;
    if (!width) return;
    columns = Math.max(1, Math.floor((width + GAP) / (MIN_CARD + GAP)));
    rowPitch = ((width - (columns - 1) * GAP) / columns) / 1.12 + GAP;
    updateWindow();
  }

  function updateWindow() {
    if (!wrapEl || !rowPitch) return;
    const gridTop = wrapEl.getBoundingClientRect().top + window.scrollY;
    const relativeTop = Math.max(0, window.scrollY - gridTop);
    const firstRow = Math.max(0, Math.floor(relativeTop / rowPitch) - OVERSCAN_ROWS);
    const visibleRows = Math.ceil(window.innerHeight / rowPitch) + OVERSCAN_ROWS * 2 + 1;
    const nextStart = Math.min(visible.length, firstRow * columns);
    const nextEnd = Math.min(visible.length, (firstRow + visibleRows) * columns);
    if (nextStart !== windowStart) windowStart = nextStart;
    if (nextEnd !== windowEnd) windowEnd = nextEnd;
  }

  function onScroll() {
    if (scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      updateWindow();
    });
  }

  function toggleSelect(photo, index) {
    if (selected.has(photo.id)) selected.delete(photo.id);
    else selected.add(photo.id);
    lastIndex = index;
    selected = selected;
    emit();
  }

  function onCellClick(event, photo, index) {
    if (suppressClick) { suppressClick = false; return; }
    if (photo.exists === false || imageErrors.has(photo.id)) return;
    const multi = selectMode || event.ctrlKey || event.metaKey || event.shiftKey;
    if (multi) {
      if (event.shiftKey && lastIndex >= 0) {
        const [start, end] = [Math.min(lastIndex, index), Math.max(lastIndex, index)];
        for (let i = start; i <= end; i++) if (visible[i]?.exists !== false) selected.add(visible[i].id);
        selected = selected;
        emit();
      } else toggleSelect(photo, index);
    } else {
      dispatch("select", { id: photo.id, ids, photos: visible });
    }
  }

  async function onCellKey(event, index) {
    if (!event.key.startsWith("Arrow")) return;
    const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns }[event.key];
    if (delta === undefined) return;
    event.preventDefault();
    const nextIndex = index + delta;
    if (nextIndex < 0 || nextIndex >= visible.length) return;
    forcedFocusIndex = nextIndex;
    const row = Math.floor(nextIndex / columns);
    const viewTop = Math.floor(windowStart / columns);
    const viewBottom = Math.ceil(windowEnd / columns);
    if (row < viewTop || row >= viewBottom) {
      const visibleRows = Math.ceil(window.innerHeight / rowPitch) + OVERSCAN_ROWS * 2 + 1;
      const startRow = Math.max(0, row - Math.floor(visibleRows / 2));
      windowStart = startRow * columns;
      windowEnd = Math.min(visible.length, (startRow + visibleRows) * columns);
    }
    await tick();
    wrapEl?.querySelector(`[data-photo-index="${nextIndex}"]`)?.focus();
    forcedFocusIndex = null;
  }

  function onImageError(event, photo) {
    const next = new Set(imageErrors);
    next.add(photo.id);
    imageErrors = next;
  }

  function cellNodes() {
    return wrapEl ? [...wrapEl.querySelectorAll("[data-photo-index]")] : [];
  }

  let dragging = false, dragMoved = false, suppressClick = false;
  let baseSel = null;
  let sx = 0, sy = 0, cx = 0, cy = 0;
  function rel(event) {
    const rect = wrapEl.getBoundingClientRect();
    return [event.clientX - rect.left + wrapEl.scrollLeft, event.clientY - rect.top + wrapEl.scrollTop];
  }
  function box() {
    return { left: Math.min(sx, cx), top: Math.min(sy, cy), right: Math.max(sx, cx), bottom: Math.max(sy, cy) };
  }
  function intersects(element, bounds) {
    const container = wrapEl.getBoundingClientRect();
    const rect = element.getBoundingClientRect();
    const left = rect.left - container.left + wrapEl.scrollLeft;
    const top = rect.top - container.top + wrapEl.scrollTop;
    return !(left + rect.width < bounds.left || left > bounds.right || top + rect.height < bounds.top || top > bounds.bottom);
  }
  function onDown(event) {
    if (!selectMode || event.button !== 0 || event.target.closest("a,input,select")) return;
    [sx, sy] = rel(event); cx = sx; cy = sy;
    dragging = true; dragMoved = false; baseSel = new Set(selected);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    event.preventDefault();
  }
  function onMove(event) {
    if (!dragging) return;
    [cx, cy] = rel(event);
    if (!dragMoved && (Math.abs(cx - sx) > 4 || Math.abs(cy - sy) > 4)) dragMoved = true;
    if (!dragMoved) return;
    const bounds = box();
    const next = new Set(baseSel);
    for (const element of cellNodes()) {
      const index = Number(element.dataset.photoIndex);
      const photo = visible[index];
      if (photo && photo.exists !== false && intersects(element, bounds)) next.add(photo.id);
    }
    selected = next;
    emit();
  }
  function onUp() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    if (dragMoved) suppressClick = true;
    dragging = false; dragMoved = false;
  }

  onMount(() => {
    resizeObserver = new ResizeObserver(measureAndWindow);
    if (wrapEl) resizeObserver.observe(wrapEl);
    measureAndWindow();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", measureAndWindow, { passive: true });
  });
  onDestroy(() => {
    resizeObserver?.disconnect();
    window.removeEventListener("scroll", onScroll);
    window.removeEventListener("resize", measureAndWindow);
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    if (scrollFrame) cancelAnimationFrame(scrollFrame);
  });
</script>

{#if visible.length === 0}
  <p class="muted">No photos to show.</p>
{:else}
  <div class="gridwrap" class:dragging bind:this={wrapEl} on:mousedown={onDown} role="presentation">
    <div class="grid" style={gridStyle}>
      {#if topSpace > 0}<div class="spacer" style={`height:${topSpace}px`} aria-hidden="true"></div>{/if}
      {#each windowPhotos as photo, offset (photo.id)}
        {@const index = windowStart + offset}
        {@const matchLabels = readableMatchFields(photo.match_fields)}
        {#if photo.exists === false || imageErrors.has(photo.id)}
          <div class="cell unavailable" class:selected={selected.has(photo.id)} data-photo-index={index} aria-label={`Unavailable: ${photo.filename || "photo"}`}>
            <span class="unavailable-icon" aria-hidden="true">⌁</span>
            <span>{photo.exists === false ? "File unavailable" : "Preview unavailable"}</span>
          </div>
        {:else}
          <button class="cell" class:selected={selected.has(photo.id)} class:video={photo.media_type === "video"}
            type="button" data-photo-index={index}
            aria-label={`${selectMode ? (selected.has(photo.id) ? "Deselect" : "Select") : "Open"} ${photo.caption || photo.filename || "memory"}${matchLabels.length ? `. Matched in ${matchLabels.join(", ")}` : ""}`}
            title={matchLabels.length ? `Matched in ${matchLabels.join(", ")}` : undefined}
            aria-pressed={selectMode ? selected.has(photo.id) : undefined}
            on:click={(event) => onCellClick(event, photo, index)}
            on:keydown={(event) => onCellKey(event, index)}>
            <img src={api.thumbUrl(photo.id)} alt={photo.caption || photo.filename || "Photo"}
              loading="lazy" decoding="async" draggable="false" on:error={(event) => onImageError(event, photo)} />
            {#if photo.media_type === "video"}
              <span class="play" aria-hidden="true">▶</span>
              {#if fmtDuration(photo.duration_s)}<span class="duration">{fmtDuration(photo.duration_s)}</span>{/if}
            {/if}
            {#if matchLabels.length}<span class="match-reason" title={`Matched in ${matchLabels.join(", ")}`}>{matchLabels[0]}</span>{/if}
            {#if photo.caption}<span class="caption">{photo.caption}</span>{/if}
            {#if selectMode || selected.has(photo.id)}
              <span class="selection-mark" class:on={selected.has(photo.id)} aria-hidden="true">{selected.has(photo.id) ? "✓" : ""}</span>
            {/if}
          </button>
        {/if}
      {/each}
      {#if bottomSpace > 0}<div class="spacer" style={`height:${bottomSpace}px`} aria-hidden="true"></div>{/if}
    </div>
    {#if dragging && dragMoved}
      <div class="marquee" style={`left:${box().left}px;top:${box().top}px;width:${box().right-box().left}px;height:${box().bottom-box().top}px`}></div>
    {/if}
  </div>
{/if}

<style>
  .gridwrap { position: relative; min-height: 1px; }
  .gridwrap.dragging { user-select: none; cursor: crosshair; }
  .grid { display: grid; grid-template-columns: repeat(var(--columns), minmax(0, 1fr)); gap: 7px; }
  .spacer { grid-column: 1 / -1; pointer-events: none; }
  .cell { position: relative; display: block; width: 100%; aspect-ratio: 1.12; padding: 0; overflow: hidden;
    border: 1px solid rgba(30,45,39,.07); border-radius: 8px; background: #e9ede8; cursor: pointer; }
  .cell:hover:not(:disabled) { transform: none; filter: none; box-shadow: 0 3px 12px rgba(25,42,35,.14); }
  .cell img { display: block; width: 100%; height: 100%; object-fit: cover; transition: transform .25s ease; }
  .cell:hover img { transform: scale(1.025); }
  .cell:focus-visible { z-index: 1; outline: 3px solid #3b8a80; outline-offset: 1px; }
  .cell.selected { outline: 3px solid #398a7d; outline-offset: -3px; }
  .caption { position: absolute; inset: auto 0 0; padding: 21px 9px 8px; color: #fff; text-align: left;
    background: linear-gradient(transparent,rgba(10,20,17,.68)); font-size: 10px; line-height: 1.35;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; opacity: .97; }
  .play { position: absolute; inset: 50% auto auto 50%; display: grid; place-items: center; width: 39px; height: 39px;
    padding-left: 3px; color: #fff; background: rgba(13,24,20,.5); border: 1px solid rgba(255,255,255,.55);
    border-radius: 50%; transform: translate(-50%,-50%); font-size: 14px; pointer-events: none; }
  .duration { position: absolute; right: 7px; bottom: 7px; padding: 3px 5px; color: #fff;
    background: rgba(10,18,16,.74); border-radius: 4px; font-size: 10px; font-variant-numeric: tabular-nums; }
  .match-reason { position: absolute; top: 7px; right: 7px; max-width: calc(100% - 14px); overflow: hidden; padding: 3px 6px; color: #fff; background: rgba(17,43,37,.82); border: 1px solid rgba(255,255,255,.35); border-radius: 20px; font-size: 9px; line-height: 1.3; text-overflow: ellipsis; white-space: nowrap; pointer-events: none; }
  .selection-mark { position: absolute; top: 7px; left: 7px; display: grid; place-items: center; width: 21px; height: 21px;
    color: #fff; border: 1.5px solid #fff; border-radius: 50%; background: rgba(20,34,29,.25); font-size: 12px; }
  .selection-mark.on { border-color: #398a7d; background: #398a7d; }
  .unavailable { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 5px;
    padding: 10px; color: #718079; text-align: center; font-size: 10px; border: 1px dashed #bdc7bf; cursor: default; }
  .unavailable-icon { color: #92a19a; font-size: 23px; }
  .marquee { position: absolute; z-index: 4; pointer-events: none; background: rgba(57,138,125,.18);
    border: 1px solid #398a7d; border-radius: 3px; }
  .muted { color: var(--muted); }
  @media (max-width: 760px) { .cell { border-radius: 5px; } .caption { padding: 16px 5px 5px; font-size: 9px; } }
</style>
