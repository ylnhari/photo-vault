<script>
  import { api } from "./api.js";
  import { createEventDispatcher } from "svelte";
  export let photos = [];
  export let cols = 5;
  export let selectMode = false;
  export let resetToken = 0;       // bump from parent to clear the selection
  const dispatch = createEventDispatcher();

  let selected = new Set();
  let lastIndex = -1;

  $: ids = photos.map((p) => p.id);

  // Clear selection when the parent bumps resetToken (e.g. after a batch delete).
  let _lastReset = resetToken;
  $: if (resetToken !== _lastReset) { _lastReset = resetToken; selected = new Set(); emit(); }

  // Reset cached cell element refs whenever the photo set changes.
  let cellEls = [];
  $: if (photos) cellEls = [];

  function emit() { dispatch("selectionchange", Array.from(selected)); }

  function onCellClick(e, p, i) {
    if (suppressClick) { suppressClick = false; return; }
    if (p.exists === false) return;
    const multi = selectMode || e.ctrlKey || e.metaKey || e.shiftKey;
    if (multi) {
      if (e.shiftKey && lastIndex >= 0) {
        const [a, b] = [Math.min(lastIndex, i), Math.max(lastIndex, i)];
        for (let k = a; k <= b; k++) if (photos[k].exists !== false) selected.add(photos[k].id);
      } else {
        if (selected.has(p.id)) selected.delete(p.id);
        else selected.add(p.id);
        lastIndex = i;
      }
      selected = selected;  // trigger reactivity
      emit();
    } else {
      dispatch("select", { id: p.id, ids });
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
    cellEls.forEach((el, i) => {
      const p = photos[i];
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
</script>

{#if photos.length === 0}
  <p class="muted">No photos to show.</p>
{:else}
  <div class="gridwrap" class:dragging bind:this={wrapEl} on:mousedown={onDown} role="presentation">
    <div class="grid" style="--cols:{cols}">
      {#each photos as p, i (p.id)}
        <div class="cell" class:missing={p.exists === false} class:selected={selected.has(p.id)}
             bind:this={cellEls[i]}
             on:click={(e) => onCellClick(e, p, i)}
             on:keydown={(e) => e.key === "Enter" && dispatch("select", { id: p.id, ids })}
             role="button" tabindex="0">
          {#if p.exists === false}
            <div class="gone">🚫<br /><small>{p.filename}</small></div>
          {:else}
            <img src={api.thumbUrl(p.id)} alt={p.filename} loading="lazy" decoding="async" draggable="false" />
            {#if p.caption}<div class="cap">{p.caption}</div>{/if}
            {#if selectMode || selected.has(p.id)}
              <div class="check" class:on={selected.has(p.id)}>{selected.has(p.id) ? "✓" : ""}</div>
            {/if}
          {/if}
        </div>
      {/each}
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
  .grid {
    display: grid;
    grid-template-columns: repeat(var(--cols), 1fr);
    gap: 8px;
  }
  .cell {
    position: relative; border-radius: 8px; overflow: hidden;
    background: var(--surface); aspect-ratio: 1; cursor: pointer;
    border: 1px solid var(--border);
    /* Skip rendering/layout of off-screen cells so a 10k-photo grid stays
       smooth. The intrinsic size keeps the scrollbar accurate. */
    content-visibility: auto;
    contain-intrinsic-size: 220px;
  }
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
