<script>
  // Modal folder browser backed by GET /api/fs/list — browsers can't hand a
  // real filesystem path to a web page, so the server lists drives/folders
  // and the user clicks down to the one they want.
  import { createEventDispatcher } from "svelte";
  import { api } from "./api.js";
  const dispatch = createEventDispatcher();

  export let open = false;
  export let title = "Select a folder";

  let cur = null;      // current path (null = drive list)
  let parent = null;
  let dirs = [];
  let loading = false;
  let err = "";

  $: if (open) load(null);

  async function load(path) {
    loading = true; err = "";
    try {
      const r = await api.fsList(path);
      cur = r.path; parent = r.parent; dirs = r.dirs;
    } catch (e) { err = e.message; }
    loading = false;
  }

  function enter(name) {
    // At the drive level entries are already full roots like "D:\".
    load(cur ? cur.replace(/\\$/, "") + "\\" + name : name);
  }

  function choose() {
    if (cur) { dispatch("select", { path: cur }); dispatch("close"); }
  }
  function cancel() { dispatch("close"); }
  function onKey(e) { if (e.key === "Escape") cancel(); }
</script>

{#if open}
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <div class="overlay" role="dialog" aria-modal="true" aria-label={title}
       tabindex="-1" on:keydown={onKey}>
    <div class="picker">
      <div class="head">
        <b>{title}</b>
        <button class="ghost sm" on:click={cancel} aria-label="Close">✕</button>
      </div>
      <div class="crumb" title={cur || "Drives"}>
        {#if cur}
          <button class="ghost sm" on:click={() => load(parent)} disabled={loading}
                  aria-label="Up one level">↑ Up</button>
          <span class="path">{cur}</span>
        {:else}
          <span class="path">Select a drive</span>
        {/if}
      </div>
      {#if err}<p class="err-text" style="padding:0 14px">{err}</p>{/if}
      <div class="list" aria-busy={loading}>
        {#if loading}
          <p class="hint" style="padding:10px 14px">Loading…</p>
        {:else if !dirs.length}
          <p class="hint" style="padding:10px 14px">No subfolders here.</p>
        {:else}
          {#each dirs as d (d)}
            <button class="row-btn" on:click={() => enter(d)}>📁 {d}</button>
          {/each}
        {/if}
      </div>
      <div class="foot">
        <button class="ghost" on:click={cancel}>Cancel</button>
        <button class="primary" on:click={choose} disabled={!cur}>
          Use this folder{cur ? `: ${cur}` : ""}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .overlay {
    position: fixed; inset: 0; z-index: 200;
    background: rgba(0,0,0,.45);
    display: flex; align-items: center; justify-content: center;
  }
  .picker {
    width: min(560px, 92vw); max-height: 78vh;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); box-shadow: var(--shadow-2);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .head { display: flex; justify-content: space-between; align-items: center;
          padding: 12px 14px; border-bottom: 1px solid var(--border); }
  .crumb { display: flex; gap: 8px; align-items: center; padding: 8px 14px;
           border-bottom: 1px solid var(--border); }
  .crumb .path { font-family: ui-monospace, monospace; font-size: 12px;
                 overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .list { overflow-y: auto; flex: 1; padding: 6px 8px; min-height: 180px; }
  .row-btn { display: block; width: 100%; text-align: left; padding: 7px 10px;
             background: none; border: none; border-radius: 8px; font-size: 13px;
             cursor: pointer; color: var(--text); }
  .row-btn:hover { background: var(--surface2); }
  .foot { display: flex; justify-content: flex-end; gap: 10px;
          padding: 12px 14px; border-top: 1px solid var(--border); }
  .foot .primary { max-width: 70%; overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap; }
  .hint { color: var(--muted); font-size: 13px; }
  .err-text { color: var(--danger); font-size: 13px; }
  .sm { padding: 4px 9px; font-size: 12px; }
</style>
