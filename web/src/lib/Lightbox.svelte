<script>
  import { api } from "./api.js";
  import { createEventDispatcher, onMount } from "svelte";
  export let id;
  const dispatch = createEventDispatcher();

  let meta = null;
  let err = "";
  let confirmDelete = false;
  let busy = false;

  const ATTRS = [
    ["caption", "Caption"], ["scene", "Scene"], ["weather", "Weather"],
    ["occasion", "Occasion"], ["mood", "Mood"], ["location_type", "Location"],
    ["group_size", "Group"], ["clothing_style", "Clothing"], ["season", "Season"],
    ["time_of_day", "Time"], ["year", "Year"], ["embedding_source", "Embedding"],
  ];

  onMount(load);
  async function load() {
    try { meta = await api.meta(id); } catch (e) { err = e.message; }
  }
  function close() { dispatch("close"); }
  async function remove(deleteFile) {
    busy = true;
    try {
      await api.deleteImage(id, deleteFile);
      dispatch("deleted", id);
    } catch (e) { err = e.message; busy = false; }
  }
</script>

<div class="overlay" on:click|self={close} role="presentation">
  <div class="box">
    <button class="ghost close" on:click={close}>✕</button>
    <div class="content">
      <div class="imgwrap">
        <img src={api.fullUrl(id)} alt="" />
      </div>
      <div class="side col">
        {#if err}<p style="color:var(--danger)">{err}</p>{/if}
        {#if meta}
          {#each ATTRS as [k, label]}
            {#if meta[k] && meta[k] !== "unknown"}
              <div><span class="muted">{label}:</span> {meta[k]}</div>
            {/if}
          {/each}
        {:else if !err}
          <p class="muted">Loading…</p>
        {/if}

        <div class="section-label" style="margin-top:16px">Manage</div>
        <button on:click={() => remove(false)} disabled={busy}>Remove from index</button>
        <label class="row" style="font-size:13px">
          <input type="checkbox" bind:checked={confirmDelete} style="width:auto" />
          Confirm permanent delete
        </label>
        <button class="danger" on:click={() => remove(true)} disabled={!confirmDelete || busy}>
          🗑 Delete file from disk
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
  .content { display: grid; grid-template-columns: 1.6fr 1fr; gap: 0; max-height: 90vh; }
  .imgwrap { background: #000; display: flex; align-items: center; justify-content: center; }
  .imgwrap img { max-width: 100%; max-height: 90vh; object-fit: contain; }
  .side { padding: 24px; overflow-y: auto; }
  @media (max-width: 700px) { .content { grid-template-columns: 1fr; } }
</style>
