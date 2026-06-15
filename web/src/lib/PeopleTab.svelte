<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount } from "svelte";
  export let indexedCount = 0;
  const dispatch = createEventDispatcher();

  let persons = [];
  let active = null;
  let results = [];
  let loadingFind = false;
  let err = "";

  let newName = "";
  let refDir = "";
  let regMsg = "";
  let regBusy = false;

  onMount(refresh);
  async function refresh() {
    try { persons = (await api.people()).people; } catch (e) { err = e.message; }
  }
  async function find(name) {
    active = name; loadingFind = true; results = [];
    try { results = (await api.search("person photo", {}, name)).results; }
    catch (e) { err = e.message; }
    loadingFind = false;
  }
  async function register() {
    regBusy = true; regMsg = "";
    try {
      await api.addPerson(newName, refDir);
      regMsg = `✅ Registered ${newName}.`;
      newName = ""; refDir = "";
      await refresh();
    } catch (e) { regMsg = "⚠️ " + e.message; }
    regBusy = false;
  }
</script>

<div class="layout">
  <section class="card col" style="flex:1; min-width:0">
    <div class="section-label">Registered People</div>
    {#if err}<p style="color:var(--danger)">{err}</p>{/if}
    {#if persons.length === 0}
      <p class="muted">No people registered yet. Add someone on the right.</p>
    {:else}
      {#if indexedCount === 0}
        <p class="pill" style="color:var(--warn)">Index photos first to search by person.</p>
      {/if}
      {#each persons as name}
        <div class="row" style="justify-content:space-between">
          <span>👤 <b>{name}</b></span>
          <button on:click={() => find(name)} disabled={indexedCount === 0}>Find</button>
        </div>
        {#if active === name}
          {#if loadingFind}
            <p class="muted">Finding photos of {name}…</p>
          {:else}
            <PhotoGrid photos={results} cols={5}
                       on:select={(e) => dispatch("select", e.detail)} />
          {/if}
        {/if}
      {/each}
    {/if}
  </section>

  <aside class="card col" style="width:320px; flex-shrink:0">
    <div class="section-label">Register New Person</div>
    <p class="muted" style="font-size:13px">Folder of clear face photos (5–20 work best).</p>
    <input bind:value={newName} placeholder="Name" />
    <input bind:value={refDir} placeholder="Reference photos folder path" />
    <button class="primary" on:click={register} disabled={regBusy}>Register</button>
    {#if regMsg}<p style="font-size:13px">{regMsg}</p>{/if}
  </aside>
</div>

<style>
  .layout { display: flex; gap: 16px; align-items: flex-start; }
  @media (max-width: 800px) { .layout { flex-direction: column; } aside { width: 100% !important; } }
</style>
