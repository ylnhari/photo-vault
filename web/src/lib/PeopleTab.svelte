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

  // face clustering / review
  let fstatus = null;
  let clusters = [];
  let clustering = false;
  let loadingClusters = false;
  let clusterErr = "";
  let names = {};        // cluster_id → typed name
  let rowBusy = {};

  onMount(async () => {
    await refresh();
    await loadFaceStatus();
    await loadClusters();
  });

  async function refresh() {
    try { persons = (await api.people()).people; } catch (e) { err = e.message; }
  }
  async function loadFaceStatus() {
    try { fstatus = await api.facesStatus(); } catch {}
  }
  async function loadClusters() {
    loadingClusters = true;
    try { clusters = (await api.facesClusters()).clusters || []; } catch (e) { clusterErr = e.message; }
    loadingClusters = false;
  }

  async function runClustering() {
    clustering = true; clusterErr = "";
    try {
      await api.facesCluster(null, null);
      await Promise.all([loadFaceStatus(), loadClusters()]);
    } catch (e) { clusterErr = e.message; }
    clustering = false;
  }

  let reindexing = false;
  async function reindexFaces() {
    reindexing = true; clusterErr = "";
    try {
      await api.facesReindex();
      await loadFaceStatus();
    } catch (e) { clusterErr = e.message; }
    reindexing = false;
  }

  async function nameCluster(cid) {
    const name = (names[cid] || "").trim();
    if (!name) return;
    rowBusy = { ...rowBusy, [cid]: true };
    try {
      await api.nameCluster(cid, name);
      clusters = clusters.filter((c) => c.cluster_id !== cid);
      await refresh();
    } catch (e) { clusterErr = e.message; }
    rowBusy = { ...rowBusy, [cid]: false };
  }
  async function ignoreCluster(cid) {
    rowBusy = { ...rowBusy, [cid]: true };
    try {
      await api.ignoreCluster(cid);
      clusters = clusters.filter((c) => c.cluster_id !== cid);
    } catch (e) { clusterErr = e.message; }
    rowBusy = { ...rowBusy, [cid]: false };
  }

  async function renamePersonUi(name) {
    const newName = prompt(`Rename “${name}” to:`, name);
    if (!newName || !newName.trim() || newName.trim() === name) return;
    err = "";
    try {
      await api.renamePerson(name, newName.trim());
      if (active === name) active = newName.trim();
      await refresh();
    } catch (e) { err = e.message; }
  }
  async function deletePersonUi(name) {
    if (!confirm(`Remove “${name}”? Their photos stay indexed — only the person label is deleted.`)) return;
    err = "";
    try {
      await api.deletePerson(name);
      if (active === name) { active = null; results = []; }
      await refresh();
    } catch (e) { err = e.message; }
  }

  async function find(name) {
    active = name; loadingFind = true; results = [];
    // Empty query = person-only browse: the backend returns ALL of the
    // person's photos from the face index, not a semantic top-k intersection.
    try { results = (await api.search("", {}, name)).results; }
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

<!-- Face groups review -->
<div class="card">
  <div class="row" style="justify-content:space-between; align-items:center">
    <div class="section-label">Face groups <span class="hint">(auto-grouped faces to review &amp; name)</span></div>
    <div class="row" style="gap:8px">
      {#if fstatus}
        <span class="hint" style="font-size:12px">
          {fstatus.detected}/{fstatus.total} photos scanned for faces
          {#if fstatus.pending > 0}· {fstatus.pending} pending (run “Face detection” in Index &amp; Manage){/if}
          {#if fstatus.detected > 0 && fstatus.ann_index_count === 0}· ⚠ face index not built{/if}
        </span>
      {/if}
      {#if fstatus && fstatus.detected > 0 && fstatus.ann_index_count === 0}
        <button class="ghost sm" on:click={reindexFaces} disabled={reindexing}>
          {reindexing ? "Building…" : "Build face index"}
        </button>
      {/if}
      <button class="primary sm" on:click={runClustering} disabled={clustering}>
        {clustering ? "Grouping…" : "Find face groups"}
      </button>
      <button class="ghost sm" on:click={loadClusters} disabled={loadingClusters}>Refresh</button>
    </div>
  </div>
  {#if clusterErr}<p class="err-text">{clusterErr}</p>{/if}

  {#if loadingClusters}
    <p class="muted">Loading groups…</p>
  {:else if clusters.length === 0}
    <p class="muted">No face groups yet. Detect faces first (Index &amp; Manage → Face detection), then click “Find face groups”.</p>
  {:else}
    <div class="clusters">
      {#each clusters as c (c.cluster_id)}
        <div class="cluster">
          <div class="faces">
            {#each c.samples as s}
              <img class="face" src={api.faceCropUrl(s.image_id, s.face_index)} alt="face"
                   loading="lazy" decoding="async"
                   on:click={() => dispatch("select", { id: s.image_id, ids: c.samples.map((x) => x.image_id) })} />
            {/each}
          </div>
          <div class="meta">{c.size} face{c.size === 1 ? "" : "s"}</div>
          <div class="row" style="gap:6px; margin-top:6px">
            <input placeholder="Name this person" bind:value={names[c.cluster_id]}
                   on:keydown={(e) => e.key === "Enter" && nameCluster(c.cluster_id)} />
            <button class="sm primary" on:click={() => nameCluster(c.cluster_id)}
                    disabled={rowBusy[c.cluster_id] || !(names[c.cluster_id] || '').trim()}>Save</button>
            <button class="sm ghost" on:click={() => ignoreCluster(c.cluster_id)}
                    disabled={rowBusy[c.cluster_id]}>Ignore</button>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<div class="layout">
  <section class="card col" style="flex:1; min-width:0">
    <div class="section-label">Registered People</div>
    {#if err}<p style="color:var(--danger)">{err}</p>{/if}
    {#if persons.length === 0}
      <p class="muted">No people registered yet. Name a face group above, or add someone on the right.</p>
    {:else}
      {#if indexedCount === 0}
        <p class="pill" style="color:var(--warn)">Index photos first to search by person.</p>
      {/if}
      {#each persons as name}
        <div class="row" style="justify-content:space-between">
          <span>👤 <b>{name}</b></span>
          <span class="row" style="gap:6px">
            <button class="sm ghost" on:click={() => renamePersonUi(name)} title="Rename">✎</button>
            <button class="sm ghost" on:click={() => deletePersonUi(name)} title="Remove person">✕</button>
            <button on:click={() => find(name)} disabled={indexedCount === 0}>Find</button>
          </span>
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
    <div class="section-label">Register from folder</div>
    <p class="muted" style="font-size:13px">Folder of clear face photos (5–20 work best).</p>
    <input bind:value={newName} placeholder="Name" />
    <input bind:value={refDir} placeholder="Reference photos folder path" />
    <button class="primary" on:click={register} disabled={regBusy}>Register</button>
    {#if regMsg}<p style="font-size:13px">{regMsg}</p>{/if}
  </aside>
</div>

<style>
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; margin-bottom: 14px; }
  .layout { display: flex; gap: 16px; align-items: flex-start; }
  @media (max-width: 800px) { .layout { flex-direction: column; } aside { width: 100% !important; } }
  .hint { color: var(--muted); font-size: 13px; font-weight: 400; }
  .muted { color: var(--muted); }
  .err-text { color: var(--danger); font-size: 13px; }
  .sm { padding: 5px 10px; font-size: 13px; }
  .ghost { background: transparent; border-color: var(--border); }

  .clusters { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin-top: 10px; }
  .cluster { border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
  .faces { display: flex; gap: 4px; flex-wrap: wrap; }
  .face { width: 52px; height: 52px; object-fit: cover; border-radius: 6px; cursor: pointer; border: 1px solid var(--border); }
  .face:hover { outline: 2px solid var(--accent); }
  .meta { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .cluster input { flex: 1; min-width: 0; }
</style>
