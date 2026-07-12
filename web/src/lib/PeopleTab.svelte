<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount } from "svelte";
  import { onActivateKey } from "./keyboard.js";
  export let indexedCount = 0;
  const dispatch = createEventDispatcher();

  let persons = [];
  let active = null;
  let results = [];
  let loadingFind = false;
  let loading = true;
  let err = "";
  let findSeq = 0;  // request-sequencing guard so out-of-order find() responses can't clobber each other

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
    // The three are independent of each other's results, so run them in
    // parallel instead of a serial await chain (which delayed all three by
    // the sum of their latencies and left the empty-state messages showing
    // the whole time).
    try {
      await Promise.all([refresh(), loadFaceStatus(), loadClusters()]);
    } finally {
      loading = false;
    }
  });

  async function refresh() {
    try {
      const r = await api.people();
      // Prefer the detailed list (name + relation + is_family); fall back to
      // bare names for older servers.
      persons = r.detailed
        || (r.people || []).map((n) => ({ name: n, relation: "", is_family: false }));
    } catch (e) { err = e.message; }
  }

  const RELATION_OPTIONS = ["", "self", "spouse", "partner", "mother", "father",
    "son", "daughter", "brother", "sister", "grandmother", "grandfather",
    "aunt", "uncle", "cousin", "niece", "nephew", "friend", "colleague", "other"];
  let relationBusy = {};
  let familyOnly = false;
  $: shownPersons = familyOnly ? persons.filter((p) => p.is_family) : persons;

  async function setRelation(name, relation) {
    relationBusy = { ...relationBusy, [name]: true };
    try {
      await api.setPersonRelation(name, relation, null);
      await refresh();
    } catch (e) { err = e.message; }
    relationBusy = { ...relationBusy, [name]: false };
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
    const seq = ++findSeq;
    // Empty query = person-only browse: the backend returns ALL of the
    // person's photos from the face index, not a semantic top-k intersection.
    try {
      const r = await api.search("", {}, name);
      if (seq !== findSeq) return;  // a newer find() started; ignore this stale response
      results = r.results;
    } catch (e) {
      if (seq === findSeq) err = e.message;
    }
    if (seq === findSeq) loadingFind = false;
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
              <span class="face-wrap" role="button" tabindex="0"
                   on:click={() => dispatch("select", { id: s.image_id, ids: c.samples.map((x) => x.image_id) })}
                   on:keydown={(e) => onActivateKey(e, () => dispatch("select", { id: s.image_id, ids: c.samples.map((x) => x.image_id) }))}>
                <img class="face" src={api.faceCropUrl(s.image_id, s.face_index)} alt="face"
                     decoding="async" />
              </span>
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
    <div class="row" style="justify-content:space-between; align-items:center">
      <div class="section-label" style="margin:0">Registered People</div>
      {#if persons.some((p) => p.is_family)}
        <label class="row" style="gap:6px; font-size:12px; color:var(--muted)">
          <input type="checkbox" bind:checked={familyOnly} style="width:auto" /> Family only
        </label>
      {/if}
    </div>
    {#if err}<p style="color:var(--danger)">{err}</p>{/if}
    {#if loading}
      <p class="muted">Loading…</p>
    {:else if persons.length === 0}
      <p class="muted">No people registered yet. Name a face group above, or add someone on the right.</p>
    {:else}
      {#if indexedCount === 0}
        <p class="pill" style="color:var(--warn)">Index photos first to search by person.</p>
      {/if}
      {#if familyOnly && shownPersons.length === 0}
        <p class="muted">No one tagged as family yet — set a relation below.</p>
      {/if}
      {#each shownPersons as person (person.name)}
        <div class="row" style="justify-content:space-between; align-items:center">
          <span>👤 <b>{person.name}</b>
            {#if person.is_family}<span class="fam-badge" title="Family member">family</span>{/if}
          </span>
          <span class="row" style="gap:6px; flex-wrap:wrap; align-items:center">
            <select class="relation-select" value={person.relation}
                    on:change={(e) => setRelation(person.name, e.target.value)}
                    disabled={relationBusy[person.name]} title="Relationship (kept separate from the name)">
              <option value="">— relation —</option>
              {#each RELATION_OPTIONS.filter((r) => r) as rel}
                <option value={rel}>{rel}</option>
              {/each}
            </select>
            <button class="sm ghost" on:click={() => renamePersonUi(person.name)} title="Rename">✎</button>
            <button class="sm ghost" on:click={() => deletePersonUi(person.name)} title="Remove person">✕</button>
            <button on:click={() => find(person.name)} disabled={indexedCount === 0}>Find</button>
          </span>
        </div>
        {#if active === person.name}
          {#if loadingFind}
            <p class="muted">Finding photos of {person.name}…</p>
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
  .face-wrap { display: inline-block; cursor: pointer; border-radius: 6px; }
  .face-wrap:hover .face, .face-wrap:focus-visible { outline: 2px solid var(--accent); }
  .face { width: 52px; height: 52px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); display: block; }
  .meta { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .cluster input { flex: 1; min-width: 0; }
  .fam-badge { font-size: 10px; text-transform: uppercase; letter-spacing: .04em;
    background: color-mix(in srgb, var(--accent) 22%, transparent);
    color: var(--accent); border-radius: 4px; padding: 1px 6px; margin-left: 6px;
    vertical-align: middle; }
  .relation-select { font-size: 12px; padding: 3px 6px; max-width: 140px; }
</style>
