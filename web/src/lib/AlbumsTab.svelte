<script>
  import { api } from "./api.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createEventDispatcher, onMount } from "svelte";
  import { onActivateKey } from "./keyboard.js";
  const dispatch = createEventDispatcher();

  let albums = [];
  let err = "";
  let newName = "";
  let creating = false;
  let loading = true;

  // detail view
  let openId = null;
  let openName = "";
  let photos = [];
  let loadingDetail = false;
  let selectMode = false;
  let selectedIds = [];
  let resetToken = 0;
  let busy = false;

  onMount(async () => {
    try { await loadAlbums(); } finally { loading = false; }
  });
  async function loadAlbums() {
    try { albums = (await api.albums()).albums; } catch (e) { err = e.message; }
  }

  async function create() {
    if (!newName.trim()) return;
    creating = true; err = "";
    try { await api.createAlbum(newName.trim()); newName = ""; await loadAlbums(); }
    catch (e) { err = e.message; }
    creating = false;
  }

  async function open(a) {
    openId = a.id; openName = a.name; selectMode = false; selectedIds = []; loadingDetail = true;
    try { const d = await api.getAlbum(a.id); photos = d.photos; openName = d.name; }
    catch (e) { err = e.message; }
    loadingDetail = false;
  }
  function back() { openId = null; photos = []; loadAlbums(); }

  async function rename() {
    const name = prompt("Rename album", openName);
    if (!name || !name.trim()) return;
    try { await api.renameAlbum(openId, name.trim()); openName = name.trim(); }
    catch (e) { err = e.message; }
  }
  async function removeAlbum() {
    if (!confirm(`Delete album “${openName}”? Photos themselves are not deleted.`)) return;
    try { await api.deleteAlbum(openId); back(); } catch (e) { err = e.message; }
  }

  function onSelectionChange(e) { selectedIds = e.detail; }
  async function removeSelected() {
    if (!selectedIds.length) return;
    busy = true;
    try {
      await api.albumRemove(openId, selectedIds);
      const set = new Set(selectedIds);
      photos = photos.filter((p) => !set.has(p.id));
      selectedIds = []; resetToken += 1; selectMode = false;
    } catch (e) { err = e.message; }
    busy = false;
  }
</script>

{#if err}<div class="note">{err} <button class="ghost sm" on:click={() => err = ""}>×</button></div>{/if}

{#if openId === null}
  <!-- album list -->
  <div class="row" style="gap:8px; margin-bottom:14px; flex-wrap:wrap">
    <input bind:value={newName} placeholder="New album name"
           on:keydown={(e) => e.key === "Enter" && create()} />
    <button class="primary" on:click={create} disabled={creating || !newName.trim()}>
      {creating ? "Creating…" : "Create album"}
    </button>
  </div>

  {#if loading}
    <p class="muted">Loading…</p>
  {:else if albums.length === 0}
    <p class="muted">No albums yet. Create one above, or select photos in Search and “Add to album”.</p>
  {:else}
    <div class="albumgrid">
      {#each albums as a (a.id)}
        <div class="album" on:click={() => open(a)} on:keydown={(e) => onActivateKey(e, () => open(a))}
             role="button" tabindex="0">
          <div class="cover">
            {#if a.cover}
              <img src={api.thumbUrl(a.cover)} alt={a.name} loading="lazy" decoding="async" />
            {:else}
              <div class="empty">📁</div>
            {/if}
          </div>
          <div class="aname" title={a.name}>{a.name}</div>
          <div class="muted" style="font-size:12px">{a.count} photo{a.count === 1 ? "" : "s"}</div>
        </div>
      {/each}
    </div>
  {/if}
{:else}
  <!-- album detail -->
  <div class="row" style="gap:10px; margin-bottom:12px; flex-wrap:wrap; align-items:center">
    <button class="ghost sm" on:click={back}>← Albums</button>
    <h2 style="margin:0; font-size:18px">{openName}</h2>
    <span class="muted" style="font-size:13px">{photos.length} photo{photos.length === 1 ? "" : "s"}</span>
    <div style="flex:1"></div>
    <button class="ghost sm" on:click={rename}>Rename</button>
    <button class="ghost sm danger" on:click={removeAlbum}>Delete album</button>
  </div>

  <div class="row" style="gap:10px; margin-bottom:10px; min-height:30px; flex-wrap:wrap">
    <button class="ghost sm" class:active={selectMode}
            on:click={() => { selectMode = !selectMode; if (!selectMode) { selectedIds = []; resetToken += 1; } }}>
      {selectMode ? "✓ Selecting" : "Select"}
    </button>
    {#if selectedIds.length}
      <span class="muted" style="font-size:13px">{selectedIds.length} selected</span>
      <button class="sm danger" on:click={removeSelected} disabled={busy}>
        {busy ? "Removing…" : `Remove ${selectedIds.length} from album`}
      </button>
    {/if}
  </div>

  {#if loadingDetail}
    <p class="muted">Loading…</p>
  {:else}
    <PhotoGrid {photos} {selectMode} {resetToken}
               on:select={(e) => dispatch("select", e.detail)}
               on:selectionchange={onSelectionChange} />
  {/if}
{/if}

<style>
  .note { background: var(--surface); border: 1px solid var(--warn); color: var(--warn);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;
    display: flex; justify-content: space-between; gap: 12px; }
  .muted { color: var(--muted); }
  .sm { padding: 5px 10px; font-size: 13px; }
  .ghost { background: transparent; border-color: var(--border); }
  .ghost.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .danger { color: var(--danger); border-color: var(--danger); }
  .danger.sm:hover { background: var(--danger); color: #fff; }

  .albumgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 14px; }
  .album { cursor: pointer; border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    background: var(--surface); }
  .album:hover { border-color: var(--accent); }
  .album:focus-visible { outline: 2px solid var(--accent); }
  .cover { aspect-ratio: 1; background: var(--bg); display: flex; align-items: center; justify-content: center; }
  .cover img { width: 100%; height: 100%; object-fit: cover; }
  .empty { font-size: 40px; opacity: .5; }
  .aname { font-weight: 600; padding: 8px 10px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .album .muted { padding: 0 10px 10px; }
</style>
