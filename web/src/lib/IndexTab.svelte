<script>
  import { api } from "./api.js";
  import { onMount, onDestroy, createEventDispatcher } from "svelte";
  import { health, status, models, refreshHealth, refreshStatus, refreshModels } from "./stores.js";
  import StatusPill from "./StatusPill.svelte";
  import JobPanel from "./JobPanel.svelte";
  const dispatch = createEventDispatcher();

  // ── job / poll state ────────────────────────────────────────────────────────
  let job = null;
  let poll = null;
  let busy = {};
  let err = "";
  let rechecking = false;

  // ── provider model catalogue ─────────────────────────────────────────────────
  let pmodels = { lm_studio: [], lm_studio_types: {}, gemini_vision: [], gemini_embed: [] };

  // ── app settings ─────────────────────────────────────────────────────────────
  let settings = {
    vision_provider: "auto", vision_model: null,
    embed_provider: "auto",  embed_model: null,
    caption_source_model: null,
    max_fail: 5,
  };
  let settingsDirty = false;
  let settingsSaving = false;

  // ── folder management ─────────────────────────────────────────────────────────
  let folderConfig = { included: [], excluded: [] };
  let newFolderPath = "";
  let newExcludePath = "";
  let defaults = [];
  let scanResults = null;
  let confirmRemove = null;
  let showExcluded = false;

  // ── orphaned ──────────────────────────────────────────────────────────────────
  let orphaned = { orphaned: [], total: 0 };
  let showOrphaned = false;
  let orphanedBusy = false;

  const PROVIDERS = [
    ["auto", "Auto"],
    ["lm_studio", "LM Studio"],
    ["gemini", "Gemini"],
  ];

  onMount(async () => {
    if (!$health.loaded) refreshHealth();
    if (!$status.loaded) refreshStatus();
    if (!$models.loaded) refreshModels();
    await Promise.all([
      loadSettings(),
      loadFolderConfig(),
      loadOrphaned(),
      api.providerModels().then(r => { pmodels = r; }).catch(() => {}),
    ]);
    job = await api.indexProgress();
    if (job.active) startPolling();
  });

  async function loadSettings() {
    try { settings = await api.getSettings(); settingsDirty = false; } catch {}
  }
  async function loadFolderConfig() {
    try { folderConfig = await api.getFolderConfig(); } catch {}
  }
  async function loadOrphaned() {
    try { orphaned = await api.getOrphaned(); } catch {}
  }

  // ── settings derivations ─────────────────────────────────────────────────────
  $: lmTypes = pmodels.lm_studio_types || {};

  // LM Studio models filtered by type
  $: lmVisionModels = pmodels.lm_studio.filter(m => {
    const t = lmTypes[m]?.type;
    return t === "vision" || t === "unknown";
  });
  $: lmEmbedModels = pmodels.lm_studio.filter(m => {
    const t = lmTypes[m]?.type;
    return t === "embed" || t === "unknown";
  });

  // Dropdown options for current provider selection
  $: visionModelOpts = settings.vision_provider === "lm_studio" ? lmVisionModels
                     : settings.vision_provider === "gemini" ? pmodels.gemini_vision
                     : [];
  $: embedModelOpts  = settings.embed_provider === "lm_studio" ? lmEmbedModels
                     : settings.embed_provider === "gemini" ? pmodels.gemini_embed
                     : [];

  // All previously used vision model labels (from caption_history summary)
  $: usedVisionModels = Object.keys($status.model_status?.vision?.model_summary || {});

  // Vision model label for the current settings (e.g. "lm_studio:qwen2-vl-7b")
  $: selectedVisionLabel = (settings.vision_provider && settings.vision_provider !== "auto" && settings.vision_model)
    ? `${settings.vision_provider}:${settings.vision_model}` : null;

  // Model-aware counts from extended status
  $: ms = $status.model_status || {};
  $: mVision = ms.vision || {};
  $: mEmbed = ms.embed || {};
  $: totalScanned = $status.stage?.total_scanned || 0;
  $: visionPending = $status.vision_pending || 0;
  $: embedPending = $status.embed_pending || 0;
  $: missingFull = $status.missing_full || 0;
  $: missingAttrs = $status.missing_attrs || 0;
  $: facesPending = $status.faces_pending || 0;
  $: facesDone = $status.faces_done || 0;

  // ── settings helpers ──────────────────────────────────────────────────────────
  function markDirty() { settingsDirty = true; }

  async function saveSettings() {
    settingsSaving = true; err = "";
    try {
      settings = await api.saveSettings(settings);
      settingsDirty = false;
      await refreshStatus();
    } catch (e) { err = e.message; }
    settingsSaving = false;
  }

  // When provider changes, clear model selection
  function onVisionProviderChange() { settings.vision_model = null; markDirty(); }
  function onEmbedProviderChange()  { settings.embed_model = null;  markDirty(); }

  // ── job helpers ───────────────────────────────────────────────────────────────
  function buildCfg(type) {
    return {
      type,
      vision_provider: settings.vision_provider,
      vision_model: settings.vision_provider === "auto" ? null : (settings.vision_model || null),
      embed_provider: settings.embed_provider,
      embed_model: settings.embed_provider === "auto" ? null : (settings.embed_model || null),
      caption_source_model: settings.caption_source_model || null,
      max_fail: settings.max_fail,
    };
  }

  onDestroy(() => clearInterval(poll));

  function startPolling() {
    clearInterval(poll);
    poll = setInterval(async () => {
      job = await api.indexProgress();
      if (!job.active) {
        clearInterval(poll);
        await Promise.all([refreshStatus(), refreshModels()]);
        dispatch("indexed");
      }
    }, 1000);
  }

  $: noServices = $health.loaded && !$health.lm_studio && !$health.gemini;
  $: running = job && job.active;
  $: st = $status;

  async function start(type) {
    err = "";
    // Save settings first if dirty
    if (settingsDirty) await saveSettings();
    try {
      job = await api.indexStart(buildCfg(type));
      if (job.active) startPolling();
      else { await refreshStatus(); dispatch("indexed"); }
    } catch (e) { err = e.message; }
  }
  async function stop() { job = await api.indexStop(); }
  async function retry() {
    const type = job.type;
    await api.indexReset();
    err = "";
    try { job = await api.indexStart(buildCfg(type)); if (job.active) startPolling(); }
    catch (e) { err = e.message; }
  }
  async function clearJob() { await api.indexReset(); job = await api.indexProgress(); }

  async function recheckHealth() {
    rechecking = true;
    await refreshHealth();
    rechecking = false;
  }

  // ── scan ──────────────────────────────────────────────────────────────────────
  async function scan() {
    busy = { ...busy, scan: true }; err = ""; scanResults = null;
    try {
      const res = await api.scan();
      scanResults = res.summary;
      await Promise.all([loadFolderConfig(), loadOrphaned(), refreshStatus()]);
    } catch (e) { err = e.message; }
    busy = { ...busy, scan: false };
  }

  // ── folder actions ────────────────────────────────────────────────────────────
  async function addFolder() {
    const path = newFolderPath.trim(); if (!path) return;
    busy = { ...busy, addFolder: true }; err = "";
    try {
      const res = await api.addIncludedFolder(path);
      if (res.status === "redundant") err = `Already covered by "${res.covered_by}".`;
      else if (res.status === "duplicate") err = "Folder already in list.";
      else { newFolderPath = ""; await loadFolderConfig(); }
    } catch (e) { err = e.message; }
    busy = { ...busy, addFolder: false };
  }
  async function suggestDefaults() {
    try { defaults = (await api.getFolderDefaults()).defaults || []; } catch {}
  }
  async function addDefault(p) { newFolderPath = p; await addFolder(); defaults = []; }
  function requestRemoveFolder(f) { confirmRemove = { path: f.path, imageCount: f.image_count || 0 }; }
  async function confirmRemoveFolder() {
    if (!confirmRemove) return;
    busy = { ...busy, removeFolder: true }; err = "";
    try {
      await api.removeIncludedFolder(confirmRemove.path, true);
      confirmRemove = null;
      await Promise.all([loadFolderConfig(), loadOrphaned(), refreshStatus()]);
    } catch (e) { err = e.message; }
    busy = { ...busy, removeFolder: false };
  }
  async function addExclude() {
    const path = newExcludePath.trim(); if (!path) return;
    busy = { ...busy, addExclude: true }; err = "";
    try {
      const res = await api.addExcludedFolder(path);
      if (res.status === "duplicate") err = "Already excluded.";
      else { newExcludePath = ""; await loadFolderConfig(); }
    } catch (e) { err = e.message; }
    busy = { ...busy, addExclude: false };
  }
  async function removeExclude(path) {
    busy = { ...busy, removeExclude: path };
    try { await api.removeExcludedFolder(path); await loadFolderConfig(); }
    catch (e) { err = e.message; }
    busy = { ...busy, removeExclude: null };
  }

  // ── orphaned ──────────────────────────────────────────────────────────────────
  async function removeOrphanedAll() {
    orphanedBusy = true; err = "";
    try { await api.cleanupOrphaned([]); await Promise.all([loadOrphaned(), refreshStatus()]); }
    catch (e) { err = e.message; }
    orphanedBusy = false;
  }
  async function removeOrphanedOne(id) {
    busy = { ...busy, [id]: true };
    try { await api.cleanupOrphaned([id]); await Promise.all([loadOrphaned(), refreshStatus()]); }
    catch (e) { err = e.message; }
    busy = { ...busy, [id]: false };
  }

  // ── model management ──────────────────────────────────────────────────────────
  async function switchActiveModel(name) {
    await api.setActiveModel(name);
    await Promise.all([refreshModels(), refreshStatus()]);
    dispatch("indexed");
  }

  const jobIs = (t) => job && (job.active || job.finished) && job.type === t;
  function fmtDate(iso) { return iso ? iso.slice(0, 16).replace("T", " ") : "never scanned"; }

  function modelTypeLabel(id, provider) {
    if (provider === "lm_studio") {
      const t = lmTypes[id]?.type;
      return t === "embed" ? "[EMBED ONLY]" : t === "unknown" ? "[?]" : "";
    }
    return "";
  }
</script>

{#if err}
  <div class="note-card">{err} <button class="ghost sm" on:click={() => err = ""}>×</button></div>
{/if}

{#if confirmRemove}
  <div class="confirm-card">
    <span class="warn-text">⚠ Removing <b>{confirmRemove.path}</b> will delete
    {confirmRemove.imageCount > 0 ? `~${confirmRemove.imageCount}` : "all"} indexed images from this folder out of your library.
    Files on disk are NOT deleted.</span>
    <div class="row" style="gap:10px; margin-top:10px">
      <button on:click={() => confirmRemove = null}>Cancel</button>
      <button class="danger" on:click={confirmRemoveFolder} disabled={busy.removeFolder}>
        {busy.removeFolder ? "Removing…" : "Confirm Remove"}
      </button>
    </div>
  </div>
{/if}

<!-- Services -->
<div class="card">
  <div class="row" style="justify-content:space-between">
    <div class="section-label">Services</div>
    <button class="ghost sm" on:click={recheckHealth} disabled={rechecking}>
      {rechecking ? "⟳ Checking…" : "Recheck"}
    </button>
  </div>
  <div class="row" style="flex-wrap:wrap; gap:10px; margin-top:8px">
    {#if !$health.loaded}
      <StatusPill label="Checking…" state="unknown" />
    {:else}
      <StatusPill label="LM Studio" state={$health.lm_studio ? "on" : "off"}
                  detail={$health.lm_studio ? "online" : "offline"} />
      <StatusPill label="Gemini" state={$health.gemini ? "on" : ($health.gemini_key_set ? "warn" : "off")}
                  detail={$health.gemini ? "fallback ready" : ($health.gemini_key_set ? "unreachable" : "no key")} />
    {/if}
  </div>
  {#if noServices}
    <p class="err-text">Start LM Studio or set GEMINI_API_KEY in .env to enable indexing.</p>
  {/if}
</div>

<!-- Pipeline overview -->
<div class="card">
  <div class="row" style="justify-content:space-between">
    <div class="section-label">Pipeline</div>
    <button class="ghost sm" on:click={refreshStatus}>Refresh</button>
  </div>
  <div class="pipeline">
    <div class="stage">
      <div class="num">{totalScanned}</div>
      <div class="lbl">Scanned</div>
    </div>
    <div class="arrow">→</div>
    <div class="stage" class:pending={visionPending > 0}>
      <div class="num">{mVision.done ?? st.stage?.vision_done ?? 0}</div>
      <div class="lbl">Captioned</div>
      {#if visionPending > 0}<div class="pend">{visionPending} pending</div>{/if}
      {#if mVision.selected_label}
        <div class="model-badge">{mVision.selected_label}</div>
      {/if}
    </div>
    <div class="arrow">→</div>
    <div class="stage" class:pending={embedPending > 0}>
      <div class="num">{mEmbed.done ?? st.stage?.active_model_embedded ?? 0}</div>
      <div class="lbl">Embedded</div>
      {#if embedPending > 0}<div class="pend">{embedPending} pending</div>{/if}
      {#if mEmbed.selected_model}
        <div class="model-badge">{mEmbed.selected_model}</div>
      {/if}
    </div>
  </div>
  {#if mVision.model_summary && Object.keys(mVision.model_summary).length > 1}
    <div class="model-summary">
      {#each Object.entries(mVision.model_summary) as [label, count]}
        <span class="badge">{label}: {count}</span>
      {/each}
    </div>
  {/if}
</div>

<!-- Orphaned images -->
{#if orphaned.total > 0}
<div class="card warn-card">
  <div class="row" style="justify-content:space-between; align-items:center">
    <span class="warn-text">⚠ {orphaned.total} image{orphaned.total === 1 ? "" : "s"} point to missing files</span>
    <div class="row" style="gap:8px">
      <button class="ghost sm" on:click={() => showOrphaned = !showOrphaned}>
        {showOrphaned ? "Hide" : "Show"}
      </button>
      <button class="sm danger" on:click={removeOrphanedAll} disabled={orphanedBusy}>
        {orphanedBusy ? "Removing…" : `Remove all`}
      </button>
    </div>
  </div>
  {#if showOrphaned}
    <div class="orphan-list">
      {#each orphaned.orphaned as img}
        <div class="orphan-row">
          <span class="orphan-path" title={img.path}>{img.path}</span>
          <button class="sm ghost" on:click={() => removeOrphanedOne(img.id)} disabled={busy[img.id]}>
            {busy[img.id] ? "…" : "Remove"}
          </button>
        </div>
      {/each}
    </div>
  {/if}
</div>
{/if}

<!-- ── MODEL CONFIGURATION ─────────────────────────────────────────────────── -->
<div class="card">
  <div class="row" style="justify-content:space-between; align-items:center">
    <div class="section-label">Model Configuration</div>
    {#if settingsDirty}
      <div class="row" style="gap:8px">
        <span class="warn-text" style="font-size:12px">Unsaved changes</span>
        <button class="sm primary" on:click={saveSettings} disabled={settingsSaving}>
          {settingsSaving ? "Saving…" : "Save"}
        </button>
        <button class="sm ghost" on:click={loadSettings}>Discard</button>
      </div>
    {:else}
      <button class="ghost sm" on:click={loadSettings}>Reload</button>
    {/if}
  </div>

  <div class="cfg-grid">
    <!-- Vision model -->
    <div class="col">
      <div class="cfg-label">Vision (captioning)</div>
      <div class="row" style="flex-wrap:wrap; gap:10px; margin-bottom:8px">
        {#each PROVIDERS as [val, label]}
          <label class="radio">
            <input type="radio" bind:group={settings.vision_provider} value={val}
                   on:change={onVisionProviderChange} /> {label}
          </label>
        {/each}
      </div>
      {#if settings.vision_provider !== "auto"}
        {#if visionModelOpts.length}
          <select bind:value={settings.vision_model} on:change={markDirty}>
            <option value={null}>— auto-pick —</option>
            {#each visionModelOpts as m}
              <option value={m}>{m} {modelTypeLabel(m, settings.vision_provider)}</option>
            {/each}
          </select>
        {:else}
          <p class="warn-text hint">No {settings.vision_provider === "lm_studio" ? "LM Studio" : "Gemini"} vision models found.</p>
        {/if}
      {:else}
        <p class="hint">LM Studio first → Gemini fallback. Model auto-detected.</p>
      {/if}
      {#if selectedVisionLabel}
        <p class="hint" style="margin-top:4px">Label in history: <code>{selectedVisionLabel}</code></p>
      {/if}
    </div>

    <!-- Embed model -->
    <div class="col">
      <div class="cfg-label">Embedding (search vectors)</div>
      <div class="row" style="flex-wrap:wrap; gap:10px; margin-bottom:8px">
        {#each PROVIDERS as [val, label]}
          <label class="radio">
            <input type="radio" bind:group={settings.embed_provider} value={val}
                   on:change={onEmbedProviderChange} /> {label}
          </label>
        {/each}
      </div>
      {#if settings.embed_provider !== "auto"}
        {#if embedModelOpts.length}
          <select bind:value={settings.embed_model} on:change={markDirty}>
            <option value={null}>— auto-pick —</option>
            {#each embedModelOpts as m}
              <option value={m}>{m}</option>
            {/each}
          </select>
        {:else}
          <p class="warn-text hint">No {settings.embed_provider === "lm_studio" ? "LM Studio" : "Gemini"} embed models found.</p>
        {/if}
      {:else}
        <p class="hint">LM Studio embed model first → Gemini text-embedding fallback.</p>
      {/if}
    </div>
  </div>

  <!-- Caption source for embedding -->
  <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:12px">
    <div class="cfg-label">Caption source for embedding</div>
    <p class="hint" style="margin-bottom:8px">
      Which vision analysis to convert to embeddings. When an image has been captioned
      by multiple models, this selects which caption to use.
    </p>
    <select bind:value={settings.caption_source_model} on:change={markDirty} style="max-width:400px">
      <option value={null}>Latest caption (any model)</option>
      {#each usedVisionModels as label}
        <option value={label}>{label}</option>
      {/each}
    </select>
    {#if settings.caption_source_model}
      <p class="hint" style="margin-top:4px">
        Embedding eligible: {mEmbed.eligible ?? "?"} images with a caption from
        <code>{settings.caption_source_model}</code>
      </p>
    {/if}
  </div>

  <!-- Max failures + face toggle -->
  <div style="margin-top:12px; display:flex; flex-direction:column; gap:10px">
    <label class="row" style="gap:8px; font-size:13px">
      Abort job after
      <input type="number" bind:value={settings.max_fail} min="1" max="50"
             on:change={markDirty} style="width:60px" />
      consecutive failures
    </label>
    <label class="row" style="gap:8px; font-size:13px">
      <input type="checkbox" bind:checked={settings.faces_during_embed}
             on:change={markDirty} style="width:auto" />
      Detect faces during embedding
      <span class="hint">(off → run face detection separately below)</span>
    </label>
  </div>
</div>

<!-- A: Folder Management -->
<div class="card">
  <div class="section-label">A · Folder Management</div>
  <p class="hint">Manages which directories are scanned. Photos are tracked by content — moves and renames within scanned folders are detected automatically.</p>

  <div class="subsection-label">Included folders</div>
  {#if folderConfig.included.length === 0}
    <p class="hint" style="margin-bottom:10px">No folders added yet.</p>
  {:else}
    <div class="folder-list">
      {#each folderConfig.included as folder}
        <div class="folder-row">
          <div class="folder-info">
            <span class="folder-path" title={folder.path}>{folder.path}</span>
            <span class="hint">
              {folder.image_count > 0 ? `${folder.image_count} files` : "not yet scanned"}
              {#if folder.last_scanned_at} · {fmtDate(folder.last_scanned_at)}{/if}
            </span>
          </div>
          <button class="sm ghost danger-hover" on:click={() => requestRemoveFolder(folder)} disabled={busy.scan}>Remove</button>
        </div>
        {#if scanResults?.per_folder?.[folder.path]}
          {@const s = scanResults.per_folder[folder.path]}
          <div class="scan-result-row">
            {#if s.error}<span class="err-text">Error: {s.error}</span>
            {:else}
              <span class="ok-text">+{s.added} new</span>
              {#if s.moved}<span class="warn-text"> · {s.moved} moved</span>{/if}
              <span class="hint"> · {s.unchanged} unchanged · {s.scanned} found</span>
            {/if}
          </div>
        {/if}
      {/each}
    </div>
  {/if}

  <div class="add-row">
    <input bind:value={newFolderPath} placeholder="Folder path…"
           on:keydown={(e) => e.key === "Enter" && addFolder()}
           disabled={busy.scan} />
    <button on:click={addFolder} disabled={!newFolderPath.trim() || busy.addFolder || busy.scan}>
      {busy.addFolder ? "Adding…" : "Add folder"}
    </button>
    <button class="ghost" on:click={suggestDefaults} disabled={busy.scan}>Suggest defaults</button>
  </div>
  {#if defaults.length}
    <div class="defaults-list">
      <span class="hint">Suggested:</span>
      {#each defaults as d}
        <button class="ghost sm" on:click={() => addDefault(d)}>{d}</button>
      {/each}
    </div>
  {/if}

  <div class="subsection-label" style="margin-top:14px">
    <button class="ghost sm" on:click={() => showExcluded = !showExcluded} style="font-size:inherit; padding:0">
      Excluded subfolders {folderConfig.excluded?.length ? `(${folderConfig.excluded.length})` : ""} {showExcluded ? "▴" : "▾"}
    </button>
  </div>
  {#if showExcluded}
    {#if !folderConfig.excluded?.length}
      <p class="hint">No exclusions.</p>
    {:else}
      <div class="folder-list" style="margin-bottom:8px">
        {#each folderConfig.excluded as ex}
          <div class="folder-row">
            <span class="folder-path" title={ex.path}>{ex.path}</span>
            <button class="sm ghost" on:click={() => removeExclude(ex.path)} disabled={busy.scan}>Remove exclusion</button>
          </div>
        {/each}
      </div>
    {/if}
    <div class="add-row">
      <input bind:value={newExcludePath} placeholder="Subfolder to skip…"
             on:keydown={(e) => e.key === "Enter" && addExclude()}
             disabled={busy.scan} />
      <button on:click={addExclude} disabled={!newExcludePath.trim() || busy.addExclude || busy.scan}>
        {busy.addExclude ? "Adding…" : "Exclude folder"}
      </button>
    </div>
  {/if}

  <div style="margin-top:16px">
    <button class="primary" on:click={scan}
            disabled={busy.scan || running || folderConfig.included.length === 0}>
      {busy.scan ? "Scanning…" : `Scan ${folderConfig.included.length} folder${folderConfig.included.length === 1 ? "" : "s"}`}
    </button>
  </div>
  {#if busy.scan}
    <div class="scan-progress"><span></span></div>
  {/if}
  {#if scanResults && !busy.scan}
    <div class="scan-summary">
      <span class="ok-text">+{scanResults.added} new</span>
      {#if scanResults.moved}<span class="warn-text"> · {scanResults.moved} moved</span>{/if}
      <span class="hint"> · {scanResults.unchanged} unchanged · {scanResults.total} total</span>
      {#if scanResults.reconciled}<span class="hint"> · {scanResults.reconciled} index paths updated</span>{/if}
    </div>
  {/if}
</div>

<!-- B: Vision analysis -->
<div class="card">
  <div class="section-label">B · Vision analysis
    <span class="hint">(image → caption + 12 attributes)</span>
  </div>
  {#if selectedVisionLabel}
    <p class="hint">Running with: <code>{selectedVisionLabel}</code>
      · {visionPending} image{visionPending === 1 ? "" : "s"} pending for this model</p>
  {:else}
    <p class="hint">{visionPending} image{visionPending === 1 ? "" : "s"} pending (any model)</p>
  {/if}
  {#if jobIs("vision")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if visionPending === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All photos have captions{selectedVisionLabel ? ` from ${selectedVisionLabel}` : ""}.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if running}
    <p class="hint">Another job is running — stop it first.</p>
  {:else}
    <button class="primary" on:click={() => start("vision")} disabled={noServices}>
      ▶ Caption {visionPending} photo{visionPending === 1 ? "" : "s"}
    </button>
  {/if}
</div>

<!-- C: Embed -->
<div class="card">
  <div class="section-label">C · Embed
    <span class="hint">(caption → searchable vector)</span>
  </div>
  {#if settings.caption_source_model}
    <p class="hint">Source: captions from <code>{settings.caption_source_model}</code>
      → embed model: <code>{settings.embed_model || "auto"}</code>
      · {embedPending} pending</p>
  {:else}
    <p class="hint">Source: latest caption · {embedPending} pending</p>
  {/if}
  {#if jobIs("embed")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if embedPending === 0 && mEmbed.eligible > 0}
    <p class="ok-text">✓ All eligible captions are embedded.</p>
  {:else if embedPending === 0}
    <p class="hint">Run vision analysis first (B).</p>
  {:else if running}
    <p class="hint">Another job is running — stop it first.</p>
  {:else}
    <button class="primary" on:click={() => start("embed")} disabled={noServices}>
      ▶ Embed {embedPending} photo{embedPending === 1 ? "" : "s"}
    </button>
  {/if}
</div>

<!-- C2: Face detection (separate, user-controlled stage) -->
<div class="card">
  <div class="section-label">Face detection
    <span class="hint">(detect + embed faces for person search)</span>
  </div>
  <p class="hint">{facesDone} done · {facesPending} pending.
    {settings.faces_during_embed ? "Also runs automatically during embedding." : "Runs only when you start it here."}</p>
  {#if jobIs("faces")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if facesPending === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All photos have been scanned for faces.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if running}
    <p class="hint">Another job is running — stop it first.</p>
  {:else}
    <button class="primary" on:click={() => start("faces")}>
      ▶ Detect faces in {facesPending} photo{facesPending === 1 ? "" : "s"}
    </button>
  {/if}
</div>

<!-- D: Full index -->
<div class="card">
  <div class="section-label">D · Full index <span class="hint">(caption + embed in one pass)</span></div>
  {#if jobIs("full")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if missingFull === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All scanned photos are in the index.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if running}
    <p class="hint">Another job is running — stop it first.</p>
  {:else}
    <p class="hint">First-time setup: runs B then C for everything not yet indexed.</p>
    <button class="primary" on:click={() => start("full")} disabled={noServices}>
      ▶ Index {missingFull} photo{missingFull === 1 ? "" : "s"} from scratch
    </button>
  {/if}
</div>

<!-- E: Re-analyze -->
<div class="card">
  <div class="section-label">E · Re-analyze <span class="hint">(refresh captions + re-embed)</span></div>
  {#if jobIs("reanalyze")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if missingAttrs === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All indexed photos have full attributes.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if running}
    <p class="hint">Another job is running — stop it first.</p>
  {:else}
    <button on:click={() => start("reanalyze")} disabled={noServices}>
      🔄 Re-analyze {missingAttrs} stale photo{missingAttrs === 1 ? "" : "s"}
    </button>
  {/if}
</div>

<!-- F: Active search model -->
<div class="card">
  <div class="section-label">F · Active search model <span class="hint">(index used by Search + Timeline)</span></div>
  {#if !$models.loaded}
    <p class="hint">Loading…</p>
  {:else if Object.keys($models.models).length === 0}
    <p class="hint">No embedding models yet — run C or D first.</p>
  {:else}
    {#each Object.entries($models.models) as [name, info]}
      <div class="row" style="justify-content:space-between; padding:4px 0">
        <span>
          {#if name === $models.active}<span class="ok-text">✓ </span>{/if}
          <b>{name}</b>
          <span class="hint">{info.source} · {info.dimension}d · {info.indexed_count} photos</span>
        </span>
        {#if name !== $models.active}
          <button class="sm" on:click={() => switchActiveModel(name)}>Use for search</button>
        {/if}
      </div>
    {/each}
  {/if}
</div>

<style>
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; margin-bottom: 14px; }
  .warn-card { border-color: color-mix(in srgb, var(--warn) 50%, var(--border)); }
  .note-card { background: var(--surface); border: 1px solid var(--warn);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; color: var(--warn);
    display: flex; justify-content: space-between; align-items: flex-start; gap:12px; }
  .confirm-card { background: var(--surface); border: 1px solid var(--danger);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
  .section-label { font-size: 14px; font-weight: 600; margin-bottom: 8px; }
  .subsection-label { font-size: 12px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing:.05em; margin: 12px 0 6px; }
  .cfg-label { font-size: 12px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing:.05em; margin-bottom: 8px; }
  .hint { color: var(--muted); font-size: 13px; font-weight: 400; margin: 0; }
  .ok-text { color: var(--success); font-size: 14px; }
  .err-text { color: var(--danger); font-size: 13px; }
  .warn-text { color: var(--warn); font-size: 13px; }
  .sm { padding: 5px 10px; font-size: 13px; }
  .ghost { background: transparent; border-color: var(--border); }
  .primary { background: var(--accent, #5b8def); color: #fff; border-color: var(--accent, #5b8def); }
  .danger { background: var(--danger); color: #fff; border-color: var(--danger); }
  .danger-hover:hover { color: var(--danger); border-color: var(--danger); }
  .radio { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; }
  .radio input { width: auto; }
  code { background: var(--surface2); padding: 1px 6px; border-radius: 4px; font-size: 11px; }

  /* model config grid */
  .cfg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 12px; }
  .cfg-grid .col { display: flex; flex-direction: column; gap: 6px; }
  @media (max-width: 720px) { .cfg-grid { grid-template-columns: 1fr; } }

  /* pipeline */
  .pipeline { display: flex; align-items: stretch; gap: 8px; margin-top: 8px; }
  .stage { flex: 1; text-align: center; padding: 14px 8px; border-radius: 10px;
    background: var(--bg); border: 1px solid var(--border); }
  .stage.pending { border-color: color-mix(in srgb, var(--warn) 45%, var(--border)); }
  .stage .num { font-size: 26px; font-weight: 700; }
  .stage .lbl { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing:.06em; }
  .stage .pend { font-size: 11px; color: var(--warn); margin-top: 4px; }
  .stage .model-badge { font-size: 10px; color: var(--muted); margin-top: 4px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .arrow { color: var(--muted); font-size: 18px; align-self: center; }
  .model-summary { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
  .badge { font-size: 11px; background: var(--surface2); border-radius: 4px;
    padding: 2px 7px; color: var(--muted); }

  /* folder management */
  .folder-list { margin: 8px 0 4px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .folder-row { display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; border-bottom: 1px solid var(--border); gap: 12px; }
  .folder-row:last-child { border-bottom: none; }
  .folder-info { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .folder-path { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .scan-result-row { padding: 4px 12px 6px; font-size: 12px; background: var(--bg);
    border-bottom: 1px solid var(--border); }
  .scan-result-row:last-child { border-bottom: none; }
  .add-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  .add-row input { flex: 1; min-width: 200px; }
  .defaults-list { display: flex; align-items: center; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
  .scan-summary { margin-top: 10px; font-size: 13px; }
  .scan-progress { height: 4px; border-radius: 99px; background: var(--surface2);
    overflow: hidden; margin: 10px 0; }
  .scan-progress > span { display: block; height: 100%; background: var(--accent);
    width: 30%; animation: scan-indeterminate 1.5s ease-in-out infinite; border-radius: 99px; }
  @keyframes scan-indeterminate { 0% { transform: translateX(-100%); } 100% { transform: translateX(400%); } }

  /* orphaned */
  .orphan-list { margin-top: 10px; border: 1px solid color-mix(in srgb, var(--warn) 40%, var(--border));
    border-radius: 8px; overflow: hidden; }
  .orphan-row { display: flex; justify-content: space-between; align-items: center;
    padding: 6px 12px; border-bottom: 1px solid var(--border); gap: 12px; }
  .orphan-row:last-child { border-bottom: none; }
  .orphan-path { font-size: 12px; color: var(--muted); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
</style>
