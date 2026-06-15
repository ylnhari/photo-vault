<script>
  import { api } from "./api.js";
  import { onMount, onDestroy, createEventDispatcher } from "svelte";
  import { health, status, models, refreshHealth, refreshStatus, refreshModels } from "./stores.js";
  import StatusPill from "./StatusPill.svelte";
  import JobPanel from "./JobPanel.svelte";
  const dispatch = createEventDispatcher();

  let provider = "auto";
  let maxFail = 5;
  let scanDirs = "";
  let busy = "";
  let err = "";
  let job = null;
  let poll = null;

  const PROVIDERS = [
    ["auto", "Auto (LM Studio → Gemini)"],
    ["lm_studio", "LM Studio only"],
    ["gemini", "Gemini only"],
  ];

  onMount(async () => {
    if (!$health.loaded) refreshHealth();
    if (!$status.loaded) refreshStatus();
    if (!$models.loaded) refreshModels();
    job = await api.indexProgress();
    if (job.active) startPolling();
  });
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
    try {
      job = await api.indexStart(type, provider, maxFail);
      if (job.active) startPolling();
      else { await refreshStatus(); dispatch("indexed"); }
    } catch (e) { err = e.message; }
  }
  async function stop() { job = await api.indexStop(); }
  async function retry() {
    const failed = job.failed_ids, type = job.type;
    await api.indexReset();
    err = "";
    try {
      job = await api.indexStart(type, provider, maxFail);  // backend recomputes pending
      if (job.active) startPolling();
    } catch (e) { err = e.message; }
  }
  async function clearJob() { await api.indexReset(); job = await api.indexProgress(); }

  async function scan() {
    busy = "scan"; err = "";
    try {
      const dirs = scanDirs.split("\n").map((s) => s.trim()).filter(Boolean);
      await api.scan(dirs);
      await refreshStatus();
    } catch (e) { err = e.message; }
    busy = "";
  }
  async function cleanup() {
    busy = "cleanup";
    try { const r = await api.cleanupMissing(); await refreshStatus();
      err = `Removed ${r.removed} stale entries.`; }
    catch (e) { err = e.message; }
    busy = "";
  }
  async function switchModel(m) { await api.setActiveModel(m); await Promise.all([refreshModels(), refreshStatus()]); dispatch("indexed"); }

  // Show the inline job panel inside a section when the active job matches it.
  const jobIs = (t) => job && (job.active || job.finished) && job.type === t;
</script>

{#if err}<div class="note-card">{err}</div>{/if}

<!-- Services -->
<div class="card">
  <div class="section-label">Services</div>
  <div class="row" style="flex-wrap:wrap; gap:10px">
    {#if !$health.loaded}
      <StatusPill label="Checking services…" state="unknown" />
    {:else}
      <StatusPill label="LM Studio" state={$health.lm_studio ? "on" : "off"}
                  detail={$health.lm_studio ? "vision + embeddings" : "offline"} />
      <StatusPill label="Gemini" state={$health.gemini ? "on" : ($health.gemini_key_set ? "warn" : "off")}
                  detail={$health.gemini ? "fallback ready" : ($health.gemini_key_set ? "unreachable" : "no key")} />
    {/if}
    <button class="ghost sm" on:click={refreshHealth} style="margin-left:auto">Recheck</button>
  </div>
  {#if noServices}
    <p class="err-text">Start LM Studio (vision + embedding model) or set GEMINI_API_KEY in .env to enable indexing.</p>
  {/if}
</div>

<!-- Pipeline -->
<div class="card">
  <div class="row" style="justify-content:space-between">
    <div class="section-label">Pipeline</div>
    <button class="ghost sm" on:click={refreshStatus}>Refresh</button>
  </div>
  <div class="pipeline">
    <div class="stage">
      <div class="num">{st.stage.total_scanned}</div><div class="lbl">Scanned</div>
    </div>
    <div class="arrow">→</div>
    <div class="stage" class:pending={st.vision_pending}>
      <div class="num">{st.stage.vision_done}</div><div class="lbl">Captioned</div>
      {#if st.vision_pending}<div class="pend">{st.vision_pending} to go</div>{/if}
    </div>
    <div class="arrow">→</div>
    <div class="stage" class:pending={st.embed_pending}>
      <div class="num">{st.stage.active_model_embedded}</div><div class="lbl">Embedded</div>
      {#if st.embed_pending}<div class="pend">{st.embed_pending} to go</div>{/if}
    </div>
  </div>
  {#if st.missing_files}
    <div class="row" style="justify-content:space-between; margin-top:10px">
      <span class="warn-text">⚠️ {st.missing_files} catalog entries point to missing files.</span>
      <button class="sm" on:click={cleanup} disabled={busy === 'cleanup'}>Clean up</button>
    </div>
  {/if}
</div>

<!-- Vision provider -->
<div class="card">
  <div class="section-label">Vision provider</div>
  <div class="row" style="flex-wrap:wrap; gap:16px">
    {#each PROVIDERS as [val, label]}
      <label class="radio"><input type="radio" bind:group={provider} value={val} /> {label}</label>
    {/each}
  </div>
  <label class="row" style="gap:8px; font-size:13px; margin-top:6px">
    Stop after <input type="number" bind:value={maxFail} min="1" max="20" style="width:60px" /> consecutive failures
  </label>
</div>

<!-- A: Scan -->
<div class="card">
  <div class="section-label">A · Scan for photos</div>
  <p class="hint">Discovers image files on disk (no AI). Re-scanning a folder only adds new files.</p>
  <textarea bind:value={scanDirs} rows="2" placeholder="One folder per line, e.g. C:\Users\you\Pictures"></textarea>
  <button on:click={scan} disabled={busy === 'scan' || running}>
    {busy === "scan" ? "Scanning…" : "Scan folders"}
  </button>
</div>

<!-- B: Vision -->
<div class="card">
  <div class="section-label">B · Vision analysis <span class="hint">(image → caption + attributes)</span></div>
  {#if jobIs("vision")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if st.vision_pending === 0}
    <p class="ok-text">✓ All {st.stage.total_scanned} scanned photos have captions.</p>
  {:else if running}
    <p class="hint">Another operation is running — finish or stop it first.</p>
  {:else}
    <button class="primary" on:click={() => start("vision")} disabled={noServices}>
      ▶ Caption {st.vision_pending} photo{st.vision_pending === 1 ? "" : "s"}
    </button>
  {/if}
</div>

<!-- C: Embed -->
<div class="card">
  <div class="section-label">C · Embed <span class="hint">(caption → searchable vector)</span></div>
  {#if jobIs("embed")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if st.embed_pending === 0}
    <p class="ok-text">{st.stage.vision_done > 0 ? "✓ All captioned photos are embedded." : "Caption some photos first (B)."}</p>
  {:else if running}
    <p class="hint">Another operation is running — finish or stop it first.</p>
  {:else}
    <button class="primary" on:click={() => start("embed")} disabled={noServices}>
      ▶ Embed {st.embed_pending} photo{st.embed_pending === 1 ? "" : "s"}
    </button>
  {/if}
</div>

<!-- D: Full index -->
<div class="card">
  <div class="section-label">D · Full index <span class="hint">(caption + embed in one pass)</span></div>
  {#if jobIs("full")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if st.missing_full === 0}
    <p class="ok-text">✓ All scanned photos are in the index.</p>
  {:else if running}
    <p class="hint">Another operation is running — finish or stop it first.</p>
  {:else}
    <p class="hint">Best for first-time setup: runs B then C for everything not yet indexed.</p>
    <button class="primary" on:click={() => start("full")} disabled={noServices}>
      ▶ Index {st.missing_full} photo{st.missing_full === 1 ? "" : "s"} from scratch
    </button>
  {/if}
</div>

<!-- E: Re-analyze -->
<div class="card">
  <div class="section-label">E · Re-analyze <span class="hint">(refresh captions + re-embed)</span></div>
  {#if jobIs("reanalyze")}
    <JobPanel {job} on:stop={stop} on:retry={retry} on:clear={clearJob} />
  {:else if st.missing_attrs === 0}
    <p class="ok-text">✓ All indexed photos have full attributes.</p>
  {:else if running}
    <p class="hint">Another operation is running — finish or stop it first.</p>
  {:else}
    <button on:click={() => start("reanalyze")} disabled={noServices}>
      🔄 Re-analyze {st.missing_attrs} stale photo{st.missing_attrs === 1 ? "" : "s"}
    </button>
  {/if}
</div>

<!-- F: Active embedding model -->
<div class="card">
  <div class="section-label">F · Active embedding model <span class="hint">(used by Search)</span></div>
  {#if !$models.loaded}
    <p class="hint">Loading…</p>
  {:else if Object.keys($models.models).length === 0}
    <p class="hint">No embedding models yet — run C or D first.</p>
  {:else}
    {#each Object.entries($models.models) as [name, info]}
      <div class="row" style="justify-content:space-between; padding:4px 0">
        <span>
          {#if name === $models.active}<span class="ok-text">✓</span>{/if}
          <b>{name}</b> <span class="hint">{info.source} · {info.dimension}d</span>
        </span>
        {#if name !== $models.active}
          <button class="sm" on:click={() => switchModel(name)}>Use for search</button>
        {/if}
      </div>
    {/each}
  {/if}
</div>

<style>
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; margin-bottom: 14px; }
  .note-card { background: var(--surface); border: 1px solid var(--warn);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; color: var(--warn); }
  .hint { color: var(--muted); font-size: 13px; font-weight: 400; }
  .ok-text { color: var(--success); font-size: 14px; }
  .err-text { color: var(--danger); font-size: 13px; margin-top: 8px; }
  .warn-text { color: var(--warn); font-size: 13px; }
  .sm { padding: 5px 10px; font-size: 13px; }
  .radio { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; }
  .radio input { width: auto; }
  textarea { margin-bottom: 10px; }

  .pipeline { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
  .stage { flex: 1; text-align: center; padding: 14px 8px; border-radius: 10px;
    background: var(--bg); border: 1px solid var(--border); position: relative; }
  .stage.pending { border-color: color-mix(in srgb, var(--warn) 45%, var(--border)); }
  .stage .num { font-size: 26px; font-weight: 700; }
  .stage .lbl { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
  .stage .pend { font-size: 11px; color: var(--warn); margin-top: 4px; }
  .arrow { color: var(--muted); font-size: 18px; }
</style>
