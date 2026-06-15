<script>
  import { api } from "./api.js";
  import { onMount, onDestroy, createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();

  let svc = {};
  let st = null;          // /api/status
  let job = null;         // /api/index/progress
  let models = { active: null, models: {} };
  let provider = "auto";
  let maxFail = 5;
  let scanDirs = "";
  let busy = "";
  let err = "";
  let poll = null;

  const PROVIDERS = [
    ["auto", "Auto (LM Studio → Gemini)"],
    ["lm_studio", "LM Studio only"],
    ["gemini", "Gemini only"],
  ];

  onMount(async () => {
    await refreshAll();
    job = await api.indexProgress();
    if (job.active) startPolling();
  });
  onDestroy(() => clearInterval(poll));

  async function refreshAll() {
    try {
      [svc, st, models] = await Promise.all([api.health(), api.status(), api.models()]);
    } catch (e) { err = e.message; }
  }

  function startPolling() {
    clearInterval(poll);
    poll = setInterval(async () => {
      job = await api.indexProgress();
      if (!job.active) {
        clearInterval(poll);
        await refreshAll();
        dispatch("indexed");
      }
    }, 1000);
  }

  const noServices = () => svc && !svc.lm_studio && !svc.gemini;

  async function start(type) {
    err = "";
    try {
      job = await api.indexStart(type, provider, maxFail);
      if (job.active) startPolling();
      else { await refreshAll(); dispatch("indexed"); }
    } catch (e) { err = e.message; }
  }
  async function stop() { job = await api.indexStop(); }
  async function clearJob() { await api.indexReset(); job = await api.indexProgress(); }

  async function scan() {
    busy = "scan"; err = "";
    try {
      const dirs = scanDirs.split("\n").map((s) => s.trim()).filter(Boolean);
      st = await api.scan(dirs);
    } catch (e) { err = e.message; }
    busy = "";
  }
  async function cleanup() {
    busy = "cleanup";
    try { const r = await api.cleanupMissing(); await refreshAll();
      err = `Removed ${r.removed} stale entries.`; }
    catch (e) { err = e.message; }
    busy = "";
  }
  async function switchModel(m) {
    await api.setActiveModel(m); await refreshAll(); dispatch("indexed");
  }

  $: pct = job && job.total ? Math.round((job.done / job.total) * 100) : 0;
  $: running = job && job.active;
</script>

{#if err}<div class="card" style="border-color:var(--warn); margin-bottom:12px">{err}</div>{/if}

<!-- Live job panel -->
{#if job && (job.active || job.finished)}
  <div class="card" style="margin-bottom:16px; border-color:var(--accent)">
    <div class="row" style="justify-content:space-between">
      <b>⏳ {job.type} — {running ? "running" : "finished"}</b>
      {#if running}
        <button class="danger" on:click={stop}>🛑 Stop</button>
      {:else}
        <div class="row">
          {#if job.failed_ids?.length}
            <button on:click={() => start(job.type)}>Retry failed ({job.failed_ids.length})</button>
          {/if}
          <button class="ghost" on:click={clearJob}>Clear</button>
        </div>
      {/if}
    </div>
    <div class="bar" style="margin:10px 0"><span style="width:{pct}%"></span></div>
    <div class="muted" style="font-size:13px">
      {pct}% · processed {job.done}/{job.total} · ✅ {job.ok} · ❌ {job.fail}
      {#if job.aborted}<span style="color:var(--danger)"> · auto-stopped (too many failures)</span>{/if}
      {#if job.stopped}<span style="color:var(--warn)"> · stopped by you</span>{/if}
    </div>
    {#if job.log?.length}
      <div class="log">
        {#each job.log.slice(-25) as l}
          <div>{l.kind === "fail" ? "❌" : l.kind === "cloud" ? "☁️" : "✅"}
            {l.id.split(/[\\/]/).pop()}: {l.note}</div>
        {/each}
      </div>
    {/if}
  </div>
{/if}

<!-- Services -->
<div class="card col" style="margin-bottom:16px">
  <div class="section-label">Services</div>
  <div class="row" style="gap:24px">
    <div>LM Studio: <b style="color:{svc.lm_studio ? 'var(--success)' : 'var(--danger)'}">
      {svc.lm_studio ? "online" : "offline"}</b></div>
    <div>Gemini: <b style="color:{svc.gemini ? 'var(--success)' : 'var(--muted)'}">
      {svc.gemini ? "online" : svc.gemini_key_set ? "unreachable" : "no key"}</b></div>
  </div>
  {#if noServices()}
    <p style="color:var(--danger)">No services available — start LM Studio or set GEMINI_API_KEY in .env.</p>
  {/if}
</div>

<!-- Status -->
{#if st}
  <div class="card col" style="margin-bottom:16px">
    <div class="row" style="justify-content:space-between">
      <div class="section-label">Index Status</div>
      <button class="ghost" on:click={refreshAll}>Refresh</button>
    </div>
    <div class="row" style="gap:28px">
      <div><div class="muted" style="font-size:12px">Scanned</div><b style="font-size:22px">{st.stage.total_scanned}</b></div>
      <div><div class="muted" style="font-size:12px">Captioned</div><b style="font-size:22px">{st.stage.vision_done}</b>
        {#if st.vision_pending}<span class="pill">-{st.vision_pending}</span>{/if}</div>
      <div><div class="muted" style="font-size:12px">Embedded</div><b style="font-size:22px">{st.stage.active_model_embedded}</b>
        {#if st.embed_pending}<span class="pill">-{st.embed_pending}</span>{/if}</div>
    </div>
    {#if st.missing_files}
      <div class="row" style="justify-content:space-between; margin-top:8px">
        <span style="color:var(--warn)">⚠️ {st.missing_files} catalog entries point to missing files.</span>
        <button on:click={cleanup} disabled={busy === 'cleanup'}>Clean up</button>
      </div>
    {/if}
  </div>
{/if}

<!-- Provider -->
<div class="card col" style="margin-bottom:16px">
  <div class="section-label">Vision Provider</div>
  <div class="row" style="flex-wrap:wrap">
    {#each PROVIDERS as [val, label]}
      <label class="row" style="gap:6px; font-size:14px">
        <input type="radio" bind:group={provider} value={val} style="width:auto" /> {label}
      </label>
    {/each}
  </div>
  <label class="row" style="gap:8px; font-size:14px">
    Stop after <input type="number" bind:value={maxFail} min="1" max="20" style="width:64px" />
    consecutive failures
  </label>
</div>

<!-- Actions -->
<div class="card col" style="margin-bottom:16px">
  <div class="section-label">A — Scan for new photos</div>
  <textarea bind:value={scanDirs} rows="2" placeholder="One folder per line, e.g. C:\Users\you\Pictures"></textarea>
  <button on:click={scan} disabled={busy === 'scan' || running} style="width:max-content">
    {busy === "scan" ? "Scanning…" : "Scan folders"}
  </button>
</div>

{#if st}
<div class="card col" style="margin-bottom:16px">
  <div class="section-label">B — Vision analysis (image → text)</div>
  {#if st.vision_pending === 0}<p class="muted">All scanned photos have captions.</p>
  {:else}<p>{st.vision_pending} photos need vision analysis.</p>
    <button class="primary" on:click={() => start("vision")} disabled={running || noServices()} style="width:max-content">
      ▶ Run vision on {st.vision_pending}</button>{/if}
</div>

<div class="card col" style="margin-bottom:16px">
  <div class="section-label">C — Embed (text → searchable vector)</div>
  {#if st.embed_pending === 0}<p class="muted">All captioned photos are embedded in active model.</p>
  {:else}<p>{st.embed_pending} captioned photos not embedded yet.</p>
    <button class="primary" on:click={() => start("embed")} disabled={running || noServices()} style="width:max-content">
      ▶ Embed {st.embed_pending}</button>{/if}
</div>

<div class="card col" style="margin-bottom:16px">
  <div class="section-label">D — Full index (vision + embed)</div>
  {#if st.missing_full === 0}<p class="muted">All scanned photos are in the active index.</p>
  {:else}<p>{st.missing_full} photos not in active index.</p>
    <button class="primary" on:click={() => start("full")} disabled={running || noServices()} style="width:max-content">
      ▶ Full index {st.missing_full}</button>{/if}
</div>

<div class="card col" style="margin-bottom:16px">
  <div class="section-label">E — Re-analyze stale photos</div>
  {#if st.missing_attrs === 0}<p class="muted">All indexed photos have rich attributes.</p>
  {:else}<p>{st.missing_attrs} photos lack rich attributes.</p>
    <button on:click={() => start("reanalyze")} disabled={running || noServices()} style="width:max-content">
      🔄 Re-analyze {st.missing_attrs}</button>{/if}
</div>
{/if}

<!-- Models -->
<div class="card col">
  <div class="section-label">F — Active embedding model</div>
  {#if Object.keys(models.models).length === 0}
    <p class="muted">No embedding models registered yet — run C or D first.</p>
  {:else}
    {#each Object.entries(models.models) as [name, info]}
      <div class="row" style="justify-content:space-between">
        <span>{name === models.active ? "✓ " : ""}<b>{name}</b>
          <span class="muted">— {info.source} · {info.dimension}d</span></span>
        {#if name !== models.active}
          <button on:click={() => switchModel(name)}>Use for search</button>
        {/if}
      </div>
    {/each}
  {/if}
</div>

<style>
  .log { font-size: 11px; color: var(--muted); max-height: 160px; overflow-y: auto;
    margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px; }
</style>
