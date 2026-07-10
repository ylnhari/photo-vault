<script>
  import { createEventDispatcher } from "svelte";
  import { fly } from "svelte/transition";
  export let job;
  const dispatch = createEventDispatcher();

  const TITLES = { vision: "Vision analysis", embed: "Embedding",
                   full: "Full index", reanalyze: "Re-analyze",
                   faces: "Face detection", thumbs: "Thumbnails",
                   dhash: "Duplicate scan", scan: "Scanning folders" };

  // Each job type gets its own accent so the panel reads as a continuation
  // of that section's identity, not a generic grey status box.
  const THEME = {
    vision:   { icon: "👁", color: "#6366f1" },
    embed:    { icon: "🧬", color: "#8b5cf6" },
    faces:    { icon: "🙂", color: "#ec4899" },
    thumbs:   { icon: "🖼", color: "#f59e0b" },
    dhash:    { icon: "🔍", color: "#06b6d4" },
    scan:     { icon: "📂", color: "#3b82f6" },
    full:     { icon: "⚡", color: "#22c55e" },
    reanalyze:{ icon: "🔄", color: "#fb923c" },
  };
  $: theme = THEME[job.type] || { icon: "⚙", color: "#6366f1" };

  // Stop is honored between work items/batches — with slow local models the
  // in-flight batch can take minutes, so show an explicit "stopping" state
  // instead of a Stop button that looks ignored. Bindable so the parent can
  // reset it back to false if the stop request itself fails (otherwise the
  // button would stay stuck on "Stopping…" forever).
  export let stopRequested = false;
  $: if (!job.active) stopRequested = false;  // reset once the job halts
  function requestStop() { stopRequested = true; dispatch("stop"); }

  $: running = job.active;
  $: pct = job.total ? Math.round((job.done / job.total) * 100) : (running ? 0 : 100);

  // ETA comes from the backend (cumulative average rate, includes time spent
  // paused on rate limits). rate_wait says which provider is currently
  // sleeping on a full rate-limit window — without it a throttled job just
  // looks frozen.
  function fmtDur(s) {
    s = Math.max(0, Math.round(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m ${sec}s`;
    return `${sec}s`;
  }
  const RL_NAMES = { lm_studio: "LM Studio", gemini: "Gemini", "9router": "9Router" };
  $: rateWaits = Object.entries(job.rate_wait || {});
</script>

<div class="panel" class:running class:aborted={job.aborted} style="--tile:{theme.color}"
     transition:fly={{ y: -6, duration: 220 }}>
  <div class="head">
    <span class="tile" style="width:30px;height:30px;font-size:15px">
      {#if running}<span class="spin"></span>{:else if job.aborted}✕{:else}✓{/if}
    </span>
    <span class="title">
      {theme.icon} {TITLES[job.type] || "Working"}
      {#if running && stopRequested}
        <span class="warn" style="font-weight:400; font-size:12px">— stopping after the current batch…</span>
      {/if}
    </span>
    <div class="spacer"></div>
    {#if running}
      <button class="danger sm" on:click={requestStop} disabled={stopRequested}>
        {stopRequested ? "Stopping…" : "🛑 Stop"}
      </button>
    {:else}
      <div class="row" style="gap:8px">
        {#if job.failed_ids?.length}
          <button class="sm" on:click={() => dispatch("retry")}>↻ Retry {job.failed_ids.length} failed</button>
        {/if}
        <button class="ghost sm" on:click={() => dispatch("clear")}>Done</button>
      </div>
    {/if}
  </div>

  <div class="bar" class:indeterminate={running && !job.total}>
    <span class="fill" style="width:{pct}%"></span>
  </div>

  <div class="stats">
    <span class="pct">{pct}%</span>
    <span class="muted">{job.done}/{job.total}</span>
    <span class="chip ok">✅ {job.ok}</span>
    {#if job.fail}<span class="chip fail">❌ {job.fail}</span>{/if}
    {#if running && job.eta_seconds != null}
      <span class="chip">⏱ ~{fmtDur(job.eta_seconds)} left{job.rate_per_min ? ` · ${job.rate_per_min}/min` : ""}</span>
    {/if}
    {#each rateWaits as [prov, w] (prov)}
      <span class="chip warn">⏸ {RL_NAMES[prov] || prov} rate limit ({w.limit}) — resumes in {fmtDur(w.seconds)}</span>
    {/each}
    {#if job.aborted}<span class="chip fail">auto-stopped · too many failures</span>{/if}
    {#if job.stopped}<span class="chip warn">stopped by you</span>{/if}
  </div>

  <!-- Per-model tally: which models actually produced this run's captions.
       Matters with 9Router, whose gateway can substitute serving models
       mid-run — one run can legitimately span several models. Lives only in
       job state, so it disappears when the job is cleared. -->
  {#if job.model_counts && Object.keys(job.model_counts).length}
    <div class="models">
      <span class="muted" style="font-size:12px">Models used:</span>
      {#each Object.entries(job.model_counts).sort((a, b) => b[1] - a[1]) as [label, count]}
        <span class="chip">{label} × {count}</span>
      {/each}
    </div>
  {/if}

  {#if job.log?.length}
    <div class="log">
      {#each job.log.slice(-12) as l, i (l.id + '-' + l.note + '-' + i)}
        <div class="line {l.kind}">
          <span class="dot">{l.kind === "fail" ? "❌" : l.kind === "cloud" ? "☁️" : "✅"}</span>
          <span class="fn" title={l.id}>{l.file || l.id.slice(0, 12)}</span>
          <span class="note">{l.note}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .panel {
    border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 16px; background: var(--surface);
    box-shadow: var(--shadow-1);
  }
  .panel.running {
    border-color: color-mix(in srgb, var(--tile) 55%, var(--border));
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--tile) 25%, transparent), var(--shadow-2);
  }
  .panel.aborted { border-color: color-mix(in srgb, var(--danger) 55%, var(--border)); }
  .head { display: flex; align-items: center; gap: 10px; }
  .spacer { flex: 1; }
  .title { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; font-size: 15px; }
  .spin {
    display: block; width: 14px; height: 14px; border: 2px solid color-mix(in srgb, var(--tile) 30%, transparent);
    border-top-color: var(--tile); border-radius: 50%; animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .bar {
    height: 10px; border-radius: 99px; background: var(--surface2); overflow: hidden;
    margin: 14px 0 10px; position: relative;
  }
  .bar .fill {
    display: block; height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--tile), color-mix(in srgb, var(--tile) 60%, var(--accent-2)));
    transition: width .35s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
  }
  /* Subtle moving shimmer while a job is actively running, so a bar that's
     barely moving (slow local model) still visibly reads as "alive". */
  .bar .fill::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.28), transparent);
    animation: shimmer 1.6s linear infinite;
  }
  @keyframes shimmer { from { transform: translateX(-100%); } to { transform: translateX(100%); } }
  .bar.indeterminate .fill { width: 30% !important; animation: indeterminate 1.4s ease-in-out infinite; }
  @keyframes indeterminate { 0% { transform: translateX(-100%); } 100% { transform: translateX(330%); } }

  .stats { display: flex; gap: 10px; align-items: baseline; font-size: 13px; flex-wrap: wrap; }
  .models { display: flex; gap: 6px; align-items: baseline; flex-wrap: wrap; margin-top: 8px; }
  .pct { font-size: 22px; font-weight: 700; color: var(--tile); }
  .chip { padding: 2px 9px; border-radius: 99px; background: var(--surface2); font-size: 12px; }
  .chip.ok { color: var(--success); }
  .chip.fail { color: var(--danger); background: color-mix(in srgb, var(--danger) 14%, var(--surface2)); }
  .chip.warn { color: var(--warn); background: color-mix(in srgb, var(--warn) 14%, var(--surface2)); }

  .sm { padding: 6px 12px; font-size: 13px; }
  .log {
    margin-top: 12px; max-height: 160px; overflow-y: auto;
    border-top: 1px solid var(--border); padding-top: 8px;
    display: flex; flex-direction: column; gap: 1px;
  }
  .line {
    display: flex; gap: 8px; font-size: 11px; padding: 3px 6px; border-radius: 6px;
    align-items: center;
  }
  .line:nth-child(odd) { background: color-mix(in srgb, var(--surface2) 45%, transparent); }
  .line .dot { flex-shrink: 0; }
  .line .fn { color: var(--text); min-width: 140px; font-family: ui-monospace, monospace; }
  .line .note { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .line.fail .note { color: var(--danger); }
</style>
