<script>
  import { createEventDispatcher } from "svelte";
  export let job;
  const dispatch = createEventDispatcher();

  const TITLES = { vision: "Vision analysis", embed: "Embedding",
                   full: "Full index", reanalyze: "Re-analyze",
                   faces: "Face detection", thumbs: "Thumbnails",
                   dhash: "Duplicate scan", scan: "Scanning folders" };

  // Stop is honored between work items/batches — with slow local models the
  // in-flight batch can take minutes, so show an explicit "stopping" state
  // instead of a Stop button that looks ignored.
  let stopRequested = false;
  $: if (!job.active) stopRequested = false;  // reset once the job halts
  function requestStop() { stopRequested = true; dispatch("stop"); }

  $: running = job.active;
  $: pct = job.total ? Math.round((job.done / job.total) * 100) : (running ? 0 : 100);
</script>

<div class="panel" class:running>
  <div class="head">
    <span class="title">
      {#if running}<span class="spin"></span>{:else}<span class="check">✓</span>{/if}
      {TITLES[job.type] || "Working"}
      {#if running && stopRequested}
        <span class="warn" style="font-weight:400; font-size:12px">
          — stopping after the current batch…</span>
      {/if}
    </span>
    {#if running}
      <button class="danger sm" on:click={requestStop} disabled={stopRequested}>
        {stopRequested ? "Stopping…" : "🛑 Stop"}
      </button>
    {:else}
      <div class="row">
        {#if job.failed_ids?.length}
          <button class="sm" on:click={() => dispatch("retry")}>Retry {job.failed_ids.length} failed</button>
        {/if}
        <button class="ghost sm" on:click={() => dispatch("clear")}>Done</button>
      </div>
    {/if}
  </div>

  <div class="bar"><span style="width:{pct}%"></span></div>

  <div class="stats">
    <span class="big">{pct}%</span>
    <span class="muted">{job.done}/{job.total}</span>
    <span class="ok">✅ {job.ok}</span>
    <span class="fail">❌ {job.fail}</span>
    {#if job.aborted}<span class="fail">· auto-stopped (too many failures)</span>{/if}
    {#if job.stopped}<span class="warn">· stopped by you</span>{/if}
  </div>

  {#if job.log?.length}
    <div class="log">
      {#each job.log.slice(-12) as l}
        <div class="line {l.kind}">
          <span>{l.kind === "fail" ? "❌" : l.kind === "cloud" ? "☁️" : "✅"}</span>
          <span class="fn" title={l.id}>{l.file || l.id.slice(0, 12)}</span>
          <span class="note">{l.note}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px; background: var(--bg); }
  .panel.running { border-color: var(--accent); }
  .head { display: flex; justify-content: space-between; align-items: center; }
  .title { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }
  .check { color: var(--success); }
  .spin { width: 13px; height: 13px; border: 2px solid var(--surface2);
    border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .bar { height: 8px; border-radius: 99px; background: var(--surface2); overflow: hidden; margin: 12px 0 8px; }
  .bar > span { display: block; height: 100%; background: var(--accent); transition: width .25s; }
  .stats { display: flex; gap: 12px; align-items: baseline; font-size: 13px; flex-wrap: wrap; }
  .big { font-size: 18px; font-weight: 700; }
  .ok { color: var(--success); } .fail { color: var(--danger); } .warn { color: var(--warn); }
  .sm { padding: 5px 10px; font-size: 13px; }
  .log { margin-top: 10px; max-height: 150px; overflow-y: auto; border-top: 1px solid var(--border); padding-top: 8px; }
  .line { display: flex; gap: 8px; font-size: 11px; padding: 1px 0; }
  .line .fn { color: var(--text); min-width: 120px; }
  .line .note { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .line.fail .note { color: var(--danger); }
</style>
