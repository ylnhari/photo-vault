<script>
  import { onMount } from "svelte";
  import { health, status, refreshHealth, refreshStatus, lastDeleted,
           jobStatus, refreshJob } from "./lib/stores.js";
  import { onDestroy } from "svelte";
  import SearchTab from "./lib/SearchTab.svelte";
  import TimelineTab from "./lib/TimelineTab.svelte";
  import MapTab from "./lib/MapTab.svelte";
  import AlbumsTab from "./lib/AlbumsTab.svelte";
  import PeopleTab from "./lib/PeopleTab.svelte";
  import IndexTab from "./lib/IndexTab.svelte";
  import Lightbox from "./lib/Lightbox.svelte";

  const TABS = ["Search", "Timeline", "Map", "Albums", "People", "Index & Manage"];
  let tab = "Search";
  let selectedId = null;
  let selectedIds = null;
  let selectedIndex = 0;

  // Single health/status fetch for the whole app — fixes the old contradiction
  // where two tabs fetched health separately and disagreed.
  let jobPoll;
  onMount(() => {
    refreshHealth(); refreshStatus(); refreshJob();
    jobPoll = setInterval(refreshJob, 4000);
  });
  onDestroy(() => clearInterval(jobPoll));

  $: jobPct = $jobStatus.total
    ? Math.round(($jobStatus.done / $jobStatus.total) * 100) : 0;

  $: indexedCount = $status.stage.active_model_embedded || 0;
  $: noServices = $health.loaded && !$health.lm_studio && !$health.gemini;

  function onSelect(e) {
    const d = e.detail;
    if (d && typeof d === "object") {
      selectedIds = d.ids || [d.id];
      selectedId = d.id;
      selectedIndex = selectedIds.indexOf(d.id);
    } else {
      selectedId = d;
      selectedIds = [d];
      selectedIndex = 0;
    }
  }
  function onClose() { selectedId = null; selectedIds = null; }
  function onDeleted(e) {
    if (e?.detail) lastDeleted.set(e.detail);  // grids hide the photo instantly
    refreshStatus();
  }
</script>

<header>
  <h1>📷 Photo Vault</h1>
  <nav>
    {#each TABS as t}
      <button class:active={tab === t} class="ghost" on:click={() => (tab = t)}>{t}</button>
    {/each}
  </nav>
  {#if $jobStatus.active}
    <button class="jobpill" title="Open Index & Manage"
            on:click={() => (tab = "Index & Manage")}>
      <span class="jobspin"></span>
      {$jobStatus.type} · {jobPct}% ({$jobStatus.done}/{$jobStatus.total})
    </button>
  {/if}
</header>

{#if noServices}
  <div class="warn">
    No AI service online — indexing is disabled. Start LM Studio (vision + embedding model loaded),
    or add <code>GEMINI_API_KEY</code> to <code>.env</code>.
  </div>
{/if}

<main>
  {#if tab === "Search"}
    <SearchTab {indexedCount} on:select={onSelect} on:deleted={onDeleted}
               on:goto-index={() => (tab = "Index & Manage")} />
  {:else if tab === "Timeline"}
    <TimelineTab on:select={onSelect} />
  {:else if tab === "Map"}
    <MapTab {indexedCount} on:select={onSelect} />
  {:else if tab === "Albums"}
    <AlbumsTab on:select={onSelect} />
  {:else if tab === "People"}
    <PeopleTab {indexedCount} on:select={onSelect} />
  {:else}
    <IndexTab on:select={onSelect} />
  {/if}
</main>

{#if selectedId}
  <Lightbox id={selectedId} ids={selectedIds} index={selectedIndex}
            on:close={onClose} on:deleted={onDeleted} />
{/if}

<style>
  header {
    display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
    padding: 14px 22px; border-bottom: 1px solid var(--border);
    position: sticky; top: 0; background: var(--bg); z-index: 10;
  }
  h1 { font-size: 20px; }
  nav { display: flex; gap: 6px; }
  nav button.active { background: var(--surface2); color: var(--text); }
  main { padding: 22px; max-width: 1400px; margin: 0 auto; }
  .warn { background: #2a1a00; color: var(--warn); border: 1px solid var(--warn);
    padding: 10px 16px; margin: 12px 22px; border-radius: 8px; }
  code { background: var(--surface2); padding: 1px 6px; border-radius: 4px; font-size: 12px; }
  .jobpill {
    margin-left: auto; display: inline-flex; align-items: center; gap: 8px;
    font-size: 12px; padding: 5px 12px; border-radius: 99px;
    background: color-mix(in srgb, var(--accent) 18%, var(--surface));
    border: 1px solid var(--accent); color: var(--text);
  }
  .jobspin {
    width: 11px; height: 11px; border-radius: 50%;
    border: 2px solid var(--surface2); border-top-color: var(--accent);
    animation: appjobspin .7s linear infinite;
  }
  @keyframes appjobspin { to { transform: rotate(360deg); } }
</style>
