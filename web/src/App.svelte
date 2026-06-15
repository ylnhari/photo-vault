<script>
  import { onMount } from "svelte";
  import { health, status, refreshHealth, refreshStatus } from "./lib/stores.js";
  import SearchTab from "./lib/SearchTab.svelte";
  import TimelineTab from "./lib/TimelineTab.svelte";
  import PeopleTab from "./lib/PeopleTab.svelte";
  import IndexTab from "./lib/IndexTab.svelte";
  import Lightbox from "./lib/Lightbox.svelte";

  const TABS = ["Search", "Timeline", "People", "Index & Manage"];
  let tab = "Search";
  let selectedId = null;

  // Single health/status fetch for the whole app — fixes the old contradiction
  // where two tabs fetched health separately and disagreed.
  onMount(() => { refreshHealth(); refreshStatus(); });

  $: indexedCount = $status.stage.active_model_embedded || 0;
  $: noServices = $health.loaded && !$health.lm_studio && !$health.gemini;

  function onSelect(e) { selectedId = e.detail; }
  function onDeleted() { selectedId = null; refreshStatus(); }
</script>

<header>
  <h1>📷 Photo Vault</h1>
  <nav>
    {#each TABS as t}
      <button class:active={tab === t} class="ghost" on:click={() => (tab = t)}>{t}</button>
    {/each}
  </nav>
</header>

{#if noServices}
  <div class="warn">
    No AI service online — indexing is disabled. Start LM Studio (vision + embedding model loaded),
    or add <code>GEMINI_API_KEY</code> to <code>.env</code>.
  </div>
{/if}

<main>
  {#if tab === "Search"}
    <SearchTab {indexedCount} on:select={onSelect} />
  {:else if tab === "Timeline"}
    <TimelineTab on:select={onSelect} />
  {:else if tab === "People"}
    <PeopleTab {indexedCount} on:select={onSelect} />
  {:else}
    <IndexTab on:select={onSelect} />
  {/if}
</main>

{#if selectedId}
  <Lightbox id={selectedId} on:close={() => (selectedId = null)} on:deleted={onDeleted} />
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
</style>
