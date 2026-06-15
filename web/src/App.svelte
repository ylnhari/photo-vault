<script>
  import { onMount } from "svelte";
  import { api } from "./lib/api.js";
  import SearchTab from "./lib/SearchTab.svelte";
  import TimelineTab from "./lib/TimelineTab.svelte";
  import PeopleTab from "./lib/PeopleTab.svelte";
  import IndexTab from "./lib/IndexTab.svelte";
  import Lightbox from "./lib/Lightbox.svelte";

  const TABS = ["Search", "Timeline", "People", "Index & Manage"];
  let tab = "Search";
  let indexedCount = 0;
  let selectedId = null;
  let healthWarn = "";

  onMount(refresh);
  async function refresh() {
    try {
      const st = await api.status();
      indexedCount = st.stage.active_model_embedded || 0;
      const h = await api.health();
      if (!h.lm_studio && !h.gemini)
        healthWarn = "No AI service online — indexing disabled. Start LM Studio or set GEMINI_API_KEY.";
      else healthWarn = "";
    } catch (e) { healthWarn = "Backend unreachable: " + e.message; }
  }
  function onSelect(e) { selectedId = e.detail; }
  function onDeleted() { selectedId = null; refresh(); }
</script>

<header>
  <h1>📷 Photo Vault</h1>
  <nav>
    {#each TABS as t}
      <button class:active={tab === t} class="ghost" on:click={() => (tab = t)}>{t}</button>
    {/each}
  </nav>
</header>

{#if healthWarn}<div class="warn">{healthWarn}</div>{/if}

<main>
  {#if tab === "Search"}
    <SearchTab {indexedCount} on:select={onSelect} />
  {:else if tab === "Timeline"}
    <TimelineTab on:select={onSelect} />
  {:else if tab === "People"}
    <PeopleTab {indexedCount} on:select={onSelect} />
  {:else}
    <IndexTab on:indexed={refresh} on:select={onSelect} />
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
</style>
