<script>
  import { onDestroy, onMount } from "svelte";
import { jobStatus, lastDeleted, refreshJob, status } from "./lib/stores.js";
import { api } from "./lib/api.js";

  const TABS = [
    { name: "Library", icon: "▦", load: () => import("./lib/SearchTab.svelte") },
    { name: "Timeline", icon: "◷", load: () => import("./lib/TimelineTab.svelte") },
    { name: "Map", icon: "⌖", load: () => import("./lib/MapTab.svelte") },
    { name: "Albums", icon: "▤", load: () => import("./lib/AlbumsTab.svelte") },
    { name: "People", icon: "♧", load: () => import("./lib/PeopleTab.svelte") },
    { name: "Manage", icon: "⚙", load: () => import("./lib/IndexTab.svelte") },
  ];

  let tab = "Library";
  let components = {};
  let imports = {};
  let selectedId = null;
  let selectedIds = null;
  let selectedIndex = 0;
  let selectedCards = [];
  let lightboxComponent = null;
  let lightboxLoading = false;
  let lightboxImport;
  let catalogSummary = { total: 0, photos: 0, videos: 0, captioned: 0, years: [] };
  let jobPoll = null;

  async function ensureTab(name) {
    if (components[name]) return components[name];
    if (!imports[name]) {
      const definition = TABS.find((entry) => entry.name === name);
      imports[name] = definition.load().then((module) => {
        components = { ...components, [name]: module.default };
        return module.default;
      }).finally(() => { delete imports[name]; });
    }
    return imports[name];
  }

  async function navigate(name) {
    tab = name;
    await ensureTab(name);
    if (name === "Manage") {
      void refreshJob();
      syncJobPolling();
    }
  }

  function syncJobPolling() {
    const shouldPoll = typeof document !== "undefined" && !document.hidden &&
      (tab === "Manage" || $jobStatus.active);
    if (shouldPoll && !jobPoll) {
      jobPoll = setInterval(async () => {
        await refreshJob();
        if (document.hidden || (tab !== "Manage" && !$jobStatus.active)) syncJobPolling();
      }, 4000);
    } else if (!shouldPoll && jobPoll) {
      clearInterval(jobPoll);
      jobPoll = null;
    }
  }

  function onVisibilityChange() {
    if (!document.hidden && (tab === "Manage" || $jobStatus.active)) void refreshJob();
    syncJobPolling();
  }

  function onSelect(e) {
    const detail = e.detail;
    if (detail && typeof detail === "object") {
      selectedIds = detail.ids || [detail.id];
      selectedCards = detail.photos || [];
      selectedId = detail.id;
      selectedIndex = selectedIds.indexOf(detail.id);
    } else {
      selectedId = detail;
      selectedIds = [detail];
      selectedIndex = 0;
      selectedCards = [];
    }
    void openLightbox();
  }

  async function openLightbox() {
    if (lightboxComponent) return;
    if (lightboxLoading) return lightboxImport;
    lightboxLoading = true;
    lightboxImport = import("./lib/Lightbox.svelte").then((module) => {
      lightboxComponent = module.default;
      return module.default;
    }).finally(() => { lightboxLoading = false; });
    await lightboxImport;
  }

  function closeLightbox() {
    selectedId = null;
    selectedIds = null;
    selectedCards = [];
  }

  function onDeleted(e) {
    if (e?.detail) {
      // SearchTab updates the shared deletion store; this event also refreshes
      // the cheap Library tally after a single-item removal from the lightbox.
      const deleted = Array.isArray(e.detail) ? e.detail : [e.detail];
      lastDeleted.set(deleted);
      catalogSummary = { ...catalogSummary, total: Math.max(0, catalogSummary.total - deleted.length) };
      void api.librarySummary().then(updateSummary).catch(() => {});
    }
  }

  function updateSummary(e) {
    catalogSummary = { ...catalogSummary, ...(e?.detail || e || {}) };
  }

  onMount(() => {
    document.addEventListener("visibilitychange", onVisibilityChange);
    void ensureTab("Library");
  });
  onDestroy(() => {
    document.removeEventListener("visibilitychange", onVisibilityChange);
    if (jobPoll) clearInterval(jobPoll);
  });

  $: jobPct = $jobStatus.total ? Math.round(($jobStatus.done / $jobStatus.total) * 100) : 0;
  $: indexedCount = catalogSummary.total || $status.stage?.total_scanned || 0;
</script>

<div class="shell">
  <aside class="sidebar" aria-label="Photo Vault">
    <a class="brand" href="#library" on:click|preventDefault={() => navigate("Library")} aria-label="Photo Vault home">
      <span class="brand-mark" aria-hidden="true">◩</span>
      <span>Photo Vault</span>
    </a>

    <nav aria-label="Main navigation">
      <div class="nav-label">Your library</div>
      {#each TABS as item}
        <button class="nav-item" class:active={tab === item.name}
                aria-current={tab === item.name ? "page" : undefined}
                on:click={() => navigate(item.name)}>
          <span class="nav-icon" aria-hidden="true">{item.icon}</span>
          <span>{item.name}</span>
          {#if item.name === "Library" && catalogSummary.total > 0}
            <span class="nav-count">{catalogSummary.total.toLocaleString()}</span>
          {/if}
        </button>
      {/each}
    </nav>

    <div class="sidebar-bottom">
      {#if $jobStatus.active}
        <button class="job-card" on:click={() => navigate("Manage")}>
          <span class="job-dot" aria-hidden="true"></span>
          <span><b>Library is updating</b><small>{$jobStatus.preparing ? "Preparing items…" : `${$jobStatus.type} · ${jobPct}%`}</small></span>
          <span class="job-arrow" aria-hidden="true">›</span>
        </button>
      {/if}
      <div class="local-note"><span class="local-dot"></span>Your local library</div>
    </div>
  </aside>

  <main class="main-shell">
    <div class="mobile-topbar">
      <span class="brand-mark" aria-hidden="true">◩</span><b>Photo Vault</b>
      {#if $jobStatus.active}<span class="mobile-job" aria-label="Library update in progress">●</span>{/if}
    </div>

    {#each TABS as item (item.name)}
      {#if components[item.name]}
        <section class="view-panel" class:library={item.name === "Library"} class:current={tab === item.name}
                 aria-hidden={tab !== item.name}>
          <svelte:component this={components[item.name]}
            indexedCount={indexedCount}
            on:select={onSelect}
            on:deleted={onDeleted}
            on:goto-manage={() => navigate("Manage")}
            on:summary={updateSummary} />
        </section>
      {:else if tab === item.name}
        <div class="view-loading" role="status">Opening {item.name.toLowerCase()}…</div>
      {/if}
    {/each}
  </main>

  <nav class="mobile-nav" aria-label="Main navigation">
    {#each TABS as item}
      <button class="mobile-nav-item" class:active={tab === item.name}
              aria-current={tab === item.name ? "page" : undefined}
              on:click={() => navigate(item.name)}>
        <span aria-hidden="true">{item.icon}</span><small>{item.name}</small>
      </button>
    {/each}
  </nav>
</div>

{#if selectedId && lightboxComponent}
  <svelte:component this={lightboxComponent} id={selectedId} ids={selectedIds}
    cards={selectedCards} index={selectedIndex} on:close={closeLightbox} on:deleted={onDeleted} />
{/if}

<style>
  .shell { min-height: 100vh; }
  .sidebar {
    position: fixed; inset: 0 auto 0 0; z-index: 20; width: 248px;
    display: flex; flex-direction: column; padding: 27px 17px 18px;
    background: var(--surface); border-right: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 11px; padding: 0 12px 29px;
    color: var(--ink); text-decoration: none; font-size: 17px; font-weight: 750; letter-spacing: -.03em; }
  .brand-mark { display: inline-grid; place-items: center; width: 30px; height: 30px;
    color: #fff; background: var(--accent); border-radius: 9px; font-size: 17px; }
  .nav-label { padding: 0 12px 8px; color: var(--muted); font-size: 10px; font-weight: 750;
    letter-spacing: .12em; text-transform: uppercase; }
  nav { display: flex; flex-direction: column; gap: 4px; }
  .nav-item { display: flex; align-items: center; gap: 12px; width: 100%; padding: 11px 12px;
    background: transparent; border: 0; border-radius: 10px; color: #52606a; text-align: left;
    font-size: 13px; font-weight: 590; transition: background .16s, color .16s; }
  .nav-item:hover { background: #f1f3ef; color: var(--ink); transform: none; filter: none; }
  .nav-item.active { color: #145b59; background: #e5f2ef; font-weight: 720; }
  .nav-icon { display: inline-grid; place-items: center; width: 23px; font-size: 17px; }
  .nav-count { margin-left: auto; color: #66756d; font-size: 11px; font-variant-numeric: tabular-nums; }
  .sidebar-bottom { display: flex; flex-direction: column; gap: 14px; margin-top: auto; }
  .local-note { display: flex; align-items: center; gap: 9px; padding: 10px 12px; color: #53635b; font-size: 11px; }
  .local-dot, .job-dot { width: 8px; height: 8px; border-radius: 50%; background: #6eaa84; flex: 0 0 auto; }
  .job-card { display: flex; align-items: center; gap: 10px; padding: 12px; background: #f2f4f1;
    color: var(--ink); border: 1px solid var(--border); text-align: left; }
  .job-card:hover { transform: none; filter: none; border-color: #87b9ae; }
  .job-card small { display: block; color: var(--muted); font-size: 11px; margin-top: 3px; }
  .job-arrow { margin-left: auto; font-size: 22px; color: var(--muted); }
  .main-shell { min-height: 100vh; margin-left: 248px; }
  .view-panel { display: none; min-height: 100vh; }
  .view-panel.current { display: block; }
  .view-loading { padding: 42px; color: var(--muted); }
  .mobile-topbar, .mobile-nav { display: none; }
  @media (max-width: 760px) {
    .sidebar { display: none; }
    .main-shell { margin-left: 0; padding-bottom: 78px; }
    .mobile-topbar { display: flex; align-items: center; gap: 10px; height: 54px; padding: 0 17px;
      border-bottom: 1px solid var(--border); background: var(--surface); color: var(--ink); }
    .mobile-topbar .brand-mark { width: 27px; height: 27px; font-size: 15px; }
    .mobile-job { margin-left: auto; color: var(--accent); font-size: 11px; }
    .mobile-nav { position: fixed; inset: auto 0 0; z-index: 30; display: grid;
      grid-template-columns: repeat(6, 1fr); padding: 7px 5px calc(7px + env(safe-area-inset-bottom));
      border-top: 1px solid var(--border); background: color-mix(in srgb, var(--surface) 94%, transparent);
      backdrop-filter: blur(14px); }
    .mobile-nav-item { display: flex; min-width: 0; flex-direction: column; align-items: center; gap: 3px;
      padding: 5px 1px; border: 0; border-radius: 9px; color: #66726f; background: transparent; }
    .mobile-nav-item > span { font-size: 17px; line-height: 1.1; }
    .mobile-nav-item small { font-size: 9px; white-space: nowrap; }
    .mobile-nav-item.active { color: #145b59; background: #e5f2ef; }
  }
</style>
