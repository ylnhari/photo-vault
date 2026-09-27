<script>
  import { createEventDispatcher, onDestroy, onMount } from "svelte";
  import { api } from "./api.js";
  import { lastDeleted } from "./stores.js";
  import PhotoGrid from "./PhotoGrid.svelte";
  import { createRequestGate } from "./requestGate.js";
  import { MAX_SAVED_SEARCHES, buildBrowseParams, readSavedSearches, validateBrowseQuery, validateSavedSearch, writeSavedSearches } from "./browseSearch.js";

  const dispatch = createEventDispatcher();
  const LIMIT = 60;
  const FILTER_ORDER = [
    ["year", "Year"], ["month", "Month"], ["place", "Place"], ["weather", "Weather"],
    ["occasion", "Occasion"], ["festival_name", "Festival"], ["scene", "Scene"],
    ["group_size", "Group size"], ["person_count", "People"], ["clothing_style", "Clothing"],
    ["mood", "Mood"], ["location_type", "Location"], ["season", "Season"],
    ["photo_type", "Photo type"],
  ];

  const browseGate = createRequestGate();
  const summaryGate = createRequestGate();
  const smartGate = createRequestGate();
  const filterGate = createRequestGate();
  const facetGate = createRequestGate();
  const peopleGate = createRequestGate();
  let mode = "browse";
  let q = "";
  let person = "";
  let mediaType = "all";
  let year = "";
  let summary = { total: 0, photos: 0, videos: 0, captioned: 0, years: [] };
  let photos = [];
  let smartResults = null;
  let total = 0;
  let hasMore = false;
  let loading = false;
  let initialLoading = true;
  let pageBusy = false;
  let summaryReady = false;
  let summaryError = false;
  let browseError = "";
  let smartError = "";
  let personNotFound = false;
  let filterError = false;
  let filtersOpen = false;
  let filtersLoading = false;
  let filtersLoaded = false;
  let filterVals = {};
  let selected = {};
  let browseFilters = { text_scope: "all", date_from: "", date_to: "", month: "", day: "", photo_type: "", scene: "", weather: "", occasion: "", time_of_day: "", camera: "", has_text: "", has_location: "", has_caption: "", sort: "newest" };
  let browseFiltersOpen = false;
  let browseFacets = {};
  let browseFacetTruncated = {};
  let browseFacetsLoaded = false;
  let browseFacetsLoading = false;
  let browseFacetError = "";
  let browseFacetsPromise = null;
  let smartPeople = [];
  let smartPeopleLoaded = false;
  let smartPeopleError = false;
  let savedSearches = [];
  let savedSearchName = "";
  let savedSearchesOpen = false;
  let savePanelOpen = false;
  let savedSearchMessage = "";
  let browseFeedback = "";

  let selectMode = false;
  let selectedIds = [];
  let resetToken = 0;
  let deleteFiles = false;
  let batchBusy = false;
  let albums = [];
  let albumMsg = "";
  let debounceTimer;

  $: activeFilterCount = Object.values(selected).filter((value) => value && value !== "All").length;
  $: displayedPhotos = mode === "smart" ? (smartResults || []) : photos;
  $: displayedTotal = mode === "smart" ? (smartResults?.length || 0) : total;
  $: years = summary.years.map((entry) => typeof entry === "object"
    ? { year: String(entry.year), count: entry.count || 0 }
    : { year: String(entry), count: 0 });
  $: mediaParam = mediaType === "photos" ? "image" : mediaType === "videos" ? "video" : "";

  function normalizedSummary(raw) {
    const entries = Array.isArray(raw?.years) ? raw.years : [];
    return {
      total: Number(raw?.total) || 0,
      photos: Number(raw?.photos) || 0,
      videos: Number(raw?.videos) || 0,
      captioned: Number(raw?.captioned) || 0,
      years: entries.map((entry) => typeof entry === "object"
        ? { year: String(entry.year), count: Number(entry.count) || 0 }
        : { year: String(entry), count: 0 }),
    };
  }

  async function loadSummary() {
    const request = summaryGate.begin();
    try {
      const response = await api.librarySummary({ signal: request.signal });
      if (!request.isCurrent()) return;
      summary = normalizedSummary(response);
      summaryReady = true;
      summaryError = false;
      dispatch("summary", summary);
    } catch (error) {
      if (request.isCurrent()) {
        summaryReady = true;
        summaryError = true;
        browseError = browseError || `Library counts could not be refreshed. ${error.message}`;
      }
    }
  }

  function browseParams(offset = 0) {
    return buildBrowseParams({ q, mediaType, year, filters: browseFilters }, offset, LIMIT);
  }

  async function loadPage(offset = 0) {
    const validationError = validateBrowseQuery(q, browseFilters);
    if (validationError) {
      browseGate.cancel();
      loading = false;
      pageBusy = false;
      browseError = validationError;
      return;
    }
    const request = browseGate.begin();
    loading = true;
    browseError = "";
    if (offset > 0) pageBusy = true;
    try {
      const page = await api.library(browseParams(offset), { signal: request.signal });
      if (!request.isCurrent()) return;
      const pageRows = Array.isArray(page.results) ? page.results : [];
      if (offset === 0) {
        photos = pageRows;
      }
      else {
        const existing = new Set(photos.map((photo) => photo.id));
        photos = [...photos, ...pageRows.filter((photo) => !existing.has(photo.id))];
      }
      total = Number(page.total) || 0;
      hasMore = !!page.has_more;
      initialLoading = false;
    } catch (error) {
      if (request.isCurrent()) {
        initialLoading = false;
        browseError = error.name === "AbortError" ? "" : error.message;
      }
    } finally {
      if (request.isCurrent()) {
        loading = false;
        pageBusy = false;
      }
    }
  }

  function scheduleBrowse({ immediate = false } = {}) {
    clearTimeout(debounceTimer);
    browseGate.cancel();
    clearSelection();
    const validationError = validateBrowseQuery(q, browseFilters);
    loading = !validationError;
    browseError = validationError;
    debounceTimer = setTimeout(() => {
      if (mode === "browse") void loadPage(0);
    }, immediate ? 0 : 240);
  }

  function changeMode(nextMode) {
    if (mode === nextMode) return;
    mode = nextMode;
    clearTimeout(debounceTimer);
    if (nextMode === "browse") {
      smartGate.cancel();
      loading = false;
      scheduleBrowse({ immediate: true });
    } else {
      browseGate.cancel();
      loading = false;
      if (!filtersLoaded) void loadFilters();
      if (!smartPeopleLoaded) void loadSmartPeople();
    }
  }

  async function loadBrowseFacets() {
    if (browseFacetsLoaded) return;
    if (browseFacetsPromise) return browseFacetsPromise;
    const request = facetGate.begin();
    browseFacetsLoading = true;
    browseFacetError = "";
    browseFacetsPromise = (async () => {
      try {
        const response = await api.libraryFacets({ signal: request.signal });
        if (!request.isCurrent()) return;
        browseFacets = response?.facets || {};
        browseFacetTruncated = response?.truncated || response?.truncated_facets || response?.truncation || {};
        browseFacetsLoaded = true;
      } catch (error) {
        if (request.isCurrent()) browseFacetError = `Available filter choices could not be loaded. ${error.message}`;
      } finally {
        if (request.isCurrent()) browseFacetsLoading = false;
        browseFacetsPromise = null;
      }
    })();
    return browseFacetsPromise;
  }

  async function loadSmartPeople() {
    const request = peopleGate.begin();
    try {
      const response = await api.people();
      if (!request.isCurrent()) return;
      smartPeople = Array.isArray(response) ? response.map((person) => typeof person === "string" ? person : person.name).filter(Boolean) : (response?.people || []).map((person) => typeof person === "string" ? person : person.name).filter(Boolean);
      smartPeopleLoaded = true;
      smartPeopleError = false;
    } catch {
      if (request.isCurrent()) smartPeopleError = true;
    }
  }

  function openBrowseFilters() {
    browseFiltersOpen = !browseFiltersOpen;
    if (browseFiltersOpen && !browseFacetsLoaded) void loadBrowseFacets();
  }

  function setBrowseFilter(key, value, { immediate = false } = {}) {
    browseFilters = { ...browseFilters, [key]: String(value ?? "") };
    browseFeedback = "";
    scheduleBrowse({ immediate });
  }

  function clearBrowseFilters() {
    savedSearchMessage = "";
    browseFeedback = "";
    browseFilters = { text_scope: "all", date_from: "", date_to: "", month: "", day: "", photo_type: "", scene: "", weather: "", occasion: "", time_of_day: "", camera: "", has_text: "", has_location: "", has_caption: "", sort: "newest" };
    q = ""; year = ""; mediaType = "all";
    scheduleBrowse({ immediate: true });
  }

  async function quickScreenshot() {
    const choices = Array.isArray(browseFacets.photo_type) ? browseFacets.photo_type : [];
    const canonical = choices.map((item) => typeof item === "string" ? item : item.value).find((value) => String(value).toLowerCase() === "screenshot");
    setBrowseFilter("photo_type", canonical || "screenshot");
    if (!canonical && browseFacetsLoaded) browseFeedback = "No screenshot label is in the saved photo details; trying the standard Screenshot label.";
  }

  function onThisDay() {
    const today = new Date();
    year = "";
    browseFilters = { ...browseFilters, date_from: "", date_to: "", month: String(today.getMonth() + 1), day: String(today.getDate()) };
    scheduleBrowse();
  }

  function normalizedFacetOptions(key, facets, filters) {
    const values = Array.isArray(facets[key]) ? facets[key] : [];
    const options = values.map((item) => typeof item === "string" ? { value: item, count: null } : item).filter((item) => typeof item.value === "string" && item.value.trim());
    const selectedValue = filters[key];
    if (selectedValue && !options.some((item) => item.value === selectedValue)) options.push({ value: selectedValue, count: null });
    return options;
  }

  function facetsTruncated(key, truncated) {
    if (Array.isArray(truncated)) return truncated.includes(key);
    if (truncated && typeof truncated === "object") return !!truncated[key];
    return !!truncated;
  }

  function activeBrowseChips(query, currentYear, currentMediaType, filters) {
    const labels = { text_scope: "Search in", date_from: "From", date_to: "Through", month: "Month", day: "Day", photo_type: "Photo type", scene: "Scene", weather: "Weather", occasion: "Occasion", time_of_day: "Time of day", camera: "Camera", has_text: "Recognized text", has_location: "Location", has_caption: "Caption", sort: "Sort" };
    const chips = [];
    if (query.trim()) chips.push({ key: "q", label: `Search: ${query.trim()}` });
    if (currentMediaType !== "all") chips.push({ key: "mediaType", label: currentMediaType === "photos" ? "Photos" : "Videos" });
    if (currentYear) chips.push({ key: "year", label: `Year: ${currentYear}` });
    for (const [key, value] of Object.entries(filters)) {
      if (!value || value === "all" || (key === "sort" && value === "newest")) continue;
      const display = key === "text_scope" ? ({ filename: "Filename", caption: "Caption", text: "Text in photo", place: "Place", all: "Everything" }[value] || value)
        : key === "has_text" ? ({ yes: "Yes", no: "No" }[value])
        : key === "has_location" ? ({ yes: "Yes", no: "No" }[value])
        : key === "has_caption" ? ({ yes: "Yes", no: "No" }[value])
        : value;
      chips.push({ key, label: `${labels[key] || key}: ${display}` });
    }
    if (filters.month && filters.day) {
      const monthChip = chips.find((chip) => chip.key === "month");
      const dayChip = chips.find((chip) => chip.key === "day");
      if (monthChip) monthChip.label = `Month/day: ${filters.month}/${filters.day}`;
      if (dayChip) chips.splice(chips.indexOf(dayChip), 1);
    }
    return chips;
  }

  $: browseChips = activeBrowseChips(q, year, mediaType, browseFilters);
  $: facetTruncationNote = ["photo_type", "scene", "weather", "occasion", "time_of_day", "camera"]
    .some((key) => facetsTruncated(key, browseFacetTruncated));

  function removeBrowseChip(key) {
    if (key === "q") q = "";
    else if (key === "mediaType") mediaType = "all";
    else if (key === "year") year = "";
    else if (key === "month") browseFilters = { ...browseFilters, month: "", day: "" };
    else if (key === "day") browseFilters = { ...browseFilters, day: "" };
    else browseFilters = { ...browseFilters, [key]: key === "text_scope" ? "all" : key === "sort" ? "newest" : "" };
    scheduleBrowse({ immediate: true });
  }

  function saveCurrentBrowseSearch() {
    const item = { name: savedSearchName.trim(), q: q.trim(), mediaType, year, filters: { ...browseFilters } };
    if (!validateSavedSearch(item)) { savedSearchMessage = "Add a name of up to 60 characters and check the search filters."; return; }
    let next = [...savedSearches];
    const existingIndex = next.findIndex((saved) => saved.name.toLocaleLowerCase() === item.name.toLocaleLowerCase());
    if (existingIndex >= 0) next[existingIndex] = item;
    else if (next.length >= MAX_SAVED_SEARCHES) { savedSearchMessage = `You can save up to ${MAX_SAVED_SEARCHES} searches. Remove one to make room.`; return; }
    else next.push(item);
    if (!writeSavedSearches(next)) { savedSearchMessage = "This browser could not save searches locally."; return; }
    savedSearches = next;
    savedSearchName = ""; savedSearchMessage = "Saved on this device."; savePanelOpen = false;
  }

  function applySavedSearch(item) {
    if (!validateSavedSearch(item)) { savedSearchMessage = "This saved search is invalid and was not applied."; return; }
    q = item.q; mediaType = item.mediaType; year = item.year; browseFilters = { ...browseFilters, ...item.filters };
    savedSearchesOpen = false; scheduleBrowse({ immediate: true });
  }

  function removeSavedSearch(item) {
    const next = savedSearches.filter((saved) => saved !== item && saved.name !== item.name);
    if (!writeSavedSearches(next)) { savedSearchMessage = "This browser could not update saved searches."; return; }
    savedSearches = next;
  }


  async function loadFilters() {
    const request = filterGate.begin();
    filtersLoading = true;
    try {
      const values = await api.filters({ signal: request.signal });
      if (!request.isCurrent()) return;
      filterVals = values || {};
      filtersLoaded = true;
      selected = Object.fromEntries(FILTER_ORDER.map(([key]) => [key, "All"]));
    } catch (error) {
      if (request.isCurrent()) smartError = `Search filters could not be loaded. ${error.message}`;
    } finally {
      if (request.isCurrent()) filtersLoading = false;
    }
  }

  async function runSmartSearch(event) {
    event?.preventDefault?.();
    if (loading) smartGate.cancel();
    const filters = {};
    for (const [key] of FILTER_ORDER) {
      if (selected[key] && selected[key] !== "All") filters[key] = selected[key];
    }
    const request = smartGate.begin();
    clearSelection();
    loading = true;
    smartError = "";
    personNotFound = false;
    filterError = false;
    try {
      const response = await api.search(q.trim(), filters, person.trim(), 200, { signal: request.signal });
      if (!request.isCurrent()) return;
      smartResults = Array.isArray(response.results) ? response.results : [];
      personNotFound = !!response.person_not_found;
      filterError = !!response.filter_error;
    } catch (error) {
      if (request.isCurrent()) smartError = error.name === "AbortError" ? ""
        : `${error.message}${smartResults !== null ? " Your last results are still shown." : ""}`;
    } finally {
      if (request.isCurrent()) loading = false;
    }
  }

  function clearSearch() {
    clearTimeout(debounceTimer);
    browseGate.cancel();
    smartGate.cancel();
    loading = false;
    pageBusy = false;
    q = "";
    person = "";
    year = "";
    mediaType = "all";
    selected = Object.fromEntries(FILTER_ORDER.map(([key]) => [key, "All"]));
    smartResults = null;
    smartError = "";
    browseError = "";
    personNotFound = false;
    filterError = false;
    browseFilters = { text_scope: "all", date_from: "", date_to: "", month: "", day: "", photo_type: "", scene: "", weather: "", occasion: "", time_of_day: "", camera: "", has_text: "", has_location: "", has_caption: "", sort: "newest" };
    if (mode === "browse") scheduleBrowse();
  }

  function onSelectionChange(event) { selectedIds = event.detail; }
  function onSelect(event) { dispatch("select", event.detail); }
  function clearSelection() { resetToken += 1; selectedIds = []; }
  function toggleSelectMode() {
    selectMode = !selectMode;
    if (!selectMode) clearSelection();
    else if (!albums.length) void loadAlbums();
  }

  async function loadAlbums() {
    try { albums = (await api.albums()).albums || []; }
    catch (error) { albumMsg = `Albums could not be loaded: ${error.message}`; }
  }

  async function addToAlbum(event) {
    const value = event.target.value;
    event.target.value = "";
    if (!value || !selectedIds.length) return;
    albumMsg = "";
    try {
      let albumId = value;
      if (value === "__new__") {
        const name = prompt("Album name");
        if (!name?.trim()) return;
        albumId = (await api.createAlbum(name.trim())).id;
        await loadAlbums();
      }
      const response = await api.albumAdd(albumId, selectedIds);
      const album = albums.find((item) => item.id === albumId);
      albumMsg = `Added ${selectedIds.length} photos to ${album?.name || "the album"}.`;
      clearSelection();
    } catch (error) { albumMsg = error.message; }
  }

  async function deleteSelected() {
    if (!selectedIds.length || batchBusy) return;
    batchBusy = true;
    browseError = "";
    try {
      const deleting = new Set(selectedIds);
      const deletedCards = displayedPhotos.filter((photo) => deleting.has(photo.id));
      const response = await api.batchDelete(selectedIds, deleteFiles);
      photos = photos.filter((photo) => !deleting.has(photo.id));
      if (smartResults) smartResults = smartResults.filter((photo) => !deleting.has(photo.id));
      total = Math.max(0, total - selectedIds.length);
      summary = {
        ...summary,
        total: Math.max(0, summary.total - selectedIds.length),
        photos: Math.max(0, summary.photos - deletedCards.filter((photo) => photo.media_type !== "video").length),
        videos: Math.max(0, summary.videos - deletedCards.filter((photo) => photo.media_type === "video").length),
      };
      dispatch("summary", summary);
      lastDeleted.set([...selectedIds]);
      const failedCount = response?.files_failed?.length ?? response?.failed_files?.length ?? 0;
      if (failedCount) browseError = `${failedCount} file${failedCount === 1 ? " was" : "s were"} removed from the catalog but could not be deleted from disk.`;
      clearSelection();
      selectMode = false;
      deleteFiles = false;
      dispatch("deleted");
    } catch (error) { browseError = error.message; }
    finally { batchBusy = false; }
  }

  function removeFilter(key) {
    selected = { ...selected, [key]: "All" };
    void runSmartSearch();
  }

  onMount(() => {
    savedSearches = readSavedSearches();
    void loadSummary();
    void loadPage(0);
  });
  onDestroy(() => {
    clearTimeout(debounceTimer);
    browseGate.cancel();
    summaryGate.cancel();
    smartGate.cancel();
    filterGate.cancel();
    facetGate.cancel();
    peopleGate.cancel();
  });

  function clearQuery() {
    q = "";
    if (mode === "browse") scheduleBrowse();
    else {
      smartGate.cancel();
      loading = false;
      smartResults = null;
      smartError = "";
    }
  }
</script>

<section class="library-page">
  <div class="page-heading">
    <div>
      <p class="eyebrow">YOUR PHOTO LIBRARY</p>
      <h1>Moments, all in one place.</h1>
      <p class="intro">Browse your collection or describe a memory to find it.</p>
    </div>
    {#if summaryReady && summary.total > 0}
      <div class="library-count" aria-label="Library size">
        <b>{summary.total.toLocaleString()}</b><span>memories</span>
      </div>
    {/if}
  </div>

  <form class="search-panel" on:submit|preventDefault={mode === "smart" ? runSmartSearch : () => scheduleBrowse({ immediate: true })}>
    <div class="search-row">
      <label class="search-field">
        <span class="search-icon" aria-hidden="true">⌕</span>
        <span class="sr-only">{mode === "smart" ? "Describe a memory" : "Search photos"}</span>
        <input bind:value={q} on:input={() => mode === "browse" && scheduleBrowse()}
          placeholder={mode === "smart" ? "Describe a moment, place, or feeling…" : "Search filenames, captions, text, and places…"}
          autocomplete="off" maxlength="500" />
        {#if q}<button class="clear-query" type="button" aria-label="Clear search" on:click={clearQuery}>×</button>{/if}
      </label>
      {#if mode === "smart"}
        <button class="primary search-submit" type="submit" disabled={loading || (!q.trim() && !person.trim() && !activeFilterCount)}>
          {loading ? "Searching…" : "Find memory"}
        </button>
      {/if}
    </div>
    <div class="search-footer">
      <div class="mode-switch" role="group" aria-label="Search mode">
        <button type="button" class:active={mode === "browse"} aria-pressed={mode === "browse"} on:click={() => changeMode("browse")}>Browse</button>
        <button type="button" class:active={mode === "smart"} aria-pressed={mode === "smart"} on:click={() => changeMode("smart")}>Smart search <span class="sparkle" aria-hidden="true">✳</span></button>
      </div>
      {#if mode === "browse"}
        <div class="browse-controls" aria-label="Browse filters">
          <div class="media-switch" role="group" aria-label="Media type">
            <button type="button" class:active={mediaType === "all"} aria-pressed={mediaType === "all"} on:click={() => { mediaType = "all"; scheduleBrowse(); }}>All</button>
            <button type="button" class:active={mediaType === "photos"} aria-pressed={mediaType === "photos"} on:click={() => { mediaType = "photos"; scheduleBrowse(); }}>Photos</button>
            <button type="button" class:active={mediaType === "videos"} aria-pressed={mediaType === "videos"} on:click={() => { mediaType = "videos"; scheduleBrowse(); }}>Videos</button>
          </div>
          <label class="year-filter"><span class="sr-only">Filter by year</span>
            <select bind:value={year} on:change={() => scheduleBrowse()} aria-label="Filter by year">
              <option value="">Any year</option>
              {#each years as item}<option value={item.year}>{item.year}{item.count ? ` · ${item.count}` : ""}</option>{/each}
            </select>
          </label>
        </div>
      {:else}
        <span class="smart-hint">Searches descriptions from your indexed photos</span>
      {/if}
    </div>
  </form>

  {#if mode === "browse"}<p class="search-help">Use <code>"birthday cake" -screenshot</code> for an exact phrase and an excluded word. Search runs on your device; names match when they appear in saved photo details.</p>{/if}

  {#if mode === "browse"}
    <div class="discovery-row" aria-label="Quick ways to find memories">
      <span class="discovery-label">Try</span>
      <button type="button" on:click={() => setBrowseFilter("has_text", "yes")}>Recognized text</button>
      <button type="button" on:click={quickScreenshot}>Screenshots</button>
      <button type="button" on:click={() => setBrowseFilter("has_location", "no")}>Missing GPS</button>
      <button type="button" on:click={() => setBrowseFilter("has_caption", "no")}>No caption</button>
      <button type="button" title="Find this month and day across all years" on:click={onThisDay}>On this day</button>
    </div>

    <div class="browse-tools">
      <button class="filter-toggle" class:open={browseFiltersOpen} type="button" aria-expanded={browseFiltersOpen} on:click={openBrowseFilters}>
        <span aria-hidden="true">☷</span> Filters {#if browseChips.length}<b>{browseChips.length}</b>{/if}
      </button>
      <button class="text-button" type="button" on:click={() => { savedSearchesOpen = !savedSearchesOpen; savedSearches = readSavedSearches(); }} aria-expanded={savedSearchesOpen}>Saved searches</button>
      <button class="text-button" type="button" on:click={() => { savePanelOpen = !savePanelOpen; savedSearchMessage = ""; }}>Save this search</button>
      {#if browseChips.length}<button class="clear-filters" type="button" on:click={clearBrowseFilters}>Clear all</button>{/if}
    </div>

    {#if browseFiltersOpen}
      <section class="browse-filter-panel" aria-label="Browse filters">
        <div class="filter-panel-head"><div><h2>Find by photo details</h2><p>These filters use information already in your local photo library.</p></div><button class="panel-close" type="button" on:click={() => browseFiltersOpen = false} aria-label="Close filters">×</button></div>
        <div class="filter-grid">
          <label class="scope-field"><span>Search in</span>
            <select value={browseFilters.text_scope} on:change={(event) => setBrowseFilter("text_scope", event.currentTarget.value)}>
              <option value="all">Everything</option><option value="filename">File names</option><option value="caption">Captions</option><option value="details">Objects &amp; activities</option><option value="text">Text in photos</option><option value="place">Places</option><option value="camera">Camera info</option>
            </select>
          </label>
          <label><span>From capture date</span><input type="date" value={browseFilters.date_from} on:change={(event) => setBrowseFilter("date_from", event.currentTarget.value)} /></label>
          <label><span>Through capture date</span><input type="date" value={browseFilters.date_to} on:change={(event) => setBrowseFilter("date_to", event.currentTarget.value)} /></label>
          <label><span>Month</span><select value={browseFilters.month} on:change={(event) => { browseFilters = { ...browseFilters, month: event.currentTarget.value, day: "" }; scheduleBrowse(); }}><option value="">Any month</option>{#each Array.from({ length: 12 }, (_, index) => index + 1) as item}<option value={item}>{new Date(2000, item - 1, 1).toLocaleString(undefined, { month: "long" })}</option>{/each}</select></label>
          <label><span>Day</span><select value={browseFilters.day} disabled={!browseFilters.month} on:change={(event) => setBrowseFilter("day", event.currentTarget.value)}><option value="">Any day</option>{#each Array.from({ length: browseFilters.month ? new Date(2000, Number(browseFilters.month), 0).getDate() : 31 }, (_, index) => index + 1) as item}<option value={item}>{item}</option>{/each}</select></label>
          {#each [["photo_type", "Photo type"], ["scene", "Scene"], ["weather", "Weather"], ["occasion", "Occasion"], ["time_of_day", "Time of day"], ["camera", "Camera"]] as [key, label]}
            <label><span>{label}</span><select value={browseFilters[key]} on:change={(event) => setBrowseFilter(key, event.currentTarget.value)}><option value="">Any {label.toLowerCase()}</option>{#each normalizedFacetOptions(key, browseFacets, browseFilters) as option}<option value={option.value}>{option.value}{option.count != null ? ` · ${Number(option.count).toLocaleString()}` : ""}</option>{/each}</select></label>
          {/each}
          <label><span>Recognized text</span><select value={browseFilters.has_text} on:change={(event) => setBrowseFilter("has_text", event.currentTarget.value)}><option value="">Any</option><option value="yes">Has recognized text</option><option value="no">No recognized text saved</option></select></label>
          <label><span>GPS location</span><select value={browseFilters.has_location} on:change={(event) => setBrowseFilter("has_location", event.currentTarget.value)}><option value="">Any</option><option value="yes">Has GPS location</option><option value="no">Missing GPS location</option></select></label>
          <label><span>Caption</span><select value={browseFilters.has_caption} on:change={(event) => setBrowseFilter("has_caption", event.currentTarget.value)}><option value="">Any</option><option value="yes">Has caption</option><option value="no">No caption saved</option></select></label>
          <label><span>Order</span><select value={browseFilters.sort} on:change={(event) => setBrowseFilter("sort", event.currentTarget.value)}><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="added">Recently added</option><option value="filename">File name</option></select></label>
        </div>
        <p class="date-note">Dates use the capture date, then a date in the file name, then the date added to your library. Missing details mean no information is saved. “Missing GPS location” checks for saved coordinates.</p>
        {#if facetTruncationNote}<p class="date-note">Some filter lists show up to 100 common choices. A selected value stays visible, and results still search the full library.</p>{/if}
        {#if browseFacetsLoading}<p class="filter-hint" role="status">Loading available choices…</p>{/if}
        {#if browseFacetError}<p class="facet-error" role="alert">{browseFacetError} <button type="button" on:click={() => { browseFacetsLoaded = false; void loadBrowseFacets(); }}>Try again</button></p>{/if}
      </section>
    {/if}

    {#if browseChips.length}
      <div class="active-filters" aria-label="Active Browse filters">
        {#each browseChips as chip}<button class="filter-chip" type="button" on:click={() => removeBrowseChip(chip.key)} aria-label={`Remove ${chip.label}`}>{chip.label} <span aria-hidden="true">×</span></button>{/each}
      </div>
    {/if}

    {#if savePanelOpen}
      <section class="saved-panel" aria-label="Save this Browse search"><label><span>Name this search</span><input bind:value={savedSearchName} maxlength="60" placeholder="e.g. Summer hikes" on:keydown={(event) => event.key === "Enter" && saveCurrentBrowseSearch()} /></label><button class="primary" type="button" on:click={saveCurrentBrowseSearch}>Save search</button><button class="text-button" type="button" on:click={() => savePanelOpen = false}>Cancel</button></section>
    {/if}
    {#if savedSearchesOpen}
      <section class="saved-searches" aria-label="Saved Browse searches"><div><b>Saved on this device</b><p>These searches stay in this browser only. They are not synced or uploaded.</p></div>
        {#if savedSearches.length}<ul>{#each savedSearches as item (item.name)}<li><button class="saved-name" type="button" on:click={() => applySavedSearch(item)}>{item.name}</button><button class="saved-remove" type="button" aria-label={`Remove saved search ${item.name}`} on:click={() => removeSavedSearch(item)}>Remove</button></li>{/each}</ul>{:else}<p class="filter-hint">No saved searches yet.</p>{/if}
      </section>
    {/if}
    {#if savedSearchMessage || browseFeedback}<p class="browse-feedback" role="status">{savedSearchMessage || browseFeedback}</p>{/if}
  {/if}

  {#if mode === "smart"}
    <div class="smart-tools">
      <label class="person-field"><span>Person</span>
        <input bind:value={person} list="registered-people" placeholder="Optional registered person" on:keydown={(event) => event.key === "Enter" && runSmartSearch(event)} />
        <datalist id="registered-people">{#each smartPeople as name}<option value={name}></option>{/each}</datalist>
      </label>
      <button class="filter-toggle" class:open={filtersOpen} type="button" aria-expanded={filtersOpen} on:click={() => { filtersOpen = !filtersOpen; if (filtersOpen && !filtersLoaded) loadFilters(); }}>
        <span aria-hidden="true">☷</span> More filters {#if activeFilterCount}<b>{activeFilterCount}</b>{/if}
      </button>
      {#if activeFilterCount}
        <button type="button" class="clear-filters" on:click={() => { selected = Object.fromEntries(FILTER_ORDER.map(([key]) => [key, "All"])); runSmartSearch(); }}>Clear filters</button>
      {/if}
    </div>
    {#if smartPeopleError}<p class="filter-hint people-hint">Registered people could not be loaded. You can still enter a name.</p>{/if}
    {#if filtersOpen}
      <div class="advanced-filters">
        {#if filtersLoading}<span class="filter-hint">Loading available filters…</span>
        {:else if !Object.keys(filterVals).length}<span class="filter-hint">No additional filters are available yet.</span>
        {:else}
          {#each FILTER_ORDER as [key, label]}
            {#if filterVals[key]?.length}
              <label><span>{label}</span>
                <select bind:value={selected[key]} on:change={() => runSmartSearch()}>
                  <option value="All">Any {label.toLowerCase()}</option>
                  {#each filterVals[key] as value}<option value={value}>{value}</option>{/each}
                </select>
              </label>
            {/if}
          {/each}
        {/if}
      </div>
    {/if}
  {/if}

  {#if mode === "smart" && activeFilterCount}
    <div class="active-filters" aria-label="Active filters">
      {#each FILTER_ORDER as [key, label]}
        {#if selected[key] && selected[key] !== "All"}
          <button class="filter-chip" type="button" on:click={() => removeFilter(key)}>{label}: {selected[key]} <span aria-hidden="true">×</span></button>
        {/if}
      {/each}
    </div>
  {/if}

  {#if (mode === "browse" ? browseError : smartError)}
    <div class="notice error" role="alert">{mode === "browse" ? browseError : smartError}</div>
  {/if}
  {#if personNotFound}
    <div class="notice">No one named “{person}” is registered yet. Add a name under People to search for them.</div>
  {/if}
  {#if filterError}
    <div class="notice">Some filters could not be applied, so these results may be broader than requested.</div>
  {/if}

  <div class="results-heading">
    <div><h2>{mode === "smart" ? (smartResults !== null ? "Search results" : "Search the library") : q.trim() || year ? "Matching moments" : "Your memories"}</h2>
      {#if mode === "smart" && smartResults === null}<p>Describe a memory above to search your collection.</p>
      {:else if initialLoading}<p>Opening your collection…</p>
      {:else}<p aria-live="polite">{displayedTotal.toLocaleString()} {displayedTotal === 1 ? "memory" : "memories"}{loading ? " · Updating" : ""}</p>{/if}
    </div>
    <div class="result-actions">
      {#if mode === "smart" && smartResults !== null}<button class="text-button" on:click={() => { smartResults = null; smartError = ""; }}>Back to library</button>{/if}
      {#if selectedIds.length}
        <span class="selected-count" aria-live="polite">{selectedIds.length} selected</span>
        <select class="album-select" on:change={addToAlbum} aria-label="Add selected memories to an album">
          <option value="">Add to album</option>
          {#each albums as album}<option value={album.id}>{album.name}</option>{/each}
          <option value="__new__">＋ New album</option>
        </select>
        <label class="delete-option"><input type="checkbox" bind:checked={deleteFiles} /> Delete files too</label>
        <button class="danger-button" on:click={deleteSelected} disabled={batchBusy}>{batchBusy ? "Removing…" : "Remove"}</button>
        <button class="text-button" on:click={clearSelection}>Cancel</button>
      {:else}
        <button class="select-button" class:active={selectMode} on:click={toggleSelectMode}>{selectMode ? "Done selecting" : "Select"}</button>
      {/if}
    </div>
  </div>
  {#if albumMsg}<p class="album-message" role="status">{albumMsg}</p>{/if}

  {#if initialLoading && !photos.length && mode === "browse"}
    <div class="skeleton-grid" aria-label="Loading photos" aria-busy="true">
      {#each Array(12) as _}<span></span>{/each}
    </div>
  {:else if displayedPhotos.length}
    <PhotoGrid photos={displayedPhotos} {selectMode} {resetToken}
      on:select={onSelect} on:selectionchange={onSelectionChange} />
    {#if mode === "browse" && hasMore}
      <div class="load-more-wrap"><button class="load-more" on:click={() => loadPage(photos.length)} disabled={pageBusy}>
        {pageBusy ? "Loading more…" : "Load more memories"}
      </button></div>
    {/if}
    {#if mode === "smart" && smartResults?.length === 0}
      <div class="empty-state"><span class="empty-art" aria-hidden="true">⌕</span><h3>No close matches yet</h3><p>Try a different description, person, or filter.</p></div>
    {/if}
  {:else if mode === "smart" && smartResults === null}
    <div class="empty-state"><span class="empty-art" aria-hidden="true">✳</span><h3>Search by the feeling or details you remember</h3><p>For example: “a rainy afternoon at the lake” or “everyone around the birthday cake”.</p></div>
  {:else if !initialLoading}
    <div class="empty-state">
      {#if summaryError || browseError}
        <span class="empty-art" aria-hidden="true">↻</span><h3>Your library could not be opened</h3>
        <p>Check the local app connection and try again.</p>
        <button class="load-more" on:click={() => { browseError = ""; summaryError = false; initialLoading = true; void loadSummary(); void loadPage(0); }}>Try again</button>
      {:else if summaryReady && summary.total === 0}
        <span class="empty-art" aria-hidden="true">◩</span><h3>Your photo library is ready for its first memories</h3>
        <p>Choose a folder in Manage to add photos. They stay on this device.</p>
        <button class="primary" on:click={() => dispatch("goto-manage")}>Set up your library</button>
      {:else}
        <span class="empty-art" aria-hidden="true">⌕</span><h3>No memories match those filters</h3>
        <p>Try another year or media type, or clear your search.</p>
        <button class="text-button" on:click={clearSearch}>Clear search and filters</button>
      {/if}
    </div>
  {/if}
</section>

<style>
  .library-page { width: min(100%, 1540px); margin: 0 auto; padding: 36px clamp(20px, 4vw, 62px) 56px; }
  .page-heading { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 24px; }
  .eyebrow { color: #51746b; font-size: 10px; font-weight: 800; letter-spacing: .15em; margin-bottom: 9px; }
  h1 { color: var(--ink); font-size: clamp(25px, 3vw, 35px); letter-spacing: -.055em; line-height: 1.08; }
  .intro { margin-top: 8px; color: #68756e; font-size: 14px; }
  .library-count { display: flex; align-items: baseline; gap: 7px; padding: 0 4px 4px; color: #66756d; font-size: 12px; white-space: nowrap; }
  .library-count b { color: var(--ink); font-size: 21px; font-variant-numeric: tabular-nums; letter-spacing: -.04em; }
  .search-panel { padding: 9px 10px 0; background: #fff; border: 1px solid #e5e8e2; border-radius: 15px; box-shadow: 0 7px 24px rgba(36,53,47,.045); }
  .search-row { display: flex; gap: 10px; }
  .search-field { display: flex; align-items: center; flex: 1; min-width: 0; gap: 12px; padding: 0 13px; }
  .search-icon { color: #2f766c; font-size: 25px; line-height: 1; transform: rotate(-20deg); }
  .search-field input { width: 100%; height: 47px; padding: 0; background: transparent; border: 0; outline: 0; color: var(--ink); font-size: 15px; box-shadow: none; }
  .search-field input::placeholder { color: #6c7972; }
  .search-field:focus-within { box-shadow: inset 0 -2px #6eaaa0; }
  .search-help { margin: 8px 2px 0; color: #66756d; font-size: 11px; line-height: 1.5; }
  .search-help code { padding: 1px 4px; color: #375b53; background: #edf2ec; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10px; }
  .clear-query { width: 30px; height: 30px; padding: 0; color: var(--muted); background: #f1f3ef; border: 0; border-radius: 50%; font-size: 19px; line-height: 1; }
  .clear-query:hover { transform: none; filter: none; background: #e8ece6; }
  .search-submit { min-width: 130px; margin: 2px 0; padding: 0 19px; border-radius: 10px; }
  .search-footer { display: flex; justify-content: space-between; align-items: center; gap: 12px; min-height: 51px; border-top: 1px solid #eef0ec; }
  .mode-switch, .media-switch { display: inline-flex; align-items: center; gap: 3px; padding: 3px; border-radius: 9px; background: #f3f5f1; }
  .mode-switch button, .media-switch button { padding: 7px 11px; color: #5e6d65; background: transparent; border: 0; border-radius: 7px; font-size: 12px; font-weight: 650; white-space: nowrap; }
  .mode-switch button:hover, .media-switch button:hover { transform: none; filter: none; color: var(--ink); }
  .mode-switch button.active, .media-switch button.active { color: #185b57; background: #fff; box-shadow: 0 1px 4px rgba(25,49,41,.1); }
  .sparkle { padding-left: 3px; color: #4d9287; }
  .browse-controls { display: flex; align-items: center; gap: 9px; }
  .year-filter select { min-width: 105px; width: auto; height: 34px; padding: 5px 24px 5px 10px; border-color: transparent; background-color: transparent; color: #53625b; font-size: 12px; }
  .smart-hint { padding-right: 7px; color: #69766f; font-size: 11px; }
  .discovery-row { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; padding: 12px 1px 0; }
  .discovery-label { padding-right: 2px; color: #758078; font-size: 11px; }
  .discovery-row button { padding: 5px 9px; color: #52685f; background: #f2f5f0; border: 1px solid #e5eae2; border-radius: 20px; font-size: 10px; }
  .discovery-row button:hover { transform: none; filter: none; color: #245f59; background: #eaf3ee; border-color: #d2e2d8; }
  .browse-tools { display: flex; align-items: center; gap: 8px; padding-top: 11px; }
  .browse-filter-panel, .saved-panel, .saved-searches { margin-top: 11px; padding: 15px; background: #fbfcf9; border: 1px solid #e4e9e2; border-radius: 12px; }
  .filter-panel-head { display: flex; justify-content: space-between; align-items: start; gap: 12px; margin-bottom: 13px; }
  .filter-panel-head h2 { color: #33423b; font-size: 14px; }
  .filter-panel-head p, .saved-searches > div p { margin-top: 4px; color: #68766f; font-size: 11px; line-height: 1.45; }
  .panel-close { width: 30px; height: 30px; padding: 0; color: #58665e; background: transparent; border: 0; border-radius: 50%; font-size: 21px; }
  .panel-close:hover { background: #edf1eb; transform: none; filter: none; }
  .filter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 10px; }
  .filter-grid label, .saved-panel label { display: grid; min-width: 0; gap: 5px; }
  .filter-grid label span, .saved-panel label span { color: #65736b; font-size: 11px; font-weight: 650; }
  .filter-grid input, .filter-grid select, .saved-panel input { width: 100%; min-width: 0; height: 36px; padding: 6px 9px; border-color: #e0e6de; background: #fff; font-size: 12px; }
  .date-note { margin-top: 11px; color: #6b786f; font-size: 10px; line-height: 1.5; }
  .facet-error, .browse-feedback { margin-top: 8px; color: #795333; font-size: 11px; line-height: 1.45; }
  .facet-error button { padding: 2px 4px; color: var(--accent); background: transparent; border: 0; text-decoration: underline; }
  .saved-panel { display: flex; align-items: end; gap: 8px; }
  .saved-panel label { flex: 1; }
  .saved-panel .primary { height: 36px; padding: 0 12px; font-size: 11px; }
  .saved-searches > div { margin-bottom: 10px; color: #465a51; font-size: 12px; }
  .saved-searches ul { display: grid; gap: 4px; padding: 0; margin: 0; list-style: none; }
  .saved-searches li { display: flex; align-items: center; gap: 8px; min-width: 0; padding: 6px 8px; background: #fff; border: 1px solid #edf0eb; border-radius: 7px; }
  .saved-name { flex: 1; overflow: hidden; padding: 2px; color: #315d53; background: transparent; border: 0; text-align: left; text-overflow: ellipsis; white-space: nowrap; }
  .saved-name:hover { transform: none; filter: none; text-decoration: underline; }
  .saved-remove { padding: 4px 7px; color: #6b766f; background: transparent; border: 0; font-size: 10px; }
  .saved-remove:hover { color: #9b4540; transform: none; filter: none; }
  .people-hint { margin: -3px 2px 0; }
  .smart-tools { display: flex; align-items: end; gap: 9px; padding: 12px 2px 0; }
  .person-field { display: grid; gap: 4px; min-width: 210px; max-width: 300px; flex: 1; }
  .person-field span, .advanced-filters label span { color: #64726b; font-size: 11px; font-weight: 650; }
  .person-field input, .advanced-filters select { height: 36px; padding: 7px 10px; border-color: #e2e6df; background: #fff; font-size: 12px; }
  .filter-toggle, .filter-chip, .select-button, .load-more { color: #425d56; background: #fff; border: 1px solid #dfe6df; font-size: 12px; }
  .filter-toggle { display: inline-flex; align-items: center; justify-content: center; gap: 5px; white-space: nowrap; height: 36px; padding: 0 12px; }
  .browse-tools .filter-toggle { flex: 0 0 auto; }
  .filter-toggle.open { background: #edf5f2; border-color: #b9d5cd; }
  .filter-toggle b { display: inline-grid; place-items: center; min-width: 18px; height: 18px; margin-left: 4px; border-radius: 10px; color: #fff; background: var(--accent); font-size: 10px; }
  .clear-filters, .text-button { padding: 7px 8px; color: var(--accent); background: transparent; border: 0; font-size: 12px; }
  .clear-filters:hover, .text-button:hover { color: #245f59; transform: none; filter: none; text-decoration: underline; }
  .advanced-filters { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 11px; margin-top: 11px; padding: 15px; background: #f8f9f6; border: 1px solid #e8ebe5; border-radius: 11px; }
  .advanced-filters label { display: grid; gap: 5px; }
  .filter-hint { color: #64726b; font-size: 12px; }
  .active-filters { display: flex; flex-wrap: wrap; gap: 6px; padding-top: 10px; }
  .filter-chip { padding: 5px 8px; border-radius: 20px; background: #eaf3f0; border-color: #d4e6df; color: #356d64; }
  .filter-chip span { padding-left: 4px; font-size: 14px; }
  .results-heading { display: flex; justify-content: space-between; align-items: end; gap: 16px; margin: 28px 0 14px; }
  .results-heading h2 { color: var(--ink); font-size: 17px; letter-spacing: -.025em; }
  .results-heading p { margin-top: 4px; color: #66756d; font-size: 12px; }
  .result-actions { display: flex; align-items: center; justify-content: end; flex-wrap: wrap; gap: 6px; }
  .select-button { padding: 7px 11px; border-radius: 8px; }
  .select-button.active { background: #eaf3f0; border-color: #bbd7ce; color: #276a60; }
  .selected-count { padding: 0 5px; color: #4d6f67; font-size: 12px; }
  .album-select { width: auto; min-width: 120px; height: 33px; padding: 5px 9px; background: #fff; font-size: 11px; }
  .delete-option { display: flex; align-items: center; gap: 4px; color: var(--muted); font-size: 11px; }
  .delete-option input { width: 14px; accent-color: var(--accent); }
  .danger-button { padding: 7px 10px; color: #fff; background: var(--danger); border: 0; border-radius: 8px; font-size: 11px; }
  .danger-button:hover { transform: none; filter: brightness(1.05); }
  .album-message { margin: -5px 0 10px; color: var(--accent); font-size: 12px; }
  .notice { margin: 12px 0; padding: 10px 13px; color: #496158; background: #eff5f1; border: 1px solid #dfeae2; border-radius: 9px; font-size: 12px; }
  .notice.error { color: #8f4842; background: #fbf0ee; border-color: #f0d9d5; }
  .skeleton-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(142px, 1fr)); gap: 7px; }
  .skeleton-grid span { aspect-ratio: 1.08; border-radius: 8px; background: linear-gradient(100deg,#f0f2ee 25%,#f7f8f6 42%,#f0f2ee 62%); background-size: 220% 100%; animation: shimmer 1.5s ease infinite; }
  @keyframes shimmer { to { background-position-x: -220%; } }
  .empty-state { display: grid; justify-items: center; max-width: 510px; margin: 68px auto; padding: 24px; text-align: center; }
  .empty-art { display: grid; place-items: center; width: 54px; height: 54px; margin-bottom: 14px; color: #4a9084; background: #e9f3ef; border-radius: 18px; font-size: 27px; }
  .empty-state h3 { color: #2b3935; font-size: 18px; letter-spacing: -.025em; }
  .empty-state p { margin: 7px 0 15px; color: #66756d; font-size: 13px; line-height: 1.55; }
  .empty-state .primary { padding: 9px 15px; }
  .load-more-wrap { display: flex; justify-content: center; padding: 24px 0 4px; }
  .load-more { min-width: 190px; padding: 10px 16px; border-radius: 9px; }
  .load-more:hover:not(:disabled) { transform: translateY(-1px); background: #f5f8f4; border-color: #bad3c9; }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
  @media (max-width: 760px) {
    .library-page { padding: 25px 15px 34px; }
    .page-heading { margin-bottom: 18px; }
    .library-count { display: none; }
    .intro { font-size: 12px; }
    .search-panel { padding: 7px 8px 0; }
    .search-field input { height: 43px; font-size: 13px; }
    .search-footer { align-items: flex-start; flex-direction: column; padding: 9px 0 7px; }
    .browse-controls { width: 100%; justify-content: space-between; }
    .media-switch button { padding: 7px 9px; }
    .year-filter select { min-width: 94px; }
    .search-submit { min-width: 94px; padding: 0 11px; font-size: 12px; }
    .smart-hint { display: none; }
    .smart-tools { align-items: stretch; flex-wrap: wrap; }
    .person-field { min-width: 100%; }
    .filter-toggle { flex: 1; }
    .results-heading { align-items: start; flex-direction: column; margin-top: 22px; }
    .discovery-row { gap: 5px; padding-top: 9px; }
    .discovery-row button { min-height: 32px; padding: 6px 8px; font-size: 10px; }
    .browse-tools { flex-wrap: wrap; }
    .browse-filter-panel, .saved-panel, .saved-searches { padding: 12px; }
    .filter-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; }
    .filter-grid label:first-child { grid-column: 1 / -1; }
    .saved-panel { align-items: stretch; flex-wrap: wrap; }
    .saved-panel label { min-width: 100%; }
    .saved-panel .primary { flex: 1; }
    .result-actions { width: 100%; justify-content: start; }
    .skeleton-grid { grid-template-columns: repeat(3, minmax(0,1fr)); gap: 4px; }
  }
</style>
