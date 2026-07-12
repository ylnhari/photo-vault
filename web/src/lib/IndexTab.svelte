<script>
  import { api } from "./api.js";
  import { onMount, onDestroy, createEventDispatcher } from "svelte";
  import { health, status, models, refreshHealth, refreshStatus, refreshModels, jobStatus } from "./stores.js";
  import StatusPill from "./StatusPill.svelte";
  import JobPanel from "./JobPanel.svelte";
  import SectionHead from "./SectionHead.svelte";
  import FolderPicker from "./FolderPicker.svelte";
  import ErrLine from "./ErrLine.svelte";
  import { onActivateKey } from "./keyboard.js";
  const dispatch = createEventDispatcher();

  // ── job / poll state ────────────────────────────────────────────────────────
  let job = null;
  let poll = null;
  let busy = {};
  // One error at a time, SCOPED to the section whose action raised it —
  // rendered inline there via <ErrLine>, not in a page-top banner the user
  // may have scrolled away from. errAt = "" means global (top banner).
  let err = "";
  let errAt = "";
  const TYPE_SCOPE = { scan: "folders", ingest: "import", backup: "backup",
                       dhash: "dupes", dedupe: "dupes",
                       video_vision: "video", video_faces: "video" };
  const typeScope = (t) => TYPE_SCOPE[t] || t || "";
  function fail(scope, msg) { err = msg; errAt = scope; }
  function clearErr() { err = ""; errAt = ""; }
  let rechecking = false;
  let stopRequesting = false;  // bound into JobPanel; reset on stop() failure so the button doesn't stick

  // ── provider model catalogue ─────────────────────────────────────────────────
  let pmodels = { lm_studio: [], lm_studio_types: {}, gemini_vision: [], gemini_embed: [], gemini_cooldowns: {},
                  ninerouter_vision: [], ninerouter_embed: [], ninerouter_cooldowns: {} };

  // Face-detection accelerators auto-detected from the installed onnxruntime
  // build (options), plus what the current choice actually resolves to (active).
  let faceProviders = { options: [{ id: "auto", label: "Auto (fastest available)" }], selected: "auto", active: "" };
  async function loadFaceProviders() {
    try { faceProviders = await api.faceProviders(); } catch {}
  }
  // True once any real accelerator (not just auto/cpu) is installed — drives the
  // "install a GPU/NPU wheel" hint on the picker.
  $: faceAccelAvailable = (faceProviders.options || []).some(
    (o) => o.id !== "auto" && o.id !== "cpu");

  // ── app settings ─────────────────────────────────────────────────────────────
  // Provider rate limits: 0 = unlimited. The grid below binds straight into
  // settings.rate_limits[provider][window], so every loaded settings object
  // must carry the full structure even if the server response is partial.
  const RL_PROVIDERS = [["lm_studio", "LM Studio"], ["gemini", "Gemini"], ["9router", "9Router"]];
  const RL_WINDOWS = [["rps", "req/sec"], ["rpm", "req/min"], ["rph", "req/hour"], ["rpd", "req/day"]];
  function normalizeRateLimits(s) {
    s.rate_limits = s.rate_limits || {};
    for (const [p] of RL_PROVIDERS)
      s.rate_limits[p] = { rps: 0, rpm: 0, rph: 0, rpd: 0, ...(s.rate_limits[p] || {}) };
    return s;
  }

  // "Suggest" fills a provider's row from the backend: values learned from
  // real Gemini 429 QuotaFailure metadata when we've seen one, else the
  // published free-tier table. (There is no query-my-quota API for an AI
  // Studio key, so those are the only honest sources.)
  let rlSuggestNote = "";
  async function suggestLimits(pkey) {
    rlSuggestNote = "";
    try {
      const model = pkey === "gemini"
        ? ((settings.vision_provider === "gemini" && settings.vision_model) || null)
        : null;
      const r = await api.rateLimitSuggest(pkey, model);
      if (!r.available) { rlSuggestNote = r.reason; return; }
      settings.rate_limits[pkey] = { rps: r.rps, rpm: r.rpm, rph: r.rph, rpd: r.rpd };
      settings = settings;
      markDirty();
      const srcs = Object.values(r.sources || {});
      rlSuggestNote = `Filled for ${r.model}: ` + (srcs.includes("learned")
        ? "includes exact limits learned from this account's 429 responses."
        : "published free-tier values — a real 429 teaches the exact account numbers.");
    } catch (e) { rlSuggestNote = e.message; }
  }

  let settings = normalizeRateLimits({
    vision_provider: "auto", vision_model: null,
    embed_provider: "auto",  embed_model: null,
    caption_source_model: null,
    max_fail: 5,
  });
  let settingsDirty = false;
  let settingsSaving = false;

  // ── folder management ─────────────────────────────────────────────────────────
  let folderConfig = { included: [], excluded: [] };
  let newFolderPath = "";
  let newExcludePath = "";
  let defaults = [];
  let confirmRemove = null;
  let showExcluded = false;

  // ── orphaned ──────────────────────────────────────────────────────────────────
  let orphaned = { orphaned: [], total: 0 };
  let showOrphaned = false;
  let orphanedBusy = false;

  // ── ingest / dedupe / backup ─────────────────────────────────────────────────
  let stagingPath = "";
  // Import media filter — "both" | "photos" | "videos". Never forces the user
  // to split: "both" imports a mixed folder as-is; photos → <dest>/YYYY/MM,
  // videos → their own <videoDest>/YYYY/MM. videoDest blank = server default
  // (a scanned Videos root, else <dest>/Videos).
  let ingestMedia = "both";
  let videoDest = "";
  let dedupeCount = 0;
  let backupSt = { configured: false, dest: null, available: false, days_since: null };

  async function loadDedupeCount() {
    try { dedupeCount = (await api.dedupePending()).count; } catch {}
  }
  async function loadBackupStatus() {
    try { backupSt = await api.backupStatus(); } catch {}
  }

  // Folder picker + pre-flight validation. Every folder choice goes through
  // a server-side validator that either approves or explains, in a full
  // sentence, why the operation can't happen — no silent failures, no
  // hand-typed paths required.
  let picker = null; // "source" | "ingest_dest" | "backup_dest" | null
  let srcCheck = null;       // /api/ingest/validate result for stagingPath
  let srcChecking = false;
  let backupMsg = "";

  const fmtGB = (b) => (b / 1073741824).toFixed(2);

  async function validateSource(path) {
    stagingPath = path;
    srcCheck = null;
    if (!path?.trim()) return;
    srcChecking = true;
    try { srcCheck = await api.ingestValidate(path.trim()); }
    catch (e) { srcCheck = { ok: false, reason: e.message }; }
    srcChecking = false;
  }

  async function onPickFolder(path) {
    const which = picker;
    picker = null;
    if (which === "source") return validateSource(path);
    if (which === "video_dest") { videoDest = path; return; }
    if (which === "ingest_dest") {
      settings.ingest_dest = path;
      await saveSettings();               // server validates; err shows reason
      if (err) { errAt = "import"; await loadSettings(); }  // show where the user acted
      if (stagingPath) validateSource(stagingPath);
    }
    if (which === "backup_dest") {
      backupMsg = "";
      try {
        const v = await api.backupValidate(path);
        if (!v.ok) { backupMsg = v.reason; return; }
        settings.backup_dest = path;
        await saveSettings();
        if (err) { backupMsg = err; err = ""; await loadSettings(); }
        loadBackupStatus();
      } catch (e) { backupMsg = e.message; }
    }
  }

  const PROVIDERS = [
    ["auto", "Auto"],
    ["lm_studio", "LM Studio"],
    ["gemini", "Gemini"],
    ["9router", "9Router"],
  ];

  // 9Router model ids are provider-prefixed; translate the prefix into a
  // human tag so "the same model via two providers" stays tellable-apart.
  const NINEROUTER_PREFIX_TAGS = {
    "gc/": "Gemini CLI", "gemini/": "Gemini API", "kr/": "Kiro · credits",
    "openrouter/": "OpenRouter", "oc/": "OpenCode", "mmf/": "MiMo",
  };
  function ninerouterTag(id) {
    for (const [p, tag] of Object.entries(NINEROUTER_PREFIX_TAGS))
      if (id.startsWith(p)) return tag;
    return "";
  }

  const PROVIDER_NAMES = { lm_studio: "LM Studio", gemini: "Gemini", "9router": "9Router" };

  // Human titles for the "blocked — X is running" messages below, mirrors
  // JobPanel's own TITLES map so the two never disagree on wording.
  const TITLES_BY_TYPE = { vision: "Vision analysis", embed: "Embedding",
    full: "Full index", reanalyze: "Re-analyze", faces: "Face detection",
    thumbs: "Thumbnails", dhash: "Duplicate scan", scan: "Scanning folders",
    ingest: "Import & consolidate", dedupe: "Removing duplicate copies",
    backup: "Backup", video_vision: "Video captioning",
    video_faces: "Video face detection" };

  onMount(async () => {
    if (!$health.loaded) refreshHealth();
    if (!$status.loaded) refreshStatus();
    if (!$models.loaded) refreshModels();
    // Job status first: when a job is running, the panel (with Stop) must
    // appear immediately — providerModels can take seconds while LM Studio
    // is busy doing inference, and everything below can arrive later.
    // Wrapped so a transient failure here doesn't skip the Promise.all below
    // and leave the whole tab blank.
    try {
      job = await api.indexProgress();
      if (job.any_active) startPolling();
    } catch (e) {
      console.error("indexProgress failed on mount", e);
      fail("", "Could not load job status.");
    }
    await Promise.all([
      loadSettings(),
      loadFolderConfig(),
      loadOrphaned(),
      loadDedupeCount(),
      loadBackupStatus(),
      loadFaceProviders(),
      api.providerModels().then(r => { pmodels = r; }).catch(() => {}),
    ]);
  });

  async function loadSettings() {
    try { settings = normalizeRateLimits(await api.getSettings()); settingsDirty = false; } catch {}
  }
  async function loadFolderConfig() {
    try { folderConfig = await api.getFolderConfig(); } catch {}
  }
  async function loadOrphaned() {
    try { orphaned = await api.getOrphaned(); } catch {}
  }

  // ── settings derivations ─────────────────────────────────────────────────────
  $: lmTypes = pmodels.lm_studio_types || {};

  // Loaded-in-memory models first (from LM Studio's native v0 API when
  // available — real state, not a guess), then alphabetical.
  function byLoadedFirst(a, b) {
    const la = lmTypes[a]?.state === "loaded" ? 0 : 1;
    const lb = lmTypes[b]?.state === "loaded" ? 0 : 1;
    return la - lb || a.localeCompare(b);
  }

  // LM Studio models filtered by type
  $: lmVisionModels = pmodels.lm_studio.filter(m => {
    const t = lmTypes[m]?.type;
    return t === "vision" || t === "unknown";
  }).sort(byLoadedFirst);
  $: lmEmbedModels = pmodels.lm_studio.filter(m => {
    const t = lmTypes[m]?.type;
    return t === "embed" || t === "unknown";
  }).sort(byLoadedFirst);

  // Dropdown options for current provider selection
  $: visionModelOpts = settings.vision_provider === "lm_studio" ? lmVisionModels
                     : settings.vision_provider === "gemini" ? pmodels.gemini_vision
                     : settings.vision_provider === "9router" ? (pmodels.ninerouter_vision || [])
                     : [];
  $: embedModelOpts  = settings.embed_provider === "lm_studio" ? lmEmbedModels
                     : settings.embed_provider === "gemini" ? pmodels.gemini_embed
                     : settings.embed_provider === "9router" ? (pmodels.ninerouter_embed || [])
                     : [];

  // 9Router has no auto-detect: a model choice is mandatory (the API also
  // rejects the job with a 422, this just surfaces it before the click).
  $: visionCfgIncomplete = settings.vision_provider === "9router" && !settings.vision_model;
  $: embedCfgIncomplete  = settings.embed_provider === "9router" && !settings.embed_model;

  // All previously used vision model labels (from caption_history summary)
  $: usedVisionModels = Object.keys($status.model_status?.vision?.model_summary || {});

  // Vision model label for the current settings (e.g. "lm_studio:qwen2-vl-7b").
  // Not shown for 9Router: the gateway can substitute the serving model, so
  // captions are stored under the model that actually produced them.
  $: selectedVisionLabel = (settings.vision_provider && !["auto", "9router"].includes(settings.vision_provider) && settings.vision_model)
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
  $: videoTotal = $status.video_total || 0;
  $: videoVisionPending = $status.video_vision_pending || 0;
  $: videoFacesPending = $status.video_faces_pending || 0;
  $: thumbsPending = $status.thumbs_pending || 0;
  $: dhashPending = $status.dhash_pending || 0;
  $: trashCount = $status.trash_count || 0;

  // ── settings helpers ──────────────────────────────────────────────────────────
  function markDirty() { settingsDirty = true; }

  async function saveSettings() {
    settingsSaving = true; clearErr();
    try {
      settings = normalizeRateLimits(await api.saveSettings(settings));
      settingsDirty = false;
      await refreshStatus();
    } catch (e) { fail("settings", e.message); }
    settingsSaving = false;
  }

  // One-click recovery after a 9Router embed-model substitution: adopt the
  // model the gateway actually serves (its own fresh collection — the strict
  // one-vector-space-per-collection rule is what aborted the run) and re-run.
  let switchingServed = false;
  async function switchToServedAndRerun() {
    const sub = jobOf("embed")?.substitution;
    if (!sub?.suggested) return;
    switchingServed = true;
    settings.embed_provider = "9router";
    settings.embed_model = sub.suggested;
    await saveSettings();
    if (!err) await start("embed");
    switchingServed = false;
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
      // Jobs can run concurrently now — keep polling until EVERY job is idle.
      if (!(job.any_active)) {
        clearInterval(poll);
        await Promise.all([refreshStatus(), refreshModels()]);
        loadFolderConfig();  // scan jobs update folder stats
        loadOrphaned();
        loadDedupeCount();   // scans record dup copies; dedupe consumes them
        loadBackupStatus();  // backup jobs update last-backup time
        dispatch("indexed");
      }
    }, 1000);
  }

  // ── multi-job derivations ────────────────────────────────────────────────────
  // `job` is the raw progress response: a primary snapshot PLUS `.jobs[]` (all
  // tracked jobs), `.job_resources` (type → resource list), `.any_active`.
  // Jobs whose resource sets are disjoint run concurrently, so the UI keys each
  // section off ITS OWN job, and only blocks a section when starting it would
  // conflict with a running job.
  $: jobsList = (job && Array.isArray(job.jobs)) ? job.jobs
              : (job && job.type ? [job] : []);
  $: activeJobs = jobsList.filter((j) => j.active);
  $: anyActive = job ? (job.any_active ?? activeJobs.length > 0) : false;
  $: activeResources = new Set(activeJobs.flatMap((j) => j.resources || []));

  // Most-recent job (active or recently-finished) of a given type, or null.
  function jobOf(type) { return jobsList.find((j) => j.type === type) || null; }
  function resourcesFor(type) {
    return (job && job.job_resources && job.job_resources[type]) || [];
  }
  // Display name(s) of running job(s) that would block starting `type`
  // (shared resource); "" when `type` can start right now.
  function conflictFor(type) {
    const need = new Set(resourcesFor(type));
    const blockers = activeJobs.filter((j) => (j.resources || []).some((r) => need.has(r)));
    if (!blockers.length) return "";
    return [...new Set(blockers.map((j) => TITLES_BY_TYPE[j.type] || j.type))].join(", ");
  }

  $: noServices = $health.loaded && !$health.lm_studio && !$health.gemini && !$health.ninerouter;

  // "Online" (server answered) is NOT "ready" — LM Studio lists JIT-loadable
  // models even with nothing in memory. When its v0 API tells us the real
  // loaded state, reflect it: online with no model loaded = warn, not green.
  $: lmState = $health.lm_studio_state || { known: false };
  $: lmPillState = !$health.lm_studio ? "off"
                 : (lmState.known && !lmState.vision_loaded && !lmState.embed_loaded) ? "warn"
                 : "on";
  $: lmPillDetail = !$health.lm_studio ? "offline"
                  : !lmState.known ? "online"
                  : [lmState.vision_loaded && `vision: ${lmState.vision_loaded}`,
                     lmState.embed_loaded && `embed: ${lmState.embed_loaded}`]
                      .filter(Boolean).join(" · ") || "online — no model loaded";
  $: running = anyActive;                                   // any job at all
  $: scanRunning = activeJobs.some((j) => j.type === "scan");
  $: st = $status;

  // Catch up when a job was started from elsewhere (another tab, another
  // session) while this tab was sitting idle: this component's own `job`
  // only updates via its 1s poll, which only starts if a job was ALREADY
  // active when this tab mounted. The global jobStatus store (App.svelte)
  // polls continuously regardless — when it flips to active and we're not
  // already tracking it, pull the real progress and start local polling so
  // buttons/hints/JobPanel reflect reality within one global-poll interval
  // instead of requiring a manual page refresh.
  // (Plain store subscription, not a `$:` block — assigning `job` from a
  // reactive statement that also reads `job` via `running` is a cyclical
  // dependency as far as Svelte's compiler is concerned.)
  let syncingFromGlobal = false;
  const unsubJobStatus = jobStatus.subscribe(($js) => {
    if ($js.active && !(job && job.any_active) && !syncingFromGlobal) {
      syncingFromGlobal = true;
      api.indexProgress().then((j) => {
        job = j;
        if (j.any_active) startPolling();
        syncingFromGlobal = false;
      }).catch(() => { syncingFromGlobal = false; });
    }
  });
  onDestroy(unsubJobStatus);

  async function start(type, extra = {}) {
    clearErr();
    // Save settings first if dirty
    if (settingsDirty) await saveSettings();
    try {
      // indexStart returns the started job; re-pull progress for the full
      // (possibly concurrent) picture so other running jobs stay visible.
      await api.indexStart({ ...buildCfg(type), ...extra });
      job = await api.indexProgress();
      if (job.any_active) startPolling();
      else { await refreshStatus(); dispatch("indexed"); }
    } catch (e) { fail(typeScope(type), e.message); }
  }
  async function stopJob(type) {
    const j = jobOf(type);
    try {
      await api.indexStop(j ? { job_id: j.id } : undefined);
      job = await api.indexProgress();
    } catch (e) {
      // Return the button to a clickable state instead of leaving it stuck
      // on "Stopping…" forever after a transient failure.
      stopRequesting = false;
      fail(typeScope(type), e.message);
    }
  }
  async function retry(type) {
    clearErr();
    const j = jobOf(type);
    try {
      if (j) await api.indexReset({ job_id: j.id });
      await api.indexStart(buildCfg(type));
      job = await api.indexProgress();
      if (job.any_active) startPolling();
    } catch (e) { fail(typeScope(type), e.message); }
  }
  async function clearJob(type) {
    const scope = typeScope(type);
    clearErr();
    try {
      const j = jobOf(type);
      if (j) await api.indexReset({ job_id: j.id });
      job = await api.indexProgress();
    } catch (e) { fail(scope, e.message); }
  }

  async function recheckHealth() {
    rechecking = true;
    // Re-fetch the model catalogue too — its loaded/not-loaded tags were
    // previously only fetched on mount, so a model unloaded in LM Studio
    // kept showing "● loaded" here forever.
    await Promise.all([
      refreshHealth(),
      api.providerModels().then(r => { pmodels = r; }).catch(() => {}),
    ]);
    rechecking = false;
  }

  // ── scan (runs as a background job — folder walks can take minutes) ─────────
  async function scan() { await start("scan"); }

  function gotoSection(id) {
    // Instant, not smooth: content-visibility:auto sections re-layout while a
    // smooth scroll animates, and Chrome silently aborts the animation.
    document.getElementById(id)?.scrollIntoView({ block: "start" });
  }

  // ── duplicates ────────────────────────────────────────────────────────────
  let dupes = null;
  let dupesBusy = false;
  let dupeDeleteFiles = false;
  async function loadDupes() {
    dupesBusy = true; clearErr();
    try { dupes = await api.duplicates(); } catch (e) { fail("dupes", e.message); }
    dupesBusy = false;
  }
  async function removeDupeGroup(g) {
    const ids = g.photos.slice(1).map((p) => p.id);  // keep the largest file
    busy = { ...busy, dupes: true }; clearErr();
    try {
      await api.batchDelete(ids, dupeDeleteFiles);
      dupes.groups = dupes.groups.filter((x) => x !== g);
      dupes.total_groups -= 1;
      dupes = dupes;
      refreshStatus();
    } catch (e) { fail("dupes", e.message); }
    busy = { ...busy, dupes: false };
  }

  // ── trash ─────────────────────────────────────────────────────────────────
  let trashItems = null;
  let showTrash = false;
  async function loadTrash() {
    try { trashItems = (await api.trashList()).items; } catch {}
  }
  async function toggleTrash() {
    showTrash = !showTrash;
    if (showTrash && trashItems === null) await loadTrash();
  }
  async function restoreTrash(ids = []) {
    clearErr();
    try {
      await api.trashRestore(ids);
      await Promise.all([loadTrash(), refreshStatus()]);
    } catch (e) { fail("trash", e.message); }
  }
  async function emptyTrash() {
    clearErr();
    try {
      await api.trashPurge([]);
      await Promise.all([loadTrash(), refreshStatus()]);
    } catch (e) { fail("trash", e.message); }
  }

  // ── folder actions ────────────────────────────────────────────────────────────
  async function addFolder() {
    const path = newFolderPath.trim(); if (!path) return;
    busy = { ...busy, addFolder: true }; clearErr();
    try {
      const res = await api.addIncludedFolder(path);
      if (res.status === "redundant") fail("folders", `Already covered by "${res.covered_by}".`);
      else if (res.status === "duplicate") fail("folders", "Folder already in list.");
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
    busy = { ...busy, removeFolder: true }; clearErr();
    try {
      await api.removeIncludedFolder(confirmRemove.path, true);
      confirmRemove = null;
      await Promise.all([loadFolderConfig(), loadOrphaned(), refreshStatus()]);
    } catch (e) { fail("folders", e.message); }
    busy = { ...busy, removeFolder: false };
  }
  async function addExclude() {
    const path = newExcludePath.trim(); if (!path) return;
    busy = { ...busy, addExclude: true }; clearErr();
    try {
      const res = await api.addExcludedFolder(path);
      if (res.status === "duplicate") fail("folders", "Already excluded.");
      else { newExcludePath = ""; await loadFolderConfig(); }
    } catch (e) { fail("folders", e.message); }
    busy = { ...busy, addExclude: false };
  }
  async function removeExclude(path) {
    busy = { ...busy, removeExclude: path };
    try { await api.removeExcludedFolder(path); await loadFolderConfig(); }
    catch (e) { fail("folders", e.message); }
    busy = { ...busy, removeExclude: null };
  }

  // ── orphaned ──────────────────────────────────────────────────────────────────
  async function removeOrphanedAll() {
    orphanedBusy = true; clearErr();
    try { await api.cleanupOrphaned([]); await Promise.all([loadOrphaned(), refreshStatus()]); }
    catch (e) { fail("orphaned", e.message); }
    orphanedBusy = false;
  }
  async function removeOrphanedOne(id) {
    busy = { ...busy, [id]: true };
    try { await api.cleanupOrphaned([id]); await Promise.all([loadOrphaned(), refreshStatus()]); }
    catch (e) { fail("orphaned", e.message); }
    busy = { ...busy, [id]: false };
  }

  // ── model management ──────────────────────────────────────────────────────────
  async function switchActiveModel(name) {
    await api.setActiveModel(name);
    await Promise.all([refreshModels(), refreshStatus()]);
    dispatch("indexed");
  }

  function fmtDate(iso) { return iso ? iso.slice(0, 16).replace("T", " ") : "never scanned"; }

  function modelTypeLabel(id, provider) {
    if (provider === "lm_studio") {
      const info = lmTypes[id] || {};
      const typeTag = info.type === "embed" ? "[EMBED ONLY]"
                    : info.type === "text-only" ? "[TEXT ONLY]"
                    : info.type === "unknown" ? "[?]" : "";
      // state is only meaningful when we got real data from LM Studio's v0
      // API (undefined/null means we fell back to the name-guess heuristic,
      // which can't tell loaded from not-loaded).
      const stateTag = info.state === "loaded" ? "● loaded"
                     : info.state === "not-loaded" ? "○ not loaded" : "";
      return [typeTag, stateTag].filter(Boolean).join("  ");
    }
    if (provider === "gemini") {
      const cooldown = pmodels.gemini_cooldowns?.[id];
      return cooldown ? `⏳ rate-limited (${Math.ceil(cooldown)}s)` : "";
    }
    if (provider === "9router") {
      const cooldown = pmodels.ninerouter_cooldowns?.[id];
      const tags = [ninerouterTag(id)];
      if (cooldown) tags.push(`⏳ pool exhausted (${Math.ceil(cooldown)}s)`);
      return tags.filter(Boolean).map(t => `[${t}]`).join("  ");
    }
    return "";
  }
</script>

{#if err && !errAt}
  <div class="note-card">{err} <button class="ghost sm" on:click={clearErr}>×</button></div>
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
      <StatusPill label="LM Studio" state={lmPillState} detail={lmPillDetail} />
      <StatusPill label="Gemini" state={$health.gemini ? "on" : ($health.gemini_key_set ? "warn" : "off")}
                  detail={$health.gemini ? "fallback ready" : ($health.gemini_key_set ? "unreachable" : "no key")} />
      <StatusPill label="9Router" state={$health.ninerouter ? "on" : "off"}
                  detail={$health.ninerouter ? "gateway online" : "offline"} />
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
    <button class="stage" title="Go to Folder Management (A)" on:click={() => gotoSection("sec-scan")}>
      <div class="num">{totalScanned}</div>
      <div class="lbl">Scanned</div>
    </button>
    <div class="arrow">→</div>
    <button class="stage" class:pending={visionPending > 0}
            title="Go to Vision analysis (B)" on:click={() => gotoSection("sec-vision")}>
      <div class="num">{mVision.done ?? st.stage?.vision_done ?? 0}</div>
      <div class="lbl">Captioned</div>
      {#if visionPending > 0}<div class="pend">{visionPending} pending</div>{/if}
      {#if mVision.selected_label}
        <div class="model-badge">{mVision.selected_label}</div>
      {/if}
    </button>
    <div class="arrow">→</div>
    <button class="stage" class:pending={embedPending > 0}
            title="Go to Embed (C)" on:click={() => gotoSection("sec-embed")}>
      <div class="num">{mEmbed.done ?? st.stage?.active_model_embedded ?? 0}</div>
      <div class="lbl">Embedded</div>
      {#if embedPending > 0}<div class="pend">{embedPending} pending</div>{/if}
      {#if mEmbed.selected_model}
        <div class="model-badge">{mEmbed.selected_model}</div>
      {/if}
    </button>
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
  <ErrLine {err} at={errAt} scope="orphaned" onclear={clearErr} />
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
        {#if settings.vision_provider === "9router"}
          <!-- Free-entry with suggestions: 9Router's /v1/models is a cosmetic
               catalog, not a routing restriction — it passes unlisted ids
               through to the provider (verified: gemini/gemma-4-26b-a4b-it
               works while absent from the list). A plain select would wall
               off every model the gateway can actually serve. -->
          <input class="model-input" list="nr-vision-models"
                 placeholder="pick or type a model id, e.g. gemini/gemma-4-26b-a4b-it"
                 bind:value={settings.vision_model} on:input={markDirty}
                 on:change={() => { settings.vision_model = (settings.vision_model || "").trim() || null; }} />
          <datalist id="nr-vision-models">
            {#each visionModelOpts as m}
              <option value={m}>{modelTypeLabel(m, settings.vision_provider)}</option>
            {/each}
          </datalist>
          {#if visionCfgIncomplete}
            <p class="warn-text hint">9Router has no auto-detect — pick a suggestion or type the exact
              provider-prefixed id (e.g. <code>gc/…</code> for Gemini CLI, <code>gemini/…</code> for API keys).</p>
          {:else}
            <p class="hint">9Router runs caption photos that have no caption yet. Ids not in the
              suggestion list are passed through to the provider as-is. If the gateway substitutes
              the serving model, the caption is kept and credited to the model that actually
              produced it (see "Models used" in the job panel).</p>
          {/if}
          {#if !visionModelOpts.length}
            <p class="warn-text hint">9Router's model list is empty — is it running? A typed model id
              will still be tried at job start.</p>
          {/if}
        {:else if visionModelOpts.length}
          <select bind:value={settings.vision_model} on:change={markDirty}>
            <option value={null}>— auto-pick —</option>
            {#each visionModelOpts as m}
              <option value={m}>{m} {modelTypeLabel(m, settings.vision_provider)}</option>
            {/each}
          </select>
        {:else}
          <p class="warn-text hint">No {PROVIDER_NAMES[settings.vision_provider]} vision models found.</p>
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
            {#if settings.embed_provider !== "9router"}
              <option value={null}>— auto-pick —</option>
            {:else}
              <option value={null} disabled>— choose a model (required) —</option>
            {/if}
            {#each embedModelOpts as m}
              <option value={m}>{m} {modelTypeLabel(m, settings.embed_provider)}</option>
            {/each}
          </select>
          {#if embedCfgIncomplete}
            <p class="warn-text hint">9Router has no auto-detect — pick the exact embedding model to use.
              Each model gets its own search index; changing model means re-embedding.</p>
          {/if}
        {:else}
          <p class="warn-text hint">No {PROVIDER_NAMES[settings.embed_provider]} embed models found.{settings.embed_provider === "9router" ? " Is 9Router running?" : ""}</p>
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
    <label class="row" style="gap:8px; font-size:13px; flex-wrap:wrap">
      Max caption tokens
      <input type="number" bind:value={settings.vision_max_tokens} min="256" max="16384" step="256"
             on:change={markDirty} style="width:80px" />
      <span class="hint">Output ceiling per caption. Raise it if captions fail with a
        “truncated at the token ceiling” error — “thinking” models (gemini-2.5/3.x-flash)
        spend hidden reasoning tokens against this budget. Truncated captions also grow the
        budget automatically, so this is rarely needed.</span>
    </label>
    <label class="row" style="gap:8px; font-size:13px; flex-wrap:wrap">
      Keyframes per video
      <input type="number" bind:value={settings.video_frames} min="1" max="12" step="1"
             on:change={markDirty} style="width:60px" />
      <span class="hint">How many frames each video is sampled at for captioning &amp; face
        detection — a video costs about this many vision calls (not one per actual frame).
        Higher = richer coverage of long clips but more calls/quota; lower = faster &amp;
        cheaper. Default 4.</span>
    </label>
    <label class="row" style="gap:8px; font-size:13px">
      <input type="checkbox" bind:checked={settings.faces_during_embed}
             on:change={markDirty} style="width:auto" />
      Detect faces during embedding
      <span class="hint">(off → run face detection separately below)</span>
    </label>
    <label class="row" style="gap:8px; font-size:13px; flex-wrap:wrap">
      Face detection runs on
      <select bind:value={settings.face_provider} on:change={markDirty} style="max-width:280px">
        {#each faceProviders.options as opt (opt.id)}
          <option value={opt.id}>{opt.label}</option>
        {/each}
      </select>
      <span class="hint">Auto-detected from what's installed — no hardcoding. “Auto” picks the
        fastest available (GPU/NPU over CPU); CPU is always the fallback if a chosen accelerator
        is unavailable.{#if faceProviders.active} Currently resolves to <b>{faceProviders.active}</b>.{/if}
        Applies to the next face-detection run.
        {#if !faceAccelAvailable}<br>Only CPU detected. To use an Intel GPU/NPU or a discrete GPU,
        install an accelerator wheel (<code>make accel-openvino</code> / <code>accel-nvidia</code> /
        <code>accel-directml</code>) and restart — see the README.{/if}</span>
    </label>
  </div>

  <!-- Provider rate limits -->
  <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:12px">
    <div class="cfg-label">Provider rate limits</div>
    <p class="hint" style="margin-bottom:8px">
      Ceilings on captioning/embedding requests per provider — a job pauses when a window
      fills instead of burning quota on 429 errors (the panel shows when it's waiting).
      0 = unlimited. Changes apply immediately, even mid-job. Counters are in-memory and
      reset when the server restarts.
    </p>
    <div class="rl-grid" role="group" aria-label="Provider rate limits">
      <span></span>
      {#each RL_WINDOWS as [, wlabel]}<span class="rl-head">{wlabel}</span>{/each}
      <span></span>
      {#each RL_PROVIDERS as [pkey, plabel] (pkey)}
        <span class="rl-prov">{plabel}</span>
        {#each RL_WINDOWS as [wkey, wlabel] (wkey)}
          <input type="number" min="0" placeholder="0"
                 aria-label="{plabel} {wlabel}"
                 bind:value={settings.rate_limits[pkey][wkey]}
                 on:change={markDirty} />
        {/each}
        <button class="ghost sm" on:click={() => suggestLimits(pkey)}
                title="Fill from limits learned from real 429s, else published free-tier values">
          ✨ Suggest
        </button>
      {/each}
    </div>
    {#if rlSuggestNote}
      <p class="hint" style="margin-top:6px">{rlSuggestNote}</p>
    {/if}
  </div>
  <ErrLine {err} at={errAt} scope="settings" onclear={clearErr} />
</div>

<!-- A: Folder Management -->
<div class="card">
  <SectionHead icon="📂" color="#3b82f6" title="A · Folder Management" id="sec-scan" />
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
              {folder.image_count > 0 ? `${folder.image_count} files seen at last scan` : "not yet scanned"}
              {#if folder.last_scanned_at} · {fmtDate(folder.last_scanned_at)}{/if}
            </span>
          </div>
          <button class="sm ghost danger-hover" on:click={() => requestRemoveFolder(folder)} disabled={scanRunning}>Remove</button>
        </div>
      {/each}
    </div>
  {/if}

  <div class="add-row">
    <input bind:value={newFolderPath} placeholder="Folder path…"
           on:keydown={(e) => e.key === "Enter" && addFolder()}
           disabled={scanRunning} />
    <button on:click={addFolder} disabled={!newFolderPath.trim() || busy.addFolder || scanRunning}>
      {busy.addFolder ? "Adding…" : "Add folder"}
    </button>
    <button class="ghost" on:click={suggestDefaults} disabled={scanRunning}>Suggest defaults</button>
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
            <button class="sm ghost" on:click={() => removeExclude(ex.path)} disabled={scanRunning}>Remove exclusion</button>
          </div>
        {/each}
      </div>
    {/if}
    <div class="add-row">
      <input bind:value={newExcludePath} placeholder="Subfolder to skip…"
             on:keydown={(e) => e.key === "Enter" && addExclude()}
             disabled={scanRunning} />
      <button on:click={addExclude} disabled={!newExcludePath.trim() || busy.addExclude || scanRunning}>
        {busy.addExclude ? "Adding…" : "Exclude folder"}
      </button>
    </div>
  {/if}

  <div style="margin-top:16px">
    {#if jobOf("scan")}
      <JobPanel job={jobOf("scan")} bind:stopRequested={stopRequesting} on:stop={() => stopJob("scan")} on:retry={() => retry("scan")} on:clear={() => clearJob("scan")} />
    {:else if conflictFor("scan")}
      <div class="blocked-row">⏸ Blocked — <b>{conflictFor("scan")}</b> is running, stop it first</div>
    {:else}
      <button class="primary" on:click={scan}
              disabled={folderConfig.included.length === 0}>
        Scan {folderConfig.included.length} folder{folderConfig.included.length === 1 ? "" : "s"}
      </button>
    {/if}
  </div>
  <ErrLine {err} at={errAt} scope="folders" onclear={clearErr} />
</div>

<!-- Import & consolidate -->
<div class="card">
  <SectionHead icon="📥" color="#10b981" title="Import & consolidate">
    <span class="hint" style="font-weight:400">(merge SD card / Takeout / phone dumps — duplicates skipped by content)</span>
  </SectionHead>
  <p class="hint" style="margin-bottom:10px">
    Import from anywhere — SD card, pen drive, another internal or network drive, a Takeout
    extract. Every file is identified by its content hash, so anything the library has ever
    seen is <b>skipped</b> — no matter how many times it was copied or renamed. Only new
    photos & videos are copied in, organized by year/month. Originals are never touched.
    Afterwards, run <b>Scan</b> above — only the new photos go through captioning.
  </p>
  {#if jobOf("ingest")}
    <JobPanel job={jobOf("ingest")} bind:stopRequested={stopRequesting} on:stop={() => stopJob("ingest")} on:retry={() => retry("ingest")} on:clear={() => clearJob("ingest")} />
  {:else if conflictFor("ingest")}
    <div class="blocked-row">⏸ Blocked — <b>{conflictFor("ingest")}</b> is running, stop it first</div>
  {:else}
    <div class="pickrow">
      <span class="cfg-label" style="margin:0">From</span>
      <code class="pathbox" title={stagingPath}>{stagingPath || "no folder selected"}</code>
      <button class="sm" on:click={() => picker = "source"}>📂 Browse…</button>
    </div>
    <!-- Media filter: import everything, or just photos / just videos. A mixed
         folder needs no splitting — pick "Both" and it Just Works. -->
    <div class="pickrow">
      <span class="cfg-label" style="margin:0">Include</span>
      <div class="seg" role="radiogroup" aria-label="Which media to import">
        {#each [["both","Photos + videos"],["photos","Photos only"],["videos","Videos only"]] as [val, lbl]}
          <label class="seg-opt" class:on={ingestMedia === val}>
            <input type="radio" bind:group={ingestMedia} value={val} /> {lbl}
          </label>
        {/each}
      </div>
    </div>
    {#if ingestMedia !== "videos"}
      <div class="pickrow">
        <span class="cfg-label" style="margin:0">Photos into</span>
        <code class="pathbox" title={settings.ingest_dest}>{settings.ingest_dest || "…\\Imported"}<span class="hint">\YYYY\MM</span></code>
        <button class="ghost sm" on:click={() => picker = "ingest_dest"}>Change…</button>
      </div>
    {/if}
    {#if ingestMedia !== "photos"}
      <div class="pickrow">
        <span class="cfg-label" style="margin:0">Videos into</span>
        <code class="pathbox" title={videoDest}>{videoDest || "auto — your Videos root"}<span class="hint">\YYYY\MM</span></code>
        <button class="ghost sm" on:click={() => picker = "video_dest"}>Change…</button>
      </div>
    {/if}
    {#if srcChecking}
      <p class="hint">Checking folder…</p>
    {:else if srcCheck && !srcCheck.ok}
      <p class="warn-text" style="font-size:13px">✋ {srcCheck.reason}</p>
    {:else if srcCheck}
      <p class="ok-text" style="font-size:13px">
        ✓ Ready: {(srcCheck.photo_files ?? 0).toLocaleString()} photos, {(srcCheck.video_files ?? 0).toLocaleString()} videos
        ({fmtGB(srcCheck.media_bytes)} GB){ingestMedia !== "both" ? ` — importing ${ingestMedia} only` : ""};
        duplicates are skipped automatically{srcCheck.other_files ? ` · ${srcCheck.other_files.toLocaleString()} non-media files ignored` : ""}.
      </p>
    {/if}
    <div style="margin-top:10px">
      <button class="primary" disabled={!srcCheck?.ok}
              on:click={() => start("ingest", { source_path: stagingPath.trim(),
                                                ingest_media: ingestMedia,
                                                ingest_video_dest: videoDest.trim() || null })}>
        📥 Import new files
      </button>
    </div>
  {/if}
  <ErrLine {err} at={errAt} scope="import" onclear={clearErr} />
</div>

<!-- B: Vision analysis -->
<div class="card">
  <SectionHead icon="👁" color="#6366f1" title="B · Vision analysis" id="sec-vision">
    <span class="hint" style="font-weight:400">(image → caption + 12 attributes)</span>
  </SectionHead>
  {#if selectedVisionLabel}
    <p class="hint">Running with: <code>{selectedVisionLabel}</code>
      · {visionPending} image{visionPending === 1 ? "" : "s"} pending for this model</p>
  {:else}
    <p class="hint">{visionPending} image{visionPending === 1 ? "" : "s"} pending (any model)</p>
  {/if}
  {#if jobOf("vision")}
    <JobPanel job={jobOf("vision")} on:stop={() => stopJob("vision")} on:retry={() => retry("vision")} on:clear={() => clearJob("vision")} />
  {:else if visionPending === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All photos have captions{selectedVisionLabel ? ` from ${selectedVisionLabel}` : ""}.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if conflictFor("vision")}
    <div class="blocked-row">⏸ Blocked — <b>{conflictFor("vision")}</b> is running, stop it first</div>
  {:else}
    <button class="primary" on:click={() => start("vision")} disabled={noServices || visionCfgIncomplete}
            title={visionCfgIncomplete ? "Pick a 9Router vision model in Run configuration first" : ""}>
      ▶ Caption {visionPending} photo{visionPending === 1 ? "" : "s"}
    </button>
    {#if visionCfgIncomplete}
      <p class="warn-text hint">Blocked: 9Router is selected for vision but no model is chosen.</p>
    {/if}
  {/if}
  <ErrLine {err} at={errAt} scope="vision" onclear={clearErr} />
</div>

<!-- C: Embed -->
<div class="card">
  <SectionHead icon="🧬" color="#8b5cf6" title="C · Embed" id="sec-embed">
    <span class="hint" style="font-weight:400">(caption → searchable vector)</span>
  </SectionHead>
  {#if settings.caption_source_model}
    <p class="hint">Source: captions from <code>{settings.caption_source_model}</code>
      → embed model: <code>{settings.embed_model || "auto"}</code>
      · {embedPending} pending</p>
  {:else}
    <p class="hint">Source: latest caption · {embedPending} pending</p>
  {/if}
  {#if jobOf("embed")}
    {@const ej = jobOf("embed")}
    <JobPanel job={ej} on:stop={() => stopJob("embed")} on:retry={() => retry("embed")} on:clear={() => clearJob("embed")} />
    {#if !ej.active && ej.substitution}
      <div class="confirm-card" style="margin-top:10px">
        <span class="warn-text">⚠ 9Router is serving <code>{ej.substitution.served}</code>
        instead of <code>{ej.substitution.requested}</code> — the run was stopped so vectors
        from two models never mix in one search index.</span>
        <div class="row" style="gap:10px; margin-top:10px">
          <button class="primary" on:click={switchToServedAndRerun} disabled={switchingServed}>
            {switchingServed ? "Switching…" : `Switch to ${ej.substitution.suggested} & re-run`}
          </button>
          <span class="hint">Uses its own fresh collection; pending is recomputed, nothing is lost.</span>
        </div>
      </div>
    {/if}
  {:else if embedPending === 0 && mEmbed.eligible > 0}
    <p class="ok-text">✓ All eligible captions are embedded.</p>
  {:else if embedPending === 0}
    <p class="hint">Run vision analysis first (B).</p>
  {:else if conflictFor("embed")}
    <div class="blocked-row">⏸ Blocked — <b>{conflictFor("embed")}</b> is running, stop it first</div>
  {:else}
    <button class="primary" on:click={() => start("embed")} disabled={noServices || embedCfgIncomplete}
            title={embedCfgIncomplete ? "Pick a 9Router embedding model in Run configuration first" : ""}>
      ▶ Embed {embedPending} photo{embedPending === 1 ? "" : "s"}
    </button>
    {#if embedCfgIncomplete}
      <p class="warn-text hint">Blocked: 9Router is selected for embedding but no model is chosen.</p>
    {/if}
  {/if}
  <ErrLine {err} at={errAt} scope="embed" onclear={clearErr} />
</div>

<!-- C2: Face detection (separate, user-controlled stage) -->
<div class="card">
  <SectionHead icon="🙂" color="#ec4899" title="Face detection">
    <span class="hint" style="font-weight:400">(detect + embed faces for person search)</span>
  </SectionHead>
  <p class="hint">{facesDone} done · {facesPending} pending.
    {settings.faces_during_embed ? "Also runs automatically during embedding." : "Runs only when you start it here."}</p>
  {#if jobOf("faces")}
    <JobPanel job={jobOf("faces")} on:stop={() => stopJob("faces")} on:retry={() => retry("faces")} on:clear={() => clearJob("faces")} />
  {:else if facesPending === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All photos have been scanned for faces.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if conflictFor("faces")}
    <div class="blocked-row">⏸ Blocked — <b>{conflictFor("faces")}</b> is running, stop it first</div>
  {:else}
    <button class="primary" on:click={() => start("faces")}>
      ▶ Detect faces in {facesPending} photo{facesPending === 1 ? "" : "s"}
    </button>
  {/if}
  <ErrLine {err} at={errAt} scope="faces" onclear={clearErr} />
</div>

<!-- C3: Video analysis (keyframe captioning + faces) -->
{#if videoTotal > 0}
<div class="card">
  <SectionHead icon="🎬" color="#6366f1" title="Video analysis">
    <span class="hint" style="font-weight:400">(caption + find faces in videos, from sampled keyframes)</span>
  </SectionHead>
  <p class="hint">{videoTotal.toLocaleString()} video{videoTotal === 1 ? "" : "s"} in the library ·
    {videoVisionPending.toLocaleString()} to caption · {videoFacesPending.toLocaleString()} to scan for faces.
    Videos are captioned from a few sampled frames using your Vision model above, then searchable like photos.</p>
  {#if jobOf("video_vision")}
    <JobPanel job={jobOf("video_vision")} on:stop={() => stopJob("video_vision")} on:retry={() => retry("video_vision")} on:clear={() => clearJob("video_vision")} />
  {/if}
  {#if jobOf("video_faces")}
    <JobPanel job={jobOf("video_faces")} on:stop={() => stopJob("video_faces")} on:retry={() => retry("video_faces")} on:clear={() => clearJob("video_faces")} />
  {/if}
  {#if !jobOf("video_vision") && !jobOf("video_faces")}
    <div class="row" style="gap:8px; flex-wrap:wrap">
      <button class="primary" disabled={videoVisionPending === 0 || !!conflictFor("video_vision")}
              on:click={() => start("video_vision")}
              title={conflictFor("video_vision") ? `${conflictFor("video_vision")} is running` : ""}>
        ▶ Caption {videoVisionPending.toLocaleString()} video{videoVisionPending === 1 ? "" : "s"}
      </button>
      <button disabled={videoFacesPending === 0 || !!conflictFor("video_faces")}
              on:click={() => start("video_faces")}
              title={conflictFor("video_faces") ? `${conflictFor("video_faces")} is running` : ""}>
        🙂 Find faces in {videoFacesPending.toLocaleString()} video{videoFacesPending === 1 ? "" : "s"}
      </button>
    </div>
    {#if videoVisionPending === 0 && videoFacesPending === 0}
      <p class="ok-text" style="margin-top:8px">✓ All videos captioned and scanned for faces.</p>
    {/if}
  {/if}
  <ErrLine {err} at={errAt} scope="video" onclear={clearErr} />
</div>
{/if}

<!-- Thumbnails: pregenerate grid previews -->
<div class="card">
  <SectionHead icon="🖼" color="#f59e0b" title="Thumbnails">
    <span class="hint" style="font-weight:400">(pregenerate grid previews so browsing never waits)</span>
  </SectionHead>
  <p class="hint">{thumbsPending} photo{thumbsPending === 1 ? "" : "s"} without a thumbnail.
    Missing ones are still generated on demand — this just does it ahead of time.</p>
  {#if jobOf("thumbs")}
    <JobPanel job={jobOf("thumbs")} on:stop={() => stopJob("thumbs")} on:retry={() => retry("thumbs")} on:clear={() => clearJob("thumbs")} />
  {:else if thumbsPending === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ Every photo has a thumbnail.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if conflictFor("thumbs")}
    <div class="blocked-row">⏸ Blocked — <b>{conflictFor("thumbs")}</b> is running, stop it first</div>
  {:else}
    <button class="primary" on:click={() => start("thumbs")}>
      ▶ Generate {thumbsPending} thumbnail{thumbsPending === 1 ? "" : "s"}
    </button>
  {/if}
  <ErrLine {err} at={errAt} scope="thumbs" onclear={clearErr} />
</div>

<!-- Duplicates -->
<div class="card">
  <SectionHead icon="🔍" color="#06b6d4" title="Duplicates">
    <span class="hint" style="font-weight:400">(find near-identical photos — resaves, bursts, WhatsApp copies)</span>
  </SectionHead>

  <!-- Exact byte-identical extra copies recorded by scans. Distinct from the
       dHash near-dupes below: these are the SAME file sitting in several
       places, so removing them is loss-free (one canonical copy always kept,
       removals go to the Recycle Bin). -->
  {#if jobOf("dedupe")}
    <JobPanel job={jobOf("dedupe")} bind:stopRequested={stopRequesting} on:stop={() => stopJob("dedupe")} on:retry={() => retry("dedupe")} on:clear={() => clearJob("dedupe")} />
  {:else if dedupeCount > 0}
    <div class="row" style="gap:10px; margin-bottom:12px; flex-wrap:wrap; align-items:center">
      <span class="hint"><b>{dedupeCount}</b> exact duplicate cop{dedupeCount === 1 ? "y" : "ies"} on disk
        (same bytes, multiple locations — found during scans)</span>
      {#if conflictFor("dedupe")}
        <div class="blocked-row">⏸ Blocked — <b>{conflictFor("dedupe")}</b> is running</div>
      {:else}
        <button class="sm" on:click={() => start("dedupe")}
                title="Keeps one canonical file per photo; extra copies go to the Recycle Bin">
          🧹 Reclaim space — Recycle {dedupeCount} cop{dedupeCount === 1 ? "y" : "ies"}
        </button>
      {/if}
    </div>
  {:else if totalScanned > 0}
    <p class="hint" style="margin-bottom:10px">No exact duplicate copies recorded — scans note them automatically.</p>
  {/if}
  {#if jobOf("dhash")}
    <JobPanel job={jobOf("dhash")} on:stop={() => stopJob("dhash")} on:retry={() => retry("dhash")} on:clear={() => clearJob("dhash")} />
  {:else if dhashPending > 0}
    <p class="hint">{dhashPending} photo{dhashPending === 1 ? "" : "s"} not fingerprinted yet.</p>
    {#if conflictFor("dhash")}
      <div class="blocked-row">⏸ Blocked — <b>{conflictFor("dhash")}</b> is running, stop it first</div>
    {:else}
      <button class="primary" on:click={() => start("dhash")}>
        ▶ Fingerprint {dhashPending} photo{dhashPending === 1 ? "" : "s"}
      </button>
    {/if}
  {:else if totalScanned > 0}
    <p class="ok-text">✓ All photos fingerprinted.</p>
  {/if}

  {#if totalScanned > 0 && dhashPending < totalScanned}
    <div class="row" style="gap:10px; margin-top:10px; flex-wrap:wrap">
      <button class="ghost sm" on:click={loadDupes} disabled={dupesBusy}>
        {dupesBusy ? "Scanning…" : dupes ? "Refresh duplicate groups" : "Show duplicate groups"}
      </button>
      {#if dupes}
        <span class="hint">{dupes.total_groups} group{dupes.total_groups === 1 ? "" : "s"}
          found across {dupes.hashed} fingerprinted photos</span>
      {/if}
    </div>
  {/if}

  {#if dupes && dupes.groups.length}
    <label class="row" style="gap:6px; font-size:13px; margin-top:10px">
      <input type="checkbox" bind:checked={dupeDeleteFiles} style="width:auto" />
      also move duplicate files to the Recycle Bin
    </label>
    {#if dupes.total_groups > 40}
      <p class="hint" style="margin-top:6px">Showing 40 of {dupes.total_groups} duplicate groups.</p>
    {/if}
    <div class="dupe-list">
      {#each dupes.groups.slice(0, 40) as g (g.photos[0].id)}
        <div class="dupe-group">
          <div class="dupe-thumbs">
            {#each g.photos.slice(0, 8) as p, pi}
              <div class="dupe-cell" class:keeper={pi === 0} title={p.path} role="button" tabindex="0"
                   on:click={() => dispatch("select", { id: p.id, ids: g.photos.map((x) => x.id) })}
                   on:keydown={(e) => onActivateKey(e, () => dispatch("select", { id: p.id, ids: g.photos.map((x) => x.id) }))}>
                <img src={api.thumbUrl(p.id)} alt={p.filename} decoding="async" />
                {#if pi === 0}<span class="keep-badge">keep</span>{/if}
              </div>
            {/each}
            {#if g.count > 8}<span class="hint">+{g.count - 8} more</span>{/if}
          </div>
          <div class="row" style="justify-content:space-between; margin-top:6px">
            <span class="hint">{g.count} copies · keeping the largest file</span>
            <button class="sm danger" on:click={() => removeDupeGroup(g)} disabled={busy.dupes || running}>
              Remove {g.count - 1} duplicate{g.count - 1 === 1 ? "" : "s"}
            </button>
          </div>
        </div>
      {/each}
    </div>
  {:else if dupes}
    <p class="ok-text" style="margin-top:8px">✓ No duplicate groups found.</p>
  {/if}
  <ErrLine {err} at={errAt} scope="dupes" onclear={clearErr} />
</div>

<!-- Trash -->
{#if trashCount > 0}
<div class="card warn-card">
  <div class="row" style="justify-content:space-between; align-items:center">
    <span class="warn-text">🗑 Trash — {trashCount} photo{trashCount === 1 ? "" : "s"}
      <span class="hint">(removed from the library; restorable)</span></span>
    <div class="row" style="gap:8px">
      <button class="ghost sm" on:click={toggleTrash}>{showTrash ? "Hide" : "Show"}</button>
      <button class="sm" on:click={() => restoreTrash([])} disabled={running}>Restore all</button>
      <button class="sm danger" on:click={emptyTrash} disabled={running}>Empty trash</button>
    </div>
  </div>
  {#if showTrash && trashItems}
    <div class="orphan-list">
      {#each trashItems as t (t.id)}
        <div class="orphan-row">
          <span class="orphan-path" title={t.path}>
            {t.filename}{t.file_deleted ? " · file sent to Recycle Bin" : ""}
          </span>
          <button class="sm ghost" on:click={() => restoreTrash([t.id])} disabled={running}>Restore</button>
        </div>
      {/each}
    </div>
    <p class="hint" style="margin-top:6px">Restored photos keep their caption and
      reappear as embed-pending (run C to make them searchable again).</p>
  {/if}
  <ErrLine {err} at={errAt} scope="trash" onclear={clearErr} />
</div>
{/if}

<!-- D: Full index -->
<div class="card">
  <SectionHead icon="⚡" color="#22c55e" title="D · Full index">
    <span class="hint" style="font-weight:400">(caption + embed in one pass)</span>
  </SectionHead>
  {#if jobOf("full")}
    <JobPanel job={jobOf("full")} on:stop={() => stopJob("full")} on:retry={() => retry("full")} on:clear={() => clearJob("full")} />
  {:else if missingFull === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All scanned photos are in the index.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if conflictFor("full")}
    <div class="blocked-row">⏸ Blocked — <b>{conflictFor("full")}</b> is running, stop it first</div>
  {:else}
    <p class="hint">First-time setup: runs B then C for everything not yet indexed.</p>
    <button class="primary" on:click={() => start("full")} disabled={noServices || visionCfgIncomplete || embedCfgIncomplete}
            title={visionCfgIncomplete || embedCfgIncomplete ? "Pick 9Router model(s) in Run configuration first" : ""}>
      ▶ Index {missingFull} photo{missingFull === 1 ? "" : "s"} from scratch
    </button>
    {#if visionCfgIncomplete || embedCfgIncomplete}
      <p class="warn-text hint">Blocked: 9Router is selected but no model is chosen (Run configuration).</p>
    {/if}
  {/if}
  <ErrLine {err} at={errAt} scope="full" onclear={clearErr} />
</div>

<!-- E: Re-analyze -->
<div class="card">
  <SectionHead icon="🔄" color="#fb923c" title="E · Re-analyze">
    <span class="hint" style="font-weight:400">(refresh captions + re-embed)</span>
  </SectionHead>
  {#if jobOf("reanalyze")}
    <JobPanel job={jobOf("reanalyze")} on:stop={() => stopJob("reanalyze")} on:retry={() => retry("reanalyze")} on:clear={() => clearJob("reanalyze")} />
  {:else if missingAttrs === 0}
    {#if totalScanned > 0}
      <p class="ok-text">✓ All indexed photos have full attributes.</p>
    {:else}
      <p class="hint">No photos scanned yet — start with section A.</p>
    {/if}
  {:else if conflictFor("reanalyze")}
    <div class="blocked-row">⏸ Blocked — <b>{conflictFor("reanalyze")}</b> is running, stop it first</div>
  {:else}
    <button on:click={() => start("reanalyze")} disabled={noServices || visionCfgIncomplete || embedCfgIncomplete}
            title={visionCfgIncomplete || embedCfgIncomplete ? "Pick 9Router model(s) in Run configuration first" : ""}>
      🔄 Re-analyze {missingAttrs} stale photo{missingAttrs === 1 ? "" : "s"}
    </button>
  {/if}
  <ErrLine {err} at={errAt} scope="reanalyze" onclear={clearErr} />
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

<!-- G: Backup -->
<div class="card">
  <SectionHead icon="💾" color="#0ea5e9" title="G · Backup">
    <span class="hint" style="font-weight:400">(mirror library + captions/index to the SD card)</span>
  </SectionHead>
  <p class="hint" style="margin-bottom:10px">
    One-way mirror of every scanned folder <b>plus photo-vault's own data</b> (captions,
    faces, search index, settings) — a restore brings everything back, not just pixels.
    The card doesn't need to stay plugged in: sync opportunistically whenever it is.
  </p>
  <div class="pickrow" style="margin-bottom:10px">
    <span class="cfg-label" style="margin:0">Mirror to</span>
    <code class="pathbox" title={settings.backup_dest}>{settings.backup_dest || "no folder selected"}</code>
    <button class="sm" on:click={() => { backupMsg = ""; picker = "backup_dest"; }}>📂 Browse…</button>
  </div>
  {#if backupMsg}
    <p class="warn-text" style="font-size:13px; margin-bottom:10px">✋ {backupMsg}</p>
  {/if}
  {#if jobOf("backup")}
    <JobPanel job={jobOf("backup")} bind:stopRequested={stopRequesting} on:stop={() => stopJob("backup")} on:retry={() => retry("backup")} on:clear={() => clearJob("backup")} />
  {:else if backupSt.configured}
    <div class="row" style="gap:10px; flex-wrap:wrap; align-items:center">
      {#if backupSt.available}
        <span class="chip ok-text">🟢 drive connected</span>
      {:else}
        <span class="chip warn-text">🔌 drive not connected — plug in the SD card to sync</span>
      {/if}
      {#if backupSt.days_since != null}
        <span class="chip" class:warn-text={backupSt.days_since > 14}>
          last backup {backupSt.days_since < 1 ? "today" : `${Math.round(backupSt.days_since)} day${Math.round(backupSt.days_since) === 1 ? "" : "s"} ago`}
        </span>
      {:else}
        <span class="chip warn-text">never backed up</span>
      {/if}
      {#if conflictFor("backup")}
        <div class="blocked-row">⏸ Blocked — <b>{conflictFor("backup")}</b> is running</div>
      {:else if backupSt.available}
        <button class="primary" on:click={() => start("backup")}>💾 Back up now</button>
      {/if}
    </div>
  {:else}
    <p class="hint">Set a destination folder above (on the SD card) to enable backups.</p>
  {/if}
  <ErrLine {err} at={errAt} scope="backup" onclear={clearErr} />
</div>

<FolderPicker open={picker !== null}
              title={picker === "source" ? "Import from which folder?"
                   : picker === "ingest_dest" ? "Where should imported photos be filed? (must be inside a scanned folder)"
                   : picker === "video_dest" ? "Where should imported videos be filed? (must be inside a scanned folder)"
                   : "Back up to which folder? (SD card / pen drive / other drive)"}
              on:select={(e) => onPickFolder(e.detail.path)}
              on:close={() => picker = null} />

<style>
  .pickrow { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
  .seg { display: inline-flex; gap: 4px; flex-wrap: wrap; }
  .seg-opt { display: inline-flex; align-items: center; gap: 5px; cursor: pointer;
    font-size: 13px; padding: 5px 10px; border: 1px solid var(--border);
    border-radius: 8px; background: var(--surface2); user-select: none; }
  .seg-opt.on { border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 16%, transparent); }
  .seg-opt input { width: auto; margin: 0; }
  .pathbox { flex: 1; min-width: 0; padding: 6px 10px; border: 1px solid var(--border);
    border-radius: 8px; background: var(--surface2); font-size: 12px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 18px; margin-bottom: 14px;
    box-shadow: var(--shadow-1); transition: box-shadow .2s, border-color .2s; }
  .card:hover { box-shadow: var(--shadow-2); }
  .blocked-row {
    display: flex; align-items: center; gap: 6px; font-size: 13px;
    color: var(--muted); background: var(--surface2); border-radius: 8px;
    padding: 9px 12px; width: fit-content;
  }
  .blocked-row b { color: var(--text); font-weight: 600; }
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
  .chip { padding: 3px 10px; border-radius: 99px; background: var(--surface2); font-size: 12px; }
  .rl-grid { display: grid; grid-template-columns: max-content repeat(4, 90px) max-content;
    gap: 6px 10px; align-items: center; max-width: 640px; }
  .rl-grid .rl-head { font-size: 12px; color: var(--muted); text-align: center; }
  .rl-grid .rl-prov { font-size: 13px; }
  .rl-grid input { width: 100%; padding: 5px 8px; font-size: 13px; }
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
  .model-input { width: 100%; }
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

  /* duplicates */
  .dupe-list { margin-top: 10px; display: flex; flex-direction: column; gap: 10px; }
  .dupe-group { border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
  .dupe-thumbs { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .dupe-cell { position: relative; width: 72px; height: 72px; border-radius: 6px; overflow: hidden;
    border: 1px solid var(--border); cursor: pointer; }
  .dupe-cell.keeper { border: 2px solid var(--success); }
  .dupe-cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .keep-badge { position: absolute; bottom: 0; left: 0; right: 0; text-align: center;
    font-size: 9px; background: color-mix(in srgb, var(--success) 80%, black); color: #fff; }

  /* orphaned */
  .orphan-list { margin-top: 10px; border: 1px solid color-mix(in srgb, var(--warn) 40%, var(--border));
    border-radius: 8px; overflow: hidden; }
  .orphan-row { display: flex; justify-content: space-between; align-items: center;
    padding: 6px 12px; border-bottom: 1px solid var(--border); gap: 12px; }
  .orphan-row:last-child { border-bottom: none; }
  .orphan-path { font-size: 12px; color: var(--muted); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
</style>
