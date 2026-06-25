// Thin fetch wrapper. Same-origin in production; Vite proxies /api in dev.

// Per-install bearer token. In production it is injected into index.html as
// window.__PV_TOKEN__; in dev (Vite) we fetch it once from /api/token. Sent on
// every request so the hardened local API accepts us.
let _token = (typeof window !== "undefined" && window.__PV_TOKEN__) || null;
let _tokenTried = false;

async function ensureToken() {
  if (_token || _tokenTried) return;
  _tokenTried = true;
  try {
    const r = await fetch("/api/token");
    if (r.ok) _token = (await r.json()).token || null;
  } catch {}
}

async function j(method, url, body) {
  await ensureToken();
  const opts = { method, headers: {} };
  if (_token) opts.headers["Authorization"] = `Bearer ${_token}`;
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.json();
}

export const api = {
  health: () => j("GET", "/api/health"),
  status: () => j("GET", "/api/status"),

  // Scanning — dirs optional; empty array → uses folder registry
  scan: (dirs = []) => j("POST", "/api/scan", { dirs }),

  // Folder management
  getFolderConfig: () => j("GET", "/api/folders"),
  getFolderDefaults: () => j("GET", "/api/folders/defaults"),
  addIncludedFolder: (path) => j("POST", "/api/folders/include", { path }),
  removeIncludedFolder: (path, purge = true) =>
    j("DELETE", `/api/folders/include?path=${encodeURIComponent(path)}&purge=${purge}`),
  countFolderImages: (path) =>
    j("GET", `/api/folders/include/count?path=${encodeURIComponent(path)}`),
  addExcludedFolder: (path) => j("POST", "/api/folders/exclude", { path }),
  removeExcludedFolder: (path) =>
    j("DELETE", `/api/folders/exclude?path=${encodeURIComponent(path)}`),

  // Orphaned images
  getOrphaned: () => j("GET", "/api/orphaned"),
  cleanupOrphaned: (ids = []) => j("DELETE", "/api/orphaned", { ids }),

  // Settings
  getSettings: () => j("GET", "/api/settings"),
  saveSettings: (s) => j("PUT", "/api/settings", s),
  resetSettings: () => j("DELETE", "/api/settings"),

  // Indexing jobs
  indexStart: (cfg) => j("POST", "/api/index/start", cfg),
  indexStop: () => j("POST", "/api/index/stop"),
  indexProgress: () => j("GET", "/api/index/progress"),
  indexReset: () => j("POST", "/api/index/reset"),
  providerModels: () => j("GET", "/api/provider-models"),

  // Search
  filters: () => j("GET", "/api/filters"),
  search: (q, filters, person, top_k = 200) =>
    j("POST", "/api/search", { q, filters, person, top_k }),
  recent: (limit = 60) => j("GET", `/api/recent?limit=${limit}`),
  timeline: () => j("GET", "/api/timeline"),
  mapPhotos: () => j("GET", "/api/map"),

  // People
  people: () => j("GET", "/api/people"),
  addPerson: (name, ref_dir) => j("POST", "/api/people", { name, ref_dir }),

  // Faces (detection is a job via index/start type=faces; below is clustering/tagging)
  facesStatus: () => j("GET", "/api/faces/status"),
  facesCluster: (eps, min_samples) => j("POST", "/api/faces/cluster", { eps, min_samples }),
  facesReindex: () => j("POST", "/api/faces/reindex"),
  facesClusters: (samples = 6) => j("GET", `/api/faces/clusters?samples=${samples}`),
  nameCluster: (cluster_id, name) => j("POST", "/api/faces/name", { cluster_id, name }),
  ignoreCluster: (cluster_id) => j("POST", "/api/faces/ignore", { cluster_id }),
  faceCropUrl: (image_id, face_index) =>
    `/api/faces/crop?image_id=${encodeURIComponent(image_id)}&face_index=${face_index}${_tokenQS()}`,

  // Albums
  albums: () => j("GET", "/api/albums"),
  createAlbum: (name) => j("POST", "/api/albums", { name }),
  getAlbum: (id) => j("GET", `/api/albums/${encodeURIComponent(id)}`),
  renameAlbum: (id, name) => j("PUT", `/api/albums/${encodeURIComponent(id)}`, { name }),
  deleteAlbum: (id) => j("DELETE", `/api/albums/${encodeURIComponent(id)}`),
  albumAdd: (id, ids) => j("POST", `/api/albums/${encodeURIComponent(id)}/add`, { ids }),
  albumRemove: (id, ids) => j("POST", `/api/albums/${encodeURIComponent(id)}/remove`, { ids }),

  // Embedding models
  models: () => j("GET", "/api/models"),
  setActiveModel: (model) => j("POST", "/api/models/active", { model }),

  // Images
  meta: (id) => j("GET", `/api/meta?id=${encodeURIComponent(id)}`),
  explore: (id) => j("GET", `/api/explore?id=${encodeURIComponent(id)}`),
  deleteImage: (id, deleteFile) =>
    j("DELETE", `/api/image?id=${encodeURIComponent(id)}&delete_file=${deleteFile}`),
  batchDelete: (ids, deleteFile = false) =>
    j("POST", "/api/images/delete", { ids, delete_file: deleteFile }),

  // Legacy cleanup (remove all orphaned at once)
  cleanupMissing: () => j("POST", "/api/cleanup-missing"),

  // Legacy folders list (kept for compat)
  folders: () => j("GET", "/api/folders"),

  // <img> can't send Authorization headers, so carry the token as a query param.
  thumbUrl: (id) => `/api/image?size=thumb&id=${encodeURIComponent(id)}${_tokenQS()}`,
  mediumUrl: (id) => `/api/image?size=medium&id=${encodeURIComponent(id)}${_tokenQS()}`,
  fullUrl: (id) => `/api/image?size=full&id=${encodeURIComponent(id)}${_tokenQS()}`,
};

function _tokenQS() {
  return _token ? `&_t=${encodeURIComponent(_token)}` : "";
}
