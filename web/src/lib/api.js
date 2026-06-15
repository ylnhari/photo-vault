// Thin fetch wrapper. Same-origin in production; Vite proxies /api in dev.
async function j(method, url, body) {
  const opts = { method, headers: {} };
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
  scan: (dirs) => j("POST", "/api/scan", { dirs }),

  indexStart: (type, force_provider, max_fail) =>
    j("POST", "/api/index/start", { type, force_provider, max_fail }),
  indexStop: () => j("POST", "/api/index/stop"),
  indexProgress: () => j("GET", "/api/index/progress"),
  indexReset: () => j("POST", "/api/index/reset"),

  filters: () => j("GET", "/api/filters"),
  search: (q, filters, person, top_k = 200) =>
    j("POST", "/api/search", { q, filters, person, top_k }),
  recent: (limit = 60) => j("GET", `/api/recent?limit=${limit}`),
  timeline: () => j("GET", "/api/timeline"),

  people: () => j("GET", "/api/people"),
  addPerson: (name, ref_dir) => j("POST", "/api/people", { name, ref_dir }),

  models: () => j("GET", "/api/models"),
  setActiveModel: (model) => j("POST", "/api/models/active", { model }),

  meta: (id) => j("GET", `/api/meta?id=${encodeURIComponent(id)}`),
  deleteImage: (id, deleteFile) =>
    j("DELETE", `/api/image?id=${encodeURIComponent(id)}&delete_file=${deleteFile}`),
  cleanupMissing: () => j("POST", "/api/cleanup-missing"),

  thumbUrl: (id) => `/api/image?thumb=true&id=${encodeURIComponent(id)}`,
  fullUrl: (id) => `/api/image?id=${encodeURIComponent(id)}`,
};
