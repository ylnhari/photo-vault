// Shared app state. Single source of truth so every tab shows the SAME service
// health / index status — no more two components fetching health independently
// and disagreeing ("both online" vs "no services"). Values are cached across
// tab switches, so flipping tabs doesn't re-fetch and flicker.
import { writable } from "svelte/store";
import { api } from "./api.js";

export const health = writable({
  loaded: false, lm_studio: false, gemini: false, gemini_key_set: false,
  ninerouter: false, lm_studio_state: { known: false },
});

// Shared shape so a failed refreshStatus() can reset to a clear "unknown"
// state instead of leaving stale numbers in place (see refreshStatus below).
const STATUS_DEFAULT = {
  loaded: false,   // false until the FIRST successful load — UI shows a skeleton
  loading: false,  // a refresh is in flight
  error: false,    // the last refresh failed — UI keeps last-good numbers + a hint
  stage: { total_scanned: 0, vision_done: 0, vision_pending: 0,
           active_model: null, active_model_embedded: 0, embed_pending: 0, models: {} },
  vision_pending: 0, embed_pending: 0, missing_attrs: 0, missing_full: 0, missing_files: 0,
  model_status: {
    vision: { selected_label: null, done: 0, pending: 0, any_done: 0, model_summary: {} },
    embed:  { selected_model: null, caption_source: null, eligible: 0, done: 0, pending: 0 },
  },
  settings: {},
};

export const status = writable({ ...STATUS_DEFAULT });

export const models = writable({ loaded: false, active: null, models: {} });

// Ids from the most recent delete operation (single photo from the Lightbox,
// or a whole batch from a grid's multi-select). Always an array, even for a
// single id — PhotoGrid subscribes and hides every id in it, so every visible
// grid drops deleted photos without each tab having to wire up its own
// removal handling.
export const lastDeleted = writable([]);

// Lightweight background-job status for the global header pill, so a running
// index job is visible from every tab (the detailed panel lives in
// Index & Manage). Polled slowly; IndexTab keeps its own faster poll.
export const jobStatus = writable({ active: false, type: null, done: 0, total: 0 });

export async function refreshJob() {
  try {
    const j = await api.indexProgress();
    jobStatus.set({ active: !!j.active, type: j.type, done: j.done, total: j.total });
  } catch {
    // Reset to a clear "unknown" shape rather than leaving stale progress
    // numbers on screen when the server can't be reached.
    jobStatus.set({ active: false, type: null, done: 0, total: 0 });
  }
}

export async function refreshHealth(fresh = false) {
  try {
    const h = await api.health(fresh);
    health.set({ loaded: true, ...h });
  } catch {
    health.set({ loaded: true, lm_studio: false, gemini: false, gemini_key_set: false,
                 ninerouter: false, lm_studio_state: { known: false } });
  }
}

export async function refreshStatus() {
  status.update((s) => ({ ...s, loading: true }));
  try {
    const s = await api.status();
    status.set({ ...STATUS_DEFAULT, ...s, loaded: true, loading: false, error: false });
  } catch {
    // Do NOT zero the numbers on a failed/slow poll — that made a mere timeout
    // look like the whole library had been wiped (0 scanned/captioned/embedded).
    // Keep the last-good values and flag the refresh as errored so the UI can
    // show a subtle "couldn't refresh" hint instead of a scary 0. `loaded`
    // stays whatever it was: false (still skeleton) if we never succeeded once.
    status.update((s) => ({ ...s, loading: false, error: true }));
  }
}

export async function refreshModels() {
  try {
    const m = await api.models();
    models.set({ loaded: true, ...m });
  } catch {
    models.set({ loaded: true, active: null, models: {} });
  }
}

export function refreshAll() {
  return Promise.all([refreshHealth(), refreshStatus(), refreshModels()]);
}
