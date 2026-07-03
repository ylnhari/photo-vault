// Shared app state. Single source of truth so every tab shows the SAME service
// health / index status — no more two components fetching health independently
// and disagreeing ("both online" vs "no services"). Values are cached across
// tab switches, so flipping tabs doesn't re-fetch and flicker.
import { writable } from "svelte/store";
import { api } from "./api.js";

export const health = writable({
  loaded: false, lm_studio: false, gemini: false, gemini_key_set: false,
});

export const status = writable({
  loaded: false,
  stage: { total_scanned: 0, vision_done: 0, vision_pending: 0,
           active_model: null, active_model_embedded: 0, embed_pending: 0, models: {} },
  vision_pending: 0, embed_pending: 0, missing_attrs: 0, missing_full: 0, missing_files: 0,
  model_status: {
    vision: { selected_label: null, done: 0, pending: 0, any_done: 0, model_summary: {} },
    embed:  { selected_model: null, caption_source: null, eligible: 0, done: 0, pending: 0 },
  },
  settings: {},
});

export const models = writable({ loaded: false, active: null, models: {} });

// Id of the most recently deleted photo. PhotoGrid subscribes and hides it, so
// every visible grid drops a photo deleted from the Lightbox without each tab
// having to wire up its own removal handling.
export const lastDeleted = writable(null);

// Lightweight background-job status for the global header pill, so a running
// index job is visible from every tab (the detailed panel lives in
// Index & Manage). Polled slowly; IndexTab keeps its own faster poll.
export const jobStatus = writable({ active: false, type: null, done: 0, total: 0 });

export async function refreshJob() {
  try {
    const j = await api.indexProgress();
    jobStatus.set({ active: !!j.active, type: j.type, done: j.done, total: j.total });
  } catch {}
}

export async function refreshHealth() {
  try {
    const h = await api.health();
    health.set({ loaded: true, ...h });
  } catch {
    health.set({ loaded: true, lm_studio: false, gemini: false, gemini_key_set: false });
  }
}

export async function refreshStatus() {
  try {
    const s = await api.status();
    status.set({ loaded: true, ...s });
  } catch {
    status.update((v) => ({ ...v, loaded: true }));
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
