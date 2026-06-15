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
});

export const models = writable({ loaded: false, active: null, models: {} });

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
