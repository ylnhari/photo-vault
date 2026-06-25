"""Background indexing job manager.

Single-user, single-job model: one indexing pass runs at a time in a worker
thread. The API polls `status()` for live progress and calls `stop()` to abort.
Unlike the old Streamlit one-photo-per-rerun hack, this is a real background
task — the UI never blocks and Stop is honored between photos.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import settings as settings_mod
from indexer import Indexer

JOB_TYPES = ("vision", "embed", "full", "reanalyze", "faces")

# Persist the catalog every N images during sequential jobs instead of once per
# image — turns the old O(n^2) full-file rewrite into O(n).
SAVE_EVERY = 25


class JobManager:
    def __init__(self):
        self._lock = threading.RLock()  # reentrant: start() calls status() while holding it
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = self._idle_state()

    @staticmethod
    def _idle_state() -> dict:
        return {
            "active": False, "type": None, "total": 0, "done": 0,
            "ok": 0, "fail": 0, "failed_ids": [], "log": [],
            "aborted": False, "stopped": False, "finished": False,
            "started_at": None, "max_fail": 5,
            "vision_provider": "auto", "vision_model": None,
            "embed_provider": "auto", "embed_model": None,
            "caption_source_model": None,
        }

    # ── public API ──────────────────────────────────────────────────────────
    def _pending_ids(self, jtype: str, cfg: dict) -> list[str]:
        idx = Indexer()
        vm_label = cfg.get("vision_model_label")       # full label e.g. "lm_studio:qwen2-vl-7b"
        csm = cfg.get("caption_source_model")          # caption source for embed
        em = cfg.get("embed_model")                    # embed model name (not provider-prefixed)

        if jtype == "vision":
            if vm_label:
                items = idx.get_vision_pending_for_model(vm_label)
            else:
                items = idx.get_vision_pending()
        elif jtype == "embed":
            if em:
                items = idx.get_embed_pending_for_model(em, csm)
            else:
                items = idx.get_embed_pending()
        elif jtype == "full":
            items = idx.get_missing()
        elif jtype == "reanalyze":
            items = idx.get_missing_attributes()
        elif jtype == "faces":
            items = idx.get_faces_pending()
        else:
            raise ValueError(f"unknown job type: {jtype}")
        return [img_id for img_id, _ in items]

    def start(self, jtype: str, vision_provider: str = "auto", max_fail: int = 5,
              vision_model: str = None, embed_provider: str = "auto",
              embed_model: str = None, caption_source_model: str = None,
              vision_model_label: str = None) -> dict:
        if jtype not in JOB_TYPES:
            raise ValueError(f"unknown job type: {jtype}")
        with self._lock:
            if self._state["active"]:
                raise RuntimeError("a job is already running")
            cfg = {
                "vision_provider": vision_provider, "vision_model": vision_model,
                "embed_provider": embed_provider, "embed_model": embed_model,
                "caption_source_model": caption_source_model,
                "vision_model_label": vision_model_label,
                "max_fail": max_fail,
            }
            ids = self._pending_ids(jtype, cfg)
            self._stop.clear()
            self._state = self._idle_state()
            self._state.update({
                "active": True, "type": jtype, "total": len(ids),
                "started_at": time.time(),
                "vision_provider": vision_provider, "vision_model": vision_model,
                "embed_provider": embed_provider, "embed_model": embed_model,
                "caption_source_model": caption_source_model,
                "max_fail": max_fail,
            })
            if not ids:
                self._state.update({"active": False, "finished": True})
                return self.status()
            self._thread = threading.Thread(
                target=self._run, args=(jtype, ids, cfg), daemon=True
            )
            self._thread.start()
        return self.status()

    def stop(self):
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            s = dict(self._state)
            s["log"] = list(s["log"][-30:])
            s["failed_ids"] = list(s["failed_ids"])
            return s

    def reset(self):
        """Clear a finished/aborted job back to idle (only when not active)."""
        with self._lock:
            if not self._state["active"]:
                self._state = self._idle_state()

    # ── worker ────────────────────────────────────────────────────────────────

    def _run(self, jtype, ids, cfg):
        idx = Indexer()  # private mutable catalog copy for this job
        try:
            if jtype == "vision":
                self._run_vision_parallel(idx, ids, cfg)
            else:
                self._run_sequential(idx, jtype, ids, cfg)
        finally:
            with self._lock:
                self._state["active"] = False
                self._state["finished"] = True

    def _run_vision_parallel(self, idx, ids, cfg):
        """
        Vision is network-bound, so caption N images concurrently. Captions are
        computed in worker threads (no shared mutation), then applied + saved on
        this thread in batches. Stop is honored between batches.
        """
        vp, vm = cfg["vision_provider"], cfg["vision_model"]
        max_fail = cfg["max_fail"]
        try:
            conc = max(1, int(settings_mod.load().get("vision_concurrency", 4)))
        except Exception:
            conc = 4

        consecutive = 0
        i, n = 0, len(ids)
        with ThreadPoolExecutor(max_workers=conc) as ex:
            while i < n:
                if self._stop.is_set():
                    self._update(stopped=True)
                    break
                batch = ids[i:i + conc]
                i += len(batch)
                futures = {
                    ex.submit(idx.compute_caption, bid, vp, vm): bid for bid in batch
                }
                # Collect results, then apply sequentially (catalog mutation is
                # single-threaded; only the network calls ran in parallel).
                done = {}
                for fut in futures:
                    bid = futures[fut]
                    try:
                        done[bid] = (fut.result(), None)
                    except Exception as e:
                        done[bid] = (None, str(e))
                for bid in batch:
                    result, errm = done[bid]
                    if errm is None:
                        vmodel, text = result
                        idx.record_caption(bid, vmodel, text)
                        consecutive = 0
                        icon = "cloud" if vmodel and "gemini" in vmodel.lower() else "ok"
                        self._update(ok=1, done=1, log=(icon, bid, f"vision:{vmodel}"))
                    else:
                        consecutive += 1
                        self._update(fail=1, done=1, failed_id=bid, log=("fail", bid, errm))
                idx._save_catalog()  # one write per batch, not per image
                if consecutive >= max_fail:
                    self._update(aborted=True)
                    break

    def _run_sequential(self, idx, jtype, ids, cfg):
        """Embed / full / reanalyze: run one at a time (ChromaDB writes), but
        batch the images.json saves so full/reanalyze aren't O(n^2)."""
        max_fail = cfg["max_fail"]
        vp, vm = cfg["vision_provider"], cfg["vision_model"]
        ep, em = cfg["embed_provider"], cfg["embed_model"]
        csm = cfg.get("caption_source_model")
        try:
            faces_during_embed = bool(settings_mod.load().get("faces_during_embed", True))
        except Exception:
            faces_during_embed = True
        consecutive = 0
        dirty = False
        for k, img_id in enumerate(ids):
            if self._stop.is_set():
                self._update(stopped=True)
                break
            try:
                if jtype == "embed":
                    note = idx.embed_one(img_id, embed_provider=ep, embed_model=em,
                                         caption_source_model=csm, detect_faces=faces_during_embed)
                elif jtype == "faces":
                    note = idx.detect_faces_one(img_id)
                elif jtype == "full":
                    note = idx.index_one_full(img_id, use_cached=True, upsert=False,
                                              vision_provider=vp, vision_model=vm,
                                              embed_provider=ep, embed_model=em,
                                              caption_source_model=csm, persist=False,
                                              detect_faces=faces_during_embed)
                    dirty = True
                else:  # reanalyze
                    note = idx.index_one_full(img_id, use_cached=False, upsert=True,
                                              vision_provider=vp, vision_model=vm,
                                              embed_provider=ep, embed_model=em,
                                              caption_source_model=csm, persist=False,
                                              detect_faces=faces_during_embed)
                    dirty = True
                consecutive = 0
                icon = "cloud" if "gemini" in note.lower() else "ok"
                self._update(ok=1, done=1, log=(icon, img_id, note))
            except Exception as e:
                consecutive += 1
                self._update(fail=1, done=1, failed_id=img_id, log=("fail", img_id, str(e)))
                if consecutive >= max_fail:
                    self._update(aborted=True)
                    break
            if dirty and (k + 1) % SAVE_EVERY == 0:
                idx._save_catalog()
                dirty = False
        if dirty:
            idx._save_catalog()

    def _update(self, ok=0, fail=0, done=0, failed_id=None, log=None,
                stopped=False, aborted=False):
        with self._lock:
            self._state["ok"] += ok
            self._state["fail"] += fail
            self._state["done"] += done
            if failed_id is not None:
                self._state["failed_ids"].append(failed_id)
            if log is not None:
                kind, img_id, note = log
                self._state["log"].append({"kind": kind, "id": img_id, "note": note})
            if stopped:
                self._state["stopped"] = True
            if aborted:
                self._state["aborted"] = True


# module-level singleton shared by the API
manager = JobManager()
