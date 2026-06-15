"""Background indexing job manager.

Single-user, single-job model: one indexing pass runs at a time in a worker
thread. The API polls `status()` for live progress and calls `stop()` to abort.
Unlike the old Streamlit one-photo-per-rerun hack, this is a real background
task — the UI never blocks and Stop is honored between photos.
"""
import threading
import time

from indexer import Indexer

JOB_TYPES = ("vision", "embed", "full", "reanalyze")


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
        }

    # ── public API ──────────────────────────────────────────────────────────

    def _pending_ids(self, jtype: str) -> list[str]:
        idx = Indexer()
        if jtype == "vision":
            items = idx.get_vision_pending()
        elif jtype == "embed":
            items = idx.get_embed_pending()
        elif jtype == "full":
            items = idx.get_missing()
        elif jtype == "reanalyze":
            items = idx.get_missing_attributes()
        else:
            raise ValueError(f"unknown job type: {jtype}")
        return [img_id for img_id, _ in items]

    def start(self, jtype: str, vision_provider: str = "auto", max_fail: int = 5,
              vision_model: str = None, embed_provider: str = "auto",
              embed_model: str = None) -> dict:
        if jtype not in JOB_TYPES:
            raise ValueError(f"unknown job type: {jtype}")
        with self._lock:
            if self._state["active"]:
                raise RuntimeError("a job is already running")
            ids = self._pending_ids(jtype)
            self._stop.clear()
            cfg = {
                "vision_provider": vision_provider, "vision_model": vision_model,
                "embed_provider": embed_provider, "embed_model": embed_model,
                "max_fail": max_fail,
            }
            self._state = self._idle_state()
            self._state.update({
                "active": True, "type": jtype, "total": len(ids),
                "started_at": time.time(), **cfg,
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
        idx = Indexer()
        max_fail = cfg["max_fail"]
        vp, vm = cfg["vision_provider"], cfg["vision_model"]
        ep, em = cfg["embed_provider"], cfg["embed_model"]
        consecutive = 0
        for i, img_id in enumerate(ids):
            if self._stop.is_set():
                self._update(stopped=True)
                break
            try:
                if jtype == "vision":
                    note = idx.vision_one(img_id, force_provider=vp, model=vm)
                elif jtype == "embed":
                    note = idx.embed_one(img_id, embed_provider=ep, embed_model=em)
                elif jtype == "full":
                    note = idx.index_one_full(img_id, use_cached=True, upsert=False,
                                              vision_provider=vp, vision_model=vm,
                                              embed_provider=ep, embed_model=em)
                else:  # reanalyze
                    note = idx.index_one_full(img_id, use_cached=False, upsert=True,
                                              vision_provider=vp, vision_model=vm,
                                              embed_provider=ep, embed_model=em)
                consecutive = 0
                icon = "cloud" if "gemini" in note.lower() else "ok"
                self._update(ok=1, done=1, log=(icon, img_id, note))
            except Exception as e:
                consecutive += 1
                self._update(fail=1, done=1, failed_id=img_id, log=("fail", img_id, str(e)))
                if consecutive >= max_fail:
                    self._update(aborted=True)
                    break
        with self._lock:
            self._state["active"] = False
            self._state["finished"] = True

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
