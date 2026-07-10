"""Background indexing job manager.

Single-user, single-job model: one indexing pass runs at a time in a worker
thread. The API polls `status()` for live progress and calls `stop()` to abort.
Unlike the old Streamlit one-photo-per-rerun hack, this is a real background
task — the UI never blocks. Stop granularity depends on job type: "vision"
(_run_vision_parallel) and "embed" (_run_embed_batched) check between batches
(sized by vision_concurrency and EMBED_BATCH respectively, so up to one
batch's worth of in-flight work can finish after Stop is pressed); every other
job type (faces/thumbs/dhash/full/reanalyze/scan/ingest/dedupe/backup) runs
via _run_sequential and checks between every individual item.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import ratelimit
import settings as settings_mod
from indexer import Indexer, resolve_caption_json, build_embed_payload
from vision import parse_vision_attributes, build_embedding_text

JOB_TYPES = ("vision", "embed", "full", "reanalyze", "faces", "thumbs",
             "dhash", "scan", "ingest", "dedupe", "backup")

# Captions per /v1/embeddings request during embed jobs. LM Studio accepts a
# list, so one round-trip embeds the whole chunk.
EMBED_BATCH = 16

# Persist the catalog every N images during sequential jobs instead of once per
# image — turns the old O(n^2) full-file rewrite into O(n).
SAVE_EVERY = 25


class JobManager:
    def __init__(self):
        self._lock = threading.RLock()  # reentrant: start() calls status() while holding it
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = self._idle_state()
        # Let Stop interrupt a rate-limit sleep inside a provider call — a
        # full rpm/rpd window could otherwise hold the worker for minutes to
        # hours after the user pressed Stop. Providers raise
        # ratelimit.Cancelled when this fires mid-wait; the runners treat it
        # as "stopped", never as a per-image failure.
        ratelimit.set_cancel_event(self._stop)

    @staticmethod
    def _idle_state() -> dict:
        return {
            "active": False, "type": None, "total": 0, "done": 0,
            "ok": 0, "fail": 0, "skipped": 0, "failed_ids": [], "log": [],
            "aborted": False, "stopped": False, "finished": False, "error": None,
            "started_at": None, "max_fail": 5, "model_counts": {},
            "substitution": None,
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
        elif jtype == "thumbs":
            items = idx.get_thumbs_pending()
        elif jtype == "dhash":
            items = idx.get_dhash_pending()
        elif jtype == "scan":
            # One work item per configured folder (paths, not image ids).
            from folders import ensure_defaults, get_effective_scan_dirs
            ensure_defaults()
            return get_effective_scan_dirs()
        elif jtype == "ingest":
            # One work item per staging media file (paths). Listing is cheap
            # (no hashing) — safe under start()'s lock.
            from ingest import list_staging_files
            src = cfg.get("source_path")
            if not src:
                raise ValueError("ingest requires a source folder")
            return list_staging_files(src)
        elif jtype == "dedupe":
            # 'uid::path' items — recorded byte-identical extra copies.
            return idx.get_redundant_copies()
        elif jtype == "backup":
            # One work item per mirror root (source paths).
            import backup as backup_mod
            roots = [src for src, _ in backup_mod.backup_roots()]
            if not roots:
                raise ValueError("backup destination not configured")
            return roots
        else:
            raise ValueError(f"unknown job type: {jtype}")
        return [img_id for img_id, _ in items]

    def start(self, jtype: str, vision_provider: str = "auto", max_fail: int = 5,
              vision_model: str = None, embed_provider: str = "auto",
              embed_model: str = None, caption_source_model: str = None,
              vision_model_label: str = None, source_path: str = None) -> dict:
        if jtype not in JOB_TYPES:
            raise ValueError(f"unknown job type: {jtype}")
        # max_fail <= 0 would abort the job after the very first batch
        # regardless of how many images actually failed — clamp to a sane
        # floor instead of trusting whatever the caller passed.
        try:
            max_fail = max(1, int(max_fail))
        except (TypeError, ValueError):
            max_fail = 5
        # Derive the caption_history model label from vision_provider/vision_model
        # in exactly one place (matches settings.vision_model_label's formula)
        # instead of trusting a second, independently-computed value from the
        # caller — the two could otherwise silently disagree. The parameter is
        # kept for call-signature compatibility but is no longer trusted as-is.
        vision_model_label = settings_mod.vision_model_label(
            {"vision_provider": vision_provider, "vision_model": vision_model}
        )
        with self._lock:
            if self._state["active"]:
                raise RuntimeError("a job is already running")
            cfg = {
                "vision_provider": vision_provider, "vision_model": vision_model,
                "embed_provider": embed_provider, "embed_model": embed_model,
                "caption_source_model": caption_source_model,
                "vision_model_label": vision_model_label,
                "max_fail": max_fail,
                "source_path": source_path,
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
            s["model_counts"] = dict(s["model_counts"])
            # ETA from the cumulative average rate so far. Deliberately
            # includes time spent waiting on rate limits — an ETA that
            # ignored the throttle would read absurdly optimistic the moment
            # a window fills.
            s["eta_seconds"] = None
            s["rate_per_min"] = None
            if s["active"] and s["done"] and s["started_at"]:
                elapsed = time.time() - s["started_at"]
                if elapsed > 0:
                    rate = s["done"] / elapsed
                    s["eta_seconds"] = round(max(0, s["total"] - s["done"]) / rate)
                    s["rate_per_min"] = round(rate * 60, 1)
            # Which providers (if any) are currently sleeping on a rate-limit
            # window, so the UI can say WHY the bar froze.
            s["rate_wait"] = ratelimit.waiting() if s["active"] else {}
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
            elif jtype == "embed":
                self._run_embed_batched(idx, ids, cfg)
            else:
                self._run_sequential(idx, jtype, ids, cfg)
        except Exception as e:
            # Without this, an unexpected bug in a runner (as opposed to a
            # per-image failure, which the runners already catch) would only
            # ever surface via Python's default thread excepthook on stderr —
            # invisible in the UI, and the job would be left stuck "active".
            # `error` is a new, additive status field (existing consumers of
            # `aborted`/`log` keep working unchanged) that lets the UI show
            # the actual crash reason instead of a generic "too many failures".
            import traceback
            print(f"[jobs] worker thread crashed: {e}")
            traceback.print_exc()
            with self._lock:
                self._state["aborted"] = True
                self._state["error"] = str(e)
                self._state["log"].append(
                    {"kind": "fail", "id": None, "note": f"job crashed: {e}", "file": ""}
                )
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
                    except ratelimit.Cancelled:
                        # Stop pressed while this item waited on a rate-limit
                        # slot — it never reached the provider. (None, None)
                        # marks it "still pending": not ok, not failed, not
                        # done. The stop check at the top of the next loop
                        # iteration ends the job.
                        done[bid] = (None, None)
                    except Exception as e:
                        done[bid] = (None, str(e))
                for bid in batch:
                    result, errm = done[bid]
                    if result is None and errm is None:
                        continue
                    fname = idx.image_catalog["images"].get(bid, {}).get("filename", "")
                    if errm is None:
                        vmodel, text = result
                        idx.record_caption(bid, vmodel, text)
                        consecutive = 0
                        icon = "cloud" if vmodel and "gemini" in vmodel.lower() else "ok"
                        self._update(ok=1, done=1, model_used=vmodel,
                                     log=(icon, bid, f"vision:{vmodel}", fname))
                    else:
                        consecutive += 1
                        self._update(fail=1, done=1, failed_id=bid, log=("fail", bid, errm, fname))
                idx._save_catalog()  # one write per batch, not per image
                # Gated on consecutive > 0 so this can only ever fire because
                # of real accumulated failures, never as a side effect of a
                # misconfigured max_fail (max_fail is clamped to >= 1 in
                # start(), but this is cheap defense-in-depth against a
                # 0-or-negative value ever aborting a fully-successful batch).
                if consecutive > 0 and consecutive >= max_fail:
                    self._update(aborted=True)
                    break

    def _run_embed_batched(self, idx, ids, cfg):
        """
        Embed jobs: one /v1/embeddings request per EMBED_BATCH captions instead
        of per image. Face detection (when enabled) and ChromaDB writes remain
        per-batch on this thread. The dedicated 'embed' job path — 'full' and
        'reanalyze' still run item-by-item via _run_sequential.
        """
        from embeddings import (
            get_embeddings_batch,
            collection_name_for,
            last_embed_error,
            last_substitution,
            resolve_9router_embed_id,
        )
        import db

        ep, em = cfg["embed_provider"], cfg["embed_model"]
        csm = cfg.get("caption_source_model")
        max_fail = cfg["max_fail"]
        try:
            faces_during = bool(settings_mod.load().get("faces_during_embed", True))
        except Exception:
            faces_during = True

        consecutive = 0
        i, n = 0, len(ids)
        while i < n:
            if self._stop.is_set():
                self._update(stopped=True)
                break
            chunk = ids[i:i + EMBED_BATCH]
            i += len(chunk)

            # Resolve captions; per-image failures don't sink the chunk. Also
            # check Stop here (not just at the top of the outer while loop) —
            # this work is pure catalog reads/CPU, no network yet, so breaking
            # here means Stop takes effect after at most a partial chunk
            # instead of waiting for a full EMBED_BATCH-sized network round
            # trip to finish first.
            texts, members = [], []   # members: (img_id, img_data, caption_json)
            stopped_mid_chunk = False
            for iid in chunk:
                if self._stop.is_set():
                    stopped_mid_chunk = True
                    break
                img_data = idx.image_catalog["images"].get(iid)
                fname = (img_data or {}).get("filename", "")
                try:
                    if img_data is None:
                        raise RuntimeError("not in catalog")
                    cj = resolve_caption_json(img_data, csm)
                    texts.append(build_embedding_text(parse_vision_attributes(cj)))
                    members.append((iid, img_data, cj))
                except Exception as e:
                    consecutive += 1
                    self._update(fail=1, done=1, failed_id=iid,
                                 log=("fail", iid, str(e), fname))
            if stopped_mid_chunk:
                self._update(stopped=True)
                break
            if not members:
                if consecutive >= max_fail:
                    self._update(aborted=True)
                    break
                continue

            try:
                vectors, model_name, source = get_embeddings_batch(
                    texts, force_provider=ep, model=em
                )
            except ratelimit.Cancelled:
                # Stop pressed during a rate-limit wait — the chunk never
                # reached the provider; its items stay pending, not failed.
                self._update(stopped=True)
                break
            if vectors is None:
                # Surface the real provider error in the job log — "9Router
                # substituted embedding model (X → Y)" tells the user to
                # restart with a different model, while a connection error
                # just means the service needs to come back first.
                reason = last_embed_error() or "all embedding providers unavailable"
                sub = last_substitution()
                if sub:
                    # Structured record so the UI can offer one-click
                    # recovery: switch the embed model to what 9Router
                    # actually serves and re-run. "suggested" is the full
                    # selectable id (served names come back prefix-stripped).
                    with self._lock:
                        self._state["substitution"] = {
                            **sub,
                            "suggested": resolve_9router_embed_id(
                                sub["served"], sub["requested"]),
                        }
                for iid, img_data, _ in members:
                    consecutive += 1
                    self._update(fail=1, done=1, failed_id=iid,
                                 log=("fail", iid, f"embedding failed — {reason}",
                                      img_data.get("filename", "")))
                if consecutive >= max_fail:
                    self._update(aborted=True)
                    break
                continue

            # A Gemini batch can partially fail per item (see get_embeddings_batch)
            # — vectors[i] is None for those; don't let one bad item in the chunk
            # drop the successful ones.
            if any(v is None for v in vectors):
                good_members, good_vectors = [], []
                for (iid, img_data, cj), vec in zip(members, vectors):
                    if vec is None:
                        consecutive += 1
                        self._update(fail=1, done=1, failed_id=iid,
                                     log=("fail", iid, "embedding failed for this item",
                                          img_data.get("filename", "")))
                    else:
                        good_members.append((iid, img_data, cj))
                        good_vectors.append(vec)
                members, vectors = good_members, good_vectors
                if not members:
                    if consecutive >= max_fail:
                        self._update(aborted=True)
                        break
                    continue

            # Faces (optional, per image) then one batched ChromaDB add.
            if faces_during:
                from faces import detect_and_embed_faces, save_face_data, index_faces
                for iid, img_data, _ in members:
                    try:
                        fd = detect_and_embed_faces(img_data["path"])
                        save_face_data(iid, fd)
                        index_faces(iid, fd)
                    except Exception as e:
                        print(f"[jobs] face detection failed for {iid}: {e}")

            col = db.client().get_or_create_collection(
                name=collection_name_for(model_name)
            )
            b_ids = [iid for iid, _, _ in members]
            b_payloads = [
                build_embed_payload(img_data, cj, source, model_name)
                for _, img_data, cj in members
            ]
            try:
                col.upsert(ids=b_ids, embeddings=vectors, metadatas=b_payloads)
            except Exception as e:
                for iid, img_data, _ in members:
                    consecutive += 1
                    self._update(fail=1, done=1, failed_id=iid,
                                 log=("fail", iid, f"store failed: {e}",
                                      img_data.get("filename", "")))
                if consecutive >= max_fail:
                    self._update(aborted=True)
                    break
                continue

            consecutive = 0
            icon = "cloud" if "gemini" in source.lower() else "ok"
            for iid, img_data, _ in members:
                self._update(ok=1, done=1,
                             log=(icon, iid, f"embed:{source}",
                                  img_data.get("filename", "")))

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
        # Ingest runs as one session: the seen-hash set (catalog ids + media
        # hash cache, refreshing library video hashes on first use) is built
        # once up front, then reused for every staging file.
        ingest_session = None
        if jtype == "ingest":
            from ingest import IngestSession
            ingest_session = IngestSession(
                cfg["source_path"], idx.image_catalog["images"])

        consecutive = 0
        dirty = False
        for k, img_id in enumerate(ids):
            if self._stop.is_set():
                self._update(stopped=True)
                break
            if jtype in ("scan", "backup"):
                fname = img_id                      # a folder path
            elif jtype == "ingest":
                fname = os.path.basename(img_id)    # a staging file path
            elif jtype == "dedupe":
                fname = os.path.basename(img_id.partition("::")[2])
            else:
                fname = idx.image_catalog["images"].get(img_id, {}).get("filename", "")
            try:
                if jtype == "embed":
                    note = idx.embed_one(img_id, embed_provider=ep, embed_model=em,
                                         caption_source_model=csm, detect_faces=faces_during_embed)
                elif jtype == "faces":
                    note = idx.detect_faces_one(img_id)
                elif jtype == "thumbs":
                    note = idx.thumb_one(img_id)
                elif jtype == "dhash":
                    note = idx.dhash_one(img_id)
                    dirty = True
                elif jtype == "scan":
                    note = idx.scan_folder_one(img_id)  # img_id is a folder path
                elif jtype == "ingest":
                    note = ingest_session.ingest_one(img_id)
                elif jtype == "dedupe":
                    note = idx.dedupe_copy_one(img_id)
                    dirty = True
                elif jtype == "backup":
                    import backup as backup_mod
                    note = backup_mod.backup_one(img_id)
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
                if "skipped" in note.lower():
                    # A missing/moved file: neither a real failure (must not
                    # trip consecutive-failure abort) nor a silent permanent
                    # success (must stay distinguishable + retryable) — track
                    # it in its own bucket and leave `consecutive` untouched.
                    self._update(skipped=1, done=1, log=("skip", img_id, note, fname))
                else:
                    consecutive = 0
                    icon = "cloud" if "gemini" in note.lower() else "ok"
                    self._update(ok=1, done=1, log=(icon, img_id, note, fname))
            except ratelimit.Cancelled:
                # Stop during a rate-limit wait: the item stays pending.
                self._update(stopped=True)
                break
            except Exception as e:
                consecutive += 1
                self._update(fail=1, done=1, failed_id=img_id,
                             log=("fail", img_id, str(e), fname))
                if consecutive >= max_fail:
                    self._update(aborted=True)
                    break
            if dirty and (k + 1) % SAVE_EVERY == 0:
                idx._save_catalog()
                dirty = False
        if dirty:
            idx._save_catalog()
        if ingest_session is not None:
            ingest_session.close()

    def _update(self, ok=0, fail=0, skipped=0, done=0, failed_id=None, log=None,
                stopped=False, aborted=False, model_used=None):
        with self._lock:
            self._state["ok"] += ok
            self._state["fail"] += fail
            self._state["skipped"] += skipped
            self._state["done"] += done
            if model_used:
                # Per-model tally for the run summary — with 9Router the
                # gateway can substitute serving models mid-run, so one run
                # can legitimately produce captions from several models.
                mc = self._state["model_counts"]
                mc[model_used] = mc.get(model_used, 0) + 1
            if failed_id is not None:
                self._state["failed_ids"].append(failed_id)
            if log is not None:
                kind, img_id, note = log[:3]
                fname = log[3] if len(log) > 3 else ""
                self._state["log"].append(
                    {"kind": kind, "id": img_id, "note": note, "file": fname}
                )
            if stopped:
                self._state["stopped"] = True
            if aborted:
                self._state["aborted"] = True


# module-level singleton shared by the API
manager = JobManager()
