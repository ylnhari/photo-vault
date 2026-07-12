"""Background indexing job manager.

Concurrent-but-safe job model. Jobs run in worker threads; the API polls
`status()` for live progress and calls `stop()` to abort. More than one job can
run at once, but ONLY when they touch disjoint resources — see JOB_RESOURCES.
The old blanket "one job at a time" lock was over-restrictive: face detection is
local compute that writes only FACE_DIR + the faces ChromaDB collection, so it
shares nothing with a captioning/embedding pass and can overlap it. What must
still serialize: two jobs that both mutate image catalog records (last-writer
clobbers, since each job edits a private in-memory copy of the whole record),
and two jobs that both drive a provider (LM Studio model-load thrash). Those
are expressed as shared resources, and start() refuses a job whose resources
overlap a running one.

Stop granularity depends on job type: "vision" (_run_vision_parallel) and
"embed" (_run_embed_batched) check between batches (sized by vision_concurrency
and EMBED_BATCH respectively, so up to one batch's worth of in-flight work can
finish after Stop is pressed); every other job type (faces/thumbs/dhash/full/
reanalyze/scan/ingest/dedupe/backup) runs via _run_sequential and checks
between every individual item.
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
             "dhash", "scan", "ingest", "dedupe", "backup",
             "video_vision", "video_faces")

# Captions per /v1/embeddings request during embed jobs. LM Studio accepts a
# list, so one round-trip embeds the whole chunk.
EMBED_BATCH = 16

# Persist the catalog every N images during sequential jobs instead of once per
# image — turns the old O(n^2) full-file rewrite into O(n).
SAVE_EVERY = 25

# Bounds for keyframes-per-video. Floor 1 (at least one frame or there's nothing
# to caption); ceiling 12 keeps the per-video cost/quota sane on the free tier.
VIDEO_FRAMES_MIN, VIDEO_FRAMES_MAX = 1, 12

# ── Resource model (concurrency policy) ──────────────────────────────────────
# Exclusive resources a job holds for its whole run. Two jobs may run
# concurrently iff their resource sets are DISJOINT.
RES_CATALOG = "catalog"            # mutates image records in the catalog DB
RES_INFERENCE = "inference"        # drives a provider (LM Studio/Gemini/9Router)
RES_EMBED_COL = "embed_collection" # writes the embedding ChromaDB collection + registry
RES_FACES = "faces"                # writes FACE_DIR + the faces ChromaDB collection
RES_THUMBS = "thumbs"              # writes THUMB_DIR
RES_BACKUP = "backup"              # filesystem mirror to the backup dest

# Base resources per job type. embed/full/reanalyze pick up RES_FACES too when
# faces-during-embed is on (added in _resources_for, since it depends on a
# runtime setting). An unknown type is treated as needing everything, so it can
# never accidentally overlap something it shouldn't.
JOB_RESOURCES = {
    "vision":       {RES_CATALOG, RES_INFERENCE},
    "video_vision": {RES_CATALOG, RES_INFERENCE},
    "embed":        {RES_INFERENCE, RES_EMBED_COL},
    "full":         {RES_CATALOG, RES_INFERENCE, RES_EMBED_COL},
    "reanalyze":    {RES_CATALOG, RES_INFERENCE, RES_EMBED_COL},
    "faces":        {RES_FACES},
    "video_faces":  {RES_FACES},
    "thumbs":       {RES_THUMBS},
    "dhash":        {RES_CATALOG},
    "scan":         {RES_CATALOG},
    "ingest":       {RES_CATALOG},
    "dedupe":       {RES_CATALOG, RES_EMBED_COL, RES_FACES},
    "backup":       {RES_BACKUP},
}

# How many finished jobs to keep visible (for the UI's "Done"/retry panel)
# before pruning the oldest. Active jobs are never pruned.
_MAX_FINISHED_KEPT = 8


def _clamp_video_frames(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 4
    return max(VIDEO_FRAMES_MIN, min(VIDEO_FRAMES_MAX, n))


class _Job:
    """One running/finished job: its own progress state + stop flag + the
    resources it holds. Kept tiny; the state dict is what status() renders."""
    __slots__ = ("id", "type", "resources", "stop", "thread", "state")

    def __init__(self, jid: str, jtype: str, resources: set, state: dict):
        self.id = jid
        self.type = jtype
        self.resources = resources
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.state = state


class JobManager:
    def __init__(self):
        # Reentrant: start() calls status()/_render() while holding it.
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._counter = 0
        # ratelimit uses a single module-level cancel event to interrupt a
        # provider sleep on Stop. Only one inference-holding job runs at a time
        # (they share RES_INFERENCE), so each such job points this at its own
        # stop event while it runs and resets it to a neutral (never-set) event
        # when it ends — see _run. Start neutral so a stray acquire() before any
        # job never thinks it's been cancelled.
        ratelimit.set_cancel_event(threading.Event())

    @staticmethod
    def _idle_state() -> dict:
        return {
            "id": None, "resources": [],
            "active": False, "type": None, "total": 0, "done": 0,
            "ok": 0, "fail": 0, "skipped": 0, "failed_ids": [], "log": [],
            "aborted": False, "stopped": False, "finished": False, "error": None,
            "started_at": None, "max_fail": 5, "model_counts": {},
            "substitution": None,
            "vision_provider": "auto", "vision_model": None,
            "embed_provider": "auto", "embed_model": None,
            "caption_source_model": None,
        }

    # ── resource policy ───────────────────────────────────────────────────────
    @staticmethod
    def _resources_for(jtype: str, cfg: dict) -> set:
        res = set(JOB_RESOURCES.get(
            jtype, {RES_CATALOG, RES_INFERENCE, RES_EMBED_COL, RES_FACES}))
        # embed/full/reanalyze also run face detection inline when the setting
        # is on, so they then contend with a standalone faces job.
        if jtype in ("embed", "full", "reanalyze"):
            try:
                faces_on = bool(settings_mod.load().get("faces_during_embed", True))
            except Exception:
                faces_on = True
            if faces_on:
                res.add(RES_FACES)
        return res

    def _active_resources(self) -> set:
        held = set()
        for j in self._jobs.values():
            if j.state["active"]:
                held |= j.resources
        return held

    def _prune_finished(self):
        finished = [j for j in self._jobs.values() if not j.state["active"]]
        if len(finished) <= _MAX_FINISHED_KEPT:
            return
        finished.sort(key=lambda j: j.state.get("started_at") or 0)
        for j in finished[:len(finished) - _MAX_FINISHED_KEPT]:
            self._jobs.pop(j.id, None)

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
        elif jtype == "video_vision":
            items = idx.get_video_vision_pending()
        elif jtype == "video_faces":
            items = idx.get_video_faces_pending()
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
            return list_staging_files(src, media=cfg.get("ingest_media", "both"))
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
              vision_model_label: str = None, source_path: str = None,
              ingest_media: str = "both", ingest_video_dest: str = None) -> dict:
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
            cfg = {
                "vision_provider": vision_provider, "vision_model": vision_model,
                "embed_provider": embed_provider, "embed_model": embed_model,
                "caption_source_model": caption_source_model,
                "vision_model_label": vision_model_label,
                "max_fail": max_fail,
                "source_path": source_path,
                "ingest_media": ingest_media if ingest_media in
                                ("both", "photos", "videos") else "both",
                "ingest_video_dest": ingest_video_dest,
                # Keyframes sampled per video for video_vision/video_faces.
                # Read once at job start (like vision_concurrency) so the job
                # runs at a stable value even if Settings change mid-run.
                "video_frames": _clamp_video_frames(
                    settings_mod.load().get("video_frames", 4)),
            }
            # Concurrency gate: refuse only if this job's resources overlap a
            # RUNNING job's. Disjoint jobs (e.g. faces alongside embed) proceed.
            needed = self._resources_for(jtype, cfg)
            conflict = needed & self._active_resources()
            if conflict:
                others = sorted({
                    j.type for j in self._jobs.values()
                    if j.state["active"] and (j.resources & needed)
                })
                raise RuntimeError(
                    f"'{jtype}' can't start — a conflicting job is already "
                    f"running ({', '.join(others)}; shared: "
                    f"{', '.join(sorted(conflict))})"
                )

            self._prune_finished()
            self._counter += 1
            jid = f"{jtype}-{self._counter}"
            state = self._idle_state()
            ids = self._pending_ids(jtype, cfg)
            state.update({
                "id": jid, "resources": sorted(needed),
                "active": True, "type": jtype, "total": len(ids),
                "started_at": time.time(),
                "vision_provider": vision_provider, "vision_model": vision_model,
                "embed_provider": embed_provider, "embed_model": embed_model,
                "caption_source_model": caption_source_model,
                "max_fail": max_fail,
            })
            job = _Job(jid, jtype, needed, state)
            self._jobs[jid] = job
            if not ids:
                state.update({"active": False, "finished": True})
                return self._render(state)
            job.thread = threading.Thread(
                target=self._run, args=(job, jtype, ids, cfg), daemon=True
            )
            job.thread.start()
            return self._render(state)

    def stop(self, job_id: str = None):
        """Stop one job (by id) or, with no id, every active job."""
        with self._lock:
            for j in self._jobs.values():
                if j.state["active"] and (job_id is None or j.id == job_id):
                    j.stop.set()

    def _primary(self) -> "_Job | None":
        """The job status() (no id) describes for single-job/back-compat
        callers: the most-recently-started ACTIVE job, else the most recent
        job overall."""
        with self._lock:
            active = [j for j in self._jobs.values() if j.state["active"]]
            pool = active or list(self._jobs.values())
            if not pool:
                return None
            return max(pool, key=lambda j: j.state.get("started_at") or 0)

    def any_active(self) -> bool:
        with self._lock:
            return any(j.state["active"] for j in self._jobs.values())

    def active_types(self) -> set:
        with self._lock:
            return {j.type for j in self._jobs.values() if j.state["active"]}

    def _render(self, state: dict) -> dict:
        """Per-job snapshot with derived fields (ETA, rate, trimmed log)."""
        s = dict(state)
        s["log"] = list(s["log"][-30:])
        s["failed_ids"] = list(s["failed_ids"])
        s["model_counts"] = dict(s["model_counts"])
        # ETA from the cumulative average rate so far. Deliberately includes
        # time spent waiting on rate limits — an ETA that ignored the throttle
        # would read absurdly optimistic the moment a window fills.
        s["eta_seconds"] = None
        s["rate_per_min"] = None
        if s["active"] and s["done"] and s["started_at"]:
            elapsed = time.time() - s["started_at"]
            if elapsed > 0:
                rate = s["done"] / elapsed
                s["eta_seconds"] = round(max(0, s["total"] - s["done"]) / rate)
                s["rate_per_min"] = round(rate * 60, 1)
        # Which providers (if any) are currently sleeping on a rate-limit
        # window — only meaningful for the inference job that owns the provider.
        infer = RES_INFERENCE in set(s.get("resources") or [])
        s["rate_wait"] = ratelimit.waiting() if (s["active"] and infer) else {}
        return s

    def status(self, job_id: str = None) -> dict:
        """Job status. With job_id → that job (or idle if unknown). Without →
        the primary job's snapshot (back-compat single-job shape) PLUS a `jobs`
        list of every tracked job and a `job_resources` map so the UI can tell
        which job types would currently conflict."""
        with self._lock:
            if job_id is not None:
                j = self._jobs.get(job_id)
                return self._render(j.state if j else self._idle_state())
            primary = self._primary()
            base = self._render(primary.state if primary else self._idle_state())
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j.state.get("started_at") or 0, reverse=True)
            base["jobs"] = [self._render(j.state) for j in jobs]
            base["any_active"] = any(j.state["active"] for j in self._jobs.values())
            base["active_types"] = sorted(self.active_types())
            base["job_resources"] = {
                t: sorted(self._resources_for(t, {})) for t in JOB_TYPES
            }
            return base

    def reset(self, job_id: str = None):
        """Clear finished/aborted job(s) — one by id, or all non-active."""
        with self._lock:
            if job_id is not None:
                j = self._jobs.get(job_id)
                if j and not j.state["active"]:
                    self._jobs.pop(job_id, None)
                return
            for jid in [jid for jid, j in self._jobs.items()
                        if not j.state["active"]]:
                self._jobs.pop(jid, None)

    # ── worker ────────────────────────────────────────────────────────────────

    def _run(self, job, jtype, ids, cfg):
        idx = Indexer()  # private mutable catalog copy for this job
        is_infer = RES_INFERENCE in job.resources
        if is_infer:
            # Only one inference job runs at a time, so it's safe for it to own
            # the module-level cancel event for its lifetime.
            ratelimit.set_cancel_event(job.stop)
        try:
            if jtype == "vision":
                self._run_vision_parallel(job, idx, ids, cfg)
            elif jtype == "video_vision":
                # Same network-bound parallel path, but each item is a video:
                # sample keyframes → caption each → fold into one caption.
                # Bind the per-video keyframe count so the parallel path (which
                # calls compute(id, provider, model)) applies the configured value.
                import functools
                compute = functools.partial(
                    idx.compute_video_caption, frames=cfg.get("video_frames", 4))
                self._run_vision_parallel(job, idx, ids, cfg,
                                          compute=compute,
                                          label="video-vision")
            elif jtype == "embed":
                self._run_embed_batched(job, idx, ids, cfg)
            else:
                self._run_sequential(job, idx, jtype, ids, cfg)
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
                job.state["aborted"] = True
                job.state["error"] = str(e)
                job.state["log"].append(
                    {"kind": "fail", "id": None, "note": f"job crashed: {e}", "file": ""}
                )
        finally:
            if is_infer:
                ratelimit.set_cancel_event(threading.Event())  # neutral again
            with self._lock:
                job.state["active"] = False
                job.state["finished"] = True

    def _run_vision_parallel(self, job, idx, ids, cfg, compute=None, label="vision"):
        """
        Vision is network-bound, so caption N items concurrently. Captions are
        computed in worker threads (no shared mutation), then applied + saved on
        this thread in batches. Stop is honored between batches.

        `compute` is the per-item captioner — idx.compute_caption for photos,
        idx.compute_video_caption for videos; both take (id, provider, model),
        return (model_label, caption_json), and raise ratelimit.Cancelled on
        Stop. `label` only tags the progress log line.
        """
        compute = compute or idx.compute_caption
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
                if job.stop.is_set():
                    self._update(job, stopped=True)
                    break
                # A 9Router 429 means every pooled key/account for the model is
                # momentarily dry — but pools refill (per-minute windows, key
                # rotation), so a batch submitted during the cooldown would just
                # burn the consecutive-failure budget on guaranteed skips and
                # abort a job that could finish. Wait the cooldown out instead;
                # Stop stays responsive via the 1s ticks.
                if vp == "9router" and vm:
                    from vision import ninerouter_cooldowns
                    while not job.stop.is_set():
                        wait = ninerouter_cooldowns().get(vm, 0)
                        if wait <= 0:
                            break
                        time.sleep(min(1.0, wait))
                    if job.stop.is_set():
                        self._update(job, stopped=True)
                        break
                batch = ids[i:i + conc]
                i += len(batch)
                futures = {
                    ex.submit(compute, bid, vp, vm): bid for bid in batch
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
                        self._update(job, ok=1, done=1, model_used=vmodel,
                                     log=(icon, bid, f"{label}:{vmodel}", fname))
                    else:
                        consecutive += 1
                        self._update(job, fail=1, done=1, failed_id=bid, log=("fail", bid, errm, fname))
                idx._save_catalog()  # one write per batch, not per image
                # Gated on consecutive > 0 so this can only ever fire because
                # of real accumulated failures, never as a side effect of a
                # misconfigured max_fail (max_fail is clamped to >= 1 in
                # start(), but this is cheap defense-in-depth against a
                # 0-or-negative value ever aborting a fully-successful batch).
                if consecutive > 0 and consecutive >= max_fail:
                    self._update(job, aborted=True)
                    break

    def _run_embed_batched(self, job, idx, ids, cfg):
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
            if job.stop.is_set():
                self._update(job, stopped=True)
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
                if job.stop.is_set():
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
                    self._update(job, fail=1, done=1, failed_id=iid,
                                 log=("fail", iid, str(e), fname))
            if stopped_mid_chunk:
                self._update(job, stopped=True)
                break
            if not members:
                if consecutive >= max_fail:
                    self._update(job, aborted=True)
                    break
                continue

            try:
                vectors, model_name, source = get_embeddings_batch(
                    texts, force_provider=ep, model=em
                )
            except ratelimit.Cancelled:
                # Stop pressed during a rate-limit wait — the chunk never
                # reached the provider; its items stay pending, not failed.
                self._update(job, stopped=True)
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
                        job.state["substitution"] = {
                            **sub,
                            "suggested": resolve_9router_embed_id(
                                sub["served"], sub["requested"]),
                        }
                for iid, img_data, _ in members:
                    consecutive += 1
                    self._update(job, fail=1, done=1, failed_id=iid,
                                 log=("fail", iid, f"embedding failed — {reason}",
                                      img_data.get("filename", "")))
                if consecutive >= max_fail:
                    self._update(job, aborted=True)
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
                        self._update(job, fail=1, done=1, failed_id=iid,
                                     log=("fail", iid, "embedding failed for this item",
                                          img_data.get("filename", "")))
                    else:
                        good_members.append((iid, img_data, cj))
                        good_vectors.append(vec)
                members, vectors = good_members, good_vectors
                if not members:
                    if consecutive >= max_fail:
                        self._update(job, aborted=True)
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
                    self._update(job, fail=1, done=1, failed_id=iid,
                                 log=("fail", iid, f"store failed: {e}",
                                      img_data.get("filename", "")))
                if consecutive >= max_fail:
                    self._update(job, aborted=True)
                    break
                continue

            consecutive = 0
            icon = "cloud" if "gemini" in source.lower() else "ok"
            for iid, img_data, _ in members:
                self._update(job, ok=1, done=1,
                             log=(icon, iid, f"embed:{source}",
                                  img_data.get("filename", "")))

    def _run_sequential(self, job, idx, jtype, ids, cfg):
        """Embed / full / reanalyze: run one at a time (ChromaDB writes), but
        batch the images.json saves so full/reanalyze aren't O(n^2)."""
        max_fail = cfg["max_fail"]
        vp, vm = cfg["vision_provider"], cfg["vision_model"]
        ep, em = cfg["embed_provider"], cfg["embed_model"]
        csm = cfg.get("caption_source_model")
        video_frames = cfg.get("video_frames", 4)
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
                cfg["source_path"], idx.image_catalog["images"],
                video_dest=cfg.get("ingest_video_dest"),
                media=cfg.get("ingest_media", "both"))

        consecutive = 0
        dirty = False
        for k, img_id in enumerate(ids):
            if job.stop.is_set():
                self._update(job, stopped=True)
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
                elif jtype == "video_faces":
                    note = idx.video_faces_one(img_id, frames=video_frames)
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
                    self._update(job, skipped=1, done=1, log=("skip", img_id, note, fname))
                else:
                    consecutive = 0
                    icon = "cloud" if "gemini" in note.lower() else "ok"
                    self._update(job, ok=1, done=1, log=(icon, img_id, note, fname))
            except ratelimit.Cancelled:
                # Stop during a rate-limit wait: the item stays pending.
                self._update(job, stopped=True)
                break
            except Exception as e:
                consecutive += 1
                self._update(job, fail=1, done=1, failed_id=img_id,
                             log=("fail", img_id, str(e), fname))
                if consecutive >= max_fail:
                    self._update(job, aborted=True)
                    break
            if dirty and (k + 1) % SAVE_EVERY == 0:
                idx._save_catalog()
                dirty = False
        if dirty:
            idx._save_catalog()
        if ingest_session is not None:
            ingest_session.close()

    def _update(self, job, ok=0, fail=0, skipped=0, done=0, failed_id=None, log=None,
                stopped=False, aborted=False, model_used=None):
        with self._lock:
            st = job.state
            st["ok"] += ok
            st["fail"] += fail
            st["skipped"] += skipped
            st["done"] += done
            if model_used:
                # Per-model tally for the run summary — with 9Router the
                # gateway can substitute serving models mid-run, so one run
                # can legitimately produce captions from several models.
                mc = st["model_counts"]
                mc[model_used] = mc.get(model_used, 0) + 1
            if failed_id is not None:
                st["failed_ids"].append(failed_id)
            if log is not None:
                kind, img_id, note = log[:3]
                fname = log[3] if len(log) > 3 else ""
                st["log"].append(
                    {"kind": kind, "id": img_id, "note": note, "file": fname}
                )
            if stopped:
                st["stopped"] = True
            if aborted:
                st["aborted"] = True


def resources_for(jtype: str) -> set:
    """Public helper: the resources a job type would hold right now (factoring
    the current faces-during-embed setting). Lets the API/UI reason about
    conflicts without a JobManager instance."""
    return JobManager._resources_for(jtype, {})


# module-level singleton shared by the API
manager = JobManager()
