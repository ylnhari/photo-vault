import time
import pytest
from unittest.mock import patch, MagicMock

from jobs import JobManager


@pytest.fixture(autouse=True)
def _serial_vision(monkeypatch):
    # Pin vision concurrency to 1 so parallel-batch tests are deterministic while
    # still exercising the batched worker code path.
    monkeypatch.setattr("jobs.settings_mod.load", lambda: {"vision_concurrency": 1})
    yield


def _wait_idle(mgr, timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        if not mgr.status()["active"]:
            return mgr.status()
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def _fake_indexer(pending_ids, dispatch):
    fake = MagicMock()
    fake.get_vision_pending.return_value = [(i, {}) for i in pending_ids]
    fake.get_embed_pending.return_value = [(i, {}) for i in pending_ids]
    fake.get_missing.return_value = [(i, {}) for i in pending_ids]
    fake.get_missing_attributes.return_value = [(i, {}) for i in pending_ids]

    # Vision now goes through compute_caption (parallel) + record_caption.
    def _compute(img_id, *a, **k):
        dispatch(img_id)  # raises for the failure-path tests
        return ("m", '{"caption":"ok"}')
    fake.compute_caption.side_effect = _compute
    fake.record_caption = MagicMock()
    fake.vision_one.side_effect = dispatch
    fake.embed_one.side_effect = dispatch
    fake.index_one_full.side_effect = dispatch
    return fake


def test_start_with_no_pending_finishes_immediately():
    mgr = JobManager()
    with patch("jobs.Indexer", return_value=_fake_indexer([], lambda *a, **k: "ok")):
        s = mgr.start("vision")
    assert s["active"] is False
    assert s["finished"] is True
    assert s["total"] == 0


def test_runs_all_items_and_counts_ok():
    mgr = JobManager()
    fake = _fake_indexer(["a", "b", "c"], lambda img_id, **k: f"vision:m")
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision")
        s = _wait_idle(mgr)
    assert s["ok"] == 3
    assert s["fail"] == 0
    assert s["done"] == 3
    assert s["finished"] is True


def test_aborts_on_consecutive_failures():
    mgr = JobManager()
    def boom(img_id, **k):
        raise RuntimeError("vision down")
    fake = _fake_indexer(["a", "b", "c", "d", "e"], boom)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision", max_fail=3)
        s = _wait_idle(mgr)
    assert s["aborted"] is True
    assert s["fail"] == 3
    assert len(s["failed_ids"]) == 3


def test_consecutive_resets_on_success():
    mgr = JobManager()
    calls = {"n": 0}
    def flaky(img_id, **k):
        calls["n"] += 1
        if calls["n"] in (1, 2):
            raise RuntimeError("temp")
        return "vision:m"
    fake = _fake_indexer(["a", "b", "c", "d"], flaky)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision", max_fail=3)
        s = _wait_idle(mgr)
    assert s["aborted"] is False
    assert s["fail"] == 2
    assert s["ok"] == 2


def test_stop_aborts_mid_run():
    mgr = JobManager()
    def slow(img_id, **k):
        time.sleep(0.05)
        return "vision:m"
    fake = _fake_indexer([str(i) for i in range(50)], slow)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision")
        time.sleep(0.08)
        mgr.stop()
        s = _wait_idle(mgr)
    assert s["stopped"] is True
    assert s["done"] < 50


def test_cannot_start_two_jobs():
    import pytest
    mgr = JobManager()
    def slow(img_id, **k):
        time.sleep(0.05)
        return "ok"
    fake = _fake_indexer(["a", "b", "c"], slow)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision")
        # vision holds {catalog, inference}; embed holds {inference,...} —
        # they share the provider, so the second start is refused.
        with pytest.raises(RuntimeError, match="already running"):
            mgr.start("embed")
        _wait_idle(mgr)


def _wait_all_idle(mgr, timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        if not mgr.any_active():
            return
        time.sleep(0.02)
    raise AssertionError("jobs did not finish in time")


def test_disjoint_jobs_run_concurrently():
    """faces (local: FACE_DIR + faces collection) shares nothing with vision
    (catalog + provider), so the two must run at the same time."""
    mgr = JobManager()
    def slow(img_id, **k):
        time.sleep(0.05)
        return "ok"
    fake = _fake_indexer(["a", "b", "c"], slow)
    fake.get_faces_pending.return_value = [(i, {}) for i in ["x", "y", "z"]]
    fake.detect_faces_one.side_effect = lambda img_id, **k: "faces:1"
    with patch("jobs.Indexer", return_value=fake):
        v = mgr.start("vision")
        f = mgr.start("faces")            # must NOT raise — disjoint resources
        assert v["active"] is True and f["active"] is True
        # both show up as concurrently active
        st = mgr.status()
        assert set(st["active_types"]) == {"vision", "faces"}
        assert len(st["jobs"]) == 2
        _wait_all_idle(mgr)


def test_faces_conflicts_when_faces_during_embed_on():
    """With faces-during-embed ON, an embed job also owns the faces resource,
    so a standalone faces job must be refused while it runs."""
    import pytest
    mgr = JobManager()
    def slow(img_id, **k):
        time.sleep(0.05)
        return "ok"
    fake = _fake_indexer(["a", "b", "c"], slow)
    with patch("jobs.Indexer", return_value=fake), \
         patch("jobs.settings_mod.load",
               lambda: {"vision_concurrency": 1, "faces_during_embed": True}):
        mgr.start("embed")
        with pytest.raises(RuntimeError, match="already running"):
            mgr.start("faces")
        _wait_all_idle(mgr)


def test_embed_and_faces_concurrent_when_faces_during_embed_off():
    """With faces-during-embed OFF, embed drops the faces resource, so a
    GPU faces job can run alongside embedding — the user's target scenario."""
    mgr = JobManager()
    def slow(img_id, **k):
        time.sleep(0.05)
        return "ok"
    fake = _fake_indexer(["a", "b", "c"], slow)
    fake.get_faces_pending.return_value = [(i, {}) for i in ["x", "y"]]
    fake.detect_faces_one.side_effect = lambda img_id, **k: "faces:0"
    with patch("jobs.Indexer", return_value=fake), \
         patch("jobs.settings_mod.load",
               lambda: {"vision_concurrency": 1, "faces_during_embed": False}):
        mgr.start("embed")
        f = mgr.start("faces")            # allowed now
        assert f["active"] is True
        _wait_all_idle(mgr)


def test_stop_targets_single_job_by_id():
    """stop(job_id) halts only that job; a disjoint concurrent job keeps
    running."""
    mgr = JobManager()
    def slow(img_id, **k):
        time.sleep(0.05)
        return "ok"
    fake = _fake_indexer([str(i) for i in range(40)], slow)
    fake.get_faces_pending.return_value = [(str(i), {}) for i in range(40)]
    fake.detect_faces_one.side_effect = lambda img_id, **k: (time.sleep(0.05) or "faces:0")
    with patch("jobs.Indexer", return_value=fake):
        v = mgr.start("vision")
        f = mgr.start("faces")
        time.sleep(0.08)
        mgr.stop(v["id"])                 # stop only vision
        _wait_all_idle(mgr)
        vs = mgr.status(v["id"])
        fs = mgr.status(f["id"])
        assert vs["stopped"] is True
        assert fs["stopped"] is False


def test_reset_clears_finished_state():
    mgr = JobManager()
    fake = _fake_indexer(["a"], lambda img_id, **k: "ok")
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision")
        _wait_idle(mgr)
    mgr.reset()
    s = mgr.status()
    assert s["finished"] is False
    assert s["done"] == 0


def test_embed_job_batches_one_request(monkeypatch):
    """The embed job embeds the whole chunk in ONE get_embeddings_batch call
    and stores all vectors with ONE ChromaDB upsert."""
    monkeypatch.setattr(
        "jobs.settings_mod.load",
        lambda: {"vision_concurrency": 1, "faces_during_embed": False},
    )
    mgr = JobManager()
    imgs = {
        "a": {"path": "/a.jpg", "filename": "a.jpg", "caption_json": '{"caption":"one"}', "metadata": {}},
        "b": {"path": "/b.jpg", "filename": "b.jpg", "caption_json": '{"caption":"two"}', "metadata": {}},
    }
    fake = MagicMock()
    fake.get_embed_pending.return_value = list(imgs.items())
    fake.image_catalog = {"images": imgs}

    col = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = col

    with patch("jobs.Indexer", return_value=fake), \
         patch("embeddings.get_embeddings_batch",
               return_value=([[0.1], [0.2]], "test-model", "lm_studio")) as geb, \
         patch("db.client", return_value=client):
        mgr.start("embed")
        s = _wait_idle(mgr)

    assert s["ok"] == 2 and s["fail"] == 0
    assert geb.call_count == 1
    # embedding text is now the natural-language caption, not the raw JSON blob
    assert geb.call_args[0][0] == ["one", "two"]
    assert col.upsert.call_count == 1
    assert col.upsert.call_args[1]["ids"] == ["a", "b"]


def test_embed_job_bad_caption_fails_only_that_image(monkeypatch):
    monkeypatch.setattr(
        "jobs.settings_mod.load",
        lambda: {"vision_concurrency": 1, "faces_during_embed": False},
    )
    mgr = JobManager()
    imgs = {
        "good": {"path": "/g.jpg", "filename": "g.jpg", "caption_json": '{"caption":"ok"}', "metadata": {}},
        "bad": {"path": "/b.jpg", "filename": "b.jpg", "caption_json": "", "metadata": {}},
    }
    fake = MagicMock()
    fake.get_embed_pending.return_value = list(imgs.items())
    fake.image_catalog = {"images": imgs}
    col = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = col

    with patch("jobs.Indexer", return_value=fake), \
         patch("embeddings.get_embeddings_batch",
               return_value=([[0.1]], "test-model", "lm_studio")), \
         patch("db.client", return_value=client):
        mgr.start("embed")
        s = _wait_idle(mgr)

    assert s["ok"] == 1
    assert s["fail"] == 1
    assert s["failed_ids"] == ["bad"]


def test_embed_job_partial_gemini_batch_keeps_successful_items(monkeypatch):
    """One item in a batch coming back None (a per-item Gemini failure, see
    embeddings.get_embeddings_batch) must fail only that id, not the whole
    batch — the upsert should still go through for the successful ones."""
    monkeypatch.setattr(
        "jobs.settings_mod.load",
        lambda: {"vision_concurrency": 1, "faces_during_embed": False},
    )
    mgr = JobManager()
    imgs = {
        "a": {"path": "/a.jpg", "filename": "a.jpg", "caption_json": '{"caption":"one"}', "metadata": {}},
        "b": {"path": "/b.jpg", "filename": "b.jpg", "caption_json": '{"caption":"two"}', "metadata": {}},
    }
    fake = MagicMock()
    fake.get_embed_pending.return_value = list(imgs.items())
    fake.image_catalog = {"images": imgs}
    col = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = col

    with patch("jobs.Indexer", return_value=fake), \
         patch("embeddings.get_embeddings_batch",
               return_value=([None, [0.2]], "text-embedding-004", "gemini")), \
         patch("db.client", return_value=client):
        mgr.start("embed")
        s = _wait_idle(mgr)

    assert s["ok"] == 1 and s["fail"] == 1
    assert s["failed_ids"] == ["a"]
    col.upsert.assert_called_once()
    assert col.upsert.call_args[1]["ids"] == ["b"]
    assert col.upsert.call_args[1]["embeddings"] == [[0.2]]


def test_embed_job_stop_checked_inside_chunk_not_just_between_chunks(monkeypatch):
    """Item 12: Stop must be checked while resolving captions within a chunk
    (pure catalog/CPU work, no network yet), not only at the top of the outer
    while loop — otherwise clicking Stop partway through a 16-item chunk still
    waits for the rest of that chunk's captions to resolve AND the network
    embed call to fire before it takes effect."""
    monkeypatch.setattr(
        "jobs.settings_mod.load",
        lambda: {"vision_concurrency": 1, "faces_during_embed": False},
    )
    mgr = JobManager()
    ids = [str(i) for i in range(10)]
    imgs = {
        i: {"path": f"/{i}.jpg", "filename": f"{i}.jpg",
            "caption_json": '{"caption":"x"}', "metadata": {}}
        for i in ids
    }
    fake = MagicMock()
    fake.get_embed_pending.return_value = list(imgs.items())
    fake.image_catalog = {"images": imgs}

    calls = {"n": 0}

    def fake_resolve(img_data, csm):
        calls["n"] += 1
        if calls["n"] == 3:
            mgr.stop()
        return img_data["caption_json"]

    with patch("jobs.Indexer", return_value=fake),          patch("jobs.resolve_caption_json", side_effect=fake_resolve),          patch("embeddings.get_embeddings_batch") as geb:
        mgr.start("embed")
        s = _wait_idle(mgr)

    assert s["stopped"] is True
    geb.assert_not_called()  # broke out before the batch's network embed call
    assert s["done"] < len(ids)
    assert calls["n"] < len(ids)  # didn't resolve the rest of the chunk either


# ── skipped bucket (item 5: missing-file consistency) ───────────────────────

def test_sequential_job_tracks_skipped_separately_from_ok_and_fail():
    mgr = JobManager()
    fake = _fake_indexer(["a", "b", "c"], lambda *a, **k: "ok")
    fake.get_faces_pending.return_value = [("a", {}), ("b", {}), ("c", {})]
    # a: skipped (missing file), b: real success, c: real failure
    def dispatch(img_id, **k):
        if img_id == "a":
            return "faces:skipped (file missing)"
        if img_id == "c":
            raise RuntimeError("boom")
        return "faces:2"
    fake.detect_faces_one = MagicMock(side_effect=dispatch)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("faces", max_fail=5)
        s = _wait_idle(mgr)
    assert s["skipped"] == 1
    assert s["ok"] == 1
    assert s["fail"] == 1
    assert s["done"] == 3


def test_skipped_items_do_not_trip_consecutive_failure_abort():
    """A run of skips must not itself abort the job, and must not count
    towards the consecutive-failure streak that a REAL failure right after it
    would extend."""
    mgr = JobManager()
    fake = _fake_indexer(["a", "b", "c", "d"], lambda *a, **k: "ok")
    fake.get_thumbs_pending.return_value = [(i, {}) for i in ["a", "b", "c", "d"]]

    def dispatch(img_id, **k):
        if img_id == "d":
            raise RuntimeError("real failure")
        return "thumb:skipped (file missing)"
    fake.thumb_one = MagicMock(side_effect=dispatch)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("thumbs", max_fail=2)
        s = _wait_idle(mgr)
    # 3 skips then 1 real failure: consecutive failures should be 1, not 4 —
    # so max_fail=2 must NOT trip the abort.
    assert s["skipped"] == 3
    assert s["fail"] == 1
    assert s["aborted"] is False


# ── max_fail sanitization (item 11) ─────────────────────────────────────────

def test_max_fail_non_positive_clamped_to_one():
    mgr = JobManager()
    def boom(img_id, **k):
        raise RuntimeError("down")
    fake = _fake_indexer(["a", "b", "c"], boom)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision", max_fail=0)
        s = _wait_idle(mgr)
    assert s["aborted"] is True
    assert s["fail"] == 1  # aborted after the very first failure, not zero


# ── vision_model_label derived internally (item 13) ─────────────────────────

def test_vision_model_label_derived_from_provider_and_model_not_trusted_input():
    """A caller-supplied vision_model_label that disagrees with
    vision_provider/vision_model must be ignored — the label is always
    recomputed from vision_provider/vision_model in one place."""
    mgr = JobManager()
    captured_cfg = {}
    real_pending_ids = mgr._pending_ids
    def spy_pending_ids(jtype, cfg):
        captured_cfg.update(cfg)
        return real_pending_ids(jtype, cfg)
    mgr._pending_ids = spy_pending_ids
    fake = _fake_indexer([], lambda *a, **k: "ok")
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision", vision_provider="lm_studio", vision_model="qwen2-vl",
                  vision_model_label="totally-wrong-label")
    assert captured_cfg["vision_model_label"] == "lm_studio:qwen2-vl"


def test_vision_model_label_none_in_auto_mode_even_if_caller_passes_one():
    mgr = JobManager()
    captured_cfg = {}
    real_pending_ids = mgr._pending_ids
    def spy_pending_ids(jtype, cfg):
        captured_cfg.update(cfg)
        return real_pending_ids(jtype, cfg)
    mgr._pending_ids = spy_pending_ids
    fake = _fake_indexer([], lambda *a, **k: "ok")
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision", vision_provider="auto", vision_model=None,
                  vision_model_label="stale-label-from-a-previous-run")
    assert captured_cfg["vision_model_label"] is None


# ── worker-thread crash visibility (item 14) ────────────────────────────────

def test_worker_thread_crash_is_recorded_not_swallowed():
    """A bug that escapes the per-item try/except inside a runner (as opposed
    to a normal per-image failure) must still leave the job in a finished,
    non-active state with the crash visible in status/log — not stuck
    'active' forever with the exception only printed to stderr."""
    mgr = JobManager()
    fake = _fake_indexer(["a"], lambda *a, **k: "ok")
    with patch("jobs.Indexer", return_value=fake), \
         patch.object(mgr, "_run_vision_parallel", side_effect=RuntimeError("unexpected bug")):
        mgr.start("vision")
        s = _wait_idle(mgr)
    assert s["active"] is False
    assert s["finished"] is True
    assert s["aborted"] is True
    assert any("unexpected bug" in entry["note"] for entry in s["log"])
    # Additive "error" field: precise, structured crash reason (distinct from
    # "aborted", which also fires for the ordinary consecutive-failure case)
    # so the UI can tell "job code crashed" apart from "too many bad photos".
    assert s["error"] is not None
    assert "unexpected bug" in s["error"]


def test_idle_state_error_field_defaults_to_none():
    mgr = JobManager()
    assert mgr.status()["error"] is None


def test_vision_job_tallies_models_used():
    """model_counts records how many captions each ACTUAL model produced —
    the end-of-run summary for 9Router runs, where the gateway can substitute
    serving models mid-run."""
    mgr = JobManager()
    fake = _fake_indexer(["a", "b", "c"], lambda *a, **k: None)
    served = iter(["9router:gc/gemini-2.5-flash-lite",
                   "9router:gemini-3.1-flash-lite",
                   "9router:gc/gemini-2.5-flash-lite"])
    fake.compute_caption.side_effect = lambda *a, **k: (next(served), '{"caption":"ok"}')
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision")
        s = _wait_idle(mgr)
    assert s["ok"] == 3
    assert s["model_counts"] == {
        "9router:gc/gemini-2.5-flash-lite": 2,
        "9router:gemini-3.1-flash-lite": 1,
    }


def test_vision_9router_waits_out_cooldown_instead_of_failing_batches():
    """A 9Router post-429 cooldown must pause the job (pools refill), not let
    every batch insta-fail its way into the consecutive-failure abort."""
    mgr = JobManager()
    fake = _fake_indexer(["a", "b", "c"], lambda *a, **k: None)
    fake.compute_caption.side_effect = lambda *a, **k: ("9router:m", '{"caption":"ok"}')
    cooldown = {"9router:m-model": 0.3}
    calls = []
    def fake_cooldowns():
        calls.append(1)
        # cooldown active on the first check, expired afterwards
        if len(calls) == 1:
            return dict(cooldown)
        return {}
    with patch("jobs.Indexer", return_value=fake), \
         patch("vision.ninerouter_cooldowns", side_effect=fake_cooldowns):
        mgr.start("vision", vision_provider="9router",
                  vision_model="9router:m-model")
        s = _wait_idle(mgr)
    assert s["ok"] == 3 and s["fail"] == 0
    assert s["aborted"] is False
    assert len(calls) >= 2  # actually consulted the cooldown table


def test_vision_9router_stop_during_cooldown_wait_stops_cleanly():
    mgr = JobManager()
    fake = _fake_indexer(["a", "b"], lambda *a, **k: None)
    fake.compute_caption.side_effect = lambda *a, **k: ("9router:m", '{"caption":"ok"}')
    with patch("jobs.Indexer", return_value=fake), \
         patch("vision.ninerouter_cooldowns", return_value={"9router:m-model": 60.0}):
        mgr.start("vision", vision_provider="9router",
                  vision_model="9router:m-model")
        time.sleep(0.1)          # let the worker enter the cooldown wait
        mgr.stop()
        s = _wait_idle(mgr)
    assert s["stopped"] is True
    assert s["fail"] == 0 and s["done"] == 0  # nothing burned during the wait


def test_model_counts_reset_between_jobs():
    mgr = JobManager()
    fake = _fake_indexer(["a"], lambda *a, **k: None)
    fake.compute_caption.side_effect = lambda *a, **k: ("m1", '{"caption":"ok"}')
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision")
        s = _wait_idle(mgr)
    assert s["model_counts"] == {"m1": 1}
    mgr.reset()
    assert mgr.status()["model_counts"] == {}


def test_embed_job_records_substitution_for_recovery():
    """A 9Router embed substitution aborts the run AND leaves a structured
    {requested, served, suggested} record in job state so the UI can offer
    'switch to the served model & re-run'."""
    mgr = JobManager()
    fake = _fake_indexer(["a", "b"], lambda *a, **k: None)
    # An explicit embed_model routes through the model-aware pending query.
    fake.get_embed_pending_for_model.return_value = [("a", {}), ("b", {})]
    fake.image_catalog = {"images": {
        "a": {"filename": "a.jpg", "caption_json": '{"caption":"x"}'},
        "b": {"filename": "b.jpg", "caption_json": '{"caption":"y"}'},
    }}
    with patch("jobs.Indexer", return_value=fake), \
         patch("jobs.settings_mod.load", return_value={"faces_during_embed": False}), \
         patch("jobs.resolve_caption_json", side_effect=lambda d, csm: d["caption_json"]), \
         patch("embeddings.get_embeddings_batch", return_value=(None, "", "error")), \
         patch("embeddings.last_embed_error",
               return_value="9router: substituted embedding model"), \
         patch("embeddings.last_substitution",
               return_value={"requested": "gemini/gemini-embedding-001",
                             "served": "gemini-embedding-2-preview"}), \
         patch("embeddings.resolve_9router_embed_id",
               return_value="gemini/gemini-embedding-2-preview"):
        mgr.start("embed", embed_provider="9router",
                  embed_model="gemini/gemini-embedding-001", max_fail=2)
        s = _wait_idle(mgr)
    assert s["aborted"] is True
    assert s["substitution"] == {
        "requested": "gemini/gemini-embedding-001",
        "served": "gemini-embedding-2-preview",
        "suggested": "gemini/gemini-embedding-2-preview",
    }
    assert any("substituted" in (l["note"] or "") for l in s["log"])
    mgr.reset()
    assert mgr.status()["substitution"] is None


def _inject_active_job(mgr, jtype="vision", resources=("inference",), **state_fields):
    """Install a synthetic active job into the manager (no worker thread) so
    status()/_render can be exercised on a known state."""
    from jobs import _Job
    st = mgr._idle_state()
    st.update({"active": True, "type": jtype, "id": "test-1",
               "resources": list(resources), **state_fields})
    j = _Job("test-1", jtype, set(resources), st)
    with mgr._lock:
        mgr._jobs[j.id] = j
    return j


def test_status_reports_eta_and_rate_while_active():
    mgr = JobManager()
    _inject_active_job(mgr, total=10, done=5, started_at=time.time() - 50)
    s = mgr.status()
    # 5 done in 50s → 0.1/s → 5 remaining ≈ 50s, 6/min
    assert s["eta_seconds"] == pytest.approx(50, abs=3)
    assert s["rate_per_min"] == pytest.approx(6.0, abs=0.5)


def test_status_eta_none_when_idle_or_nothing_done():
    mgr = JobManager()
    s = mgr.status()
    assert s["eta_seconds"] is None
    assert s["rate_per_min"] is None
    assert s["rate_wait"] == {}
    _inject_active_job(mgr, total=10, done=0, started_at=time.time())
    s = mgr.status()
    assert s["eta_seconds"] is None


def test_rate_limit_cancel_stops_sequential_job_without_failures():
    """ratelimit.Cancelled (Stop pressed during a rate-limit sleep) must end
    the job as 'stopped' — never counted as a per-image failure."""
    import ratelimit
    mgr = JobManager()

    def boom(img_id, **k):
        raise ratelimit.Cancelled("stopped while waiting on gemini rate limit (rpm)")
    fake = _fake_indexer(["a", "b", "c"], boom)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("full")
        s = _wait_idle(mgr)
    assert s["stopped"] is True
    assert s["fail"] == 0
    assert s["done"] == 0
    assert s["failed_ids"] == []


def test_rate_limit_cancel_in_vision_batch_leaves_items_pending():
    """In the parallel vision runner a Cancelled item is neither ok nor
    failed — it simply stays pending for the next run."""
    import ratelimit
    mgr = JobManager()

    def boom(img_id, **k):
        raise ratelimit.Cancelled("stopped while waiting on gemini rate limit (rpm)")
    fake = _fake_indexer(["a", "b"], boom)
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("vision")
        s = _wait_idle(mgr)
    assert s["fail"] == 0
    assert s["ok"] == 0
    assert s["done"] == 0
    assert s["aborted"] is False


# ── video jobs ────────────────────────────────────────────────────────────────

def test_video_vision_runs_through_parallel_path_with_video_compute():
    """video_vision selects video items and captions them via
    compute_video_caption (not the still-image compute_caption)."""
    mgr = JobManager()
    fake = MagicMock()
    fake.get_video_vision_pending.return_value = [("v1", {}), ("v2", {})]
    calls = []
    def _vcompute(img_id, *a, **k):
        calls.append(img_id)
        return ("gemini:gemma", '{"caption":"a clip"}')
    fake.compute_video_caption.side_effect = _vcompute
    fake.record_caption = MagicMock()
    # If the job ever fell back to the photo path this would be hit instead:
    fake.compute_caption.side_effect = AssertionError("used photo compute for a video")
    with patch("jobs.Indexer", return_value=fake), \
         patch("jobs.settings_mod.load", lambda: {"vision_concurrency": 1}):
        mgr.start("video_vision", vision_provider="9router", vision_model="gemini/gemma")
        s = _wait_idle(mgr)
    assert s["ok"] == 2 and s["fail"] == 0
    assert set(calls) == {"v1", "v2"}
    assert fake.record_caption.call_count == 2


def test_video_faces_runs_sequentially():
    mgr = JobManager()
    fake = MagicMock()
    fake.get_video_faces_pending.return_value = [("v1", {}), ("v2", {})]
    fake.video_faces_one.side_effect = lambda img_id, **k: f"video-faces:1"
    with patch("jobs.Indexer", return_value=fake):
        mgr.start("video_faces")
        s = _wait_idle(mgr)
    assert s["ok"] == 2 and s["done"] == 2
    assert fake.video_faces_one.call_count == 2


def test_video_frames_setting_flows_to_both_video_jobs():
    """The configured keyframes-per-video reaches compute_video_caption
    (video_vision) and video_faces_one (video_faces), clamped to bounds."""
    # video_vision: settings say 7 → each compute call gets frames=7
    mgr = JobManager()
    fake = MagicMock()
    fake.get_video_vision_pending.return_value = [("v1", {})]
    seen = {}
    def _vcompute(img_id, *a, frames=4, **k):
        seen["vision"] = frames
        return ("gemini:gemma", '{"caption":"a clip"}')
    fake.compute_video_caption.side_effect = _vcompute
    with patch("jobs.Indexer", return_value=fake), \
         patch("jobs.settings_mod.load", lambda: {"vision_concurrency": 1, "video_frames": 7}):
        mgr.start("video_vision", vision_provider="9router", vision_model="gemini/gemma")
        _wait_idle(mgr)
    assert seen["vision"] == 7

    # video_faces: an out-of-range value is clamped (99 → 12)
    mgr = JobManager()
    fake = MagicMock()
    fake.get_video_faces_pending.return_value = [("v1", {})]
    def _vfaces(img_id, frames=4, **k):
        seen["faces"] = frames
        return "video-faces:1"
    fake.video_faces_one.side_effect = _vfaces
    with patch("jobs.Indexer", return_value=fake), \
         patch("jobs.settings_mod.load", lambda: {"faces_during_embed": True, "video_frames": 99}):
        mgr.start("video_faces")
        _wait_idle(mgr)
    assert seen["faces"] == 12  # clamped to VIDEO_FRAMES_MAX
