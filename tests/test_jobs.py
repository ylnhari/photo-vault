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
        with pytest.raises(RuntimeError, match="already running"):
            mgr.start("embed")
        _wait_idle(mgr)


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


def test_status_reports_eta_and_rate_while_active():
    mgr = JobManager()
    with mgr._lock:
        mgr._state.update({"active": True, "total": 10, "done": 5,
                           "started_at": time.time() - 50})
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
    with mgr._lock:
        mgr._state.update({"active": True, "total": 10, "done": 0,
                           "started_at": time.time()})
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
