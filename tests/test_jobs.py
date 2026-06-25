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
