"""ratelimit: sliding-window request throttling (RPS/RPM/RPH/RPD) with a
stop-event escape hatch. Time is faked — no test actually sleeps."""
import threading

import pytest

import ratelimit


class FakeClock:
    """Stands in for ratelimit's `time` module: sleep() just advances the
    clock, so waits are instant and deterministic."""
    def __init__(self, start=1_000.0):
        self.now = start
        self.slept = 0.0
        self.on_sleep = None  # optional hook, called before advancing

    def time(self):
        return self.now

    def sleep(self, seconds):
        if self.on_sleep:
            self.on_sleep()
        self.slept += seconds
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(ratelimit, "time", c)
    ratelimit.reset()
    ratelimit.set_cancel_event(None)
    yield c
    ratelimit.reset()


def _limits(monkeypatch, **caps):
    lims = {k: caps.get(k, 0) for k in ratelimit.WINDOWS}
    monkeypatch.setattr(ratelimit, "_limits_for", lambda provider: dict(lims))


def test_unlimited_never_waits(clock, monkeypatch):
    _limits(monkeypatch)  # all 0
    for _ in range(100):
        ratelimit.acquire("gemini")
    assert clock.slept == 0


def test_rpm_window_blocks_then_frees(clock, monkeypatch):
    _limits(monkeypatch, rpm=2)
    ratelimit.acquire("gemini")
    ratelimit.acquire("gemini")
    assert clock.slept == 0
    ratelimit.acquire("gemini")  # third within the minute must wait ~60s
    assert clock.slept == pytest.approx(60, abs=1)


def test_rps_paces_consecutive_calls(clock, monkeypatch):
    _limits(monkeypatch, rps=1)
    ratelimit.acquire("gemini")
    ratelimit.acquire("gemini")
    assert clock.slept == pytest.approx(1, abs=0.1)


def test_tightest_window_wins(clock, monkeypatch):
    # rpm allows 10 but rph only 2 — the third call waits on the HOUR window.
    _limits(monkeypatch, rpm=10, rph=2)
    ratelimit.acquire("gemini")
    ratelimit.acquire("gemini")
    ratelimit.acquire("gemini")
    assert clock.slept == pytest.approx(3600, rel=0.01)


def test_providers_are_independent(clock, monkeypatch):
    _limits(monkeypatch, rpm=1)
    ratelimit.acquire("gemini")
    ratelimit.acquire("lm_studio")  # different provider — separate history
    assert clock.slept == 0


def test_cancel_event_aborts_wait(clock, monkeypatch):
    _limits(monkeypatch, rpm=1)
    evt = threading.Event()
    ratelimit.set_cancel_event(evt)
    ratelimit.acquire("gemini")
    evt.set()
    with pytest.raises(ratelimit.Cancelled):
        ratelimit.acquire("gemini")
    # the aborted wait must not leave a stale "waiting" marker behind
    assert ratelimit.waiting() == {}


def test_waiting_reports_provider_window_and_seconds(clock, monkeypatch):
    _limits(monkeypatch, rpm=1)
    seen = {}
    clock.on_sleep = lambda: seen.update(ratelimit.waiting())
    ratelimit.acquire("gemini")
    ratelimit.acquire("gemini")  # blocks; hook samples waiting() mid-sleep
    assert "gemini" in seen
    assert seen["gemini"]["limit"] == "rpm"
    assert 0 < seen["gemini"]["seconds"] <= 60
    assert ratelimit.waiting() == {}  # cleared once the slot was taken


def test_limits_for_reads_settings_and_sanitizes(monkeypatch):
    import settings
    monkeypatch.setattr(settings, "load", lambda: {
        "rate_limits": {"gemini": {"rpm": "15", "rpd": None, "rps": -3}}
    })
    lims = ratelimit._limits_for("gemini")
    assert lims == {"rps": 0, "rpm": 15, "rph": 0, "rpd": 0}
    # provider with no config at all → fully unlimited
    assert ratelimit._limits_for("lm_studio") == {"rps": 0, "rpm": 0, "rph": 0, "rpd": 0}
