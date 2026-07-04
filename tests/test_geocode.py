import math
import pytest
from unittest.mock import patch, MagicMock


def _reset_geocode_state(monkeypatch):
    """geocode.py caches a lazy singleton + a lookup cache at module scope;
    reset both so tests don't leak state into each other."""
    import geocode
    monkeypatch.setattr(geocode, "_geocoder", None)
    monkeypatch.setattr(geocode, "_cache", {})


def test_place_for_rejects_out_of_range_latitude(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    assert geocode.place_for(200.0, 10.0) is None


def test_place_for_rejects_out_of_range_longitude(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    assert geocode.place_for(10.0, -300.0) is None


def test_place_for_rejects_nan(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    assert geocode.place_for(float("nan"), 10.0) is None


def test_place_for_rejects_infinity(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    assert geocode.place_for(float("inf"), 10.0) is None
    assert geocode.place_for(10.0, float("-inf")) is None


def test_place_for_rejects_non_numeric(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    assert geocode.place_for("not-a-number", 10.0) is None
    assert geocode.place_for(None, 10.0) is None


def test_place_for_accepts_valid_coordinates_and_uses_geocoder(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    fake_geocoder = MagicMock()
    fake_geocoder.query.return_value = [{"name": "Sanvordem", "admin1": "Goa"}]
    monkeypatch.setattr(geocode, "_get_geocoder", lambda: fake_geocoder)

    result = geocode.place_for(15.35, 74.07)
    assert result == "Sanvordem, Goa"
    fake_geocoder.query.assert_called_once()


def test_place_for_caches_by_rounded_key(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    fake_geocoder = MagicMock()
    fake_geocoder.query.return_value = [{"name": "City", "admin1": "Region"}]
    monkeypatch.setattr(geocode, "_get_geocoder", lambda: fake_geocoder)

    geocode.place_for(15.351, 74.071)
    geocode.place_for(15.352, 74.069)  # rounds to the same (15.35, 74.07) key
    assert fake_geocoder.query.call_count == 1


def test_place_for_boundary_values_are_valid(monkeypatch):
    import geocode
    _reset_geocode_state(monkeypatch)
    fake_geocoder = MagicMock()
    fake_geocoder.query.return_value = [{"name": "Edge", "admin1": ""}]
    monkeypatch.setattr(geocode, "_get_geocoder", lambda: fake_geocoder)
    assert geocode.place_for(90.0, 180.0) == "Edge"
    assert geocode.place_for(-90.0, -180.0) == "Edge"


def test_get_geocoder_builds_only_once_across_concurrent_calls(monkeypatch):
    """Regression for the missing lock: concurrent callers racing on the
    first call must not each construct a separate (expensive) RGeocoder."""
    import threading
    import time
    import geocode
    _reset_geocode_state(monkeypatch)

    build_count = {"n": 0}
    start_event = threading.Event()

    class FakeGeocoder:
        pass

    def fake_rgeocoder(mode, verbose):
        build_count["n"] += 1
        time.sleep(0.05)  # widen the window so an unlocked race would show up
        return FakeGeocoder()

    monkeypatch.setattr(geocode.rg, "RGeocoder", fake_rgeocoder)

    results = []

    def worker():
        start_event.wait(timeout=5)
        results.append(geocode._get_geocoder())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    start_event.set()
    for t in threads:
        t.join(timeout=5)

    assert build_count["n"] == 1
    assert len({id(r) for r in results}) == 1
