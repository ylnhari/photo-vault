"""Offline reverse geocoding (GPS -> place name), no network calls.

Uses reverse_geocoder's bundled GeoNames cities dataset (~30k populated places
worldwide) with an in-process KD-tree lookup. Coarser than a live geocoding API
(nearest city, not street-level) but has no external dependency, no rate limit,
and never sends coordinates anywhere.
"""
import math
import threading

import reverse_geocoder as rg

_geocoder = None
_geocoder_lock = threading.Lock()
_cache: dict[tuple[float, float], str | None] = {}
_cache_lock = threading.Lock()


def _get_geocoder():
    global _geocoder
    if _geocoder is None:
        with _geocoder_lock:
            # Double-checked locking: without this, concurrent worker threads
            # (vision_concurrency defaults to 4) racing on the first geocode
            # call could each build the ~30k-place index redundantly.
            if _geocoder is None:
                # mode=1 = single-threaded; mode=2 (default) spawns worker
                # processes, which deadlocks under Windows multiprocessing
                # when called from a non-`__main__` module (our case, called
                # from the job worker thread).
                _geocoder = rg.RGeocoder(mode=1, verbose=False)
    return _geocoder


def place_for(lat: float, lon: float) -> str | None:
    """'City, Region' label for a coordinate, e.g. 'Sanvordem, Goa'.
    Rounded to 2 decimals (~1km) so nearby photos share one cache entry and one
    filter value instead of a near-duplicate per GPS fix.

    Returns None for invalid input (non-finite, non-numeric, or out of the
    valid lat/lon range) instead of passing corrupted GPS through to the
    geocoder, which would otherwise happily reverse-geocode garbage to a
    plausible-but-bogus place name."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return None
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return None

    key = (round(lat_f, 2), round(lon_f, 2))
    with _cache_lock:
        # Check-then-populate under the same lock as the cache read/write so
        # concurrent callers can't both miss the cache and each pay the cost
        # of building/querying the geocoder for the same key.
        if key not in _cache:
            result = _get_geocoder().query([key])[0]
            name = result.get("name") or ""
            admin1 = result.get("admin1") or ""
            _cache[key] = ", ".join(p for p in (name, admin1) if p) or None
        return _cache[key]
