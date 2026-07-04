"""Offline reverse geocoding (GPS -> place name), no network calls.

Uses reverse_geocoder's bundled GeoNames cities dataset (~30k populated places
worldwide) with an in-process KD-tree lookup. Coarser than a live geocoding API
(nearest city, not street-level) but has no external dependency, no rate limit,
and never sends coordinates anywhere.
"""
import reverse_geocoder as rg

_geocoder = None
_cache: dict[tuple[float, float], str | None] = {}


def _get_geocoder():
    global _geocoder
    if _geocoder is None:
        # mode=1 = single-threaded; mode=2 (default) spawns worker processes,
        # which deadlocks under Windows multiprocessing when called from a
        # non-`__main__` module (our case, called from the job worker thread).
        _geocoder = rg.RGeocoder(mode=1, verbose=False)
    return _geocoder


def place_for(lat: float, lon: float) -> str | None:
    """'City, Region' label for a coordinate, e.g. 'Sanvordem, Goa'.
    Rounded to 2 decimals (~1km) so nearby photos share one cache entry and one
    filter value instead of a near-duplicate per GPS fix."""
    key = (round(lat, 2), round(lon, 2))
    if key not in _cache:
        result = _get_geocoder().query([key])[0]
        name = result.get("name") or ""
        admin1 = result.get("admin1") or ""
        _cache[key] = ", ".join(p for p in (name, admin1) if p) or None
    return _cache[key]
