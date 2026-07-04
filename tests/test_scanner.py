from scanner import _gps_to_decimal, _ratio_to_float


class _Ratio:
    """Mimic exifread's Ratio (num/den)."""
    def __init__(self, num, den=1):
        self.num = num
        self.den = den


def test_ratio_to_float_handles_rational():
    assert _ratio_to_float(_Ratio(1, 2)) == 0.5
    assert _ratio_to_float(_Ratio(50, 1)) == 50.0


def test_ratio_to_float_handles_plain_number():
    assert _ratio_to_float(7) == 7.0


def test_gps_north_east_positive():
    # 37°48'30" N → 37.808333
    lat = _gps_to_decimal([_Ratio(37), _Ratio(48), _Ratio(30)], "N")
    assert abs(lat - 37.808333) < 1e-5


def test_gps_south_west_negative():
    lon = _gps_to_decimal([_Ratio(122), _Ratio(25), _Ratio(0)], "W")
    assert lon < 0
    assert abs(lon - (-122.416667)) < 1e-5


def test_gps_accepts_object_with_values_attr():
    class _Tag:
        values = [_Ratio(10), _Ratio(0), _Ratio(0)]
    assert _gps_to_decimal(_Tag(), "N") == 10.0


def test_gps_bad_input_returns_none():
    assert _gps_to_decimal([], "N") is None
    assert _gps_to_decimal(None, "N") is None


def test_in_place_edit_retires_old_uid(tmp_path):
    """Editing a file in place (same path, new bytes) must replace the old
    catalog entry, not leave a stale duplicate that never shows as orphaned."""
    from scanner import scan_directory
    import catalog_db
    import os, time
    root = tmp_path / "photos"
    root.mkdir()
    f = root / "a.jpg"
    f.write_bytes(b"original-bytes-1")
    out = str(tmp_path / "catalog.db")
    scan_directory(str(root), out)
    # rewrite with different content (and nudge mtime so the sig changes)
    f.write_bytes(b"edited-bytes-22222")
    os.utime(f, (time.time() + 5, time.time() + 5))
    scan_directory(str(root), out)
    images = catalog_db.load_all(out)["images"]
    assert len(images) == 1, f"stale duplicate left behind: {list(images)}"
