import os
import sqlite3
import pytest

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


def test_gps_out_of_range_latitude_returns_none():
    # 200 degrees is not a valid latitude — malformed EXIF must not produce
    # a garbage coordinate that flows into geocoding.
    lat = _gps_to_decimal([_Ratio(200), _Ratio(0), _Ratio(0)], "N")
    assert lat is None


def test_gps_out_of_range_longitude_returns_none():
    lon = _gps_to_decimal([_Ratio(300), _Ratio(0), _Ratio(0)], "E")
    assert lon is None


def test_gps_boundary_values_are_valid():
    # Exactly at the boundary must still be accepted.
    lat = _gps_to_decimal([_Ratio(90), _Ratio(0), _Ratio(0)], "N")
    assert lat == 90.0
    lon = _gps_to_decimal([_Ratio(180), _Ratio(0), _Ratio(0)], "W")
    assert lon == -180.0


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


def test_scan_aborts_without_wiping_catalog_on_load_failure(tmp_path, monkeypatch):
    """The most important regression to lock in: a transient catalog-load
    failure (locked DB, corrupted row, transient I/O) must abort the scan
    rather than being silently treated as 'empty catalog'. scan_directory's
    checkpoint save is a full sync that deletes every row absent from the
    in-memory dict — if the failure were swallowed into {}, the very next
    save would wipe the whole catalog."""
    import catalog_db
    from scanner import scan_directory

    root = tmp_path / "photos"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"new-file")
    out = str(tmp_path / "catalog.db")

    # Seed the catalog with a pre-existing row that must survive the failed scan.
    catalog_db.save_all(out, {"existing-uid": {"path": "/somewhere/old.jpg"}}, {})

    def boom(_db_path):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as m:
        m.setattr(catalog_db, "load_all", boom)
        with pytest.raises(sqlite3.OperationalError):
            scan_directory(str(root), out)

    # Real load_all is restored now — the pre-existing row must be untouched
    # and no destructive save should have run.
    data = catalog_db.load_all(out)
    assert "existing-uid" in data["images"]
    assert data["images"]["existing-uid"]["path"] == "/somewhere/old.jpg"


def test_scan_handles_symlink_cycle_without_hanging(tmp_path):
    """os.walk(followlinks=True) would traverse a self-referential directory
    symlink/junction forever without cycle detection — the scan must
    terminate and must not re-process files under the cyclic subtree
    infinitely."""
    from scanner import scan_directory

    root = tmp_path / "photos"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"hello")
    try:
        os.symlink(str(root), str(root / "loop"), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported in this environment")

    out = str(tmp_path / "catalog.db")
    summary = scan_directory(str(root), out)
    # Only the one real file should be catalogued — the cycle must not cause
    # runaway re-discovery (this call must also simply return promptly;
    # a regression here would hang the whole test suite).
    assert summary["added"] == 1
    assert summary["total"] == 1


def test_duplicate_content_path_is_stable_across_rescans(tmp_path):
    """Two byte-identical files share one content uid. The catalog can only
    track one path per uid (a documented data-model limitation) — but which
    path it tracks must be deterministic (lexicographically first) instead
    of flip-flopping with os.walk's visitation order on every rescan."""
    from scanner import scan_directory
    import catalog_db

    root = tmp_path / "photos"
    root.mkdir()
    a = root / "a_copy.jpg"
    b = root / "b_copy.jpg"
    a.write_bytes(b"identical-bytes")
    b.write_bytes(b"identical-bytes")
    out = str(tmp_path / "catalog.db")

    scan_directory(str(root), out)
    images = catalog_db.load_all(out)["images"]
    assert len(images) == 1
    tracked_path = next(iter(images.values()))["path"]
    assert tracked_path == str(a)  # "a_copy.jpg" sorts before "b_copy.jpg"

    # Repeated rescans must not flip the tracked path.
    for _ in range(3):
        scan_directory(str(root), out)
        images = catalog_db.load_all(out)["images"]
        assert next(iter(images.values()))["path"] == tracked_path


def test_duplicate_copies_recorded_in_dup_paths(tmp_path):
    """The untracked byte-identical copy must be recorded in dup_paths so the
    dedupe job can physically reclaim it — regardless of which path the
    catalog adopted as canonical."""
    from scanner import scan_directory
    import catalog_db

    root = tmp_path / "photos"
    root.mkdir()
    a = root / "a_copy.jpg"
    b = root / "b_copy.jpg"
    c = root / "c_copy.jpg"
    for f in (a, b, c):
        f.write_bytes(b"identical-bytes")
    out = str(tmp_path / "catalog.db")

    scan_directory(str(root), out)
    entry = next(iter(catalog_db.load_all(out)["images"].values()))
    assert entry["path"] == str(a)
    assert sorted(entry["dup_paths"]) == [str(b), str(c)]

    # Rescans must not duplicate the records.
    scan_directory(str(root), out)
    entry = next(iter(catalog_db.load_all(out)["images"].values()))
    assert sorted(entry["dup_paths"]) == [str(b), str(c)]


# ── media typing (photos vs videos) ───────────────────────────────────────────

def test_is_video_path_and_media_extensions():
    from pathlib import Path
    import scanner
    assert scanner.is_video_path("a.MP4") and scanner.is_video_path(Path("b.mkv"))
    assert not scanner.is_video_path("c.jpg")
    assert ".mp4" in scanner.MEDIA_EXTENSIONS and ".jpg" in scanner.MEDIA_EXTENSIONS
    assert ".mp3" not in scanner.MEDIA_EXTENSIONS   # audio not media (for now)


def test_media_fields_tags_video_with_probe(monkeypatch):
    from pathlib import Path
    import scanner, video
    monkeypatch.setattr(video, "probe", lambda p: {
        "duration_s": 12.5, "width": 1920, "height": 1080,
        "codec": "h264", "capture_time": "2023-06-01T10:00:00Z"})
    f = scanner._media_fields(Path("clip.mp4"))
    assert f["media_type"] == "video"
    assert f["duration_s"] == 12.5 and f["width"] == 1920 and f["codec"] == "h264"
    assert f["metadata"]["date"] == "2023-06-01T10:00:00Z"


def test_media_fields_tags_image(monkeypatch):
    from pathlib import Path
    import scanner
    monkeypatch.setattr(scanner, "get_metadata", lambda p: {"camera_make": "X"})
    f = scanner._media_fields(Path("photo.jpg"))
    assert f["media_type"] == "image"
    assert f["metadata"] == {"camera_make": "X"}
    assert "duration_s" not in f


def test_media_fields_video_probe_failure_still_catalogs(monkeypatch):
    from pathlib import Path
    import scanner, video
    monkeypatch.setattr(video, "probe", lambda p: None)  # corrupt/undecodable
    f = scanner._media_fields(Path("broken.mov"))
    assert f["media_type"] == "video"
    assert f["duration_s"] is None and f["metadata"] == {}
