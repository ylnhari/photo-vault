"""Ingest/consolidate: content-hash dedupe against the library, YYYY/MM
placement, collision-safe naming, and the persistent media-hash cache."""
import json
import os
import time

import pytest

import ingest
from scanner import content_uid


@pytest.fixture
def env(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    library = tmp_path / "library"
    staging.mkdir()
    library.mkdir()
    monkeypatch.setattr(ingest, "MEDIA_HASHES_PATH", str(tmp_path / "media_hashes.json"))
    # No library folders to walk for video hashes unless a test opts in.
    import folders
    monkeypatch.setattr(folders, "get_effective_scan_dirs", lambda: [str(library)])
    return staging, library


def _write(path, data: bytes, mtime: float = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime:
        os.utime(path, (mtime, mtime))
    return path


def test_new_file_lands_in_year_month_folder(env):
    staging, library = env
    # 2023-06-15 mtime → 2023/06 (no EXIF in raw bytes → mtime fallback)
    ts = time.mktime((2023, 6, 15, 12, 0, 0, 0, 0, -1))
    _write(staging / "IMG_1.jpg", b"photo-bytes-1", ts)
    s = ingest.IngestSession(str(staging), catalog_images={}, dest=str(library))
    note = s.ingest_one(str(staging / "IMG_1.jpg"))
    s.close()
    assert note.startswith("imported")
    assert (library / "2023" / "06" / "IMG_1.jpg").read_bytes() == b"photo-bytes-1"


def test_catalog_content_is_skipped(env):
    staging, library = env
    f = _write(staging / "copy_of_known.jpg", b"already-known-bytes")
    known_uid = content_uid(f)
    s = ingest.IngestSession(str(staging), catalog_images={known_uid: {}}, dest=str(library))
    note = s.ingest_one(str(f))
    s.close()
    assert "skipped" in note
    assert not any(library.rglob("*.jpg"))


def test_intra_run_duplicate_copied_once(env):
    staging, library = env
    ts = time.mktime((2024, 1, 1, 9, 0, 0, 0, 0, -1))
    a = _write(staging / "a" / "same.jpg", b"same-bytes", ts)
    b = _write(staging / "b" / "same_renamed.jpg", b"same-bytes", ts)
    s = ingest.IngestSession(str(staging), catalog_images={}, dest=str(library))
    n1 = s.ingest_one(str(a))
    n2 = s.ingest_one(str(b))
    s.close()
    assert n1.startswith("imported")
    assert "skipped" in n2
    assert len(list(library.rglob("*.jpg"))) == 1


def test_name_collision_gets_hash_suffix(env):
    staging, library = env
    ts = time.mktime((2024, 3, 5, 9, 0, 0, 0, 0, -1))
    a = _write(staging / "x" / "IMG.jpg", b"contents-A", ts)
    b = _write(staging / "y" / "IMG.jpg", b"contents-B", ts)
    s = ingest.IngestSession(str(staging), catalog_images={}, dest=str(library))
    s.ingest_one(str(a))
    note = s.ingest_one(str(b))
    s.close()
    files = {p.name for p in (library / "2024" / "03").iterdir()}
    assert "IMG.jpg" in files and len(files) == 2
    suffixed = (files - {"IMG.jpg"}).pop()
    assert suffixed.startswith("IMG-") and note.startswith("imported")


def test_video_dedupe_persists_across_sessions(env):
    staging, library = env
    ts = time.mktime((2022, 12, 25, 9, 0, 0, 0, 0, -1))
    v = _write(staging / "clip.mp4", b"video-bytes", ts)
    s1 = ingest.IngestSession(str(staging), catalog_images={}, dest=str(library))
    assert s1.ingest_one(str(v)).startswith("imported")
    s1.close()
    # New session, same video appears again in a different staging spot —
    # the persisted media-hash cache must recognize it.
    v2 = _write(staging / "elsewhere" / "clip (1).mp4", b"video-bytes", ts)
    s2 = ingest.IngestSession(str(staging), catalog_images={}, dest=str(library))
    assert "skipped" in s2.ingest_one(str(v2))
    s2.close()


def test_library_video_hashed_on_first_run(env):
    staging, library = env
    # A video that already lives in the LIBRARY but was never ingested (it
    # predates the feature) must still be recognized as a duplicate.
    ts = time.mktime((2021, 7, 1, 9, 0, 0, 0, 0, -1))
    _write(library / "old" / "existing.mp4", b"lib-video-bytes", ts)
    incoming = _write(staging / "existing_copy.mp4", b"lib-video-bytes", ts)
    s = ingest.IngestSession(str(staging), catalog_images={}, dest=str(library))
    assert "skipped" in s.ingest_one(str(incoming))
    s.close()


def test_list_staging_files_filters_media_and_sorts(env):
    staging, _ = env
    _write(staging / "b.jpg", b"1")
    _write(staging / "a.mp4", b"2")
    _write(staging / "notes.txt", b"3")
    files = ingest.list_staging_files(str(staging))
    assert [os.path.basename(f) for f in files] == ["a.mp4", "b.jpg"]


def test_list_staging_files_missing_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        ingest.list_staging_files(str(tmp_path / "nope"))


def test_default_dest_prefers_non_onedrive_folder(monkeypatch):
    import settings, folders
    monkeypatch.setattr(settings, "load", lambda: {"ingest_dest": None})
    monkeypatch.setattr(folders, "get_effective_scan_dirs", lambda: [
        r"C:\Users\x\OneDrive\Pictures", r"C:\Users\x\Pictures",
    ])
    assert ingest.default_dest() == os.path.join(r"C:\Users\x\Pictures", "Imported")


def test_default_dest_falls_back_to_onedrive_when_only_option(monkeypatch):
    import settings, folders
    monkeypatch.setattr(settings, "load", lambda: {"ingest_dest": None})
    monkeypatch.setattr(folders, "get_effective_scan_dirs",
                        lambda: [r"C:\Users\x\OneDrive\Pictures"])
    assert ingest.default_dest() == os.path.join(r"C:\Users\x\OneDrive\Pictures", "Imported")


def test_stale_cache_entries_pruned_so_deleted_files_reimport(env):
    staging, library = env
    ts = time.mktime((2020, 5, 5, 9, 0, 0, 0, 0, -1))
    v = _write(staging / "clip.mp4", b"video-bytes-2", ts)
    # Keep the video in the library tree for this cache test (videos otherwise
    # route to their own tree — see test_videos_land_in_separate_tree).
    s1 = ingest.IngestSession(str(staging), catalog_images={},
                              dest=str(library), video_dest=str(library))
    note = s1.ingest_one(str(v))
    s1.close()
    assert note.startswith("imported")
    # Simulate the imported copy being deleted (cleanup, accident, whatever):
    # its hash must NOT keep claiming "in library" forever.
    imported = next((library / "2020" / "05").iterdir())
    imported.unlink()
    s2 = ingest.IngestSession(str(staging), catalog_images={},
                              dest=str(library), video_dest=str(library))
    assert s2.ingest_one(str(v)).startswith("imported")
    s2.close()


# ── pre-flight validators (rules are user-defined; messages are the UX) ──────

def _folders(monkeypatch, included, excluded=()):
    import folders
    monkeypatch.setattr(folders, "get_effective_scan_dirs", lambda: list(included))
    monkeypatch.setattr(folders, "get_excluded_paths", lambda: list(excluded))


def test_validate_source_refuses_included_and_excluded(tmp_path, monkeypatch):
    lib = tmp_path / "Pictures"; lib.mkdir()
    ex = lib / "private"; ex.mkdir()
    other = tmp_path / "sdcard"; other.mkdir()
    _folders(monkeypatch, [str(lib)], [str(ex)])
    assert not ingest.validate_source(str(lib))["ok"]
    assert "already part of your scanned library" in ingest.validate_source(str(lib))["reason"]
    assert not ingest.validate_source(str(ex))["ok"]
    assert "excluded" in ingest.validate_source(str(ex))["reason"]
    assert ingest.validate_source(str(other))["ok"]


def test_validate_source_missing_folder(tmp_path, monkeypatch):
    _folders(monkeypatch, [])
    r = ingest.validate_source(str(tmp_path / "nope"))
    assert not r["ok"] and "doesn't exist" in r["reason"]


def test_validate_dest_must_be_inside_included_not_excluded(tmp_path, monkeypatch):
    lib = tmp_path / "Pictures"; lib.mkdir()
    ex = lib / "private"; ex.mkdir()
    _folders(monkeypatch, [str(lib)], [str(ex)])
    assert ingest.validate_dest(str(lib / "Imported"))["ok"]
    r_out = ingest.validate_dest(str(tmp_path / "elsewhere"))
    assert not r_out["ok"] and "included scan folder" in r_out["reason"]
    r_ex = ingest.validate_dest(str(ex / "Imported"))
    assert not r_ex["ok"] and "excluded" in r_ex["reason"]


def test_source_stats_counts_media_and_ignores_rest(tmp_path):
    src = tmp_path / "dump"; src.mkdir()
    _write(src / "a.jpg", b"x" * 100)
    _write(src / "b.mp4", b"y" * 200)
    _write(src / "readme.txt", b"z")
    s = ingest.source_stats(str(src))
    assert s == {"media_files": 2, "photo_files": 1, "video_files": 1,
                 "media_bytes": 300, "other_files": 1}


# ── video routing + media filter ──────────────────────────────────────────────

def test_ext_wanted_media_filter():
    assert ingest._ext_wanted(".jpg", "both") and ingest._ext_wanted(".mp4", "both")
    assert ingest._ext_wanted(".jpg", "photos") and not ingest._ext_wanted(".mp4", "photos")
    assert ingest._ext_wanted(".mp4", "videos") and not ingest._ext_wanted(".jpg", "videos")
    assert not ingest._ext_wanted(".mp3", "both")   # audio is ignored entirely


def test_videos_land_in_separate_tree(env):
    staging, library = env
    videos = library / "Videos"; videos.mkdir()
    ts = time.mktime((2023, 6, 15, 12, 0, 0, 0, 0, -1))
    _write(staging / "clip.mp4", b"video-bytes", ts)
    _write(staging / "pic.jpg", b"photo-bytes", ts)
    s = ingest.IngestSession(str(staging), catalog_images={},
                             dest=str(library), video_dest=str(videos))
    vnote = s.ingest_one(str(staging / "clip.mp4"))
    pnote = s.ingest_one(str(staging / "pic.jpg"))
    # video under the Videos tree, photo under the library tree
    assert (videos / "2023" / "06" / "clip.mp4").exists()
    assert (library / "2023" / "06" / "pic.jpg").exists()
    assert not (library / "2023" / "06" / "clip.mp4").exists()
    assert "imported" in vnote and "imported" in pnote


def test_media_filter_skips_other_type(env):
    staging, library = env
    _write(staging / "clip.mp4", b"vv")
    _write(staging / "pic.jpg", b"pp")
    # photos-only session must skip the video
    s = ingest.IngestSession(str(staging), catalog_images={},
                             dest=str(library), media="photos")
    note = s.ingest_one(str(staging / "clip.mp4"))
    assert "skipped" in note and "filtered" in note


def test_source_stats_splits_photos_and_videos(env):
    staging, _ = env
    _write(staging / "a.jpg", b"a")
    _write(staging / "b.png", b"b")
    _write(staging / "c.mp4", b"c")
    _write(staging / "notes.txt", b"x")   # ignored
    st = ingest.source_stats(str(staging))
    assert st["photo_files"] == 2
    assert st["video_files"] == 1
    assert st["other_files"] == 1
    assert st["media_files"] == 3


def test_default_video_dest_prefers_scanned_videos_root(tmp_path, monkeypatch):
    import folders, settings as settings_mod
    vroot = tmp_path / "Videos"; vroot.mkdir()
    monkeypatch.setattr(settings_mod, "load", lambda: {})
    monkeypatch.setattr(folders, "get_effective_scan_dirs",
                        lambda: [str(tmp_path / "Pictures"), str(vroot)])
    assert ingest.default_video_dest() == str(vroot)


def test_list_staging_files_honors_media_filter(env):
    staging, _ = env
    _write(staging / "a.jpg", b"a")
    _write(staging / "b.mp4", b"b")
    both = ingest.list_staging_files(str(staging), media="both")
    vids = ingest.list_staging_files(str(staging), media="videos")
    assert len(both) == 2
    assert len(vids) == 1 and vids[0].endswith("b.mp4")
