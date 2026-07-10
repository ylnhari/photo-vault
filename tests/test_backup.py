"""Backup: root mapping, drive-availability status, robocopy invocation and
summary parsing (subprocess mocked — no real robocopy runs)."""
import json
import os
from unittest.mock import patch, MagicMock

import pytest

import backup


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "STATE_PATH", str(tmp_path / "backup_state.json"))
    import settings
    monkeypatch.setattr(settings, "load",
                        lambda: {"backup_dest": str(tmp_path / "sd" / "PhotoVaultBackup")})
    import folders
    pics = tmp_path / "Pictures"
    pics.mkdir()
    monkeypatch.setattr(folders, "get_effective_scan_dirs", lambda: [str(pics)])
    return tmp_path, pics


def test_backup_roots_maps_folders_and_data(env):
    tmp_path, pics = env
    roots = backup.backup_roots()
    srcs = [s for s, _ in roots]
    dests = dict(roots)
    assert str(pics) in srcs
    assert backup.DATA_DIR in srcs
    assert dests[backup.DATA_DIR].endswith("photo-vault-data")
    # a scan folder mirrors to <dest>/<its basename>: …\Pictures → …\Pictures
    assert dests[str(pics)].endswith(os.sep + "Pictures")


def test_backup_roots_same_basename_falls_back_to_full_label(tmp_path, monkeypatch):
    import settings, folders
    monkeypatch.setattr(backup, "STATE_PATH", str(tmp_path / "s.json"))
    monkeypatch.setattr(settings, "load", lambda: {"backup_dest": str(tmp_path / "sd")})
    a = tmp_path / "one" / "Pictures"
    b = tmp_path / "two" / "Pictures"
    a.mkdir(parents=True); b.mkdir(parents=True)
    monkeypatch.setattr(folders, "get_effective_scan_dirs", lambda: [str(a), str(b)])
    dests = dict(backup.backup_roots())
    # both are named "Pictures" — they must NOT collide on the destination
    assert dests[str(a)] != dests[str(b)]


def test_status_reports_unconfigured(monkeypatch, tmp_path):
    import settings
    monkeypatch.setattr(settings, "load", lambda: {"backup_dest": None})
    monkeypatch.setattr(backup, "STATE_PATH", str(tmp_path / "s.json"))
    s = backup.status()
    assert s["configured"] is False and s["available"] is False


def test_status_available_and_staleness(env):
    tmp_path, _ = env
    s = backup.status()
    assert s["configured"] is True
    assert s["available"] is True  # dest is on the same (existing) drive in tests
    assert s["days_since"] is None  # never backed up
    backup.record_success()
    assert backup.status()["days_since"] == 0.0


_SUMMARY = """
------------------------------------------------------------------------------
               Total    Copied   Skipped  Mismatch    FAILED    Extras
    Dirs :        10         1         9         0         0         0
   Files :       120         5       115         0         0         2
"""


def test_backup_one_photo_root_copies_without_purge_and_skips_videos(env, monkeypatch):
    tmp_path, pics = env
    monkeypatch.setattr(os, "name", "nt")  # force the robocopy engine
    proc = MagicMock(returncode=3, stdout=_SUMMARY)
    with patch("backup.subprocess.run", return_value=proc) as run:
        note = backup.backup_one(str(pics))
    cmd = run.call_args[0][0]
    assert cmd[0] == "robocopy" and cmd[1] == str(pics)
    # photo roots: additive copy (no /MIR — drive-side videos must survive)
    # with video files invisible to robocopy entirely
    assert "/MIR" not in cmd and "/E" in cmd
    assert "/XF" in cmd and "*.mp4" in cmd
    assert "5 copied" in note and "115 unchanged" in note
    assert backup.status()["last_backup_at"] is not None


def test_backup_one_data_root_uses_strict_mirror(env, monkeypatch):
    tmp_path, pics = env
    monkeypatch.setattr(os, "name", "nt")
    proc = MagicMock(returncode=1, stdout=_SUMMARY)
    with patch("backup.subprocess.run", return_value=proc) as run:
        backup.backup_one(backup.DATA_DIR)
    cmd = run.call_args[0][0]
    assert "/MIR" in cmd and "/XF" not in cmd


def test_backup_one_failure_raises(env, monkeypatch):
    tmp_path, pics = env
    monkeypatch.setattr(os, "name", "nt")
    proc = MagicMock(returncode=16, stdout="ERROR : access denied\n")
    with patch("backup.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="exit 16"):
            backup.backup_one(str(pics))
    assert backup.status()["last_backup_at"] is None


def test_backup_one_unmapped_source_raises(env):
    with pytest.raises(RuntimeError, match="no backup destination"):
        backup.backup_one(r"C:\not\a\root")


# ── cross-platform python mirror engine ──────────────────────────────────────
# Pure stdlib, so these run for real on every OS (no robocopy involved).

def _tree(base, spec):
    """Create files from {relpath: content} under base."""
    for rel, content in spec.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def test_python_mirror_copies_and_skips_videos_without_purge(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"2023/a.jpg": b"aa", "2023/clip.mp4": b"vv", "b.png": b"bb"})
    # a pre-existing extra at dest must SURVIVE a no-purge photo mirror
    _tree(dst, {"old-video.mov": b"keepme"})
    note = backup._mirror_python(str(src), str(dst), purge=False)
    assert (dst / "2023" / "a.jpg").read_bytes() == b"aa"
    assert (dst / "b.png").read_bytes() == b"bb"
    assert not (dst / "2023" / "clip.mp4").exists()   # video invisible
    assert (dst / "old-video.mov").exists()           # extras kept (no purge)
    assert "2 copied" in note


def test_python_mirror_incremental_second_run_copies_nothing(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"a.jpg": b"aa", "sub/b.jpg": b"bb"})
    backup._mirror_python(str(src), str(dst), purge=False)
    note = backup._mirror_python(str(src), str(dst), purge=False)
    assert "0 copied" in note and "2 unchanged" in note


def test_python_mirror_purge_removes_extras_and_empty_dirs(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"keep.bin": b"k"})
    _tree(dst, {"keep.bin": b"k", "gone/stale.bin": b"s"})
    note = backup._mirror_python(str(src), str(dst), purge=True)
    assert not (dst / "gone").exists()   # extra file AND its emptied dir gone
    assert (dst / "keep.bin").exists()
    assert "1 removed at dest" in note


def test_python_mirror_skips_system_dirs(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"ok.jpg": b"o", "$RECYCLE.BIN/zombie.jpg": b"z",
                ".Trash-1000/dead.jpg": b"d"})
    backup._mirror_python(str(src), str(dst), purge=False)
    assert (dst / "ok.jpg").exists()
    assert not (dst / "$RECYCLE.BIN").exists()
    assert not (dst / ".Trash-1000").exists()


def test_python_mirror_failure_raises_and_reports(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tree(src, {"a.jpg": b"aa"})
    monkeypatch.setattr(backup.shutil, "copy2",
                        lambda a, b: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError, match="disk full"):
        backup._mirror_python(str(src), str(dst), purge=False)


def test_backup_one_uses_python_engine_off_windows(env, monkeypatch):
    tmp_path, pics = env
    (pics / "x.jpg").write_bytes(b"x")
    monkeypatch.setattr(os, "name", "posix")
    with patch("backup.subprocess.run") as run:  # must NOT be called
        note = backup.backup_one(str(pics))
    run.assert_not_called()
    assert "1 copied" in note
    assert backup.status()["last_backup_at"] is not None


def test_validate_dest_rejects_library_overlap(tmp_path, monkeypatch):
    import folders
    lib = tmp_path / "Pictures"; lib.mkdir()
    monkeypatch.setattr(folders, "get_effective_scan_dirs", lambda: [str(lib)])
    monkeypatch.setattr(folders, "get_excluded_paths", lambda: [])
    inside = backup.validate_dest(str(lib / "backup"))
    assert not inside["ok"] and "recursively" in inside["reason"]
    containing = backup.validate_dest(str(tmp_path))
    assert not containing["ok"] and "double every photo" in containing["reason"]
    ok = backup.validate_dest(str(tmp_path / "separate"))
    assert ok["ok"]
