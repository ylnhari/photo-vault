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


def test_backup_one_photo_root_copies_without_purge_and_skips_videos(env):
    tmp_path, pics = env
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


def test_backup_one_data_root_uses_strict_mirror(env):
    tmp_path, pics = env
    proc = MagicMock(returncode=1, stdout=_SUMMARY)
    with patch("backup.subprocess.run", return_value=proc) as run:
        backup.backup_one(backup.DATA_DIR)
    cmd = run.call_args[0][0]
    assert "/MIR" in cmd and "/XF" not in cmd


def test_backup_one_failure_raises(env):
    tmp_path, pics = env
    proc = MagicMock(returncode=16, stdout="ERROR : access denied\n")
    with patch("backup.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="exit 16"):
            backup.backup_one(str(pics))
    assert backup.status()["last_backup_at"] is None


def test_backup_one_unmapped_source_raises(env):
    with pytest.raises(RuntimeError, match="no backup destination"):
        backup.backup_one(r"C:\not\a\root")


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
