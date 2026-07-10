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
    dests = [d for _, d in roots]
    assert str(pics) in srcs
    assert backup.DATA_DIR in srcs
    assert any(d.endswith("photo-vault-data") for d in dests)
    # library folders live under <dest>/library/<path-derived label>
    assert any(os.sep + "library" + os.sep in d for d in dests)


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


def test_backup_one_success_parses_summary_and_records(env):
    tmp_path, pics = env
    proc = MagicMock(returncode=3, stdout=_SUMMARY)
    with patch("backup.subprocess.run", return_value=proc) as run:
        note = backup.backup_one(str(pics))
    cmd = run.call_args[0][0]
    assert cmd[0] == "robocopy" and "/MIR" in cmd
    assert cmd[1] == str(pics)
    assert "5 copied" in note and "115 unchanged" in note and "2 removed" in note
    assert backup.status()["last_backup_at"] is not None


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
