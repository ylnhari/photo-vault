"""platformfs: the one home for OS-specific filesystem behavior. Each OS
branch is exercised by forcing the platform flags, so the full matrix runs
on whatever OS the tests happen to execute on."""
import os

import pytest

import platformfs


# ── system-dir filtering ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "$RECYCLE.BIN", "System Volume Information", "$SysReset", "RECYCLER",
    ".Trashes", ".Spotlight-V100", ".fseventsd",
    "lost+found", ".Trash", ".Trash-1000",
])
def test_is_system_name_matches_all_platforms(name):
    assert platformfs.is_system_name(name)


@pytest.mark.parametrize("name", [
    "Pictures", "2023", "Trash photos from goa", "$$$ receipts", "DCIM",
])
def test_is_system_name_leaves_user_dirs_alone(name):
    # "$$$ receipts" starts with '$' but is NOT $RECYCLE — only real system
    # names may be skipped; user folders with odd names must survive.
    assert not platformfs.is_system_name(name)


def test_skip_system_dirs_prunes_in_place():
    dirs = ["2023", "$RECYCLE.BIN", "DCIM", ".Trash-1000", "lost+found"]
    out = platformfs.skip_system_dirs(dirs)
    assert out is dirs  # in-place contract (os.walk pruning)
    assert dirs == ["2023", "DCIM"]


# ── roots per OS ──────────────────────────────────────────────────────────────

def test_list_roots_windows(monkeypatch):
    monkeypatch.setattr(platformfs, "IS_WINDOWS", True)
    monkeypatch.setattr(platformfs.os.path, "exists",
                        lambda p: p in ("C:\\", "D:\\"))
    assert platformfs.list_roots() == ["C:\\", "D:\\"]


def test_list_roots_macos(monkeypatch):
    monkeypatch.setattr(platformfs, "IS_WINDOWS", False)
    monkeypatch.setattr(platformfs, "IS_MACOS", True)
    monkeypatch.setattr(platformfs.os, "listdir",
                        lambda base: ["Macintosh HD", "SDCARD", ".Trashes"]
                        if base == "/Volumes" else [])
    monkeypatch.setattr(platformfs.os.path, "isdir", lambda p: True)
    roots = platformfs.list_roots()
    assert roots[0] == "/"
    assert "/Volumes/SDCARD" in roots
    assert all(".Trashes" not in r for r in roots)


def test_list_roots_linux(monkeypatch):
    monkeypatch.setattr(platformfs, "IS_WINDOWS", False)
    monkeypatch.setattr(platformfs, "IS_MACOS", False)
    monkeypatch.setattr(platformfs.getpass, "getuser", lambda: "hari")
    listing = {"/media": ["hari"], "/media/hari": ["USB64"], "/mnt": ["nas"]}
    monkeypatch.setattr(platformfs.os, "listdir",
                        lambda base: listing.get(base, []) or (_ for _ in ()).throw(OSError())
                        if base in listing else (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(platformfs.os.path, "isdir", lambda p: True)
    roots = platformfs.list_roots()
    assert roots[0] == "/"
    assert "/media/hari/USB64" in roots
    assert "/mnt/nas" in roots
    # /media/hari appears both as a listing of /media and as its own base —
    # never duplicated
    assert len(roots) == len(set(roots))


# ── destination availability ──────────────────────────────────────────────────

def test_dest_available_posix_requires_existing_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(platformfs, "IS_WINDOWS", False)
    existing = tmp_path / "backup"
    existing.mkdir()
    assert platformfs.dest_available(str(existing))                # exists
    assert platformfs.dest_available(str(tmp_path / "new"))        # parent exists
    # unmounted-mountpoint guard: neither the dir nor its parent exist
    assert not platformfs.dest_available(str(tmp_path / "usb" / "backup"))
    assert not platformfs.dest_available("")


def test_unavailable_reason_names_the_right_thing(monkeypatch):
    monkeypatch.setattr(platformfs, "IS_WINDOWS", True)
    assert "Drive E:" in platformfs.unavailable_reason(r"E:\backup")
    monkeypatch.setattr(platformfs, "IS_WINDOWS", False)
    assert "mounted" in platformfs.unavailable_reason("/media/usb/backup")


# ── linux trash (pure stdlib, testable on any OS) ─────────────────────────────

def test_trash_linux_moves_file_and_writes_trashinfo(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    victim = tmp_path / "photo.jpg"
    victim.write_bytes(b"jpegdata")
    assert platformfs._trash_linux(str(victim)) is True
    assert not victim.exists()
    files = list((tmp_path / "xdg" / "Trash" / "files").iterdir())
    infos = list((tmp_path / "xdg" / "Trash" / "info").iterdir())
    assert len(files) == 1 and files[0].read_bytes() == b"jpegdata"
    assert len(infos) == 1 and infos[0].name.endswith(".trashinfo")
    body = infos[0].read_text()
    assert "[Trash Info]" in body and "DeletionDate=" in body


def test_trash_linux_name_collision_gets_unique_name(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    for content in (b"one", b"two"):
        f = tmp_path / "same.jpg"
        f.write_bytes(content)
        assert platformfs._trash_linux(str(f)) is True
    files = list((tmp_path / "xdg" / "Trash" / "files").iterdir())
    assert len(files) == 2  # second delete didn't clobber the first


def test_trash_linux_cross_device_returns_false(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    victim = tmp_path / "photo.jpg"
    victim.write_bytes(b"x")
    monkeypatch.setattr(platformfs.os, "rename",
                        lambda a, b: (_ for _ in ()).throw(OSError(18, "cross-device")))
    assert platformfs._trash_linux(str(victim)) is False
    assert victim.exists()  # untouched — caller decides about permanent delete
    # no orphan .trashinfo left behind
    info_dir = tmp_path / "xdg" / "Trash" / "info"
    assert not info_dir.exists() or not list(info_dir.iterdir())
