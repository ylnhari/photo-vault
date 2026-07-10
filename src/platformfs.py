"""One home for every OS-specific filesystem behavior, so the rest of the app
stays platform-neutral. Everything here is stdlib-only (house rule) and each
function documents its per-OS strategy:

  - list_roots():        where the folder-picker starts (drive letters vs mounts)
  - is_system_name():    junk/system dirs no walk should descend into
  - skip_system_dirs():  in-place os.walk dirnames filter using the above
  - dest_available():    "is this destination's device actually connected?"
  - move_to_trash():     recoverable delete (Recycle Bin / Finder Trash /
                         freedesktop Trash), False → caller falls back

Windows behavior is unchanged from the original single-platform code; the
macOS/Linux paths are additive.
"""
import ctypes
import getpass
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

# Directory names that are OS plumbing, never user photos. A walk (scan,
# ingest preview, backup mirror) must not descend into them: recycle bins
# resurrect deleted photos as "new" imports, Spotlight/fseventsd churn
# endlessly, and lost+found is fsck wreckage.
_SYSTEM_DIR_NAMES = {
    # Windows
    "$recycle.bin", "system volume information", "$sysreset", "recycler",
    # macOS
    ".trashes", ".spotlight-v100", ".fseventsd", ".documentrevisions-v100",
    ".temporaryitems",
    # Linux
    "lost+found", ".trash",
}


def is_system_name(name: str) -> bool:
    """True for OS system/junk directory names (case-insensitive), including
    the per-user variants like '.Trash-1000' and '$RECYCLE.BIN'."""
    low = name.lower()
    return (low in _SYSTEM_DIR_NAMES
            or low.startswith((".trash-", "$recycle")))


def skip_system_dirs(dirnames: list[str]) -> list[str]:
    """In-place os.walk `dirnames` filter: prune system dirs so the walk never
    descends into them. Returns the same (mutated) list for convenience."""
    dirnames[:] = [d for d in dirnames if not is_system_name(d)]
    return dirnames


def list_roots() -> list[str]:
    """Filesystem roots for the folder-picker's first screen.

    Windows: connected drive letters (C:\\, D:\\, ...).
    macOS:   / plus every mounted volume under /Volumes (external disks,
             SD cards, network shares all appear there).
    Linux:   / plus mounted media under /media, /media/<user>,
             /run/media/<user>, and /mnt — the places udisks and admins
             put removable drives.
    """
    if IS_WINDOWS:
        import string
        return [f"{c}:\\" for c in string.ascii_uppercase
                if os.path.exists(f"{c}:\\")]

    import posixpath
    roots = ["/"]
    candidates = []
    if IS_MACOS:
        candidates.append("/Volumes")
    else:
        user = getpass.getuser()
        candidates += ["/media", f"/media/{user}", f"/run/media/{user}", "/mnt"]
    seen = set(roots)
    for base in candidates:
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            continue
        for name in entries:
            # posixpath explicitly: these are POSIX mount paths by definition
            p = posixpath.join(base, name)
            if os.path.isdir(p) and not is_system_name(name) and p not in seen:
                seen.add(p)
                roots.append(p)
    return roots


def dest_available(dest: str) -> bool:
    """Is the destination's device connected/mounted right now?

    Windows: the drive letter exists.
    POSIX:   the folder (or, for a first-ever backup, its parent) exists.
             Requiring an existing parent is the unmounted-mount guard: with
             /media/usb unplugged, blindly creating /media/usb/backup would
             silently mirror 46GB onto the ROOT filesystem — the classic
             rsync-into-a-dead-mountpoint trap.
    """
    if not dest:
        return False
    if IS_WINDOWS:
        drive = os.path.splitdrive(dest)[0]
        return os.path.exists(drive + os.sep) if drive else os.path.isdir(dest)
    p = Path(dest)
    return p.is_dir() or p.parent.is_dir()


def unavailable_reason(dest: str) -> str:
    """Human sentence for the validate/preflight message when dest_available
    is False — names the actual thing to fix per OS."""
    if IS_WINDOWS:
        drive = os.path.splitdrive(dest)[0] or dest
        return (f"Drive {drive} isn't connected right now — "
                "plug it in and try again.")
    return (f"{dest} isn't reachable — is the drive mounted? "
            "Plug it in / mount it and try again.")


# ── recoverable delete ────────────────────────────────────────────────────────

def _trash_windows(path: str) -> bool:
    """Recycle Bin via SHFileOperationW — stdlib ctypes, no dependency."""
    try:
        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", ctypes.c_void_p),
                ("wFunc", ctypes.c_uint),
                ("pFrom", ctypes.c_wchar_p),
                ("pTo", ctypes.c_wchar_p),
                ("fFlags", ctypes.c_uint16),
                ("fAnyOperationsAborted", ctypes.c_int),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", ctypes.c_wchar_p),
            ]

        FO_DELETE = 3
        FOF_ALLOWUNDO = 0x0040
        FOF_NOCONFIRMATION = 0x0010
        FOF_SILENT = 0x0004
        FOF_NOERRORUI = 0x0400

        op = SHFILEOPSTRUCTW()
        op.wFunc = FO_DELETE
        op.pFrom = os.path.abspath(path) + "\0"  # double-NUL-terminated list
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return result == 0 and not op.fAnyOperationsAborted
    except Exception as e:
        print(f"[platformfs] recycle-bin delete failed for {path}: {e}")
        return False


def _trash_macos(path: str) -> bool:
    """Finder Trash via osascript — no dependency, genuinely restorable from
    the Trash with 'Put Back'."""
    try:
        proc = subprocess.run(
            ["osascript", "-e",
             'tell application "Finder" to delete POSIX file '
             f'"{os.path.abspath(path)}"'],
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode == 0
    except Exception as e:
        print(f"[platformfs] Finder trash failed for {path}: {e}")
        return False


def _trash_linux(path: str) -> bool:
    """freedesktop.org Trash spec, home-trash flavor: move the file into
    ~/.local/share/Trash/files and write the matching .trashinfo so desktop
    trash UIs can restore it. os.rename can't cross filesystems, so a file on
    another device returns False and the caller falls back to plain delete —
    honest, rather than a fake 'trash' that actually copies+deletes."""
    src = Path(path).resolve()
    trash = Path(os.environ.get("XDG_DATA_HOME",
                                Path.home() / ".local" / "share")) / "Trash"
    files_dir = trash / "files"
    info_dir = trash / "info"
    try:
        files_dir.mkdir(parents=True, exist_ok=True)
        info_dir.mkdir(parents=True, exist_ok=True)
        name = src.name
        target = files_dir / name
        while target.exists() or (info_dir / (target.name + ".trashinfo")).exists():
            target = files_dir / f"{src.stem}-{uuid.uuid4().hex[:8]}{src.suffix}"
        # Write info first (spec order): a crash between the two leaves an
        # orphan .trashinfo, which trash UIs tolerate; the reverse leaves an
        # unrestorable file.
        from urllib.parse import quote
        info = (f"[Trash Info]\nPath={quote(str(src), safe='/')}\n"
                f"DeletionDate={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        (info_dir / (target.name + ".trashinfo")).write_text(info)
        os.rename(src, target)  # same-filesystem move; raises across devices
        return True
    except OSError as e:
        # EXDEV (cross-device) or anything else: clean up the orphan info
        try:
            (info_dir / (target.name + ".trashinfo")).unlink(missing_ok=True)
        except Exception:
            pass
        print(f"[platformfs] trash move failed for {path}: {e}")
        return False
    except Exception as e:
        print(f"[platformfs] trash failed for {path}: {e}")
        return False


def move_to_trash(path: str) -> bool:
    """Delete a file recoverably via the OS trash. Returns True when the
    trash path was used; False means the caller should decide about a plain
    (permanent) delete — same contract on every OS."""
    if IS_WINDOWS:
        return _trash_windows(path)
    if IS_MACOS:
        return _trash_macos(path)
    return _trash_linux(path)
