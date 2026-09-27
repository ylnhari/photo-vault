import os
import sys
import tempfile
import socket
import threading
from pathlib import Path

import pytest

# Keep every test import and default storage operation away from the user's
# real .env and data/ directory. This must be set before importing app modules.
_test_data_dir = tempfile.TemporaryDirectory(prefix="photo-vault-tests-")
os.environ["PHOTO_VAULT_DATA_DIR"] = _test_data_dir.name
os.environ["PHOTO_VAULT_ENV_FILE"] = "-"
os.environ["GEMINI_API_KEY"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "False"


@pytest.fixture(autouse=True)
def block_test_network(monkeypatch):
    """A missing provider mock must never reach a real service."""
    original_connect = socket.socket.connect
    original_pair = socket.socketpair
    control_pipe = threading.local()

    def blocked(*args, **kwargs):
        raise AssertionError("Network access disabled in tests; mock the provider")

    def connect(sock, address):
        # On Windows, asyncio's internal self-pipe uses the standard library's
        # loopback socketpair implementation. Permit only that synchronous
        # construction, never arbitrary loopback or provider connections.
        if getattr(control_pipe, "creating", False):
            return original_connect(sock, address)
        return blocked()

    def socketpair(*args, **kwargs):
        control_pipe.creating = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            control_pipe.creating = False

    monkeypatch.setattr(socket, "socketpair", socketpair)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def pytest_unconfigure(config):
    # The app keeps Chroma and SQLite connections open for its lifetime. Close
    # clients created under this temporary test directory before the
    # TemporaryDirectory removes it (required on Windows).
    try:
        import db
        if db._client is not None:
            db._client.close()
            db._client = None
        import catalog_db
        for path, conn in list(catalog_db._conn_cache.items()):
            if os.path.commonpath([path, _test_data_dir.name]) == _test_data_dir.name:
                conn.close()
                catalog_db._conn_cache.pop(path, None)
    except Exception:
        pass
    _test_data_dir.cleanup()

# Add src/ to path so modules can import each other (e.g. indexer imports vision)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
