"""Portable data roots must isolate every persistent store before imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import socket

import pytest


def test_isolated_data_root_in_fresh_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PHOTO_VAULT_DATA_DIR=str(tmp_path),
               PHOTO_VAULT_ENV_FILE="-", PHOTO_VAULT_PORT="8768")
    result = subprocess.run(
        [sys.executable, "-c", "import constants as c, json; "
         "print(json.dumps([c.DATA_DIR,c.IMAGE_CATALOG_PATH,c.CHROMA_DB_PATH,"
         "c.FACE_DIR,c.THUMB_DIR,c.PERSON_MAP_PATH,c.PERSON_RELATIONS_PATH,"
         "c.EMBEDDING_REGISTRY_PATH,c.FOLDERS_CONFIG_PATH,c.SETTINGS_PATH,c.ALBUMS_PATH]))"],
        cwd=root / "src", env=env, capture_output=True, text=True, check=True,
    )
    assert all(Path(p).is_relative_to(tmp_path) for p in json.loads(result.stdout))


def test_env_loading_can_be_disabled(monkeypatch):
    import constants
    monkeypatch.setenv("PHOTO_VAULT_ENV_FILE", "-")
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("Disabled dotenv must never read a file")))
    constants._load_env()


def test_network_guard_blocks_services_but_allows_event_loop_control_pipe():
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match="Network access disabled"):
            sock.connect(("127.0.0.1", 1234))
    with pytest.raises(AssertionError, match="Network access disabled"):
        socket.create_connection(("example.invalid", 443))
    left, right = socket.socketpair()
    try:
        left.send(b"local-control")
        assert right.recv(32) == b"local-control"
    finally:
        left.close()
        right.close()
