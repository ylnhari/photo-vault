"""Concurrent first requests must not race NumPy's package initialization."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_concurrent_people_routes_initialize_optional_modules_safely(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update({
        "PHOTO_VAULT_DATA_DIR": str(tmp_path),
        "PHOTO_VAULT_ENV_FILE": "-",
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": str(root / "src"),
        "GEMINI_API_KEY": "",
    })
    env.pop("NINEROUTER_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    script = textwrap.dedent("""
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        import urllib.request
        import requests

        def deny_network(*args, **kwargs):
            raise AssertionError("Unexpected network access in read-only route test")

        urllib.request.urlopen = deny_network
        requests.sessions.Session.request = deny_network

        from fastapi.testclient import TestClient
        import api

        paths = ["/api/people", "/api/faces/status", "/api/faces/clusters"]
        barrier = Barrier(len(paths))
        client = TestClient(api.app, raise_server_exceptions=False)

        def request(path):
            barrier.wait()
            response = client.get(path)
            return path, response.status_code, response.text

        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            results = list(pool.map(request, paths))
        failures = [(path, status, body[:500]) for path, status, body in results
                    if status != 200]
        assert not failures, failures
        import sys
        assert "sklearn" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_face_clusters_read_does_not_import_clustering_stack(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update({
        "PHOTO_VAULT_DATA_DIR": str(tmp_path),
        "PHOTO_VAULT_ENV_FILE": "-",
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": str(root / "src"),
        "GEMINI_API_KEY": "",
    })
    script = textwrap.dedent("""
        import urllib.request
        import requests

        def deny_network(*args, **kwargs):
            raise AssertionError("Unexpected network access in read-only route test")

        urllib.request.urlopen = deny_network
        requests.sessions.Session.request = deny_network

        from fastapi.testclient import TestClient
        import api
        response = TestClient(api.app, raise_server_exceptions=False).get(
            "/api/faces/clusters")
        assert response.status_code == 200, response.text
        import sys
        assert "sklearn" not in sys.modules
        assert "scipy" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
