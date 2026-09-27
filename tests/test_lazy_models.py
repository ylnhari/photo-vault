"""Face lookup must not start the detector; concurrent starts share one model."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_face_module_does_not_import_inference_runtimes():
    result = subprocess.run(
        [sys.executable, "-c", "import faces,sys,json; "
         "print(json.dumps([m for m in ['insightface','onnxruntime','openvino','cv2','chromadb'] if m in sys.modules]))"],
        cwd=Path(__file__).resolve().parents[1] / "src",
        env=os.environ.copy(), capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == []


def test_concurrent_detector_initialization_builds_once(monkeypatch):
    import faces
    fake = MagicMock()
    factory = MagicMock(return_value=fake)
    monkeypatch.setattr(faces, "_face_app", None)
    monkeypatch.setattr(faces, "_face_app_choice", None)
    monkeypatch.setattr(faces, "insightface", SimpleNamespace(app=SimpleNamespace(FaceAnalysis=factory)))
    monkeypatch.setattr(faces.settings_mod, "load", lambda: {"face_provider": "cpu"})
    monkeypatch.setattr(faces, "_resolve_providers", lambda choice: (["CPUExecutionProvider"], [{}]))
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: faces._get_app(), range(8)))
    assert all(value is fake for value in results)
    factory.assert_called_once()
    fake.prepare.assert_called_once()
