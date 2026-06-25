import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open


def test_project_root_is_parent_of_src():
    import constants
    assert constants.PROJECT_ROOT.endswith("photo-vault") or "photo-vault" in constants.PROJECT_ROOT


def test_data_dir_under_project_root():
    import constants
    assert constants.DATA_DIR == os.path.join(constants.PROJECT_ROOT, "data")


def test_all_data_paths_under_data_dir():
    import constants
    for path in [constants.IMAGE_CATALOG_PATH, constants.CHROMA_DB_PATH,
                 constants.FACE_DIR, constants.PERSON_MAP_PATH]:
        assert path.startswith(constants.DATA_DIR), f"{path} not under DATA_DIR"


def test_load_port_env_override(monkeypatch):
    import constants
    monkeypatch.setenv("PHOTO_VAULT_PORT", "9999")
    assert constants._load_port() == 9999


def test_load_port_ignores_bad_env(monkeypatch):
    import constants
    monkeypatch.setenv("PHOTO_VAULT_PORT", "not-a-number")
    # Falls through to ports.json / default without raising
    assert isinstance(constants._load_port(), int)


def test_gemini_models_ordered_lite_first():
    import constants
    first = constants.GEMINI_VISION_MODELS[0]
    assert "lite" in first or "flash-lite" in first, "First model should be a lite (high rate limit) variant"


def test_gemini_models_non_empty():
    import constants
    assert len(constants.GEMINI_VISION_MODELS) >= 3


def test_load_env_sets_variable(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR_XYZ=hello123\n")

    # Temporarily patch PROJECT_ROOT so _load_env reads our tmp .env
    import constants
    original = constants.PROJECT_ROOT
    constants.PROJECT_ROOT = str(tmp_path)
    os.environ.pop("TEST_VAR_XYZ", None)
    constants._load_env()
    constants.PROJECT_ROOT = original

    assert os.environ.get("TEST_VAR_XYZ") == "hello123"
    del os.environ["TEST_VAR_XYZ"]


def test_load_env_skips_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# this is a comment\nVALID_VAR=yes\n")

    import constants
    original = constants.PROJECT_ROOT
    constants.PROJECT_ROOT = str(tmp_path)
    os.environ.pop("VALID_VAR", None)
    constants._load_env()
    constants.PROJECT_ROOT = original

    assert os.environ.get("VALID_VAR") == "yes"
    del os.environ["VALID_VAR"]


def test_load_env_no_file_no_crash(tmp_path):
    import constants
    original = constants.PROJECT_ROOT
    constants.PROJECT_ROOT = str(tmp_path / "nonexistent")
    constants._load_env()  # should not raise
    constants.PROJECT_ROOT = original
