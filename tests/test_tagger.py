import json
import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import io


def _make_image(tmp_path, name="face.jpg"):
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    path = tmp_path / name
    path.write_bytes(buf.getvalue())
    return str(path)


# ── get_all_persons ───────────────────────────────────────────────────────────

def test_get_all_persons_empty_when_no_file(tmp_path):
    from tagger import get_all_persons
    with patch("tagger.PERSON_MAP_PATH", str(tmp_path / "person_map.json")):
        result = get_all_persons()
    assert result == []


def test_get_all_persons_returns_names(tmp_path):
    from tagger import get_all_persons
    person_map = {"Alice": [0.1, 0.2], "Bob": [0.3, 0.4]}
    person_map_path = tmp_path / "person_map.json"
    person_map_path.write_text(json.dumps(person_map))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        result = get_all_persons()

    assert set(result) == {"Alice", "Bob"}


# ── get_person_embedding ──────────────────────────────────────────────────────

def test_get_person_embedding_known(tmp_path):
    from tagger import get_person_embedding
    embedding = [0.1, 0.2, 0.3]
    person_map_path = tmp_path / "person_map.json"
    person_map_path.write_text(json.dumps({"Alice": embedding}))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        result = get_person_embedding("Alice")

    assert result == embedding


def test_get_person_embedding_unknown_returns_none(tmp_path):
    from tagger import get_person_embedding
    person_map_path = tmp_path / "person_map.json"
    person_map_path.write_text(json.dumps({"Alice": [0.1]}))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        result = get_person_embedding("Charlie")

    assert result is None


# ── add_person_reference ──────────────────────────────────────────────────────

def test_add_person_creates_entry(tmp_path):
    from tagger import add_person_reference, get_person_embedding
    img_path = _make_image(tmp_path)
    person_map_path = tmp_path / "person_map.json"

    fake_face = {"embedding": [0.5, 0.6, 0.7]}

    with patch("tagger.detect_and_embed_faces", return_value=[fake_face]), \
         patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        add_person_reference("Alice", str(tmp_path))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        emb = get_person_embedding("Alice")

    assert emb is not None
    assert len(emb) == 3


def test_add_person_no_faces_does_not_create(tmp_path):
    from tagger import add_person_reference, get_all_persons
    person_map_path = tmp_path / "person_map.json"
    _make_image(tmp_path)

    with patch("tagger.detect_and_embed_faces", return_value=[]), \
         patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        add_person_reference("Nobody", str(tmp_path))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        persons = get_all_persons()

    assert "Nobody" not in persons


def test_add_person_averages_multiple_faces(tmp_path):
    from tagger import add_person_reference, get_person_embedding
    person_map_path = tmp_path / "person_map.json"
    _make_image(tmp_path, "img1.jpg")
    _make_image(tmp_path, "img2.jpg")

    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    call_count = [0]
    def fake_detect(path):
        e = embeddings[call_count[0] % 2]
        call_count[0] += 1
        return [{"embedding": e}]

    with patch("tagger.detect_and_embed_faces", side_effect=fake_detect), \
         patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        add_person_reference("Alice", str(tmp_path))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        emb = get_person_embedding("Alice")

    assert abs(emb[0] - 0.5) < 0.01
    assert abs(emb[1] - 0.5) < 0.01


def test_add_person_appends_does_not_overwrite_others(tmp_path):
    from tagger import add_person_reference, get_all_persons
    person_map_path = tmp_path / "person_map.json"
    person_map_path.write_text(json.dumps({"Bob": [0.1, 0.2]}))
    _make_image(tmp_path)

    with patch("tagger.detect_and_embed_faces", return_value=[{"embedding": [0.9, 0.8]}]), \
         patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        add_person_reference("Alice", str(tmp_path))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        persons = get_all_persons()

    assert "Alice" in persons
    assert "Bob" in persons
