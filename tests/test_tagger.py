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


# ── add_person_reference: multi-face reference images (#13) ────────────────

def test_add_person_skips_multi_face_reference_image(tmp_path):
    """A reference image with more than one detected face (e.g. a group
    photo used by mistake) must be skipped entirely, not blended in."""
    from tagger import add_person_reference, get_person_embedding
    person_map_path = tmp_path / "person_map.json"
    _make_image(tmp_path, "solo.jpg")
    _make_image(tmp_path, "group.jpg")

    def fake_detect(path):
        if "group" in path:
            return [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        return [{"embedding": [0.5, 0.5]}]

    with patch("tagger.detect_and_embed_faces", side_effect=fake_detect), \
         patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        result = add_person_reference("Alice", str(tmp_path))

    assert result["registered"] is True
    assert result["faces_used"] == 1
    assert len(result["skipped_multi_face"]) == 1
    assert "group.jpg" in result["skipped_multi_face"][0]

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        emb = get_person_embedding("Alice")
    assert emb == [0.5, 0.5]


def test_add_person_all_images_multi_face_registers_nothing(tmp_path):
    from tagger import add_person_reference
    person_map_path = tmp_path / "person_map.json"
    _make_image(tmp_path, "group.jpg")

    with patch("tagger.detect_and_embed_faces",
               return_value=[{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]), \
         patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        result = add_person_reference("Nobody", str(tmp_path))

    assert result["registered"] is False
    assert result["faces_used"] == 0
    assert len(result["skipped_multi_face"]) == 1


def test_add_person_reference_strips_name(tmp_path):
    """#14: the name must be .strip()'d before storing, matching the
    cluster-naming path's behavior."""
    from tagger import add_person_reference, get_all_persons
    person_map_path = tmp_path / "person_map.json"
    _make_image(tmp_path)

    with patch("tagger.detect_and_embed_faces", return_value=[{"embedding": [0.5, 0.5]}]), \
         patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        add_person_reference("  Alice  ", str(tmp_path))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        persons = get_all_persons()

    assert persons == ["Alice"]


# ── rename_person (#16) ──────────────────────────────────────────────────────

def test_rename_person_strips_new_name(tmp_path):
    from tagger import rename_person, get_all_persons
    person_map_path = tmp_path / "person_map.json"
    person_map_path.write_text(json.dumps({"Alice": [0.1, 0.2]}))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        rename_person("Alice", "  Alicia  ")
        persons = get_all_persons()

    assert persons == ["Alicia"]


def test_rename_person_rejects_whitespace_only_new_name(tmp_path):
    from tagger import rename_person
    person_map_path = tmp_path / "person_map.json"
    person_map_path.write_text(json.dumps({"Alice": [0.1, 0.2]}))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        with pytest.raises(ValueError):
            rename_person("Alice", "   ")


def test_rename_person_stripped_duplicate_is_rejected(tmp_path):
    """A whitespace-padded name that would collide with an existing person
    once stripped must be rejected, not silently create a near-duplicate."""
    from tagger import rename_person
    person_map_path = tmp_path / "person_map.json"
    person_map_path.write_text(json.dumps({"Alice": [0.1, 0.2], "Bob": [0.3, 0.4]}))

    with patch("tagger.PERSON_MAP_PATH", str(person_map_path)):
        with pytest.raises(ValueError):
            rename_person("Alice", "  Bob  ")


# ── relation / family metadata ────────────────────────────────────────────────

def _relation_env(tmp_path):
    return (patch("tagger.PERSON_MAP_PATH", str(tmp_path / "person_map.json")),
            patch("tagger.PERSON_RELATIONS_PATH", str(tmp_path / "person_relations.json")))


def test_set_relation_and_detailed(tmp_path):
    import tagger
    (tagger.PERSON_MAP_PATH, tagger.PERSON_RELATIONS_PATH)  # touch names
    mp, rp = _relation_env(tmp_path)
    with mp, rp:
        # unknown person → cannot set
        assert tagger.set_relation("Ghost", "daughter") is False
        tagger._save_map({"Hasi": [0.1, 0.2], "Ravi": [0.3, 0.4]})
        assert tagger.set_relation("Hasi", "daughter") is True   # family derived
        assert tagger.set_relation("Ravi", "friend") is True     # not family
        detailed = {p["name"]: p for p in tagger.get_people_detailed()}
        assert detailed["Hasi"]["relation"] == "daughter" and detailed["Hasi"]["is_family"] is True
        assert detailed["Ravi"]["relation"] == "friend" and detailed["Ravi"]["is_family"] is False


def test_set_relation_rejects_unknown_and_clears(tmp_path):
    import tagger
    mp, rp = _relation_env(tmp_path)
    with mp, rp:
        tagger._save_map({"Hasi": [0.1]})
        with pytest.raises(ValueError):
            tagger.set_relation("Hasi", "bestie")   # not in allowed set
        tagger.set_relation("Hasi", "daughter")
        tagger.set_relation("Hasi", "")              # clear
        assert tagger.get_relations() == {}


def test_relation_follows_rename_and_delete(tmp_path):
    import tagger
    mp, rp = _relation_env(tmp_path)
    with mp, rp:
        tagger._save_map({"Hasi": [0.1]})
        tagger.set_relation("Hasi", "daughter")
        tagger.rename_person("Hasi", "Hasini")
        assert tagger.get_relations()["Hasini"]["relation"] == "daughter"
        assert "Hasi" not in tagger.get_relations()
        tagger.delete_person("Hasini")
        assert tagger.get_relations() == {}
