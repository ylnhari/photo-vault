"""Duplicate grouping + trash round-trip + person management."""
import json
from unittest.mock import patch, MagicMock

from PIL import Image
import pytest


# ── dupes ─────────────────────────────────────────────────────────────────────

def test_dhash_stable_and_resize_invariant(tmp_path):
    from dupes import dhash
    img = Image.new("RGB", (200, 150))
    for x in range(200):
        for y in range(150):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    p1 = tmp_path / "a.jpg"; img.save(p1, quality=95)
    p2 = tmp_path / "b.jpg"; img.resize((100, 75)).save(p2, quality=70)
    h1, h2 = dhash(str(p1)), dhash(str(p2))
    assert h1 == dhash(str(p1))                       # deterministic
    diff = bin(int(h1, 16) ^ int(h2, 16)).count("1")
    assert diff <= 6                                   # resize/recompress ≈ same


def test_group_duplicates_groups_and_sorts():
    from dupes import group_duplicates
    catalog = {
        "a": {"dhash": "00000000000000ff"},
        "b": {"dhash": "00000000000000fe"},  # 1 bit from a
        "c": {"dhash": "ffffffffffffffff"},  # unrelated
        "d": {"dhash": "00000000000000ff"},  # identical to a
        "e": {},                              # not hashed → skipped
    }
    groups = group_duplicates(catalog, threshold=4)
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b", "d"}


# ── trash round-trip ──────────────────────────────────────────────────────────

def test_trash_add_take_purge(tmp_path, monkeypatch):
    import trash
    monkeypatch.setattr(trash, "TRASH_PATH", str(tmp_path / "trash.json"))
    trash.add("id1", {"path": "/a.jpg", "filename": "a.jpg"})
    trash.add("id2", {"path": "/b.jpg", "filename": "b.jpg"}, file_deleted=True)
    items = trash.list_items()
    assert set(items) == {"id1", "id2"}
    assert items["id2"]["file_deleted"] is True

    got = trash.take(["id1"])
    assert got == {"id1": {"path": "/a.jpg", "filename": "a.jpg"}}
    assert set(trash.list_items()) == {"id2"}

    dropped = trash.purge(None)
    assert dropped == ["id2"]
    assert trash.list_items() == {}


# ── person management ─────────────────────────────────────────────────────────

def test_person_rename_and_delete(tmp_path, monkeypatch):
    import tagger
    monkeypatch.setattr(tagger, "PERSON_MAP_PATH", str(tmp_path / "pm.json"))
    tagger.add_person_embedding("Alice", [0.1, 0.2])
    tagger.rename_person("Alice", "Alicia")
    assert tagger.get_all_persons() == ["Alicia"]
    with pytest.raises(KeyError):
        tagger.rename_person("Nobody", "X")
    tagger.add_person_embedding("Bob", [0.3])
    with pytest.raises(ValueError):
        tagger.rename_person("Bob", "Alicia")
    assert tagger.delete_person("Bob") is True
    assert tagger.delete_person("Bob") is False
    assert tagger.get_all_persons() == ["Alicia"]
