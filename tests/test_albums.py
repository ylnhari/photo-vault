import json
import pytest


@pytest.fixture
def albums_file(tmp_path, monkeypatch):
    p = tmp_path / "albums.json"
    monkeypatch.setattr("albums.ALBUMS_PATH", str(p))
    monkeypatch.setattr("albums.DATA_DIR", str(tmp_path))
    return p


def test_create_and_list(albums_file):
    import albums
    a = albums.create_album("Vacation")
    assert a["name"] == "Vacation"
    listed = albums.list_albums()
    assert len(listed) == 1
    assert listed[0]["count"] == 0
    assert listed[0]["cover"] is None


def test_create_requires_name(albums_file):
    import albums
    with pytest.raises(ValueError):
        albums.create_album("   ")


def test_add_dedups_and_sets_cover(albums_file):
    import albums
    aid = albums.create_album("A")["id"]
    albums.add_to_album(aid, ["x", "y", "x"])  # duplicate x
    assert albums.add_to_album(aid, ["y", "z"]) == 3  # y already present
    listed = albums.list_albums()
    assert listed[0]["count"] == 3
    assert listed[0]["cover"] == "x"


def test_remove_and_remove_from_all(albums_file):
    import albums
    a1 = albums.create_album("A")["id"]
    a2 = albums.create_album("B")["id"]
    albums.add_to_album(a1, ["x", "y"])
    albums.add_to_album(a2, ["x"])
    albums.remove_from_album(a1, ["y"])
    assert albums.get_album(a1)["image_ids"] == ["x"]
    albums.remove_image_from_all("x")
    assert albums.get_album(a1)["image_ids"] == []
    assert albums.get_album(a2)["image_ids"] == []


def test_rename_and_delete(albums_file):
    import albums
    aid = albums.create_album("Old")["id"]
    albums.rename_album(aid, "New")
    assert albums.get_album(aid)["name"] == "New"
    assert albums.delete_album(aid) is True
    assert albums.get_album(aid) is None


def test_add_to_missing_album_raises(albums_file):
    import albums
    with pytest.raises(KeyError):
        albums.add_to_album("nope", ["x"])
