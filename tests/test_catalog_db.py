import catalog_db


def _db(tmp_path, name="catalog.db"):
    return str(tmp_path / name)


def test_load_all_on_missing_file_returns_empty(tmp_path):
    path = _db(tmp_path)
    assert catalog_db.load_all(path) == {"images": {}, "folders": {}}


def test_upsert_then_load_roundtrips(tmp_path):
    path = _db(tmp_path)
    catalog_db.upsert_images(path, {"a": {"path": "/a.jpg", "filename": "a.jpg"}})
    data = catalog_db.load_all(path)
    assert data["images"] == {"a": {"path": "/a.jpg", "filename": "a.jpg"}}


def test_upsert_overwrites_existing_row(tmp_path):
    path = _db(tmp_path)
    catalog_db.upsert_images(path, {"a": {"filename": "old.jpg"}})
    catalog_db.upsert_images(path, {"a": {"filename": "new.jpg"}})
    data = catalog_db.load_all(path)
    assert data["images"]["a"]["filename"] == "new.jpg"


def test_upsert_is_purely_additive_does_not_delete_other_rows(tmp_path):
    """The incremental path (used for per-batch job saves) must never drop
    rows outside the given dict — only save_all (full scan sync) does that."""
    path = _db(tmp_path)
    catalog_db.upsert_images(path, {"a": {}, "b": {}})
    catalog_db.upsert_images(path, {"a": {"updated": True}})
    data = catalog_db.load_all(path)
    assert set(data["images"]) == {"a", "b"}


def test_delete_images_removes_only_given_ids(tmp_path):
    path = _db(tmp_path)
    catalog_db.upsert_images(path, {"a": {}, "b": {}, "c": {}})
    catalog_db.delete_images(path, ["b"])
    data = catalog_db.load_all(path)
    assert set(data["images"]) == {"a", "c"}


def test_save_all_syncs_removes_rows_missing_from_dict(tmp_path):
    """save_all (scan checkpoint path) must match the old images.json
    full-overwrite semantics: a row absent from the given dict is deleted."""
    path = _db(tmp_path)
    catalog_db.save_all(path, {"a": {}, "b": {}}, {})
    catalog_db.save_all(path, {"a": {}}, {})  # "b" retired/removed by scanner
    data = catalog_db.load_all(path)
    assert set(data["images"]) == {"a"}


def test_save_all_syncs_folders_too(tmp_path):
    path = _db(tmp_path)
    catalog_db.save_all(path, {}, {"/old/folder": {"count": 5}})
    catalog_db.save_all(path, {}, {"/new/folder": {"count": 2}})
    data = catalog_db.load_all(path)
    assert data["folders"] == {"/new/folder": {"count": 2}}


def test_upsert_images_noop_on_empty_dict(tmp_path):
    path = _db(tmp_path)
    catalog_db.upsert_images(path, {"a": {}})
    catalog_db.upsert_images(path, {})
    assert set(catalog_db.load_all(path)["images"]) == {"a"}


def test_delete_images_noop_on_empty_list(tmp_path):
    path = _db(tmp_path)
    catalog_db.upsert_images(path, {"a": {}})
    catalog_db.delete_images(path, [])
    assert set(catalog_db.load_all(path)["images"]) == {"a"}


def test_version_unwritten_path_is_zero(tmp_path):
    assert catalog_db.version(str(tmp_path / "nope.db")) == 0


def test_version_increments_on_write(tmp_path):
    path = _db(tmp_path)
    v0 = catalog_db.version(path)
    catalog_db.upsert_images(path, {"a": {}})
    v1 = catalog_db.version(path)
    assert v1 > v0


def test_version_unchanged_on_noop_write(tmp_path):
    path = _db(tmp_path)
    catalog_db.upsert_images(path, {"a": {}})
    v1 = catalog_db.version(path)
    catalog_db.upsert_images(path, {})  # no-op: empty dict
    assert catalog_db.version(path) == v1


def test_version_bumps_on_save_all_delete(tmp_path):
    path = _db(tmp_path)
    catalog_db.save_all(path, {"a": {}, "b": {}}, {})
    v1 = catalog_db.version(path)
    catalog_db.save_all(path, {"a": {}}, {})  # "b" removed
    assert catalog_db.version(path) > v1
