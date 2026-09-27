"""Synthetic regression coverage for Browse query indexes and FTS fallback."""

import json
import sqlite3

import pytest

import catalog_db


def _photo(filename, caption, *, date="2024:01:02 12:00:00", history=None):
    row = {
        "path": f"/synthetic/{filename}",
        "filename": filename,
        "created_at": 0,
        "metadata": {"date": date} if date else {},
        "caption_json": json.dumps(caption),
    }
    if history is not None:
        row["caption_history"] = history
    return row


def _seed_v2(path, records):
    """Create projection data, then roll its marker back to emulate v2."""
    catalog_db.save_all(path, records, {})
    conn = catalog_db._connect(path)
    with catalog_db._lock:
        triggers = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='library_projection'"
        )]
        for trigger in triggers:
            conn.execute(f'DROP TRIGGER "{trigger}"')
        objects = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='library_search_fts'"
        )]
        if objects:
            conn.execute("DROP TABLE library_search_fts")
        conn.execute(
            "UPDATE catalog_meta SET value='2' WHERE key='library_projection_version'"
        )
        conn.commit()


def _fts_available():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(value)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _traced_page(path, **query):
    conn = catalog_db._connect(path)
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        page = catalog_db.library_page(path, limit=20, offset=0, **query)
    finally:
        conn.set_trace_callback(None)
    return page, statements


@pytest.mark.parametrize("direction", ["newest", "oldest"])
def test_capture_sort_index_avoids_temporary_ordering_tree(tmp_path, direction):
    path = str(tmp_path / f"sort-{direction}.db")
    _seed_v2(path, {
        "a": _photo("a.jpg", {"caption": "A"}),
        "b": _photo("b.jpg", {"caption": "B"}, date="2023:01:02 09:30:00"),
    })
    conn = catalog_db._connect(path)
    order = (
        "(capture_sort_date = '') ASC, capture_sort_timestamp DESC, created_at DESC, id DESC"
        if direction == "newest" else
        "(capture_sort_date = '') ASC, capture_sort_timestamp ASC, created_at DESC, id DESC"
    )
    plan = [row[3] for row in conn.execute(
        f"EXPLAIN QUERY PLAN SELECT id FROM library_projection ORDER BY {order} LIMIT 20"
    )]
    assert not any("USE TEMP B-TREE FOR ORDER BY" in detail for detail in plan)
    assert any("library_browse_" in detail for detail in plan)


def test_v2_migration_rebuilds_fts_and_current_caption_search_only(tmp_path):
    if not _fts_available():
        pytest.skip("SQLite build does not include FTS5")
    path = str(tmp_path / "fts-migration.db")
    _seed_v2(path, {
        "one": _photo(
            "trail.jpg",
            {"caption": "Quiet lake", "objects": ["canoe"], "text_in_image": "TRAIL 42"},
            history=[{"caption_json": json.dumps({"caption": "birthday cake"})}],
        ),
        "two": _photo("cake.jpg", {"caption": "Birthday cake"}),
    })

    assert catalog_db.library_page(
        path, limit=20, offset=0, q="canoe", text_scope="details"
    )["total"] == 1
    conn = catalog_db._connect(path)
    version = conn.execute(
        "SELECT value FROM catalog_meta WHERE key='library_projection_version'"
    ).fetchone()[0]
    fts_count = conn.execute(
        "SELECT COUNT(*) FROM library_search_fts WHERE library_search_fts MATCH 'birthday'"
    ).fetchone()[0]
    assert version == "3"
    assert fts_count == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM library_search_fts WHERE library_search_fts MATCH 'canoe'"
    ).fetchone()[0] == 1


def test_ascii_queries_use_fts_candidates_but_keep_exact_semantics(tmp_path):
    if not _fts_available():
        pytest.skip("SQLite build does not include FTS5")
    path = str(tmp_path / "fts-candidates.db")
    _seed_v2(path, {
        "exact": _photo("cat.jpg", {"caption": "A cat by the gate"}),
        "substring": _photo("vacation.jpg", {"caption": "Summer trip"}),
        "excluded": _photo("dog.jpg", {"caption": "Dog near a cat"}),
    })
    page, statements = _traced_page(path, q="cat -dog", text_scope="all")
    assert [row["id"] for row in page["results"]] == ["exact"]
    assert any("library_search_fts" in sql and "MATCH" in sql.upper() for sql in statements)

    # FTS token matching is only a coarse candidate selector; the original
    # boundary-aware INSTR predicate remains authoritative for final results.
    without_substring, _ = _traced_page(path, q="cat", text_scope="all")
    assert {row["id"] for row in without_substring["results"]} == {"exact", "excluded"}


@pytest.mark.parametrize("query", ["雪", '"%"', '"_"'])
def test_unicode_and_punctuation_only_queries_bypass_fts(tmp_path, query):
    if not _fts_available():
        pytest.skip("SQLite build does not include FTS5")
    path = str(tmp_path / "fts-fallback.db")
    _seed_v2(path, {
        "one": _photo("one.jpg", {"caption": "雪 % _"}),
        "two": _photo("two.jpg", {"caption": "clear day"}),
    })
    page, statements = _traced_page(path, q=query, text_scope="caption")
    assert [row["id"] for row in page["results"]] == ["one"]
    assert not any("library_search_fts" in sql and "MATCH" in sql.upper()
                   for sql in statements)


def test_fts_triggers_track_projection_upserts_and_deletes(tmp_path):
    if not _fts_available():
        pytest.skip("SQLite build does not include FTS5")
    path = str(tmp_path / "fts-triggers.db")
    _seed_v2(path, {"one": _photo("one.jpg", {"caption": "green forest"})})
    catalog_db.library_page(path, limit=10, offset=0)  # upgrade/rebuild
    assert catalog_db.library_page(path, limit=10, offset=0, q="forest")["total"] == 1

    catalog_db.upsert_images(path, {
        "one": _photo("one.jpg", {"caption": "blue ocean"}),
    })
    assert catalog_db.library_page(path, limit=10, offset=0, q="forest")["total"] == 0
    assert catalog_db.library_page(path, limit=10, offset=0, q="ocean")["total"] == 1

    catalog_db.delete_images(path, ["one"])
    assert catalog_db.library_page(path, limit=10, offset=0, q="ocean")["total"] == 0


def test_plain_sql_fallback_preserves_results_when_fts_table_is_unavailable(tmp_path, monkeypatch):
    path = str(tmp_path / "fts-unavailable.db")
    _seed_v2(path, {
        "exact": _photo("cat.jpg", {"caption": "A cat beside a window"}),
        "substring": _photo("vacation.jpg", {"caption": "Sunny coast"}),
    })
    # Simulate a SQLite build without FTS5 while preserving the v3 migration.
    monkeypatch.setattr(catalog_db, "_create_search_index_locked", lambda conn: None)
    page, statements = _traced_page(path, q="cat", text_scope="all")
    assert [row["id"] for row in page["results"]] == ["exact"]
    assert not any("library_search_fts" in sql and "MATCH" in sql.upper()
                   for sql in statements)


def test_failed_fts_migration_rolls_back_and_retry_rebuilds(tmp_path, monkeypatch):
    if not _fts_available():
        pytest.skip("SQLite build does not include FTS5")
    path = str(tmp_path / "fts-migration-rollback.db")
    _seed_v2(path, {"one": _photo("one.jpg", {"caption": "green forest"})})
    real_create = catalog_db._create_search_index_locked

    def fail_after_projection_rebuild(conn):
        raise RuntimeError("synthetic FTS migration failure")

    monkeypatch.setattr(catalog_db, "_create_search_index_locked", fail_after_projection_rebuild)
    with pytest.raises(RuntimeError, match="synthetic FTS migration failure"):
        catalog_db.library_page(path, limit=10, offset=0)

    conn = catalog_db._connect(path)
    assert conn.execute(
        "SELECT value FROM catalog_meta WHERE key='library_projection_version'"
    ).fetchone()[0] == "2"
    assert conn.execute("SELECT COUNT(*) FROM library_projection WHERE id='one'").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='library_search_fts'"
    ).fetchone()[0] == 0

    monkeypatch.setattr(catalog_db, "_create_search_index_locked", real_create)
    assert catalog_db.library_page(path, limit=10, offset=0, q="forest")["total"] == 1
    assert conn.execute(
        "SELECT value FROM catalog_meta WHERE key='library_projection_version'"
    ).fetchone()[0] == "3"
