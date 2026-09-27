"""Synthetic coverage for v2 local Browse search and projection migration."""

import json
import sqlite3
from datetime import datetime

import catalog_db


def _photo(name, *, date="", created=0, caption=None, metadata=None):
    row = {
        "path": f"/synthetic/{name}",
        "filename": name,
        "created_at": created,
        "metadata": dict(metadata or {}),
    }
    if date:
        row["metadata"]["date"] = date
    if caption is not None:
        row["caption_json"] = caption if isinstance(caption, dict) else json.dumps(caption)
    return row


def _put(path, rows):
    catalog_db.upsert_images(path, dict(rows))


def _ids(page):
    return [row["id"] for row in page["results"]]


def test_v1_projection_migrates_transactionally_and_reindexes_fields(tmp_path):
    path = str(tmp_path / "projection-v1.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE images (id TEXT PRIMARY KEY, data_json TEXT NOT NULL)")
    conn.execute("CREATE TABLE folders (root TEXT PRIMARY KEY, data_json TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE library_projection ("
        "id TEXT PRIMARY KEY, path TEXT NOT NULL, filename TEXT NOT NULL, "
        "created_at REAL NOT NULL, capture_date TEXT NOT NULL, year TEXT NOT NULL, "
        "month TEXT NOT NULL, media_type TEXT NOT NULL, duration_s REAL NOT NULL, "
        "gps_lat REAL, gps_lon REAL, caption TEXT NOT NULL, occasion TEXT NOT NULL, "
        "search_text TEXT NOT NULL)"
    )
    conn.execute("CREATE TABLE catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO catalog_meta VALUES ('library_projection_version','1')")
    row = _photo(
        "hike.jpg", date="2023:09:17 10:11:12",
        caption={"caption": "Caf\u00e9 hike", "text_in_image": "Trail 42", "scene": "outdoors"},
    )
    conn.execute("INSERT INTO images VALUES ('hike', ?)", (json.dumps(row),))
    conn.commit()
    conn.close()

    page = catalog_db.library_page(path, limit=10, offset=0, q="caf\u00e9 trail")
    assert _ids(page) == ["hike"]
    conn = sqlite3.connect(path)
    version = conn.execute(
        "SELECT value FROM catalog_meta WHERE key='library_projection_version'"
    ).fetchone()[0]
    columns = {row[1] for row in conn.execute("PRAGMA table_info(library_projection)")}
    conn.close()
    assert version == "3"
    assert {"capture_sort_timestamp", "caption_search", "text_search", "camera"} <= columns


def test_query_parser_uses_word_boundaries_phrases_exclusions_and_match_fields(tmp_path):
    path = str(tmp_path / "token-search.db")
    _put(path, [
        ("phrase", _photo("sunset.jpg", caption={"caption": "Beach at sunset"})),
        ("nonphrase", _photo("vacation.jpg", caption={"caption": "Sunset, then beach"})),
        ("cross", _photo("Beach-trip.jpg", caption={"caption": "A dog beside the gate"})),
    ])
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, q='"beach at sunset"', text_scope="caption"
    )) == ["phrase"]
    assert catalog_db.library_page(
        path, limit=10, offset=0, q="-cat", text_scope="caption"
    )["total"] == 3  # -cat does not exclude vacation.
    all_fields = catalog_db.library_page(
        path, limit=10, offset=0, q="dog beach", text_scope="all"
    )
    assert _ids(all_fields) == ["cross"]
    assert all_fields["results"][0]["match_fields"] == ["filename", "caption"]


def test_field_scopes_search_only_latest_caption_and_current_attributes(tmp_path):
    path = str(tmp_path / "scopes.db")
    _put(path, [(
        "photo",
        {
            **_photo("beach-day.jpg", caption={
                "caption": "Quiet shore", "objects": ["red kayak"],
                "text_in_image": "MUSEUM 25", "landmark": "Old Pier",
            }),
            "caption_history": [{
                "caption_json": json.dumps({"caption": "birthday cake"}),
            }],
            "metadata": {"camera_make": "Canon", "camera_model": "Canon EOS R5"},
        },
    )])
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, q="kayak", text_scope="details"
    )) == ["photo"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, q="museum", text_scope="text"
    )) == ["photo"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, q="pier", text_scope="place"
    )) == ["photo"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, q="canon eos", text_scope="camera"
    )) == ["photo"]
    assert catalog_db.library_page(
        path, limit=10, offset=0, q="birthday", text_scope="all"
    )["total"] == 0


def test_iso_date_ranges_calendar_shortcuts_and_capture_time_sort(tmp_path):
    path = str(tmp_path / "dates.db")
    epoch = datetime(2022, 5, 6, 8, 30).timestamp()
    _put(path, [
        ("morning", _photo("morning.jpg", date="2024:05:06 08:30:00", created=epoch)),
        ("evening", _photo("evening.jpg", date="2024:05:06 19:45:00", created=epoch - 1000)),
        ("other-year", _photo("other.jpg", date="2022:05:06 12:00:00", created=epoch)),
        # Invalid EXIF and filename candidates fall through to created_at.
        ("fallback", _photo(
            "IMG_20240231.jpg", date="2024:13:40 99:00:00",
            created=datetime(2021, 3, 4, 15, 16).timestamp(),
        )),
    ])
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, sort="newest"
    )) == ["evening", "morning", "other-year", "fallback"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, sort="oldest"
    )) == ["fallback", "other-year", "morning", "evening"]
    on_this_day = catalog_db.library_page(
        path, limit=10, offset=0, month=5, day=6, sort="oldest"
    )
    assert set(_ids(on_this_day)) == {"morning", "evening", "other-year"}
    ranged = catalog_db.library_page(
        path, limit=10, offset=0, date_from="2024-05-06", date_to="2024-05-06"
    )
    assert set(_ids(ranged)) == {"morning", "evening"}
    fallback = next(row for row in catalog_db.library_page(
        path, limit=10, offset=0, year="2021"
    )["results"] if row["id"] == "fallback")
    assert fallback["date"].startswith("2021:03:04")


def test_capture_date_parser_rejects_garbage_suffix_before_fallback(tmp_path):
    path = str(tmp_path / "strict-date-parse.db")
    imported_at = datetime(2020, 7, 8, 9, 10).timestamp()
    _put(path, [(
        "invalid-suffix",
        _photo("no-date-name.jpg", date="2024-02-10 12:garbage", created=imported_at),
    )])
    row = catalog_db.library_page(path, limit=10, offset=0)["results"][0]
    assert row["year"] == "2020"
    assert row["date"].startswith("2020:07:08")


def test_exact_facets_canonical_values_counts_and_stored_presence(tmp_path):
    path = str(tmp_path / "facets.db")
    _put(path, [
        ("a", _photo("a.jpg", caption={
            "photo_type": " Photo ", "scene": "Outdoor", "weather": "Sunny",
            "occasion": "Hike", "time_of_day": "Morning", "text_in_image": "Trail",
            "caption": "Trail notes",
        }, metadata={"camera_make": "Canon", "camera_model": "Canon EOS R5",
                     "gps_lat": 0, "gps_lon": 0})),
        ("b", _photo("b.jpg", caption={
            "photo_type": "photo", "scene": " outdoor ", "weather": "sunny",
            "occasion": "hike", "time_of_day": " morning ", "caption": "Unknown",
        }, metadata={"camera_make": "canon", "camera_model": "eos r5"})),
        ("c", _photo("c.jpg", caption={"scene": "unknown", "text_in_image": "unknown"},
                     metadata={"gps_lat": 91, "gps_lon": 0})),
    ])
    facets = catalog_db.library_facets(path)
    assert facets["facets"]["scene"] == [{"value": "outdoor", "count": 2}]
    assert facets["facets"]["photo_type"] == [{"value": "photo", "count": 2}]
    assert facets["facets"]["camera"] == [{"value": "canon eos r5", "count": 2}]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, photo_type=" PHOTO "
    )) == ["b", "a"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, has_location="yes"
    )) == ["a"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, has_location="no"
    )) == ["c", "b"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, has_text="yes"
    )) == ["a"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, has_text="no"
    )) == ["c", "b"]
    assert _ids(catalog_db.library_page(
        path, limit=10, offset=0, has_caption="no"
    )) == ["c", "b"]


def test_recognized_text_presence_uses_the_same_effective_text_as_search(tmp_path):
    path = str(tmp_path / "text-presence.db")
    _put(path, [
        ("empty", _photo("empty.jpg", caption={"text_in_image": ""}, metadata={"ocr": "invoice"})),
        ("placeholder", _photo("placeholder.jpg", caption={"text_in_image": "unknown"}, metadata={"recognized_text": "invoice"})),
        ("missing", _photo("missing.jpg", caption={"text_in_image": "unknown"}, metadata={"ocr": "none"})),
    ])
    assert _ids(catalog_db.library_page(path, limit=10, offset=0, text_scope="text", q="invoice", has_text="yes")) == ["placeholder", "empty"]
    assert _ids(catalog_db.library_page(path, limit=10, offset=0, has_text="no")) == ["missing"]
    assert catalog_db.library_page(path, limit=10, offset=0, text_scope="text", q="unknown")["total"] == 0


def test_query_and_date_validation_rejects_instead_of_dropping_constraints(tmp_path):
    import pytest
    path = str(tmp_path / "validation.db")
    _put(path, [("a", _photo("a.jpg", caption={"caption": "safe"}))])
    for query in ('"unfinished', 'left"right', '""', "-", " ".join(["x"] * 21)):
        with pytest.raises(ValueError):
            catalog_db.library_page(path, limit=10, offset=0, q=query)
    with pytest.raises(ValueError, match="date_from"):
        catalog_db.library_page(path, limit=10, offset=0, date_from="2025-02-29")
    with pytest.raises(ValueError, match="day requires month"):
        catalog_db.library_page(path, limit=10, offset=0, day=2)
