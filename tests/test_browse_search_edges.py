"""Adversarial source/synthetic tests for the local Browse search contract.

These tests use temporary SQLite rows only. They must never initialize Chroma,
providers, or read the user's catalog.
"""
import json

import pytest

import catalog_db


def _row(name, *, date="", created=1, caption="", metadata=None,
         media_type="image", caption_history=None):
    payload = {
        "path": f"/synthetic/{name}",
        "filename": name,
        "created_at": created,
        "media_type": media_type,
        "metadata": dict(metadata or {}),
        "caption_json": json.dumps(caption) if isinstance(caption, dict) else
        json.dumps({"caption": caption}),
    }
    if date:
        payload["metadata"]["date"] = date
    if caption_history is not None:
        payload["caption_history"] = caption_history
    return payload


def _seed(path, records):
    catalog_db.upsert_images(path, {key: value for key, value in records})


def _ids(page):
    return [row["id"] for row in page["results"]]


def test_unknown_capture_dates_stay_last_for_newest_and_oldest(tmp_path):
    path = str(tmp_path / "dates.db")
    _seed(path, [
        ("new", _row("new.jpg", date="2024:06:01 12:00:00", created=2)),
        ("old", _row("old.jpg", date="2018:06:01 12:00:00", created=3)),
        ("unknown-z", _row("unknown-z.jpg", created=0)),
        ("unknown-a", _row("unknown-a.jpg", created=0)),
    ])

    newest = catalog_db.library_page(path, limit=10, offset=0, sort="newest")
    oldest = catalog_db.library_page(path, limit=10, offset=0, sort="oldest")
    assert _ids(newest) == ["new", "old", "unknown-z", "unknown-a"]
    assert _ids(oldest) == ["old", "new", "unknown-z", "unknown-a"]


def test_created_timestamp_is_used_as_capture_date_fallback(tmp_path):
    from datetime import datetime

    path = str(tmp_path / "import-date.db")
    imported_at = 1717200000
    _seed(path, [("imported", _row("undated.jpg", created=imported_at))])
    result = catalog_db.library_page(path, limit=10, offset=0)
    assert result["results"][0]["date"] == datetime.fromtimestamp(imported_at).strftime(
        "%Y:%m:%d %H:%M:%S"
    )


def test_caption_scope_uses_current_caption_not_caption_history(tmp_path):
    path = str(tmp_path / "caption-history.db")
    _seed(path, [
        ("item", _row(
            "memory.jpg",
            caption={"caption": "Quiet lake", "objects": ["canoe"]},
            caption_history=[{
                "model": "old-model",
                "caption_json": json.dumps({"caption": "birthday cake"}),
            }],
        )),
    ])

    current = catalog_db.library_page(
        path, limit=10, offset=0, q="quiet", text_scope="caption"
    )
    stale = catalog_db.library_page(
        path, limit=10, offset=0, q="birthday", text_scope="caption"
    )
    current_all = catalog_db.library_page(
        path, limit=10, offset=0, q="birthday", text_scope="all"
    )
    details = catalog_db.library_page(
        path, limit=10, offset=0, q="canoe", text_scope="details"
    )
    assert _ids(current) == ["item"]
    assert stale["total"] == current_all["total"] == 0
    assert _ids(details) == ["item"]


def test_unicode_normalization_phrase_punctuation_and_literal_wildcards(tmp_path):
    path = str(tmp_path / "unicode.db")
    _seed(path, [
        ("literal", _row(
            "literal.jpg",
            caption="Café 雪 100%_done O'Neil",
        )),
        ("near", _row("near.jpg", caption="Cafe 雪 1000 done O Neil")),
    ])

    composed = catalog_db.library_page(
        path, limit=10, offset=0, q="CAFÉ", text_scope="caption"
    )
    decomposed = catalog_db.library_page(
        path, limit=10, offset=0, q="CAFE\u0301", text_scope="caption"
    )
    phrase = catalog_db.library_page(
        path, limit=10, offset=0, q='"雪 100%_done"', text_scope="caption"
    )
    punctuation = catalog_db.library_page(
        path, limit=10, offset=0, q='"O\'Neil"', text_scope="caption"
    )
    wildcard = catalog_db.library_page(
        path, limit=10, offset=0, q="100%_done", text_scope="caption"
    )
    assert _ids(composed) == ["literal"]
    assert _ids(decomposed) == ["literal"]
    assert _ids(phrase) == ["literal"]
    assert _ids(punctuation) == ["literal"]
    assert _ids(wildcard) == ["literal"]


@pytest.mark.parametrize("kwargs", [
    {"date_from": "2025-02-29"},
    {"date_to": "2024-13-01"},
    {"date_from": "2024-02-03", "date_to": "2024-02-02"},
    {"month": 2, "day": 30},
    {"day": 1},
])
def test_invalid_date_filters_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        catalog_db.library_page(
            str(tmp_path / "invalid-query-date.db"),
            limit=10, offset=0, **kwargs,
        )


def test_malformed_stored_capture_date_is_not_a_range_match(tmp_path):
    path = str(tmp_path / "malformed-capture.db")
    _seed(path, [
        ("bad-date", _row("bad.jpg", date="2024:02:30 00:00:00")),
        ("good-date", _row("good.jpg", date="2024:02:29 00:00:00")),
    ])
    page = catalog_db.library_page(
        path, limit=10, offset=0,
        date_from="2024-02-01", date_to="2024-02-29",
    )
    assert _ids(page) == ["good-date"]


def test_and_terms_can_match_different_fields_and_match_fields_explain(tmp_path):
    path = str(tmp_path / "scopes.db")
    _seed(path, [
        ("cross", _row("Beach-trip.jpg", caption="A dog beside the gate")),
        ("caption-only", _row("ordinary.jpg", caption="A dog at the beach")),
    ])
    all_fields = catalog_db.library_page(
        path, limit=10, offset=0, q="dog beach", text_scope="all"
    )
    caption_only = catalog_db.library_page(
        path, limit=10, offset=0, q="dog beach", text_scope="caption"
    )
    excluded_only = catalog_db.library_page(
        path, limit=10, offset=0, q="-gate", text_scope="all"
    )

    assert _ids(all_fields) == ["cross", "caption-only"]
    cross = next(row for row in all_fields["results"] if row["id"] == "cross")
    assert cross["match_fields"] == ["filename", "caption"]
    assert _ids(caption_only) == ["caption-only"]
    assert all(row["match_fields"] == [] for row in excluded_only["results"])


def test_gps_zero_zero_is_valid_but_invalid_and_absent_pairs_are_not_yes(tmp_path):
    path = str(tmp_path / "gps-state.db")
    _seed(path, [
        ("zero", _row("zero.jpg", metadata={"gps_lat": 0, "gps_lon": 0})),
        ("invalid", _row("invalid.jpg", metadata={"gps_lat": 91, "gps_lon": 0})),
        ("absent", _row("absent.jpg")),
        ("place-text", _row("place.jpg", metadata={"place": "Paris"})),
    ])
    yes = catalog_db.library_page(
        path, limit=10, offset=0, has_location="yes"
    )
    no = catalog_db.library_page(
        path, limit=10, offset=0, has_location="no"
    )

    assert _ids(yes) == ["zero"]
    # "No" means no valid stored GPS pair, including partial/invalid data and
    # textual place labels. It makes no claim about the actual photo location.
    assert _ids(no) == ["place-text", "invalid", "absent"]
