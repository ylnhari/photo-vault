"""SQLite catalog persistence plus an indexed, lightweight gallery projection.

The full per-image record remains authoritative in the JSON payload for
indexing and compatibility. A narrow SQL projection supports bounded gallery
reads without deserializing the whole catalog. Writes keep both representations
in one transaction. Incremental upserts also avoid rewriting the old flat
images.json snapshot for each indexing batch.
"""
import json
import calendar
import os
import re
import sqlite3
import threading
import unicodedata
from photo_date import _date_from_filename

_lock = threading.Lock()
_conn_cache: dict[str, sqlite3.Connection] = {}
# In-process write counter per db_path, used for cache invalidation instead
# of file mtime: under WAL mode, writes land in the .db-wal file and the main
# .db file's mtime doesn't necessarily change, so an mtime-keyed cache (the
# old images.json approach) can silently go stale. Since the cache itself is
# an in-memory dict that never survives a process restart anyway, an
# in-process counter is both correct and simpler.
_version: dict[str, int] = {}
_PROJECTION_VERSION = 3

_SEARCH_COLUMNS = {
    "filename": "filename_search",
    "caption": "caption_search",
    "details": "details_search",
    "text": "text_search",
    "place": "place_search",
    "camera": "camera_search",
}
_SCOPE_FIELDS = {
    "all": tuple(_SEARCH_COLUMNS),
    "filename": ("filename",),
    "caption": ("caption",),
    "details": ("details",),
    "text": ("text",),
    "place": ("place",),
    "camera": ("camera",),
}
_FACET_COLUMNS = {
    "photo_type": "photo_type",
    "scene": "scene",
    "weather": "weather",
    "occasion": "occasion",
    "time_of_day": "time_of_day",
    "camera": "camera",
}
_NEW_PROJECTION_COLUMNS = {
    "capture_sort_date": "TEXT NOT NULL DEFAULT ''",
    "capture_sort_timestamp": "TEXT NOT NULL DEFAULT ''",
    "filename_search": "TEXT NOT NULL DEFAULT ''",
    "caption_search": "TEXT NOT NULL DEFAULT ''",
    "details_search": "TEXT NOT NULL DEFAULT ''",
    "text_search": "TEXT NOT NULL DEFAULT ''",
    "place_search": "TEXT NOT NULL DEFAULT ''",
    "camera_search": "TEXT NOT NULL DEFAULT ''",
    "photo_type": "TEXT NOT NULL DEFAULT ''",
    "scene": "TEXT NOT NULL DEFAULT ''",
    "weather": "TEXT NOT NULL DEFAULT ''",
    "time_of_day": "TEXT NOT NULL DEFAULT ''",
    "camera": "TEXT NOT NULL DEFAULT ''",
    "text_state": "INTEGER",
    "location_state": "INTEGER",
    "caption_state": "INTEGER NOT NULL DEFAULT 0",
}
_PROJECTION_COLUMNS = (
    "id", "path", "filename", "created_at", "capture_date", "year", "month",
    "media_type", "duration_s", "gps_lat", "gps_lon", "caption", "occasion",
    "search_text", *_NEW_PROJECTION_COLUMNS,
)
_UNKNOWN_FACETS = {"unknown", "n/a", "na", "none", "null", "unspecified", "not known"}
_QUERY_TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def _bump(db_path: str):
    _version[db_path] = _version.get(db_path, 0) + 1


def _connect(db_path: str) -> sqlite3.Connection:
    with _lock:
        conn = _conn_cache.get(db_path)
        if conn is None:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            # Without this, sqlite3's default ~5s connect timeout means any
            # contention (e.g. a concurrent writer holding the lock a beat
            # too long) raises "database is locked" immediately instead of
            # waiting — and that error used to feed straight into
            # scanner.load_existing_data()'s catch-all, which treated it as
            # "empty catalog" and wiped the DB on the next save. Give
            # concurrent access a real window to wait and retry.
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS images "
                "(id TEXT PRIMARY KEY, data_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS folders "
                "(root TEXT PRIMARY KEY, data_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS library_projection ("
                "id TEXT PRIMARY KEY, path TEXT NOT NULL, filename TEXT NOT NULL, "
                "created_at REAL NOT NULL, capture_date TEXT NOT NULL, "
                "year TEXT NOT NULL, month TEXT NOT NULL, media_type TEXT NOT NULL, "
                "duration_s REAL NOT NULL, gps_lat REAL, gps_lon REAL, "
                "caption TEXT NOT NULL, occasion TEXT NOT NULL, search_text TEXT NOT NULL)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS library_recent_idx ON library_projection(created_at DESC, id DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS library_timeline_idx ON library_projection(capture_date DESC, created_at DESC, id DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS library_year_idx ON library_projection(year, capture_date DESC, created_at DESC, id DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS library_media_idx ON library_projection(media_type, capture_date DESC, created_at DESC, id DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS library_gps_idx ON library_projection(gps_lat, gps_lon) WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL")
            conn.commit()
            _conn_cache[db_path] = conn
        return conn


def load_all(db_path: str) -> dict:
    """Returns {"images": {id: data}, "folders": {root: data}} — the same
    shape images.json used to hold, so callers don't need to change."""
    conn = _connect(db_path)
    with _lock:
        images = {
            row[0]: json.loads(row[1])
            for row in conn.execute("SELECT id, data_json FROM images")
        }
        folders = {
            row[0]: json.loads(row[1])
            for row in conn.execute("SELECT root, data_json FROM folders")
        }
    return {"images": images, "folders": folders}


def upsert_images(db_path: str, images: dict):
    """Insert/update only the given {id: data} rows — the incremental-write
    path used for per-batch job saves."""
    if not images:
        return
    serialized = [(iid, json.dumps(data)) for iid, data in images.items()]
    projection_rows = [_projection_row(iid, data) for iid, data in images.items()]
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _upsert_prepared_locked(conn, serialized, projection_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    _bump(db_path)


def delete_images(db_path: str, ids):
    ids = list(ids)
    if not ids:
        return
    conn = _connect(db_path)
    with _lock:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany("DELETE FROM images WHERE id = ?", [(i,) for i in ids])
            conn.executemany("DELETE FROM library_projection WHERE id = ?", [(i,) for i in ids])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    _bump(db_path)


def save_folders(db_path: str, folders: dict):
    if not folders:
        return
    conn = _connect(db_path)
    with _lock:
        prepared = [(root, json.dumps(data)) for root, data in folders.items()]
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "INSERT INTO folders(root, data_json) VALUES (?, ?) "
                "ON CONFLICT(root) DO UPDATE SET data_json=excluded.data_json", prepared,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    _bump(db_path)


def save_all(db_path: str, images: dict, folders: dict):
    """Full sync for both tables — used by scan checkpoints, which always pass
    the COMPLETE current in-memory catalog. Matches the old images.json
    full-overwrite semantics: upserts every given row AND deletes any row no
    longer present (e.g. a uid retired by scanner.py's in-place-edit
    handling) — unlike upsert_images(), which is purely additive and used for
    the vision job's per-batch partial saves."""
    prepared_images = [(iid, json.dumps(data)) for iid, data in images.items()]
    projection_rows = [_projection_row(iid, data) for iid, data in images.items()]
    prepared_folders = [(root, json.dumps(data)) for root, data in folders.items()]
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _upsert_prepared_locked(conn, prepared_images, projection_rows)
            conn.executemany(
                "INSERT INTO folders(root, data_json) VALUES (?, ?) "
                "ON CONFLICT(root) DO UPDATE SET data_json=excluded.data_json", prepared_folders,
            )
            stale_images = {row[0] for row in conn.execute("SELECT id FROM images")} - set(images.keys())
            stale_folders = {row[0] for row in conn.execute("SELECT root FROM folders")} - set(folders.keys())
            if stale_images:
                conn.executemany("DELETE FROM images WHERE id = ?", [(i,) for i in stale_images])
                conn.executemany("DELETE FROM library_projection WHERE id = ?", [(i,) for i in stale_images])
            if stale_folders:
                conn.executemany("DELETE FROM folders WHERE root = ?", [(r,) for r in stale_folders])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    _bump(db_path)


def _caption_attributes(data: dict) -> dict:
    """Return only the current caption attributes, accepting stored dicts or JSON."""
    value = data.get("caption_json")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _search_tokens(value) -> str:
    """NFC/casefold text as space-delimited Unicode word and punctuation tokens."""
    normalized = unicodedata.normalize("NFC", _text(value)).casefold()
    tokens = _QUERY_TOKEN_RE.findall(normalized)
    return " " + " ".join(tokens) + " " if tokens else ""


def _facet_value(value) -> str:
    value = unicodedata.normalize("NFC", " ".join(_text(value).split())).casefold()
    return "" if value in _UNKNOWN_FACETS else value


def _camera_label(metadata: dict) -> str:
    make = _text(metadata.get("camera_make"))
    model = _text(metadata.get("camera_model"))
    if make and model:
        return model if model.casefold().startswith(make.casefold()) else f"{make} {model}"
    return model or make


def _parse_capture_date(value: str):
    """Parse EXIF/ISO date strings, preserving valid time precision for sorting."""
    value = (value or "").strip()
    exif = re.fullmatch(
        r"(\d{4}):(\d{2}):(\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?)?",
        value,
    )
    iso = re.fullmatch(
        r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?",
        value,
    )
    if not exif and not iso:
        return None
    from datetime import datetime
    try:
        if exif:
            year, month, day = (int(exif.group(i)) for i in (1, 2, 3))
            hour, minute, second = (int(exif.group(i) or 0) for i in (4, 5, 6))
            micros = int(((exif.group(7) or "") + "000000")[:6])
            return datetime(year, month, day, hour, minute, second, micros)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_projection_capture(data: dict) -> tuple[str, str, str]:
    """Choose the first valid EXIF, filename, then import date candidate.

    The shared photo_date resolver remains untouched for embedding freshness;
    this stricter projection resolver is only for calendar filters and ordering.
    """
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    exif_date = _text(metadata.get("date"))
    filename_date = _date_from_filename(_text(data.get("filename")))
    candidates = [exif_date, filename_date]
    timestamp = data.get("created_at")
    if timestamp:
        try:
            from datetime import datetime
            candidates.append(datetime.fromtimestamp(float(timestamp)).strftime("%Y:%m:%d %H:%M:%S"))
        except (OSError, OverflowError, TypeError, ValueError):
            pass
    for candidate in candidates:
        parsed = _parse_capture_date(candidate)
        if parsed is not None:
            ordered = parsed
            if parsed.tzinfo is not None:
                from datetime import timezone
                ordered = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return (
                candidate, parsed.date().isoformat(),
                ordered.isoformat(timespec="microseconds"),
            )
    return "", "", ""


def _valid_gps(metadata: dict) -> bool:
    try:
        lat, lon = float(metadata["gps_lat"]), float(metadata["gps_lon"])
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)
    except (KeyError, TypeError, ValueError):
        return False


def _projection_row(img_id: str, data: dict) -> tuple:
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    parsed = _caption_attributes(data)
    raw_date, capture_sort_date, capture_sort_timestamp = _resolve_projection_capture(data)
    year = capture_sort_date[:4] if capture_sort_date else "Unknown"
    month = capture_sort_date[5:7] if capture_sort_date else "00"
    caption = _text(parsed.get("caption"))
    if "caption" not in parsed:
        caption = _text(data.get("caption"))
    occasion = _text(parsed.get("occasion"))
    filename = _text(data.get("filename")) or os.path.basename(_text(data.get("path")))
    path = _text(data.get("path"))

    details_keys = (
        "photo_type", "scene", "weather", "season", "time_of_day", "occasion",
        "festival_name", "group_size", "person_count", "clothing_style", "mood",
        "objects", "animals", "vehicles", "food_items", "activities",
        "location_type", "people_description", "dominant_colors",
    )
    details = " ".join(_text(parsed.get(key)) for key in details_keys)
    text_value = _facet_value(parsed.get("text_in_image"))
    if not text_value:
        text_value = " ".join(filter(None, (
            _facet_value(metadata.get(key))
            for key in ("text_in_image", "ocr", "recognized_text")
        )))
    place_value = " ".join(
        _text(source.get(key))
        for source in (parsed, metadata, data)
        for key in ("place", "location", "landmark", "city", "country")
    )
    camera = _facet_value(_camera_label(metadata))
    camera_value = " ".join(filter(None, (camera, _text(metadata.get("lens")))))
    search_fields = {
        "filename": _search_tokens(filename),
        "caption": _search_tokens(caption),
        "details": _search_tokens(details),
        "text": _search_tokens(text_value),
        "place": _search_tokens(place_value),
        "camera": _search_tokens(camera_value),
    }
    search_text = "".join(search_fields.values())
    photo_type = _facet_value(parsed.get("photo_type"))
    scene = _facet_value(parsed.get("scene"))
    weather = _facet_value(parsed.get("weather"))
    occasion_value = _facet_value(parsed.get("occasion"))
    time_of_day = _facet_value(parsed.get("time_of_day"))

    has_text = int(bool(text_value))
    # These are presence flags for usable stored data, not observations about
    # the actual photo. Partial/invalid coordinates mean GPS data is missing.
    has_location = int(_valid_gps(metadata))
    caption_state = int(bool(_facet_value(caption)))
    gps_lat = metadata.get("gps_lat")
    gps_lon = metadata.get("gps_lon")
    if _valid_gps(metadata):
        gps_lat, gps_lon = float(gps_lat), float(gps_lon)
    else:
        gps_lat = gps_lon = None
    try:
        created_at = float(data.get("created_at") or 0)
    except (TypeError, ValueError):
        created_at = 0.0
    try:
        duration = float(data.get("duration_s") or 0)
    except (TypeError, ValueError):
        duration = 0.0

    return (
        img_id, path, filename, created_at, raw_date, year, month,
        "video" if data.get("media_type") == "video" else "image", duration,
        gps_lat, gps_lon, caption, occasion_value,
        search_text, capture_sort_date, capture_sort_timestamp, *search_fields.values(),
        photo_type, scene, weather, time_of_day, camera,
        has_text, has_location, caption_state,
    )


def _upsert_prepared_locked(conn: sqlite3.Connection, serialized: list, projection_rows: list):
    if not serialized:
        return
    conn.executemany(
        "INSERT INTO images(id, data_json) VALUES (?, ?) "
        "ON CONFLICT(id) DO UPDATE SET data_json=excluded.data_json",
        serialized,
    )
    _insert_projection_rows(conn, projection_rows, upsert=True)


def _ensure_projection_locked(conn: sqlite3.Connection):
    row = conn.execute("SELECT value FROM catalog_meta WHERE key='library_projection_version'").fetchone()
    if row and row[0] == str(_PROJECTION_VERSION):
        return
    # A single write transaction makes migration atomic. On interruption SQLite
    # rolls back both the projection and its version marker, so the next read retries.
    conn.execute("BEGIN IMMEDIATE")
    try:
        for trigger in ("library_search_insert", "library_search_delete", "library_search_update"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE IF EXISTS library_search_fts")
        existing = {r[1] for r in conn.execute("PRAGMA table_info(library_projection)")}
        for column, declaration in _NEW_PROJECTION_COLUMNS.items():
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE library_projection ADD COLUMN {column} {declaration}"
                )
        conn.execute("DELETE FROM library_projection")
        batch = []
        for img_id, payload in conn.execute("SELECT id, data_json FROM images"):
            batch.append(_projection_row(img_id, json.loads(payload)))
            if len(batch) >= 512:
                _insert_projection_rows(conn, batch)
                batch.clear()
        if batch:
            _insert_projection_rows(conn, batch)
        conn.execute(
            "INSERT INTO catalog_meta(key,value) VALUES ('library_projection_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(_PROJECTION_VERSION),),
        )
        conn.execute("CREATE INDEX IF NOT EXISTS library_capture_sort_idx "
                     "ON library_projection(capture_sort_timestamp, created_at, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS library_filename_idx "
                     "ON library_projection(filename COLLATE NOCASE, id)")
        for facet in _FACET_COLUMNS.values():
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS library_facet_{facet}_idx "
                f"ON library_projection({facet}, capture_sort_date, id)"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS library_presence_idx "
                     "ON library_projection(text_state, location_state, caption_state)")
        for name, direction in (("newest", "DESC"), ("oldest", "ASC")):
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS library_browse_{name}_idx ON library_projection "
                f"((capture_sort_date = '') ASC, capture_sort_timestamp {direction}, created_at DESC, id DESC)"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS library_summary_idx "
                     "ON library_projection(media_type, caption_state, year)")
        _create_search_index_locked(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _create_search_index_locked(conn: sqlite3.Connection):
    """Optional inverted index; exact predicates remain the search authority.

    Index our already normalized, space-delimited tokens rather than raw text.
    The ASCII tokenizer can safely narrow positive ASCII words; other scripts
    and punctuation retain the exact, provider-independent SQL fallback.
    """
    fields = ",".join(_SEARCH_COLUMNS.values())
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE library_search_fts USING fts5({fields}, "
            "content='library_projection', content_rowid='rowid', tokenize='ascii')"
        )
    except sqlite3.OperationalError as exc:
        if "no such module: fts5" not in str(exc).lower():
            raise
        return
    new_fields = ",".join(f"new.{field}" for field in _SEARCH_COLUMNS.values())
    old_fields = ",".join(f"old.{field}" for field in _SEARCH_COLUMNS.values())
    insert = f"INSERT INTO library_search_fts(rowid,{fields}) VALUES(new.rowid,{new_fields});"
    delete = (f"INSERT INTO library_search_fts(library_search_fts,rowid,{fields}) "
              f"VALUES('delete',old.rowid,{old_fields});")
    conn.execute(f"CREATE TRIGGER library_search_insert AFTER INSERT ON library_projection BEGIN {insert} END")
    conn.execute(f"CREATE TRIGGER library_search_delete AFTER DELETE ON library_projection BEGIN {delete} END")
    conn.execute(f"CREATE TRIGGER library_search_update AFTER UPDATE ON library_projection BEGIN {delete}{insert} END")
    conn.execute("INSERT INTO library_search_fts(library_search_fts) VALUES('rebuild')")


def _search_candidates_query(positive: list[str]) -> str:
    # Each indexed field already has spaces around every normalized word or
    # punctuation token. An ASCII alphanumeric token is therefore a necessary
    # (but not sufficient) condition for every exact positive phrase match.
    words = dict.fromkeys(
        token for term in positive for token in term.split()
        if token.isascii() and token.isalnum()
    )
    return " AND ".join(f'"{word}"' for word in words)


def _insert_projection_rows(conn: sqlite3.Connection, rows: list[tuple], upsert: bool = False):
    columns = ",".join(_PROJECTION_COLUMNS)
    placeholders = ",".join("?" for _ in _PROJECTION_COLUMNS)
    sql = f"INSERT INTO library_projection ({columns}) VALUES ({placeholders})"
    if upsert:
        updates = ",".join(
            f"{column}=excluded.{column}"
            for column in _PROJECTION_COLUMNS if column != "id"
        )
        sql += f" ON CONFLICT(id) DO UPDATE SET {updates}"
    conn.executemany(sql, rows)


def _parse_query(query: str) -> tuple[list[str], list[str]]:
    """Return positive/excluded token phrases using explicit token boundaries.

    Unquoted whitespace terms match complete Unicode tokens; quoted terms match
    adjacent token sequences. Punctuation is tokenized and retained, so a
    literal hyphen in a model or filename remains part of the searched phrase.
    """
    if not isinstance(query, str) or len(query) > 500:
        raise ValueError("q must be a string of at most 500 characters")
    positive, excluded = [], []
    i, count = 0, 0
    while i < len(query):
        while i < len(query) and query[i].isspace():
            i += 1
        if i >= len(query):
            break
        negative = query[i] == "-"
        if negative:
            i += 1
            if i >= len(query) or query[i].isspace():
                raise ValueError("an exclusion marker must be followed by a term")
        if query[i] == '"':
            start = i + 1
            end = query.find('"', start)
            if end < 0:
                raise ValueError("q contains an unmatched quote")
            term = query[start:end]
            if not term.strip():
                raise ValueError("quoted search phrases cannot be empty")
            i = end + 1
            if i < len(query) and not query[i].isspace():
                raise ValueError("quoted phrases must be separated by whitespace")
        else:
            start = i
            while i < len(query) and not query[i].isspace():
                if query[i] == '"':
                    raise ValueError("quotes must wrap a complete search phrase")
                i += 1
            term = query[start:i]
        tokens = _QUERY_TOKEN_RE.findall(unicodedata.normalize("NFC", term).casefold())
        if not tokens:
            raise ValueError("each search term must contain a word or punctuation")
        count += 1
        if count > 20:
            raise ValueError("q supports at most 20 terms")
        normalized = " " + " ".join(tokens) + " "
        (excluded if negative else positive).append(normalized)
    return positive, excluded


def _iso_date(value: str | None, field: str):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD)")
    from datetime import date
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid calendar date") from exc


def _query_value(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty value of at most 200 characters")
    return value


def library_page(
    db_path: str, *, limit: int, offset: int, q: str = "",
    media_type: str | None = None, year: str | None = None,
    text_scope: str = "all", date_from: str | None = None,
    date_to: str | None = None, month: int | None = None, day: int | None = None,
    photo_type: str | None = None, scene: str | None = None,
    weather: str | None = None, occasion: str | None = None,
    time_of_day: str | None = None, camera: str | None = None,
    has_text: str | None = None, has_location: str | None = None,
    has_caption: str | None = None, sort: str = "newest",
) -> dict:
    if not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be non-negative")
    if text_scope not in _SCOPE_FIELDS:
        raise ValueError("text_scope must be all, filename, caption, details, text, place, or camera")
    if sort not in {"newest", "oldest", "added", "filename"}:
        raise ValueError("sort must be newest, oldest, added, or filename")
    if media_type not in (None, "image", "video"):
        raise ValueError("media_type must be image or video")
    if any(flag not in (None, "yes", "no") for flag in (has_text, has_location, has_caption)):
        raise ValueError("presence filters must be yes or no")
    date_from = _iso_date(date_from, "date_from")
    date_to = _iso_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise ValueError("date_from must be on or before date_to")
    if month is not None and (not isinstance(month, int) or not 1 <= month <= 12):
        raise ValueError("month must be between 1 and 12")
    if day is not None:
        if month is None:
            raise ValueError("day requires month")
        if not isinstance(day, int) or not 1 <= day <= calendar.monthrange(2000, month)[1]:
            raise ValueError("day is not valid for the selected month")

    filters = {}
    for name, value in {
        "photo_type": photo_type, "scene": scene, "weather": weather,
        "occasion": occasion, "time_of_day": time_of_day, "camera": camera,
    }.items():
        if value is not None:
            filters[name] = _facet_value(_query_value(value, name))
            if not filters[name]:
                raise ValueError(f"{name} must be a known facet value")
    positive, excluded = _parse_query(q)
    search_fields = _SCOPE_FIELDS[text_scope]
    where, params = [], []
    for term in positive:
        alternatives = " OR ".join(
            f"instr({_SEARCH_COLUMNS[field]}, ?) > 0" for field in search_fields
        )
        where.append(f"({alternatives})")
        params.extend([term] * len(search_fields))
    for term in excluded:
        alternatives = " OR ".join(
            f"instr({_SEARCH_COLUMNS[field]}, ?) > 0" for field in search_fields
        )
        where.append(f"NOT ({alternatives})")
        params.extend([term] * len(search_fields))
    if media_type:
        where.append("media_type = ?")
        params.append(media_type)
    if year:
        where.append("year = ?")
        params.append(year)
    for name, value in filters.items():
        if value is not None:
            where.append(f"{_FACET_COLUMNS[name]} = ?")
            params.append(value)
    for name, state in (("text_state", has_text), ("location_state", has_location),
                        ("caption_state", has_caption)):
        if state is not None:
            where.append(f"{name} = ?")
            params.append(1 if state == "yes" else 0)
    if date_from or date_to or month is not None:
        where.append("capture_sort_date <> ''")
    if date_from:
        where.append("capture_sort_date >= ?")
        params.append(date_from)
    if date_to:
        where.append("capture_sort_date <= ?")
        params.append(date_to)
    if month is not None:
        where.append("substr(capture_sort_date, 6, 2) = ?")
        params.append(f"{month:02d}")
    if day is not None:
        where.append("substr(capture_sort_date, 9, 2) = ?")
        params.append(f"{day:02d}")

    ordering = {
        "newest": "(capture_sort_date = '') ASC, capture_sort_timestamp DESC, created_at DESC, id DESC",
        "oldest": "(capture_sort_date = '') ASC, capture_sort_timestamp ASC, created_at DESC, id DESC",
        "added": "created_at DESC, id DESC",
        "filename": "filename COLLATE NOCASE ASC, filename ASC, id DESC",
    }[sort]
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        candidates = _search_candidates_query(positive)
        if candidates and conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='library_search_fts'"
        ).fetchone():
            where.append("rowid IN (SELECT rowid FROM library_search_fts WHERE library_search_fts MATCH ?)")
            params.append(candidates)
        clause = " WHERE " + " AND ".join(where) if where else ""
        total = conn.execute(
            "SELECT COUNT(*) FROM library_projection" + clause, params
        ).fetchone()[0]
        select = [
            "id", "filename", "caption", "year", "occasion", "media_type",
            "duration_s", "capture_date",
        ]
        if positive:
            select.extend(_SEARCH_COLUMNS.values())
        rows = conn.execute(
            f"SELECT {','.join(select)} FROM library_projection{clause} "
            f"ORDER BY {ordering} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    results = []
    for row in rows:
        values = dict(zip(select, row))
        card = {key: values[key] for key in (
            "id", "filename", "caption", "year", "occasion", "media_type", "duration_s"
        )}
        card["date"] = values["capture_date"]
        card["match_fields"] = [
            field for field in search_fields
            if positive and any(term in values[_SEARCH_COLUMNS[field]] for term in positive)
        ]
        results.append(card)
    return {"results": results, "total": total}


def library_facets(db_path: str, *, limit: int = 100) -> dict:
    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("facet limit must be between 1 and 100")
    conn = _connect(db_path)
    facets, truncated = {}, {}
    with _lock:
        _ensure_projection_locked(conn)
        for field, column in _FACET_COLUMNS.items():
            rows = conn.execute(
                f"SELECT {column}, COUNT(*) FROM library_projection "
                f"WHERE {column} <> '' "
                f"GROUP BY {column} ORDER BY COUNT(*) DESC, {column} COLLATE NOCASE ASC, {column} ASC LIMIT ?",
                [limit + 1],
            ).fetchall()
            truncated[field] = len(rows) > limit
            facets[field] = [
                {"value": value, "count": count} for value, count in rows[:limit]
            ]
    return {"facets": facets, "truncated": truncated}


def library_summary(db_path: str) -> dict:
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        total, photos, videos, captioned = conn.execute(
            "SELECT COUNT(*), SUM(media_type='image'), SUM(media_type='video'), "
            "SUM(caption_state) FROM library_projection"
        ).fetchone()
        year_rows = conn.execute(
            "SELECT year, COUNT(*) FROM library_projection GROUP BY year ORDER BY year DESC"
        ).fetchall()
        years = [{"year": year, "count": count} for year, count in year_rows if year != "Unknown"]
        years.extend({"year": year, "count": count} for year, count in year_rows if year == "Unknown")
    return {"total": total or 0, "photos": photos or 0, "videos": videos or 0,
            "captioned": captioned or 0, "years": years}


def timeline_month_counts(db_path: str) -> dict[str, dict[str, int]]:
    """Year/month counts for quick-jump navigation, grouped in SQLite."""
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        rows = conn.execute(
            "SELECT year, month, COUNT(*) FROM library_projection GROUP BY year, month"
        ).fetchall()
    summary: dict[str, dict[str, int]] = {}
    for year, month, count in rows:
        summary.setdefault(year, {})[month] = count
    return summary


def timeline_year_counts(db_path: str) -> list[dict]:
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        rows = conn.execute(
            "SELECT year, COUNT(*) FROM library_projection GROUP BY year"
        ).fetchall()
    rows.sort(key=lambda row: (row[0] != "Unknown", row[0]), reverse=True)
    return [{"year": year, "count": count} for year, count in rows]


def timeline_year_page(db_path: str, year: str, *, limit: int, offset: int = 0) -> dict:
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        count = conn.execute(
            "SELECT COUNT(*) FROM library_projection WHERE year=?", (year,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT id,filename,capture_date,media_type,duration_s "
            "FROM library_projection WHERE year=? "
            "ORDER BY capture_date DESC,created_at DESC,id DESC LIMIT ? OFFSET ?",
            (year, limit, offset),
        ).fetchall()
    return {
        "count": count,
        "photos": [
            {"id": img_id, "filename": filename, "date": date,
             "media_type": media_type, "duration_s": duration}
            for img_id, filename, date, media_type, duration in rows
        ],
    }


def map_points(db_path: str) -> list[dict]:
    """All geotagged catalog items without checking source files on disk."""
    conn = _connect(db_path)
    with _lock:
        _ensure_projection_locked(conn)
        rows = conn.execute(
            "SELECT id,gps_lat,gps_lon,filename FROM library_projection "
            "WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL ORDER BY id"
        ).fetchall()
    return [
        {"id": img_id, "lat": lat, "lon": lon, "filename": filename, "exists": True}
        for img_id, lat, lon, filename in rows
    ]


def image_path(db_path: str, img_id: str) -> str | None:
    """Indexed point lookup that does not hydrate the full catalog."""
    conn = _connect(db_path)
    with _lock:
        row = conn.execute("SELECT json_extract(data_json, '$.path') FROM images WHERE id=?", (img_id,)).fetchone()
    return row[0] if row else None


def get_image(db_path: str, img_id: str) -> dict | None:
    """Indexed point lookup for lightweight image metadata."""
    conn = _connect(db_path)
    with _lock:
        row = conn.execute("SELECT data_json FROM images WHERE id=?", (img_id,)).fetchone()
    return json.loads(row[0]) if row else None


def version(db_path: str) -> int:
    """In-process write counter for cache-invalidation keys (see _version)."""
    return _version.get(db_path, 0)
