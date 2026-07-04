"""SQLite-backed catalog store — replaces the flat images.json file.

The app already does all catalog filtering/querying in Python over an
in-memory dict (indexer.py, scanner.py), not via SQL WHERE clauses — so this
module keeps that shape intact (each row's payload is stored as one JSON
blob) rather than normalizing every field into its own column. What changes
is persistence: incremental per-row upserts instead of rewriting the entire
file on every save. images.json was being rewritten in full on every batch
of a job (~6,000+ times over a single multi-day vision-captioning run for a
~20MB+ file) — this replaces that with targeted single-row transactions.
"""
import json
import os
import sqlite3
import threading

_lock = threading.Lock()
_conn_cache: dict[str, sqlite3.Connection] = {}
# In-process write counter per db_path, used for cache invalidation instead
# of file mtime: under WAL mode, writes land in the .db-wal file and the main
# .db file's mtime doesn't necessarily change, so an mtime-keyed cache (the
# old images.json approach) can silently go stale. Since the cache itself is
# an in-memory dict that never survives a process restart anyway, an
# in-process counter is both correct and simpler.
_version: dict[str, int] = {}


def _bump(db_path: str):
    _version[db_path] = _version.get(db_path, 0) + 1


def _connect(db_path: str) -> sqlite3.Connection:
    with _lock:
        conn = _conn_cache.get(db_path)
        if conn is None:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS images "
                "(id TEXT PRIMARY KEY, data_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS folders "
                "(root TEXT PRIMARY KEY, data_json TEXT NOT NULL)"
            )
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
    conn = _connect(db_path)
    with _lock:
        conn.executemany(
            "INSERT INTO images(id, data_json) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data_json=excluded.data_json",
            [(iid, json.dumps(data)) for iid, data in images.items()],
        )
        conn.commit()
    _bump(db_path)


def delete_images(db_path: str, ids):
    ids = list(ids)
    if not ids:
        return
    conn = _connect(db_path)
    with _lock:
        conn.executemany("DELETE FROM images WHERE id = ?", [(i,) for i in ids])
        conn.commit()
    _bump(db_path)


def save_folders(db_path: str, folders: dict):
    if not folders:
        return
    conn = _connect(db_path)
    with _lock:
        conn.executemany(
            "INSERT INTO folders(root, data_json) VALUES (?, ?) "
            "ON CONFLICT(root) DO UPDATE SET data_json=excluded.data_json",
            [(root, json.dumps(data)) for root, data in folders.items()],
        )
        conn.commit()
    _bump(db_path)


def save_all(db_path: str, images: dict, folders: dict):
    """Full sync for both tables — used by scan checkpoints, which always pass
    the COMPLETE current in-memory catalog. Matches the old images.json
    full-overwrite semantics: upserts every given row AND deletes any row no
    longer present (e.g. a uid retired by scanner.py's in-place-edit
    handling) — unlike upsert_images(), which is purely additive and used for
    the vision job's per-batch partial saves."""
    upsert_images(db_path, images)
    save_folders(db_path, folders)
    conn = _connect(db_path)
    with _lock:
        stale_images = {
            row[0] for row in conn.execute("SELECT id FROM images")
        } - set(images.keys())
        if stale_images:
            conn.executemany("DELETE FROM images WHERE id = ?", [(i,) for i in stale_images])
        stale_folders = {
            row[0] for row in conn.execute("SELECT root FROM folders")
        } - set(folders.keys())
        if stale_folders:
            conn.executemany("DELETE FROM folders WHERE root = ?", [(r,) for r in stale_folders])
        conn.commit()
    if stale_images or stale_folders:
        _bump(db_path)


def version(db_path: str) -> int:
    """In-process write counter for cache-invalidation keys (see _version)."""
    return _version.get(db_path, 0)
