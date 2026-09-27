"""Development-only synthetic benchmark for the library responsiveness revamp.

Run from the repository root with ``python tests/benchmark_revamp.py``.
This is intentionally not part of the Photo Vault application or CLI. It uses
25,000 generated catalog rows and tiny geometric images inside a temporary
directory. No real catalog, media, credentials, or provider endpoints are used.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RECORDS = 25_000


def _timed(callable_):
    start = time.perf_counter()
    result = callable_()
    return time.perf_counter() - start, result


def _isolated_env(data_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PHOTO_VAULT_DATA_DIR"] = str(data_dir)
    env["PHOTO_VAULT_ENV_FILE"] = "-"
    # Avoid consulting a sibling ports registry or inheriting provider secrets
    # and host-specific endpoints. This harness never invokes inference.
    env["PHOTO_VAULT_PORT"] = "8768"
    for name in ("GEMINI_API_KEY", "LM_STUDIO_URL", "NINEROUTER_URL"):
        env.pop(name, None)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _cold_import(data_dir: Path) -> tuple[float, float]:
    code = (
        "import time; start=time.perf_counter(); import api; "
        "print('BENCH_IMPORT_SECONDS=' + str(time.perf_counter()-start))"
    )
    started = time.perf_counter()
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_isolated_env(data_dir),
        capture_output=True,
        text=True,
        timeout=180,
    )
    elapsed = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(f"cold import api failed (exit {run.returncode}): {run.stderr[-3000:]}")
    marker = next((line for line in run.stdout.splitlines() if line.startswith("BENCH_IMPORT_SECONDS=")), None)
    if marker is None:
        raise RuntimeError("cold import subprocess did not report the import-only duration")
    return elapsed, float(marker.partition("=")[2])


def _synthetic_catalog() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for i in range(RECORDS):
        year = 2010 + (i % 15)
        video = i % 5 == 0
        caption = f"Synthetic {'video' if video else 'photo'} record {i:05d} in {year} with geometric shapes"
        month = 1 + (i % 12)
        day = 1 + (i % 27)
        rows[f"bench-{i:05d}"] = {
            "path": f"synthetic/{i:05d}.{ 'mp4' if video else 'jpg' }",
            "filename": f"{i:05d}.{ 'mp4' if video else 'jpg' }",
            "media_type": "video" if video else "image",
            "metadata": {
                "date": f"{year}:{month:02d}:{day:02d} 12:00:00",
                "Make": "Synthetic",
                "Model": "Benchmark",
                "year": year,
            },
            "created_at": float(i + 1),
            "caption": caption,
            "caption_json": json.dumps({"caption": caption, "weather": "sunny" if i % 2 else "cloudy"}),
        }
    return rows


def _seed_legacy_catalog(db_path: Path, rows: dict[str, dict]) -> None:
    """Create the pre-projection SQLite shape without invoking app writes."""
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE images (id TEXT PRIMARY KEY, data_json TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO images(id, data_json) VALUES (?, ?)",
            [(image_id, json.dumps(data)) for image_id, data in rows.items()],
        )
        connection.commit()
    finally:
        connection.close()


def _require_expected(payload: dict, *, page: bool = False) -> None:
    required = {"results", "total", "offset", "limit", "has_more"} if page else {
        "total", "photos", "videos", "captioned", "years"
    }
    missing = required - payload.keys()
    if missing:
        raise RuntimeError(f"benchmark API response missing fields: {sorted(missing)}")


def _close_isolated_catalog_connections(catalog_db, root: Path) -> None:
    """Close only catalog handles opened beneath this run's temp directory."""
    for db_path, connection in tuple(catalog_db._conn_cache.items()):
        resolved = Path(db_path).resolve()
        if resolved.is_relative_to(root.resolve()):
            connection.close()
            del catalog_db._conn_cache[db_path]
            catalog_db._version.pop(db_path, None)


def main() -> int:
    sys.path.insert(0, str(SRC))
    try:
        from PIL import Image, ImageDraw
        from fastapi.testclient import TestClient
        import catalog_db
    except Exception as exc:
        print(f"Benchmark prerequisites unavailable: {exc}")
        return 2

    with tempfile.TemporaryDirectory(prefix="photo-vault-revamp-bench-") as temporary:
        isolated = Path(temporary)
        data_dir = isolated / "data"
        data_dir.mkdir()

        try:
            cold_process_seconds, cold_import_seconds = _cold_import(isolated / "cold-data")

            # Import the app only after setting isolation variables. Force .env off
            # and avoid the sibling ports registry in this process too.
            os.environ.update(_isolated_env(data_dir))
            sys.path.insert(0, str(SRC))
            import api
            from imaging import THUMB_PX, derivative_path, ensure_derivative
            from constants import IMAGE_CATALOG_PATH

            rows = _synthetic_catalog()
            seed_seconds, _ = _timed(lambda: catalog_db.save_all(IMAGE_CATALOG_PATH, rows, {}))
            legacy_load_sort_seconds, loaded = _timed(
                lambda: sorted(
                    catalog_db.load_all(IMAGE_CATALOG_PATH)["images"].items(),
                    key=lambda item: (item[1].get("metadata", {}) or {}).get("date", ""),
                    reverse=True,
                )
            )
            if len(loaded) != RECORDS:
                raise RuntimeError("synthetic catalog load/sort lost rows")

            # Simulate a pre-projection library: only the original images table
            # exists. The first direct store query must build the SQL projection.
            legacy_db_path = isolated / "legacy" / "catalog.db"
            legacy_db_path.parent.mkdir()
            legacy_seed_seconds, _ = _timed(lambda: _seed_legacy_catalog(legacy_db_path, rows))
            migration_seconds, migration_page = _timed(
                lambda: catalog_db.library_page(str(legacy_db_path), limit=60, offset=0)
            )
            if migration_page["total"] != RECORDS or len(migration_page["results"]) != 60:
                raise RuntimeError("legacy catalog migration query returned an unexpected page")
            migration_warm_seconds, migration_warm_page = _timed(
                lambda: catalog_db.library_page(str(legacy_db_path), limit=60, offset=0)
            )
            if migration_warm_page["total"] != RECORDS or len(migration_warm_page["results"]) != 60:
                raise RuntimeError("warm legacy catalog page returned an unexpected result")

            client = TestClient(api.app)

            def get_page(**params):
                response = client.get("/api/library", params=params)
                if response.status_code != 200:
                    raise RuntimeError(f"GET /api/library returned {response.status_code}: {response.text[:1000]}")
                payload = response.json()
                _require_expected(payload, page=True)
                return payload

            first_seconds, first = _timed(lambda: get_page(offset=0, limit=60))
            if len(first["results"]) > 60 or first["total"] != RECORDS:
                raise RuntimeError("library first page did not return the expected bounded synthetic catalog")
            warm_seconds, _ = _timed(lambda: get_page(offset=0, limit=60))
            lexical_seconds, lexical = _timed(lambda: get_page(offset=0, limit=60, q="geometric shapes"))
            summary_seconds, summary_response = _timed(lambda: client.get("/api/library/summary"))
            if summary_response.status_code != 200:
                raise RuntimeError(f"GET /api/library/summary returned {summary_response.status_code}: {summary_response.text[:1000]}")
            summary = summary_response.json()
            _require_expected(summary)
            if summary["total"] != RECORDS:
                raise RuntimeError("summary count did not match synthetic catalog size")

            # A tiny generated image exercises derivative generation and an actual
            # on-disk cache hit without touching any user media.
            image_path = isolated / "geometric-source.png"
            canvas = Image.new("RGB", (320, 240), "#d6e8f2")
            draw = ImageDraw.Draw(canvas)
            draw.ellipse((45, 35, 170, 160), fill="#ed7956")
            draw.rectangle((150, 90, 275, 205), fill="#435d8c")
            canvas.save(image_path)
            thumb_path = derivative_path("benchmark-geometric-image")
            # Make sure this generated derivative path is within the temp store.
            if not Path(thumb_path).resolve().is_relative_to(data_dir.resolve()):
                raise RuntimeError("thumbnail output escaped the isolated data directory")
            thumb_cold_seconds, created = _timed(lambda: ensure_derivative(str(image_path), thumb_path, THUMB_PX))
            thumb_warm_seconds, cached = _timed(lambda: ensure_derivative(str(image_path), thumb_path, THUMB_PX))
            if not created or not cached:
                raise RuntimeError("synthetic thumbnail generation/cache check failed")

            report = {
                "catalog_rows": RECORDS,
                "synthetic_mix": "20% videos; captions, date/year and EXIF-like metadata",
                "cold_import_api_seconds": cold_import_seconds,
                "cold_subprocess_wall_seconds": cold_process_seconds,
                "catalog_seed_seconds": seed_seconds,
                "whole_catalog_load_date_sort_seconds": legacy_load_sort_seconds,
                "legacy_catalog_seed_seconds": legacy_seed_seconds,
                "legacy_first_page_with_projection_migration_seconds": migration_seconds,
                "legacy_warm_direct_store_page_seconds": migration_warm_seconds,
                "legacy_migration_page_rows": len(migration_page["results"]),
                "legacy_migration_catalog_total": migration_page["total"],
                "library_first_page_seconds": first_seconds,
                "library_first_page_rows": len(first["results"]),
                "library_warm_page_seconds": warm_seconds,
                "library_lexical_search_seconds": lexical_seconds,
                "library_lexical_search_total": lexical["total"],
                "library_summary_seconds": summary_seconds,
                "library_summary_total": summary["total"],
                "synthetic_thumbnail_generate_seconds": thumb_cold_seconds,
                "synthetic_thumbnail_cache_hit_seconds": thumb_warm_seconds,
                "measurement_note": "Local observations; no pass/fail target or performance guarantee.",
            }
            print("Synthetic responsiveness benchmark (development check; no pass/fail target)")
            print(json.dumps(report, indent=2, sort_keys=True))
        finally:
            _close_isolated_catalog_connections(catalog_db, isolated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
