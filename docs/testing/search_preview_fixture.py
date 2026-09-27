"""Create an isolated, synthetic Photo Vault search-preview library.

Development fixture only; this is not an application CLI. It writes generated
geometric PNGs and catalog rows to an explicit empty target directory. It does
not launch the server, inspect real data, or call AI providers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _on_this_day(year: int, month: int, day: int, count: int = 4) -> list[date]:
    found = []
    candidate_year = year
    while len(found) < count:
        try:
            found.append(date(candidate_year, month, day))
        except ValueError:
            pass  # For example, Feb 29 only exists in leap years.
        candidate_year -= 1
    return found


def _record(image_id: str, filename: str, caption: str = "", *, when: date,
            attributes: dict | None = None, metadata: dict | None = None,
            history: list | None = None) -> dict:
    attrs = dict(attributes or {})
    attrs["caption"] = caption
    record_metadata = {"date": when.strftime("%Y:%m:%d 12:00:00")}
    record_metadata.update(metadata or {})
    result = {
        "path": "",  # Filled with the generated local image path below.
        "filename": filename,
        "media_type": "image",
        "metadata": record_metadata,
        "created_at": float(when.toordinal()),
        "caption": caption,
        "caption_json": json.dumps(attrs, ensure_ascii=False),
    }
    if history is not None:
        result["caption_history"] = history
    return result


def _fixture_records(today: date) -> tuple[dict[str, dict], dict]:
    day_dates = _on_this_day(today.year, today.month, today.day)
    items: list[tuple[str, str, str, date, dict, dict, list | None]] = []

    def add(image_id, filename, caption, when=None, attributes=None, metadata=None, history=None):
        items.append((image_id, filename, caption, when or date(2024, 2, 10),
                      attributes or {}, metadata or {}, history))

    for n in range(1, 5):
        add(f"sv-bday-cake-{n:02d}", f"birthday-cake-{n:02d}.png",
            f"Birthday cake number {n} with candles and family gathered around the table",
            attributes={"occasion": "birthday", "scene": "birthday party", "food_items": ["cake"],
                        "objects": ["candles", "cake"], "photo_type": "photo", "weather": "clear",
                        "time_of_day": "evening"})
    add("sv-bday-card", "birthday-card.png", "Birthday guests holding a handmade card",
        attributes={"occasion": "birthday", "scene": "birthday party", "photo_type": "photo"})

    for n, animal in enumerate(("tabby", "black", "white"), start=1):
        add(f"sv-cat-{n:02d}", f"cat-{n:02d}.png", f"A {animal} cat sitting beside a sunny window",
            attributes={"animals": ["cat"], "objects": ["window"], "scene": "home"})
    for n, place in enumerate(("coast", "mountains", "island", "desert"), start=1):
        add(f"sv-vacation-{n:02d}", f"vacation-{n:02d}.png",
            f"Vacation at the {place} with a bright blue sky",
            attributes={"scene": "vacation landscape", "activities": ["travel"], "weather": "sunny"})
    add("sv-vacation-catalog", "vacation-catalog.png",
        "Vacation catalog open on a wooden table",
        attributes={"scene": "vacation planning", "objects": ["catalog", "table"]})

    add("sv-receipt-text", "receipt.png", "A small paper receipt on a counter",
        attributes={"text_in_image": "RECEIPT TOTAL $12.40 COFFEE", "photo_type": "document"})
    add("sv-invoice-text", "invoice.png", "A printed invoice beside a pen",
        attributes={"text_in_image": "INVOICE NUMBER 1042 AMOUNT DUE", "photo_type": "document"})
    add("sv-screenshot-text", "screenshot-text.png", "A screenshot of a delivery confirmation",
        attributes={"text_in_image": "DELIVERY CONFIRMED ORDER 725", "photo_type": "screenshot",
                    "scene": "screen"})
    add("sv-screenshot-clean", "screenshot-clean.png", "A screenshot with a blue weather chart",
        attributes={"photo_type": "screenshot", "scene": "screen"})

    add("sv-animal-dog", "golden-retriever.png", "A golden retriever running through a green park",
        attributes={"animals": ["golden retriever", "dog"], "scene": "park", "activities": ["running"]})
    add("sv-animal-parrot", "parrot.png", "A green parrot perched on a branch",
        attributes={"animals": ["parrot"], "objects": ["branch"]})
    add("sv-object-umbrella", "red-umbrella.png", "A red umbrella leaning beside the doorway",
        attributes={"objects": ["red umbrella", "doorway"], "dominant_colors": ["red"]})
    add("sv-camera-canon", "canon-camera.png", "A geometric still life on a table",
        metadata={"camera_make": "Canon", "camera_model": "Canon EOS R5", "lens": "RF 50mm"})
    add("sv-camera-sony", "sony-camera.png", "A geometric still life near a window",
        metadata={"camera_make": "Sony", "camera_model": "A7 IV"})
    add("sv-place-monument", "monument-valley.png", "A wide view of red sandstone formations",
        attributes={"scene": "landscape"},
        metadata={"place": "Monument Valley", "gps_lat": 0, "gps_lon": 0})
    add("sv-gps-zero", "equator-zero.png", "Geometric shapes at a marked coordinate",
        metadata={"gps_lat": 0, "gps_lon": 0})
    add("sv-gps-coordinate", "mapped-garden.png", "A geometric garden viewed from above",
        metadata={"gps_lat": 37.4219, "gps_lon": -122.0840})
    add("sv-gps-incomplete", "incomplete-coordinate.png", "A geometric blue and gold pattern",
        metadata={"gps_lat": 12.5})

    for n, captured in enumerate(day_dates, start=1):
        add(f"sv-onthisday-{n}", f"on-this-day-{captured.year}.png",
            f"Anniversary picnic photograph from {captured.year}", when=captured,
            attributes={"occasion": "anniversary", "scene": "picnic"})
    add("sv-date-early", "date-early.png", "Geometric shapes in an early spring garden",
        when=date(2019, 4, 10), attributes={"scene": "garden"})
    add("sv-date-middle", "date-middle.png", "A geometric city scene in summer",
        when=date(2021, 6, 15), attributes={"scene": "city"})
    add("sv-date-late", "date-late.png", "Geometric winter lights at night",
        when=date(2025, 11, 3), attributes={"scene": "night"})

    add("sv-history-only", "history-only.png", "",
        attributes={"caption": ""},
        history=[{"model": "synthetic-previous", "caption_json":
                  json.dumps({"caption": "retiredmarkerq7 old caption must not match"})}])
    add("sv-no-caption-empty", "no-caption-empty.png", "", attributes={"caption": ""})
    add("sv-no-caption-missing", "no-caption-missing.png", "",
        attributes={key: value for key, value in {"mood": "calm"}.items()})
    add("sv-unicode-cafe", "unicode-cafe.png", "A café façade decorated with geometric tiles",
        attributes={"objects": ["façade", "tile mosaic"]})
    add("sv-weather-rain", "rain-window.png", "Raindrops making geometric patterns on glass",
        attributes={"weather": "rainy", "scene": "window"})
    add("sv-object-bicycle", "blue-bicycle.png", "A blue bicycle beside a brick wall",
        attributes={"objects": ["bicycle", "brick wall"], "dominant_colors": ["blue"]})
    add("sv-animal-bear", "bear.png", "A brown bear walking beside a stream",
        attributes={"animals": ["bear"], "scene": "forest"})

    if len(items) != 40:
        raise RuntimeError(f"fixture definition has {len(items)} records, expected 40")

    image_ids: set[str] = set()
    records: dict[str, dict] = {}
    for index, (image_id, filename, caption, when, attributes, metadata, history) in enumerate(items):
        if image_id in image_ids:
            raise RuntimeError(f"duplicate synthetic ID: {image_id}")
        image_ids.add(image_id)
        record = _record(image_id, filename, caption, when=when, attributes=attributes,
                         metadata=metadata, history=history)
        records[image_id] = record

    expectations = {
        "birthday_cake_phrase": {"query": '"birthday cake"', "ids": [f"sv-bday-cake-{n:02d}" for n in range(1, 5)]},
        "vacation_excluding_cat": {"query": "vacation -cat", "ids": [f"sv-vacation-{n:02d}" for n in range(1, 5)] + ["sv-vacation-catalog"]},
        "cat_exact_word": {"query": "cat", "ids": [f"sv-cat-{n:02d}" for n in range(1, 4)]},
        "receipt_ocr": {"query": "receipt", "text_scope": "text", "ids": ["sv-receipt-text"]},
        "screenshots": {"photo_type": "screenshot", "ids": ["sv-screenshot-text", "sv-screenshot-clean"]},
        "valid_gps_including_zero": {"has_location": "yes", "ids": ["sv-place-monument", "sv-gps-zero", "sv-gps-coordinate"]},
        "missing_or_incomplete_gps": {"has_location": "no", "count": 37},
        "recognized_text": {"has_text": "yes", "ids": ["sv-receipt-text", "sv-invoice-text", "sv-screenshot-text"]},
        "history_only_marker": {"query": "retiredmarkerq7", "ids": []},
        "unicode_cafe": {"query": "café", "ids": ["sv-unicode-cafe"]},
        "camera_canon": {"query": "Canon EOS", "text_scope": "camera", "ids": ["sv-camera-canon"]},
        "place_monument": {"query": "Monument Valley", "text_scope": "place", "ids": ["sv-place-monument"]},
        "same_month_day": {"month": today.month, "day": today.day,
                            "ids": [f"sv-onthisday-{n}" for n in range(1, len(day_dates) + 1)]},
        "date_range_2019_2021": {"date_from": "2019-01-01", "date_to": "2021-12-31",
                                 "expected_special_ids": ["sv-date-early", "sv-date-middle"]},
        "caption_missing": {"has_caption": "no", "ids": ["sv-history-only", "sv-no-caption-empty", "sv-no-caption-missing"]},
    }
    return records, {"today": today.isoformat(), "on_this_day_years": [d.year for d in day_dates],
                    "expectations": expectations}


def _draw_geometric_image(path: Path, index: int) -> None:
    from PIL import Image, ImageDraw

    palette = ((215, 232, 241), (247, 224, 201), (219, 231, 205), (230, 215, 235))
    image = Image.new("RGB", (360, 260), palette[index % len(palette)])
    draw = ImageDraw.Draw(image)
    inset = 16 + (index % 5) * 5
    draw.rectangle((inset, inset, 180, 180), fill=(55 + index % 120, 92, 150))
    draw.ellipse((150, 55, 330 - index % 30, 235), fill=(218, 115 + index % 80, 78))
    draw.polygon(((35, 220), (105, 95), (175, 220)), fill=(77, 145, 116))
    image.save(path, format="PNG", optimize=True)


def _target_path(raw: str) -> Path:
    requested = Path(raw).expanduser()
    if requested.is_symlink():
        raise ValueError("target must not be a symlink")
    target = requested.resolve()
    repo_data = (ROOT / "data").resolve()
    if target == repo_data or target.is_relative_to(repo_data):
        raise ValueError("target must not be the repository data directory or a child of it")
    if target == ROOT or target.is_relative_to(ROOT):
        raise ValueError("target must be outside the repository")
    if target.exists():
        if not target.is_dir():
            raise ValueError("existing target must be a directory")
        if target.is_symlink():
            raise ValueError("target must not be a symlink")
        if any(target.iterdir()):
            raise ValueError("existing target directory must be empty")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True,
                        help="explicit new or empty directory for generated images and catalog")
    args = parser.parse_args()
    try:
        target = _target_path(args.target)
        target.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    # Isolation must be set before importing catalog_db/constants or any other
    # Photo Vault module. Provider credentials and local endpoints are stripped.
    os.environ["PHOTO_VAULT_DATA_DIR"] = str(target)
    os.environ["PHOTO_VAULT_ENV_FILE"] = "-"
    for name in ("GEMINI_API_KEY", "LM_STUDIO_URL", "NINEROUTER_URL"):
        os.environ.pop(name, None)
    sys.path.insert(0, str(SRC))

    import catalog_db
    from constants import IMAGE_CATALOG_PATH, PERSON_MAP_PATH

    records, manifest = _fixture_records(date.today())
    media_dir = target / "synthetic-media"
    media_dir.mkdir()
    try:
        for index, (image_id, record) in enumerate(records.items()):
            image_path = media_dir / f"{image_id}.png"
            _draw_geometric_image(image_path, index)
            record["path"] = str(image_path)
        catalog_db.save_all(IMAGE_CATALOG_PATH, records, {})
        # tagger's canonical person_map schema is {display_name: embedding list}.
        # This zero vector exists only to populate the Smart name suggestion;
        # no face files or searchable face index are created.
        with open(PERSON_MAP_PATH, "w", encoding="utf-8") as person_file:
            json.dump({"Preview Person": [0.0] * 512}, person_file)
    finally:
        connection = catalog_db._conn_cache.get(IMAGE_CATALOG_PATH)
        if connection is not None:
            connection.close()
            del catalog_db._conn_cache[IMAGE_CATALOG_PATH]
            catalog_db._version.pop(IMAGE_CATALOG_PATH, None)

    manifest["fixture_rows"] = len(records)
    manifest["registered_people"] = ["Preview Person"]
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
