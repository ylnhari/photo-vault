"""Lightweight, shared capture-date resolution for catalog and API reads."""

import re
from datetime import datetime


_FN_DATE_RE = re.compile(r"(20[0-2]\d)[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])")
_FN_EPOCH_RE = re.compile(r"(?<!\d)(1[0-9]{12})(?!\d)")


def _date_from_filename(name: str) -> str:
    """Recover a plausible phone filename date in EXIF form, or empty string."""
    if not name:
        return ""
    match = _FN_DATE_RE.search(name)
    if match:
        return f"{match.group(1)}:{match.group(2)}:{match.group(3)} 00:00:00"
    match = _FN_EPOCH_RE.search(name)
    if match:
        try:
            date = datetime.fromtimestamp(int(match.group(1)) / 1000)
            if 2005 <= date.year <= datetime.now().year:
                return date.strftime("%Y:%m:%d %H:%M:%S")
        except (OSError, OverflowError, ValueError):
            pass
    return ""


def resolve_photo_date(img_data: dict) -> str:
    """EXIF, filename date, then imported/created timestamp."""
    date = (img_data.get("metadata", {}) or {}).get("date", "") or ""
    if date and len(date) >= 4:
        return date
    date = _date_from_filename(img_data.get("filename", "") or "")
    if date:
        return date
    timestamp = img_data.get("created_at")
    if timestamp:
        try:
            return datetime.fromtimestamp(timestamp).strftime("%Y:%m:%d %H:%M:%S")
        except (OSError, OverflowError, ValueError):
            pass
    return ""
