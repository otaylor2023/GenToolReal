"""Wall-clock times for logs and metadata in US Pacific (``America/Los_Angeles``).

Uses the IANA zone (PST UTC−8 in winter, PDT UTC−7 in summer). ISO strings include
the offset so values are unambiguous.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
PACIFIC_TZ_NAME = "America/Los_Angeles"


def now_pacific() -> datetime:
    return datetime.now(tz=PACIFIC_TZ)


def format_path_stamp(dt: datetime | None = None) -> str:
    """``YYYY-mm-dd_HH-MM-SS`` in Pacific, for run folder names."""
    d = now_pacific() if dt is None else _as_pacific(dt)
    return d.strftime("%Y-%m-%d_%H-%M-%S")


def isoformat_pacific(dt: datetime | None = None, *, timespec: str = "seconds") -> str:
    """ISO 8601 in Pacific with numeric offset (e.g. ``...-07:00``)."""
    d = now_pacific() if dt is None else _as_pacific(dt)
    return d.isoformat(timespec=timespec)


def _as_pacific(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=PACIFIC_TZ)
    return dt.astimezone(PACIFIC_TZ)
