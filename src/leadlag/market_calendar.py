"""US equity (NYSE/Nasdaq) calendar and session times.

Self-contained: standard full-day holidays, early (13:00 ET) closes, and any
extra closures from config. All session logic is in America/New_York and only
converted to UTC at the end, so September (EDT, UTC-4) and November after the
DST change (EST, UTC-5) get different UTC times automatically.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

from dateutil.easter import easter

ET = ZoneInfo("America/New_York")

PRE_OPEN = dt.time(4, 0)
REGULAR_OPEN = dt.time(9, 30)
REGULAR_CLOSE = dt.time(16, 0)
EARLY_CLOSE = dt.time(13, 0)
POST_CLOSE = dt.time(20, 0)
EARLY_POST_CLOSE = dt.time(17, 0)


def _observed(d: dt.date) -> dt.date:
    """NYSE observance: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


@lru_cache(maxsize=None)
def nyse_holidays(year: int) -> frozenset[dt.date]:
    MON, THU = 0, 3
    h = {
        _nth_weekday(year, 1, MON, 3),            # MLK Day
        _nth_weekday(year, 2, MON, 3),            # Presidents' Day
        easter(year) - dt.timedelta(days=2),      # Good Friday
        _last_weekday(year, 5, MON),              # Memorial Day
        _observed(dt.date(year, 7, 4)),           # Independence Day
        _nth_weekday(year, 9, MON, 1),            # Labor Day
        _nth_weekday(year, 11, THU, 4),           # Thanksgiving
        _observed(dt.date(year, 12, 25)),         # Christmas
    }
    # New Year's Day: not observed on the prior Friday when Jan 1 is a Saturday.
    ny = dt.date(year, 1, 1)
    if ny.weekday() != 5:
        h.add(_observed(ny))
    if year >= 2022:
        h.add(_observed(dt.date(year, 6, 19)))    # Juneteenth
    return frozenset(h)


@lru_cache(maxsize=None)
def nyse_early_closes(year: int) -> frozenset[dt.date]:
    """13:00 ET closes: day after Thanksgiving, Dec 24 and Jul 3 when trading days."""
    THU = 3
    out = {_nth_weekday(year, 11, THU, 4) + dt.timedelta(days=1)}
    for d in (dt.date(year, 12, 24), dt.date(year, 7, 3)):
        if d.weekday() < 5 and d not in nyse_holidays(year):
            out.add(d)
    return frozenset(out)


def is_trading_day(d: dt.date, extra_closures: frozenset[dt.date] = frozenset()) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year) and d not in extra_closures


def previous_trading_day(d: dt.date, extra_closures: frozenset[dt.date] = frozenset()) -> dt.date:
    d -= dt.timedelta(days=1)
    while not is_trading_day(d, extra_closures):
        d -= dt.timedelta(days=1)
    return d


def session_times(d: dt.date) -> dict[str, dt.datetime]:
    """Tz-aware ET boundaries for one trading day (pre/regular/post)."""
    early = d in nyse_early_closes(d.year)
    close = EARLY_CLOSE if early else REGULAR_CLOSE
    post = EARLY_POST_CLOSE if early else POST_CLOSE
    return {
        "pre_open": dt.datetime.combine(d, PRE_OPEN, ET),
        "open": dt.datetime.combine(d, REGULAR_OPEN, ET),
        "close": dt.datetime.combine(d, close, ET),
        "post_close": dt.datetime.combine(d, post, ET),
    }
