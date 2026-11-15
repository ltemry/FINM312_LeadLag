"""US equity (NYSE) trading-day calendar.

Self-contained so the pipeline has no heavy calendar dependency. Covers the
standard NYSE full-day holidays plus any extra closures listed in config.
Early closes (half days) are not removed here; session labelling handles them.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

from dateutil.easter import easter


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
    # New Year's Day: NYSE does not observe on the prior Friday if Jan 1 is a Saturday.
    ny = dt.date(year, 1, 1)
    if ny.weekday() != 5:
        h.add(_observed(ny))
    if year >= 2022:
        h.add(_observed(dt.date(year, 6, 19)))    # Juneteenth
    return frozenset(h)


def is_trading_day(d: dt.date, extra_closures: frozenset[dt.date] = frozenset()) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year) and d not in extra_closures
