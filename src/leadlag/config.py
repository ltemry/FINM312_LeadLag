"""Load config.yaml and turn studies into concrete (source, instrument, day) tasks.

Study dates are US/Eastern calendar dates. Each source then gets the days it
needs: NYSE trading days for TAQ, and every UTC day overlapping the ET range
for the 24/7 perp sources and CME (minus Saturdays for CME).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import yaml

from .market_calendar import is_trading_day

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

SOURCES = ("taq", "dwellir", "hyperliquid", "databento")
# Which instrument leg each source needs.
SOURCE_LEG = {"taq": "equity", "dwellir": "perp", "hyperliquid": "perp", "databento": "futures"}
UTC_DAY_SOURCES = {"dwellir", "hyperliquid", "databento"}


@dataclass(frozen=True)
class Task:
    source: str
    instrument: str
    date: dt.date            # ET trading day for TAQ, UTC day for the other sources
    leg: dict = field(hash=False, compare=False)

    def __str__(self) -> str:
        return f"{self.source:<12} {self.instrument:<6} {self.date}"


class Config:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_CONFIG
        with open(self.path) as f:
            self.raw: dict[str, Any] = yaml.safe_load(f)
        proj = self.raw.get("project", {})
        self.data_dir = (REPO_ROOT / proj.get("data_dir", "data")).resolve()
        self.market_tz = proj.get("market_timezone", "America/New_York")
        self.extra_closures = frozenset(
            to_date(d) for d in proj.get("extra_market_closures", []) or []
        )
        self.instruments: dict[str, dict] = self.raw.get("instruments", {})
        self.studies: dict[str, dict] = self.raw.get("studies", {})

    def source_cfg(self, source: str) -> dict:
        return self.raw.get(source, {}) or {}

    def is_deferred(self, inst: str) -> bool:
        return bool(self.instruments.get(inst, {}).get("deferred"))

    def study(self, name: str) -> dict:
        if name not in self.studies:
            raise KeyError(f"Unknown study '{name}'. Known: {', '.join(self.studies)}")
        return self.studies[name]

    # ------------------------------------------------------------------ dates
    def study_dates(self, study: str | None, start: str | None = None, end: str | None = None,
                    dates: Iterable[str] | None = None, which: str = "retrieval") -> list[dt.date]:
        """ET calendar dates for a study block (`retrieval`, `scored`, `focus`, ...).

        CLI --dates / --start/--end override the config.
        """
        if dates:
            return sorted({to_date(d) for d in dates})
        if start or end:
            return date_range(start or end, end or start)
        if not study:
            raise ValueError("Give --study, or --start/--end, or --dates.")
        spec = self.study(study).get(which)
        if not spec:
            raise ValueError(f"Study '{study}' has no '{which}' dates in {self.path.name}; "
                             "set them or pass --dates / --start --end.")
        return parse_blocks(spec)

    def trading_days(self, et_dates: Iterable[dt.date]) -> list[dt.date]:
        return [d for d in et_dates if is_trading_day(d, self.extra_closures)]

    def utc_days(self, et_dates: Iterable[dt.date]) -> list[dt.date]:
        """Every UTC calendar day that overlaps any of the given ET days."""
        tz = ZoneInfo(self.market_tz)
        out: set[dt.date] = set()
        for d in et_dates:
            a = dt.datetime.combine(d, dt.time(), tz).astimezone(dt.timezone.utc)
            b = (dt.datetime.combine(d + dt.timedelta(days=1), dt.time(), tz)
                 .astimezone(dt.timezone.utc) - dt.timedelta(microseconds=1))
            day = a.date()
            while day <= b.date():
                out.add(day)
                day += dt.timedelta(days=1)
        return sorted(out)

    def dates_for_source(self, source: str, et_dates: list[dt.date]) -> list[dt.date]:
        if source == "taq":
            return self.trading_days(et_dates)
        days = self.utc_days(et_dates)
        if source == "databento":
            days = [d for d in days if d.weekday() != 5]
        return days

    # ------------------------------------------------------------------ tasks
    def build_tasks(self, study: str | None, sources: Iterable[str] | None = None,
                    instruments: Iterable[str] | None = None, start: str | None = None,
                    end: str | None = None, dates: Iterable[str] | None = None
                    ) -> tuple[list[Task], list[str]]:
        """Return (tasks, notes). Notes explain anything that was skipped."""
        s = self.study(study) if study else {}
        srcs = list(sources or s.get("sources") or SOURCES)
        insts = list(instruments or s.get("instruments") or self.instruments)
        et_dates = self.study_dates(study, start, end, dates)

        tasks: list[Task] = []
        notes: list[str] = []
        for inst in insts:
            if inst not in self.instruments:
                raise KeyError(f"Unknown instrument '{inst}'.")
            if self.is_deferred(inst):
                if instruments:
                    raise ValueError(f"{inst} is deferred in config.yaml; nothing will be collected.")
                notes.append(f"{inst} is deferred; skipped")
        for src in srcs:
            if src not in SOURCES:
                raise ValueError(f"Unknown source '{src}'. Choose from {SOURCES}.")
            src_dates = self.dates_for_source(src, et_dates)
            for inst in insts:
                if self.is_deferred(inst):
                    continue
                leg = self.instruments[inst].get(SOURCE_LEG[src])
                if not leg:
                    continue  # e.g. ES has no equity leg
                if src == "taq" and not leg.get("taq_sym_root"):
                    notes.append(f"taq: {inst} has no taq_sym_root; skipped")
                    continue
                if src == "databento" and not leg.get("symbol"):
                    notes.append(f"databento: {inst} futures symbol not resolved yet; skipped")
                    continue
                tasks.extend(Task(src, inst, d, leg) for d in src_dates)
        return tasks, notes


def to_date(x: Any) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def date_range(start: Any, end: Any) -> list[dt.date]:
    a, b = to_date(start), to_date(end)
    if b < a:
        raise ValueError(f"end {b} is before start {a}")
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def parse_blocks(spec: Any) -> list[dt.date]:
    """{start, end} | {dates: [...]} | [block, ...] -> sorted unique dates."""
    blocks = spec if isinstance(spec, list) else [spec]
    out: set[dt.date] = set()
    for b in blocks:
        if b.get("dates"):
            out.update(to_date(d) for d in b["dates"])
        elif b.get("start") and b.get("end"):
            out.update(date_range(b["start"], b["end"]))
        else:
            raise ValueError(f"Date block needs dates or start/end: {b}")
    excl = set()
    for b in blocks:
        excl.update(to_date(d) for d in b.get("exclude_dates", []) or [])
    return sorted(out - excl)
