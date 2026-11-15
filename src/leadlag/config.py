"""Load config.yaml and turn studies into concrete (source, instrument, date) tasks."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .market_calendar import is_trading_day

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

SOURCES = ("taq", "hyperliquid", "databento")
# Which instrument leg each source needs.
SOURCE_LEG = {"taq": "equity", "hyperliquid": "perp", "databento": "futures"}


@dataclass(frozen=True)
class Task:
    source: str
    instrument: str
    date: dt.date
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
            _to_date(d) for d in proj.get("extra_market_closures", []) or []
        )
        self.instruments: dict[str, dict] = self.raw.get("instruments", {})
        self.studies: dict[str, dict] = self.raw.get("studies", {})

    def source_cfg(self, source: str) -> dict:
        return self.raw.get(source, {}) or {}

    # ------------------------------------------------------------------ dates
    def study_dates(
        self,
        study: str | None,
        start: str | None = None,
        end: str | None = None,
        dates: Iterable[str] | None = None,
    ) -> list[dt.date]:
        """Calendar dates for a study. CLI arguments override the config."""
        s = self.studies.get(study, {}) if study else {}
        if dates:
            out = sorted({_to_date(d) for d in dates})
        elif start or end:
            out = _date_range(start or end, end or start)
        elif s.get("dates"):
            out = sorted({_to_date(d) for d in s["dates"]})
        elif s.get("start") and s.get("end"):
            out = _date_range(s["start"], s["end"])
        else:
            raise ValueError(
                f"No dates for study '{study}'. Set dates or start/end in "
                f"{self.path.name}, or pass --dates / --start --end."
            )
        excl = {_to_date(d) for d in s.get("exclude_dates", []) or []}
        return [d for d in out if d not in excl]

    def dates_for_source(self, source: str, dates: list[dt.date]) -> list[dt.date]:
        if source == "taq":
            return [d for d in dates if is_trading_day(d, self.extra_closures)]
        if source == "databento":
            return [d for d in dates if d.weekday() != 5]
        return list(dates)

    # ------------------------------------------------------------------ tasks
    def build_tasks(
        self,
        study: str | None,
        sources: Iterable[str] | None = None,
        instruments: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        dates: Iterable[str] | None = None,
    ) -> tuple[list[Task], list[str]]:
        """Return (tasks, notes). Notes explain anything that was skipped."""
        s = self.studies.get(study, {}) if study else {}
        if study and study not in self.studies:
            raise KeyError(f"Unknown study '{study}'. Known: {', '.join(self.studies)}")
        srcs = list(sources or s.get("sources") or SOURCES)
        insts = list(instruments or s.get("instruments") or self.instruments)
        all_dates = self.study_dates(study, start, end, dates)

        tasks: list[Task] = []
        notes: list[str] = []
        for src in srcs:
            if src not in SOURCES:
                raise ValueError(f"Unknown source '{src}'. Choose from {SOURCES}.")
            src_dates = self.dates_for_source(src, all_dates)
            for inst in insts:
                if inst not in self.instruments:
                    raise KeyError(f"Unknown instrument '{inst}'.")
                leg = self.instruments[inst].get(SOURCE_LEG[src])
                if not leg:
                    continue  # e.g. SP500 has no equity leg
                if src == "taq" and not leg.get("taq_sym_root"):
                    notes.append(f"taq: {inst} has no taq_sym_root yet; skipped")
                    continue
                tasks.extend(Task(src, inst, d, leg) for d in src_dates)
        return tasks, notes


def _to_date(x: Any) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def _date_range(start: Any, end: Any) -> list[dt.date]:
    a, b = _to_date(start), _to_date(end)
    if b < a:
        raise ValueError(f"end {b} is before start {a}")
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]
