"""File layout, atomic Parquet writes, skip-if-done, retries, and a run manifest.

Layout:  data/raw/<source>/<dataset>/<instrument>/<YYYY-MM-DD>.parquet
Markers: a sibling <YYYY-MM-DD>.empty file records "requested, source had no
         data" so re-runs don't hammer the API. Use --force to retry.
"""
from __future__ import annotations

import csv
import datetime as dt
import functools
import logging
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd

log = logging.getLogger("leadlag")
T = TypeVar("T")


def safe_name(s: str) -> str:
    """'xyz:TSLA' -> 'xyz_TSLA' (colons are awkward in file names)."""
    return s.replace(":", "_").replace("/", "_")


def raw_path(data_dir: Path, source: str, dataset: str, instrument: str, date: dt.date) -> Path:
    return data_dir / "raw" / source / dataset / safe_name(instrument) / f"{date.isoformat()}.parquet"


def empty_marker(path: Path) -> Path:
    return path.with_suffix(".empty")


def already_done(path: Path, force: bool) -> bool:
    if force:
        return False
    return path.exists() or empty_marker(path).exists()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    empty_marker(path).unlink(missing_ok=True)


def mark_empty(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    empty_marker(path).write_text(reason + "\n")


def retry(tries: int = 4, base_delay: float = 2.0, exceptions: tuple = (Exception,),
          giveup: Callable[[Exception], bool] | None = None):
    """Exponential-backoff retry for flaky network calls."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(1, tries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:  # noqa: PERF203
                    if attempt == tries or (giveup and giveup(e)):
                        raise
                    delay = base_delay * 2 ** (attempt - 1)
                    log.warning("%s failed (%s); retry %d/%d in %.0fs",
                                fn.__name__, e, attempt, tries - 1, delay)
                    time.sleep(delay)
            raise RuntimeError("unreachable")

        return wrapper

    return deco


class Manifest:
    """Append-only CSV log of every collection attempt (for coverage checks)."""

    FIELDS = ["run_utc", "source", "dataset", "instrument", "date", "status", "rows", "path", "message"]

    def __init__(self, data_dir: Path):
        self.path = data_dir / "manifest.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def record(self, source: str, dataset: str, instrument: str, date, status: str,
               rows: int = 0, path: Path | str = "", message: str = "") -> None:
        row = [dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), source, dataset,
               instrument, str(date), status, rows, str(path), message]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        lvl = logging.WARNING if status in ("error", "unavailable") else logging.INFO
        log.log(lvl, "%-11s %-18s %-10s %s  %-11s rows=%-7d %s",
                source, dataset, instrument, date, status, rows, message)
