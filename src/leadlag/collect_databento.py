"""Databento CME collector (ES, and TSLA single-stock futures once resolved).

Every request goes through cost_guard.CostGuard: quoted with metadata.get_cost
immediately before download and allowed only at exactly USD 0. Files are per
symbol per UTC day per schema. Credentials: DATABENTO_API_KEY (.env).
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import pandas as pd

from .config import Config, Task
from .cost_guard import CostBlocked, CostGuard, DatabentoRequest, InvalidRequest
from .storage import Manifest, raw_path, write_parquet

log = logging.getLogger("leadlag")


def make_client():
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is not set (see .env.example)")
    return db.Historical(key)


def save_store(store, path: Path) -> int:
    df = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
    if df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".empty").write_text("no records\n")
        return 0
    df = df.reset_index()
    for col in ("ts_recv", "ts_event"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True)
    if {"bid_px_00", "ask_px_00"} <= set(df.columns):
        df["mid"] = (df["bid_px_00"] + df["ask_px_00"]) / 2
    write_parquet(df, path)
    return len(df)


def day_request(leg: dict, schema: str, day: dt.date) -> DatabentoRequest:
    start = dt.datetime.combine(day, dt.time(), dt.timezone.utc)
    return DatabentoRequest(dataset=leg.get("dataset", "GLBX.MDP3"), symbols=(leg["symbol"],),
                            schema=schema, stype_in=leg.get("stype_in", "raw_symbol"),
                            start=start.isoformat(), end=(start + dt.timedelta(days=1)).isoformat())


class DatabentoCollector:
    source = "databento"

    def __init__(self, cfg: Config, manifest: Manifest, force: bool = False, dry_run: bool = False,
                 client=None):
        self.cfg = cfg
        self.c = cfg.source_cfg("databento")
        self.manifest = manifest
        self.force = force
        self.dry_run = dry_run
        self.schemas = list(self.c.get("schemas", ["bbo-1s"]))
        self._client = client
        self._guard: CostGuard | None = None

    @property
    def guard(self) -> CostGuard:
        if self._guard is None:
            self._guard = CostGuard(self._client or make_client(),
                                    self.cfg.data_dir / "ledger" / "databento_requests.jsonl",
                                    self.c.get("allowed_stype_in", ["raw_symbol", "instrument_id"]))
        return self._guard

    def run(self, task: Task) -> None:
        sym = task.leg["symbol"]
        end_utc = dt.datetime.combine(task.date + dt.timedelta(days=1), dt.time(), dt.timezone.utc)
        for schema in self.schemas:
            dataset = schema.replace("-", "_")
            path = raw_path(self.cfg.data_dir, self.source, dataset, sym, task.date)
            if path.with_suffix(".empty").exists() and not self.force:
                self.manifest.record(self.source, dataset, sym, task.date, "skipped", path=path)
                continue
            if self.force:
                path.unlink(missing_ok=True)
            if end_utc > dt.datetime.now(dt.timezone.utc):
                self.manifest.record(self.source, dataset, sym, task.date, "incomplete",
                                     message="UTC day not finished")
                continue
            req = day_request(task.leg, schema, task.date)
            try:
                if self.dry_run:
                    if path.exists():
                        log.info("[dry-run] %s %s %s cached", dataset, sym, task.date)
                        continue
                    a = self.guard.quote(req)
                    log.info("[dry-run] %s %s %s quoted $%s -> would download", dataset, sym,
                             task.date, a.cost)
                    continue
                status, rows = self.guard.fetch(req, path, save_store)
                self.manifest.record(self.source, dataset, sym, task.date,
                                     "skipped" if status == "cached" else ("ok" if rows else "empty"),
                                     rows, path, "quoted $0")
            except (CostBlocked, InvalidRequest) as e:
                self.manifest.record(self.source, dataset, sym, task.date, "blocked", message=str(e)[:200])
            except Exception as e:
                self.manifest.record(self.source, dataset, sym, task.date, "error", message=str(e)[:200])


def resolve_symbols(symbols: list[str], stype_in: str, start: str, end: str,
                    dataset: str = "GLBX.MDP3", stype_out: str = "raw_symbol") -> dict:
    """Free symbology lookup (documented as free; no data download)."""
    client = make_client()
    return client.symbology.resolve(dataset=dataset, symbols=symbols, stype_in=stype_in,
                                    stype_out=stype_out, start_date=start, end_date=end)
