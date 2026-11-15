"""Databento CME collector (ES futures) per UTC day.

Credentials: set DATABENTO_API_KEY in the environment (or in a .env file at
the repo root). Every request is cost-checked first and refused if it exceeds
`databento.max_cost_usd_per_request` in config.yaml.

Files are split by UTC calendar day, the same convention as Hyperliquid. The
CME Globex session (17:00-16:00 CT) is labelled later in the cleaning step.
"""
from __future__ import annotations

import datetime as dt
import logging
import os

import pandas as pd

from .config import Config, Task
from .storage import Manifest, already_done, mark_empty, raw_path, retry, write_parquet

log = logging.getLogger("leadlag")


class DatabentoCollector:
    source = "databento"

    def __init__(self, cfg: Config, manifest: Manifest, force: bool = False, dry_run: bool = False):
        self.cfg = cfg
        self.c = cfg.source_cfg("databento")
        self.manifest = manifest
        self.force = force
        self.dry_run = dry_run
        self.schema = self.c.get("schema", "bbo-1s")
        self.max_cost = float(self.c.get("max_cost_usd_per_request", 5.0))
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import databento as db

            key = os.environ.get("DATABENTO_API_KEY")
            if not key:
                raise RuntimeError("DATABENTO_API_KEY is not set (see .env.example)")
            self._client = db.Historical(key)
        return self._client

    def _req(self, task: Task) -> dict:
        start = dt.datetime(task.date.year, task.date.month, task.date.day, tzinfo=dt.timezone.utc)
        return dict(dataset=task.leg.get("dataset", "GLBX.MDP3"), schema=self.schema,
                    symbols=[task.leg["symbol"]], stype_in=task.leg.get("stype_in", "raw_symbol"),
                    start=start.isoformat(), end=(start + dt.timedelta(days=1)).isoformat())

    @retry(tries=3, base_delay=5)
    def _cost(self, req: dict) -> float:
        return float(self.client.metadata.get_cost(**req))

    @retry(tries=3, base_delay=5)
    def _get(self, req: dict) -> pd.DataFrame:
        store = self.client.timeseries.get_range(**req)
        return store.to_df(price_type="float", pretty_ts=True, map_symbols=True)

    def run(self, task: Task) -> None:
        sym = task.leg["symbol"]
        dataset = self.schema.replace("-", "_")
        path = raw_path(self.cfg.data_dir, self.source, dataset, sym, task.date)
        if already_done(path, self.force):
            self.manifest.record(self.source, dataset, sym, task.date, "skipped", path=path)
            return
        end_utc = dt.datetime.combine(task.date + dt.timedelta(days=1), dt.time(), dt.timezone.utc)
        if end_utc > dt.datetime.now(dt.timezone.utc):
            self.manifest.record(self.source, dataset, sym, task.date, "incomplete",
                                 message="UTC day not finished")
            return
        req = self._req(task)
        try:
            _ = self.client  # fail fast (no retries) if the API key is missing
            cost = self._cost(req)
            if self.dry_run:
                log.info("[dry-run] %s %s %s  estimated cost $%.4f", dataset, sym, task.date, cost)
                return
            if cost > self.max_cost:
                self.manifest.record(self.source, dataset, sym, task.date, "error",
                                     message=f"estimated ${cost:.2f} > max ${self.max_cost:.2f}; raise "
                                             "max_cost_usd_per_request to allow")
                return
            df = self._get(req)
            if df.empty:
                mark_empty(path, "no records")
                self.manifest.record(self.source, dataset, sym, task.date, "empty", path=path)
                return
            df = df.reset_index()
            for col in ("ts_recv", "ts_event"):
                if col in df:
                    df[col] = pd.to_datetime(df[col], utc=True)
            if {"bid_px_00", "ask_px_00"} <= set(df.columns):
                df["mid"] = (df["bid_px_00"] + df["ask_px_00"]) / 2
            write_parquet(df, path)
            self.manifest.record(self.source, dataset, sym, task.date, "ok", len(df), path,
                                 f"cost ${cost:.4f}")
        except Exception as e:
            self.manifest.record(self.source, dataset, sym, task.date, "error", message=str(e)[:200])
