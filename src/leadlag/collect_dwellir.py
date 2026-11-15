"""Dwellir Hyperliquid Index candle collector (historical 1s/1m/5m trade candles).

Endpoint: GET {base_url}/{API_KEY}/v1/candles/range
          ?market=xyz:TSLA&interval=1m&start=...Z&end=...Z&limit=5000[&cursor=...]
start inclusive, end exclusive, both aligned to the interval. Pages carry
`data`, `hasMore`, `nextCursor`; billing = X-Iris-Billing-Weight = len(data).

The API key is part of the URL, so URLs are never logged, and any error text
is redacted before it reaches logs, the manifest or the ledger.

Candles are trade-based (price_type = "trade"). A candle's close is usable at
its interval END (bar_end_utc = open + interval), not its start, and the bar
timestamp is not the time of its last trade. Empty intervals are omitted.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from typing import Callable, Iterator

import pandas as pd
import requests

from .config import Config, Task
from .dwellir_budget import (INTERVAL_MS, Budget, EntitlementNotVerified, QuotaExhausted,
                             UsageLedger, page_upper_bound)
from .storage import Manifest, already_done, mark_empty, raw_path, write_parquet

log = logging.getLogger("leadlag")


class DwellirError(Exception):
    pass


def _iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_bounds_ms(day: dt.date) -> tuple[int, int]:
    a = int(dt.datetime.combine(day, dt.time(), dt.timezone.utc).timestamp() * 1000)
    return a, a + 86_400_000


class DwellirClient:
    def __init__(self, base_url: str, api_key: str, budget: Budget, max_rps: float = 2,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep,
                 max_network_retries: int = 2, max_429: int = 4, timeout: float = 60):
        if not api_key:
            raise DwellirError("DWELLIR_API_KEY is not set (see .env.example)")
        self._base = base_url.rstrip("/")
        self._key = api_key
        self.budget = budget
        self.min_interval = 1.0 / max_rps if max_rps else 0
        self.s = session or requests.Session()
        self.sleep = sleep
        self.max_network_retries = max_network_retries
        self.max_429 = max_429
        self.timeout = timeout
        self._last = 0.0

    def redact(self, text: str) -> str:
        return str(text).replace(self._key, "***") if self._key else str(text)

    def _throttle(self) -> None:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def _page(self, params: dict, bound: int) -> dict:
        """One guarded request with reservation, settlement, retries and backoff."""
        url = f"{self._base}/{self._key}/v1/candles/range"
        meta = {"request": {k: v for k, v in params.items()}}   # sanitized: no key, no URL
        net_fail = n429 = 0
        while True:
            rid, day = self.budget.reserve(bound, meta)          # may raise QuotaExhausted
            self._throttle()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                # No response: the server may still have billed it. Charge in full.
                self.budget.settle(rid, day, bound, "no_response")
                net_fail += 1
                msg = self.redact(f"{type(e).__name__}: {e}")
                if net_fail > self.max_network_retries:
                    raise DwellirError(f"network failure after {net_fail} attempts: {msg}") from None
                log.warning("dwellir network error (%s); retry %d", msg[:120], net_fail)
                self.sleep(2 ** net_fail)
                continue
            if r.status_code == 429:
                self.budget.settle(rid, day, 0, "429")
                n429 += 1
                if n429 > self.max_429:
                    raise QuotaExhausted("repeated 429 responses: rate limit or daily quota reached; "
                                         "stopping (no plan or endpoint change)")
                delay = float(r.headers.get("Retry-After") or 2 ** n429)
                log.warning("dwellir 429; backing off %.0fs", delay)
                self.sleep(delay)
                continue
            if r.status_code in (401, 402, 403):
                self.budget.settle(rid, day, 0, f"http_{r.status_code}")
                raise QuotaExhausted(f"HTTP {r.status_code}: access/quota refused: "
                                     f"{self.redact(r.text)[:200]}")
            if r.status_code >= 400:
                self.budget.settle(rid, day, 0, f"http_{r.status_code}")
                raise DwellirError(f"HTTP {r.status_code}: {self.redact(r.text)[:300]}")
            try:
                body = r.json()
                data = body["data"]
            except Exception:
                self.budget.settle(rid, day, bound, "bad_body")
                raise DwellirError("unparseable response body") from None
            header = r.headers.get("X-Iris-Billing-Weight")
            billed = max(len(data), int(header) if header and header.isdigit() else 0)
            self.budget.settle(rid, day, billed, "ok")
            body["_billed"] = billed
            return body

    def candle_pages(self, market: str, interval: str, start_ms: int, end_ms: int,
                     limit: int = 5000, max_pages: int | None = None) -> Iterator[dict]:
        if start_ms % INTERVAL_MS[interval] or end_ms % INTERVAL_MS[interval]:
            raise DwellirError("start/end must align to the interval")
        cursor: str | None = None
        pos = start_ms
        pages = 0
        while True:
            params = {"market": market, "interval": interval, "end": _iso(end_ms), "limit": int(limit)}
            if cursor is None:
                params["start"] = _iso(start_ms)
            else:
                params["cursor"] = cursor
            bound = page_upper_bound(pos, end_ms, interval, limit)
            if bound == 0:
                return
            body = self._page(params, bound)
            yield body
            pages += 1
            if not body.get("hasMore") or (max_pages and pages >= max_pages):
                return
            cursor = body.get("nextCursor")
            if not cursor:
                raise DwellirError("hasMore=true without nextCursor")
            try:
                pos = int(cursor)
            except ValueError:  # opaque cursor: fall back to the last candle seen
                pos = (body["data"][-1]["t"] + INTERVAL_MS[interval]) if body["data"] else pos


def candles_frame(rows: list[dict], interval: str) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(rows)
    checks = {"returned": len(df)}
    df = df.drop_duplicates(subset=["t"])
    checks["duplicates_dropped"] = checks["returned"] - len(df)
    step = INTERVAL_MS[interval]
    checks["misaligned"] = int((df["t"] % step != 0).sum())
    checks["not_finalized"] = int((~df["x"].astype(bool)).sum())
    out = pd.DataFrame({
        "open_ts_utc": pd.to_datetime(df["t"], unit="ms", utc=True),
        "bar_end_utc": pd.to_datetime(df["t"] + step, unit="ms", utc=True),
        "close_ts_utc": pd.to_datetime(df["T"], unit="ms", utc=True),
        "coin": df["s"],
        "interval": df["i"],
        "open": df["o"].astype(float),
        "high": df["h"].astype(float),
        "low": df["l"].astype(float),
        "close": df["c"].astype(float),
        "volume": df["v"].astype(float),
        "notional": df["q"].astype(float),
        "n_trades": df["n"].astype("int64"),
        "finalized": df["x"].astype(bool),
    }).sort_values("open_ts_utc").reset_index(drop=True)
    out["price_type"] = "trade"
    out["source"] = "dwellir"
    return out, checks


def make_budget(cfg: Config) -> Budget:
    c = cfg.source_cfg("dwellir")
    ledger = UsageLedger(cfg.data_dir / "ledger" / "dwellir_usage.jsonl",
                         c.get("quota_reset_timezone", "UTC"))
    return Budget(ledger, c.get("daily_allowance", 100_000), c.get("headroom_fraction", 0.2),
                  c.get("other_usage_today", 0))


def make_client(cfg: Config, budget: Budget | None = None, **kw) -> DwellirClient:
    c = cfg.source_cfg("dwellir")
    return DwellirClient(c.get("base_url", "https://api-hyperliquid-index.n.dwellir.com"),
                         os.environ.get("DWELLIR_API_KEY", ""), budget or make_budget(cfg),
                         c.get("max_requests_per_second", 2), **kw)


def max_credits(tasks: list[Task], interval: str) -> int:
    """Upper bound on billed candles for a list of UTC-day tasks."""
    return len(tasks) * (86_400_000 // INTERVAL_MS[interval])


class DwellirCollector:
    source = "dwellir"

    def __init__(self, cfg: Config, manifest: Manifest, force: bool = False, dry_run: bool = False,
                 client: DwellirClient | None = None, interval: str | None = None):
        self.cfg = cfg
        self.c = cfg.source_cfg("dwellir")
        self.manifest = manifest
        self.force = force
        self.dry_run = dry_run
        self.interval = interval or self.c.get("interval", "1m")
        if self.interval not in INTERVAL_MS:
            raise ValueError(f"interval must be one of {list(INTERVAL_MS)}")
        self._client = client
        self.stopped: str | None = None    # set once quota/entitlement stops the run

    @property
    def client(self) -> DwellirClient:
        if self._client is None:
            self._client = make_client(self.cfg)
        return self._client

    def run(self, task: Task) -> None:
        coin = task.leg["coin"]
        dataset = f"candles_{self.interval}"
        path = raw_path(self.cfg.data_dir, self.source, dataset, coin, task.date)
        if already_done(path, self.force):
            self.manifest.record(self.source, dataset, coin, task.date, "skipped", path=path)
            return
        if self.stopped:
            self.manifest.record(self.source, dataset, coin, task.date, "blocked", message=self.stopped)
            return
        start, end = day_bounds_ms(task.date)
        if end > time.time() * 1000 - INTERVAL_MS[self.interval]:
            self.manifest.record(self.source, dataset, coin, task.date, "incomplete",
                                 message="UTC day not finished")
            return
        if self.dry_run:
            log.info("[dry-run] %s %s %s  max %d credits", dataset, coin, task.date,
                     86_400_000 // INTERVAL_MS[self.interval])
            return
        if not self.c.get("entitlement_verified"):
            self.stopped = ("dwellir.entitlement_verified is false: confirm free-plan access to the "
                            "candle-range endpoint and no paid overage, then set it to true "
                            "(use `leadlag dwellir-probe` to test access)")
            self.manifest.record(self.source, dataset, coin, task.date, "blocked", message=self.stopped)
            return
        try:
            rows, billed = [], 0
            for body in self.client.candle_pages(coin, self.interval, start, end,
                                                 self.c.get("page_limit", 5000)):
                rows.extend(body["data"])
                billed += body["_billed"]
            if not rows:
                mark_empty(path, "no candles (sparse: no trades, or market not listed)")
                self.manifest.record(self.source, dataset, coin, task.date, "empty", path=path,
                                     message="0 credits")
                return
            df, checks = candles_frame(rows, self.interval)
            df = df[(df.open_ts_utc >= pd.Timestamp(start, unit="ms", tz="UTC"))
                    & (df.open_ts_utc < pd.Timestamp(end, unit="ms", tz="UTC"))]
            write_parquet(df, path)
            expected = 86_400_000 // INTERVAL_MS[self.interval]
            flags = ", ".join(f"{k}={v}" for k, v in checks.items() if k != "returned" and v)
            self.manifest.record(self.source, dataset, coin, task.date, "ok", len(df), path,
                                 f"{len(df)}/{expected} bars, {billed} credits" + (f", {flags}" if flags else ""))
        except QuotaExhausted as e:
            self.stopped = str(e)
            self.manifest.record(self.source, dataset, coin, task.date, "blocked", message=str(e)[:200])
        except Exception as e:
            msg = self.client.redact(str(e)) if self._client else str(e)
            self.manifest.record(self.source, dataset, coin, task.date, "error", message=msg[:200])


def probe(cfg: Config, coin: str, day: dt.date, interval: str = "1m", client: DwellirClient | None = None
          ) -> dict:
    """Tiny access test: one page of at most `probe_limit` candles. Allowed before verification."""
    limit = int(cfg.source_cfg("dwellir").get("probe_limit", 5))
    client = client or make_client(cfg)
    start, end = day_bounds_ms(day)
    body = next(client.candle_pages(coin, interval, start, end, limit=limit, max_pages=1))
    return {"coin": coin, "day": str(day), "interval": interval, "returned": len(body["data"]),
            "billed": body["_billed"], "hasMore": body.get("hasMore"),
            "first": body["data"][0] if body["data"] else None,
            "remaining_today": client.budget.remaining()}
