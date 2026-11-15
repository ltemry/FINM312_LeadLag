"""Hyperliquid historical collector (REST API + optional S3 L2 archive).

What history is actually available for HIP-3 perps such as xyz:TSLA:
  * candleSnapshot: trade-based OHLCV bars. Only the most recent 5000 bars per
    interval are served (1m ~ 3.5 days back, 1h ~ 208 days back).
  * fundingHistory: hourly funding rate and premium, full history.
  * metaAndAssetCtxs: contract metadata and current mark/oracle/mid (snapshot only).
  * S3 requester-pays archive of L2 book snapshots (optional, needs AWS creds;
    uploaded roughly monthly; HIP-3 coverage undocumented).
Historical quotes (BBO) and mark/oracle time series are NOT served by the API,
so for forward-looking dates run `leadlag hl-record` (see hl_live.py).

All timestamps are stored as tz-aware UTC. Files are per coin per UTC day.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import time

import pandas as pd
import requests

from .config import Config, Task
from .storage import Manifest, already_done, mark_empty, raw_path, retry, write_parquet

log = logging.getLogger("leadlag")

INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
               "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
               "12h": 43_200_000, "1d": 86_400_000}
MAX_CANDLES = 5000
DAY_MS = 86_400_000


def _utc_day_ms(d: dt.date) -> tuple[int, int]:
    start = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    return start, start + DAY_MS - 1


def _now_ms() -> int:
    return int(time.time() * 1000)


class HyperliquidAPI:
    def __init__(self, url: str):
        self.url = url
        self.s = requests.Session()
        self.s.headers["Content-Type"] = "application/json"

    @retry(tries=5, base_delay=2, exceptions=(requests.RequestException,))
    def info(self, payload: dict):
        r = self.s.post(self.url, json=payload, timeout=30)
        if r.status_code == 429:
            raise requests.HTTPError("429 rate limited")
        r.raise_for_status()
        time.sleep(0.25)  # stay well under the 1200 weight/min limit
        return r.json()

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        out: list[dict] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self.info({"type": "candleSnapshot",
                               "req": {"coin": coin, "interval": interval,
                                       "startTime": cursor, "endTime": end_ms}})
            if not batch:
                break
            out.extend(batch)
            last_open = batch[-1]["t"]
            if len(batch) < MAX_CANDLES or last_open + INTERVAL_MS[interval] > end_ms:
                break
            cursor = last_open + INTERVAL_MS[interval]
        return out

    def funding(self, coin: str, start_ms: int, end_ms: int) -> list[dict]:
        out: list[dict] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self.info({"type": "fundingHistory", "coin": coin,
                               "startTime": cursor, "endTime": end_ms})
            if not batch:
                break
            out.extend(batch)
            nxt = batch[-1]["time"] + 1
            if nxt <= cursor or len(batch) < 500:
                break
            cursor = nxt
        return out

    def meta_and_ctxs(self, dex: str):
        return self.info({"type": "metaAndAssetCtxs", "dex": dex})

    def perp_dexs(self):
        return self.info({"type": "perpDexs"})


class HyperliquidCollector:
    source = "hyperliquid"

    def __init__(self, cfg: Config, manifest: Manifest, force: bool = False, dry_run: bool = False):
        self.cfg = cfg
        self.c = cfg.source_cfg("hyperliquid")
        self.api = HyperliquidAPI(self.c.get("info_url", "https://api.hyperliquid.xyz/info"))
        self.manifest = manifest
        self.force = force
        self.dry_run = dry_run
        self._s3 = None
        self._meta_done: set[str] = set()

    def coins(self, task: Task) -> list[str]:
        coins = [task.leg["coin"]]
        if self.c.get("collect_alt_coins"):
            coins += list(task.leg.get("alt_coins") or [])
        return coins

    # ------------------------------------------------------------- metadata
    def snapshot_meta(self, dex: str) -> None:
        """Save contract metadata + current mark/oracle once per run per dex.

        Re-running this over time builds a record of contract-rule changes
        (leverage, size decimals, fee scale, delistings).
        """
        if dex in self._meta_done or self.dry_run:
            return
        self._meta_done.add(dex)
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.cfg.data_dir / "raw" / self.source / "meta" / dex / f"{ts}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta, ctxs = self.api.meta_and_ctxs(dex)
        dexs = self.api.perp_dexs()
        dex_info = next((x for x in dexs if x and x.get("name") == dex), None)
        path.write_text(json.dumps({"captured_utc": ts, "dex": dex_info, "meta": meta,
                                    "asset_ctxs": ctxs}, indent=1))
        log.info("saved %s contract metadata -> %s", dex, path.relative_to(self.cfg.data_dir))

    # ------------------------------------------------------------- candles
    def _candles(self, coin: str, d: dt.date, interval: str) -> None:
        dataset = f"candles_{interval}"
        path = raw_path(self.cfg.data_dir, self.source, dataset, coin, d)
        if already_done(path, self.force):
            self.manifest.record(self.source, dataset, coin, d, "skipped", path=path)
            return
        start, end = _utc_day_ms(d)
        now = _now_ms()
        earliest = now - MAX_CANDLES * INTERVAL_MS[interval]
        if end >= now:
            self.manifest.record(self.source, dataset, coin, d, "incomplete",
                                 message="UTC day not finished; re-run later")
            return
        if start < earliest:
            self.manifest.record(self.source, dataset, coin, d, "unavailable",
                                 message=f"older than API window for {interval} "
                                         f"(earliest ~{pd.Timestamp(earliest, unit='ms', tz='UTC'):%Y-%m-%d %H:%M})")
            return
        if self.dry_run:
            log.info("[dry-run] %s %s %s", dataset, coin, d)
            return
        try:
            rows = self.api.candles(coin, interval, start, end)
            if not rows:
                mark_empty(path, "no candles (contract not trading?)")
                self.manifest.record(self.source, dataset, coin, d, "empty", path=path)
                return
            df = pd.DataFrame(rows)
            df = pd.DataFrame({
                "open_ts_utc": pd.to_datetime(df["t"], unit="ms", utc=True),
                "close_ts_utc": pd.to_datetime(df["T"], unit="ms", utc=True),
                "coin": df["s"],
                "interval": df["i"],
                "open": df["o"].astype(float),
                "high": df["h"].astype(float),
                "low": df["l"].astype(float),
                "close": df["c"].astype(float),
                "volume": df["v"].astype(float),
                "n_trades": df["n"].astype("int64"),
            }).drop_duplicates("open_ts_utc").sort_values("open_ts_utc")
            write_parquet(df, path)
            expected = DAY_MS // INTERVAL_MS[interval]
            self.manifest.record(self.source, dataset, coin, d, "ok", len(df), path,
                                 f"{len(df)}/{expected} bars (bars with no trades are absent)")
        except Exception as e:
            self.manifest.record(self.source, dataset, coin, d, "error", message=str(e)[:200])

    # ------------------------------------------------------------- funding
    def _funding(self, coin: str, d: dt.date) -> None:
        dataset = "funding"
        path = raw_path(self.cfg.data_dir, self.source, dataset, coin, d)
        if already_done(path, self.force):
            self.manifest.record(self.source, dataset, coin, d, "skipped", path=path)
            return
        start, end = _utc_day_ms(d)
        if end >= _now_ms():
            self.manifest.record(self.source, dataset, coin, d, "incomplete", message="UTC day not finished")
            return
        if self.dry_run:
            log.info("[dry-run] %s %s %s", dataset, coin, d)
            return
        try:
            rows = self.api.funding(coin, start, end)
            if not rows:
                mark_empty(path, "no funding records")
                self.manifest.record(self.source, dataset, coin, d, "empty", path=path)
                return
            df = pd.DataFrame(rows)
            df = pd.DataFrame({
                "ts_utc": pd.to_datetime(df["time"], unit="ms", utc=True),
                "coin": df["coin"],
                "funding_rate": df["fundingRate"].astype(float),
                "premium": df["premium"].astype(float),
            })
            write_parquet(df, path)
            self.manifest.record(self.source, dataset, coin, d, "ok", len(df), path)
        except Exception as e:
            self.manifest.record(self.source, dataset, coin, d, "error", message=str(e)[:200])

    # ------------------------------------------------------------- S3 L2 book
    @property
    def s3(self):
        if self._s3 is None:
            import boto3  # optional dependency

            self._s3 = boto3.client("s3")
        return self._s3

    def _s3_l2book(self, coin: str, d: dt.date) -> None:
        import lz4.frame

        sc = self.c.get("s3_l2book", {})
        levels = int(sc.get("book_levels", 5))
        dataset = "l2book_s3"
        path = raw_path(self.cfg.data_dir, self.source, dataset, coin, d)
        if already_done(path, self.force):
            self.manifest.record(self.source, dataset, coin, d, "skipped", path=path)
            return
        if self.dry_run:
            log.info("[dry-run] %s %s %s (24 hourly S3 objects)", dataset, coin, d)
            return
        bucket = sc.get("bucket", "hyperliquid-archive")
        records, missing = [], []
        for hour in range(24):
            key = f"market_data/{d:%Y%m%d}/{hour}/l2Book/{coin}.lz4"
            try:
                obj = self.s3.get_object(Bucket=bucket, Key=key, RequestPayer="requester")
                text = lz4.frame.decompress(obj["Body"].read()).decode()
            except Exception as e:
                missing.append(f"{hour}:{type(e).__name__}")
                continue
            for line in io.StringIO(text):
                if not line.strip():
                    continue
                rec = json.loads(line)
                data = rec.get("raw", {}).get("data", rec)
                bids, asks = data["levels"]
                row = {"ts_utc": pd.Timestamp(data["time"], unit="ms", tz="UTC"), "coin": coin}
                for side, book in (("bid", bids), ("ask", asks)):
                    for i in range(levels):
                        row[f"{side}_px_{i}"] = float(book[i]["px"]) if i < len(book) else None
                        row[f"{side}_sz_{i}"] = float(book[i]["sz"]) if i < len(book) else None
                records.append(row)
        if not records:
            mark_empty(path, f"no S3 objects found ({', '.join(missing[:3])}...)")
            self.manifest.record(self.source, dataset, coin, d, "empty", path=path,
                                 message="no S3 l2Book objects for this coin/day")
            return
        df = pd.DataFrame(records).sort_values("ts_utc")
        df["mid"] = (df["bid_px_0"] + df["ask_px_0"]) / 2
        write_parquet(df, path)
        self.manifest.record(self.source, dataset, coin, d, "ok", len(df), path,
                             f"missing hours: {len(missing)}")

    # ------------------------------------------------------------- run
    def run(self, task: Task) -> None:
        dex = task.leg.get("dex")
        if dex:
            try:
                self.snapshot_meta(dex)
            except Exception as e:
                log.warning("metadata snapshot for %s failed: %s", dex, e)
        for coin in self.coins(task):
            for interval in self.c.get("candle_intervals", ["1m"]):
                self._candles(coin, task.date, interval)
            if self.c.get("collect_funding", True):
                self._funding(coin, task.date)
            if self.c.get("s3_l2book", {}).get("enabled"):
                self._s3_l2book(coin, task.date)
