"""Optional forward collector for Hyperliquid (websocket).

A planned option, not a running process: start it only if historical quotes
are unavailable and a separately labelled forward sample is wanted. It cannot
reconstruct past quotes.

Hyperliquid's API does not serve historical BBO quotes or mark/oracle time
series, so this records them going forward:
  bbo             -> best bid/ask on every change          (perp quote midpoint)
  trades          -> every trade                            (activity, trade-based prices)
  activeAssetCtx  -> mark, oracle, mid, funding, OI pushes  (kept separate from quotes)

Output: data/raw/hyperliquid/live_<channel>/<coin>/<YYYY-MM-DD>/part-<HHMMSS>.parquet
(one part per flush; read a day with pd.read_parquet(<dir>)). Each row has the
exchange timestamp where provided (exch_ts_utc) and the local receive time
(recv_ts_utc); activeAssetCtx has no exchange timestamp. On every (re)subscribe
the trades channel replays recent trades, so de-duplicate on `tid` when reading.

Leave it running (e.g. in tmux or a launchd job) across the sample period.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import signal
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import websockets

from .storage import safe_name

log = logging.getLogger("leadlag")


def _ms_to_ts(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC") if ms is not None else pd.NaT


class LiveRecorder:
    def __init__(self, ws_url: str, coins: list[str], channels: list[str], data_dir: Path,
                 flush_seconds: int = 60):
        self.ws_url = ws_url
        self.coins = coins
        self.channels = channels
        self.data_dir = data_dir
        self.flush_seconds = flush_seconds
        self.buf: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.stop = asyncio.Event()
        self.counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------ parsing
    def handle(self, msg: dict) -> None:
        ch = msg.get("channel")
        data = msg.get("data")
        recv = pd.Timestamp.now(tz="UTC")
        if ch == "bbo":
            bid, ask = (data.get("bbo") or [None, None])
            row = {"exch_ts_utc": _ms_to_ts(data.get("time")), "recv_ts_utc": recv, "coin": data["coin"],
                   "bid": float(bid["px"]) if bid else None, "bid_size": float(bid["sz"]) if bid else None,
                   "bid_n": bid.get("n") if bid else None,
                   "ask": float(ask["px"]) if ask else None, "ask_size": float(ask["sz"]) if ask else None,
                   "ask_n": ask.get("n") if ask else None}
            self.buf[("bbo", data["coin"])].append(row)
        elif ch == "trades":
            for t in data:
                self.buf[("trades", t["coin"])].append({
                    "exch_ts_utc": _ms_to_ts(t["time"]), "recv_ts_utc": recv, "coin": t["coin"],
                    "side": t["side"], "px": float(t["px"]), "sz": float(t["sz"]),
                    "tid": t.get("tid"), "hash": t.get("hash")})
        elif ch == "activeAssetCtx":
            ctx = data.get("ctx", {})
            impact = ctx.get("impactPxs") or [None, None]
            self.buf[("activeAssetCtx", data["coin"])].append({
                "recv_ts_utc": recv, "coin": data["coin"],
                "mark_px": _f(ctx.get("markPx")), "oracle_px": _f(ctx.get("oraclePx")),
                "mid_px": _f(ctx.get("midPx")), "funding": _f(ctx.get("funding")),
                "premium": _f(ctx.get("premium")), "open_interest": _f(ctx.get("openInterest")),
                "impact_bid": _f(impact[0]), "impact_ask": _f(impact[1]),
                "day_ntl_vlm": _f(ctx.get("dayNtlVlm"))})
        else:
            return
        self.counts[ch] += 1

    def event(self, kind: str, detail: str = "") -> None:
        """Connection events, so gaps in the sample can be flagged later."""
        self.buf[("events", "_session")].append({"recv_ts_utc": pd.Timestamp.now(tz="UTC"),
                                                 "event": kind, "detail": detail[:300],
                                                 "coins": ",".join(self.coins)})

    # ------------------------------------------------------------ storage
    def flush(self) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        for (ch, coin), rows in list(self.buf.items()):
            if not rows:
                continue
            df = pd.DataFrame(rows)
            ts_col = "exch_ts_utc" if "exch_ts_utc" in df else "recv_ts_utc"
            # split by UTC day so files never straddle midnight
            for day, part in df.groupby(df[ts_col].fillna(df["recv_ts_utc"]).dt.date):
                d = self.data_dir / "raw" / "hyperliquid" / f"live_{ch}" / safe_name(coin) / str(day)
                d.mkdir(parents=True, exist_ok=True)
                part.to_parquet(d / f"part-{now:%H%M%S%f}.parquet", index=False)
            self.buf[(ch, coin)] = []
        log.info("flushed; messages so far: %s", dict(self.counts))

    # ------------------------------------------------------------ loop
    async def _session(self) -> None:
        async with websockets.connect(self.ws_url, ping_interval=None, max_size=2**24) as ws:
            for coin in self.coins:
                for ch in self.channels:
                    await ws.send(json.dumps({"method": "subscribe",
                                              "subscription": {"type": ch, "coin": coin}}))
            log.info("subscribed %s to %s", self.coins, self.channels)
            self.event("connected", ",".join(self.channels))
            last_ping = last_flush = time.monotonic()
            while not self.stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    self.handle(json.loads(raw))
                except asyncio.TimeoutError:
                    pass
                now = time.monotonic()
                if now - last_ping > 30:  # server drops idle connections after 60s
                    await ws.send(json.dumps({"method": "ping"}))
                    last_ping = now
                if now - last_flush > self.flush_seconds:
                    self.flush()
                    last_flush = now

    async def run(self, duration: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop.set)
        if duration:
            loop.call_later(duration, self.stop.set)
        backoff = 1
        while not self.stop.is_set():
            try:
                await self._session()
                backoff = 1
            except Exception as e:
                log.warning("websocket error: %s; reconnecting in %ds", e, backoff)
                self.event("disconnected", f"{type(e).__name__}: {e}")
                self.flush()
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60)
        self.event("stopped")
        self.flush()


def _f(x):
    return float(x) if x not in (None, "") else None
