"""Load raw Parquet extracts into one common schema per series.

Every normalized series has:
    ts_utc          when the observation is AVAILABLE (as-of joins key on this)
    source_ts_utc   the source's own timestamp of the underlying quote/trade,
                    or NaT when unknown (candle bars: the bar end is not the
                    time of the last trade, so no exact trade age is claimed)
    price           the value used for returns
    price_type      "mid" | "trade" (never silently mixed)
    source          "taq" | "dwellir" | "databento" | "hl_live"
plus source-specific extras (bid, ask, spread_bps, volume, n_trades, flags).

Raw extracts are never modified; these functions only read them.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .storage import safe_name

log = logging.getLogger("leadlag")


def _read_days(base: Path, days: Iterable[dt.date]) -> pd.DataFrame:
    parts, missing = [], []
    for d in days:
        f = base / f"{d.isoformat()}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
        elif not f.with_suffix(".empty").exists():
            missing.append(str(d))
    if missing:
        log.warning("%s: no file for %d day(s): %s", base.relative_to(base.parents[3]), len(missing),
                    ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _quote_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["locked"] = df["bid"] == df["ask"]
    df["crossed"] = df["bid"] > df["ask"]
    df["invalid"] = (df["bid"] <= 0) | (df["ask"] <= 0) | df["bid"].isna() | df["ask"].isna()
    df["spread_bps"] = np.where(df["invalid"], np.nan, (df["ask"] - df["bid"]) / df["mid"] * 1e4)
    # crossed/invalid quotes are never used as prices
    df.loc[df["crossed"] | df["invalid"], "price"] = np.nan
    return df


def taq_mid(cfg: Config, symbol: str, days: Iterable[dt.date], sample: str = "1s") -> pd.DataFrame:
    raw = _read_days(cfg.data_dir / "raw" / "taq" / f"nbbo_{sample}" / safe_name(symbol), days)
    if raw.empty:
        return raw
    df = pd.DataFrame({
        "ts_utc": raw["bucket_end_utc"], "source_ts_utc": raw["quote_ts_utc"],
        "bid": raw["bid"], "ask": raw["ask"], "mid": raw["mid"], "price": raw["mid"],
        "bid_size": raw["bid_size"], "ask_size": raw["ask_size"],
    })
    df["price_type"], df["source"] = "mid", "taq"
    df = _quote_flags(df)
    return df.drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)


def taq_trade_bars(cfg: Config, symbol: str, days: Iterable[dt.date], freq: str = "1min",
                   sample: str = "1s") -> pd.DataFrame:
    """Last-trade bars built from server-side 1s trade buckets, labelled at bar END
    (constructed the same way as the Dwellir candles for the trade-based fallback)."""
    raw = _read_days(cfg.data_dir / "raw" / "taq" / f"trades_{sample}" / safe_name(symbol), days)
    if raw.empty:
        return raw
    raw = raw.sort_values("bucket_end_utc")
    # a 1s bucket ending at 10:00:01 belongs to the bar ending 10:01
    bar_end = (raw["bucket_end_utc"] - pd.Timedelta("1ns")).dt.floor(freq) + pd.Timedelta(freq)
    g = raw.groupby(bar_end)
    df = pd.DataFrame({
        "price": g["last_price"].last(), "source_ts_utc": g["last_trade_ts_utc"].last(),
        "volume": g["volume"].sum(), "n_trades": g["n_trades"].sum(),
    }).rename_axis("ts_utc").reset_index()
    df["price_type"], df["source"] = "trade", "taq"
    return df


def dwellir_bars(cfg: Config, coin: str, days: Iterable[dt.date], interval: str = "1m") -> pd.DataFrame:
    raw = _read_days(cfg.data_dir / "raw" / "dwellir" / f"candles_{interval}" / safe_name(coin), days)
    if raw.empty:
        return raw
    df = pd.DataFrame({
        "ts_utc": raw["bar_end_utc"],             # close is usable at the interval end
        "source_ts_utc": pd.NaT,                  # last-trade time unknown from candles
        "price": raw["close"], "volume": raw["volume"], "notional": raw["notional"],
        "n_trades": raw["n_trades"], "finalized": raw["finalized"],
    })
    df["source_ts_utc"] = pd.to_datetime(df["source_ts_utc"], utc=True)
    df.loc[~df["finalized"], "price"] = np.nan   # completed bars only
    df["price_type"], df["source"] = "trade", "dwellir"
    return df.drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)


def databento_bbo(cfg: Config, symbol: str, days: Iterable[dt.date], schema: str = "bbo_1s") -> pd.DataFrame:
    """Top-of-book samples. ts_recv = sample (availability) time, ts_event = last
    book update. Verify this convention on the pilot day before relying on ages."""
    raw = _read_days(cfg.data_dir / "raw" / "databento" / schema / safe_name(symbol), days)
    if raw.empty:
        return raw
    df = pd.DataFrame({
        "ts_utc": raw["ts_recv"], "source_ts_utc": raw.get("ts_event", raw["ts_recv"]),
        "bid": raw["bid_px_00"], "ask": raw["ask_px_00"],
        "mid": (raw["bid_px_00"] + raw["ask_px_00"]) / 2,
    })
    df["price"] = df["mid"]
    df["price_type"], df["source"] = "mid", "databento"
    df = _quote_flags(df)
    return df.drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)


def databento_bars(cfg: Config, symbol: str, days: Iterable[dt.date], schema: str = "ohlcv_1m") -> pd.DataFrame:
    raw = _read_days(cfg.data_dir / "raw" / "databento" / schema / safe_name(symbol), days)
    if raw.empty:
        return raw
    step = pd.Timedelta(schema.split("_")[1].replace("m", "min"))
    df = pd.DataFrame({
        "ts_utc": raw["ts_event"] + step,        # ohlcv ts_event is the bar open
        "source_ts_utc": pd.NaT, "price": raw["close"], "volume": raw["volume"],
    })
    df["source_ts_utc"] = pd.to_datetime(df["source_ts_utc"], utc=True)
    df["price_type"], df["source"] = "trade", "databento"
    return df.drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)


def hl_live_bbo(cfg: Config, coin: str, days: Iterable[dt.date]) -> pd.DataFrame:
    """Forward-collected Hyperliquid quotes (separately labelled sample)."""
    base = cfg.data_dir / "raw" / "hyperliquid" / "live_bbo" / safe_name(coin)
    parts = [pd.read_parquet(base / str(d)) for d in days if (base / str(d)).exists()]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    df = pd.DataFrame({"ts_utc": raw["recv_ts_utc"], "source_ts_utc": raw["exch_ts_utc"],
                       "bid": raw["bid"], "ask": raw["ask"]})
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["price"] = df["mid"]
    df["price_type"], df["source"] = "mid", "hl_live"
    df = _quote_flags(df)
    return df.sort_values("ts_utc").reset_index(drop=True)
