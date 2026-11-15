"""Session grids, as-of alignment, quote ages, and session-safe returns.

Rules (docs/project_plan.md, "Data processing requirements"):
* Session logic in America/New_York via the exchange calendar, then UTC.
* Backward / as-of joins only: a grid point at t only sees observations
  AVAILABLE at or before t. Nothing is interpolated from future values.
* Every aligned observation keeps its source timestamp and age.
* Quotes older than max_age are set to NaN (flagged stale), not carried.
* Trade bars must END exactly at the grid point (exact=True): a missing bar
  is missing, never filled from an older bar as if fresh.
* Returns are computed only between consecutive grid points in the same
  session with both prices present: never across closures or gaps.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .market_calendar import ET, is_trading_day, session_times

SESSIONS = ("pre", "regular", "post")


def session_grid(et_dates: Iterable[dt.date], sessions: Sequence[str] = ("regular",),
                 freq: str = "1min", extra_closures: frozenset = frozenset(),
                 open_window_min: int = 15, close_window_min: int = 15) -> pd.DataFrame:
    """Grid of interval END times within each session, (start, end], in UTC.

    Regular session at 1min -> 09:31 ... 16:00 ET (13:00 on early-close days).
    """
    step = pd.Timedelta(freq)
    rows = []
    for d in sorted(set(et_dates)):
        if not is_trading_day(d, extra_closures):
            continue
        st = session_times(d)
        bounds = {"pre": (st["pre_open"], st["open"]), "regular": (st["open"], st["close"]),
                  "post": (st["close"], st["post_close"])}
        for s in sessions:
            a, b = bounds[s]
            idx = pd.date_range(pd.Timestamp(a) + step, pd.Timestamp(b), freq=step)
            for ts in idx:
                rows.append((ts, d, s))
    df = pd.DataFrame(rows, columns=["ts_et", "et_date", "session"])
    if df.empty:
        return df.assign(ts_utc=pd.Series(dtype="datetime64[ns, UTC]"))
    df["ts_utc"] = df["ts_et"].dt.tz_convert("UTC")
    df["session_id"] = df["et_date"].astype(str) + "_" + df["session"]
    reg = df["session"] == "regular"
    first = df[reg].groupby("et_date")["ts_et"].transform("min")
    last = df[reg].groupby("et_date")["ts_et"].transform("max")
    df["open_window"] = False
    df["close_window"] = False
    df.loc[reg, "open_window"] = df.loc[reg, "ts_et"] < first + pd.Timedelta(minutes=open_window_min)
    df.loc[reg, "close_window"] = df.loc[reg, "ts_et"] > last - pd.Timedelta(minutes=close_window_min)
    return df[["ts_utc", "ts_et", "et_date", "session", "session_id", "open_window", "close_window"]]


def asof_join(grid: pd.DataFrame, series: pd.DataFrame, name: str, max_age: str | None = "60s",
              exact: bool = False, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Attach `series` to `grid` using only observations available at or before each grid time.

    Adds {name}_price, {name}_age_s, {name}_stale, {name}_price_type, {name}_source.
    Age is measured from source_ts_utc when known, else from the availability time.
    """
    out = grid.copy()
    cols = [f"{name}_price", f"{name}_age_s", f"{name}_stale"]
    if series is None or series.empty:
        out[cols[0]] = np.nan
        out[cols[1]] = np.nan
        out[cols[2]] = True
        out[f"{name}_price_type"] = None
        out[f"{name}_source"] = None
        return out
    s = series.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    keep = ["ts_utc", "source_ts_utc", "price", "price_type", "source", *extra_cols]
    s = s[[c for c in keep if c in s.columns]].rename(columns={"ts_utc": "_avail"})
    m = pd.merge_asof(out.sort_values("ts_utc"), s, left_on="ts_utc", right_on="_avail",
                      direction="backward", allow_exact_matches=True)
    ref_ts = m["source_ts_utc"].fillna(m["_avail"]) if "source_ts_utc" in m else m["_avail"]
    age = (m["ts_utc"] - ref_ts).dt.total_seconds()
    stale = m["price"].isna()
    if exact:
        stale |= m["_avail"] != m["ts_utc"]
    if max_age is not None:
        stale |= age > pd.Timedelta(max_age).total_seconds()
    m[f"{name}_price"] = m["price"].where(~stale)
    m[f"{name}_age_s"] = age
    m[f"{name}_stale"] = stale
    m[f"{name}_price_type"] = m.get("price_type")
    m[f"{name}_source"] = m.get("source")
    for c in extra_cols:
        if c in m:
            m[f"{name}_{c}"] = m[c]
    drop = [c for c in ["_avail", "source_ts_utc", "price", "price_type", "source", *extra_cols] if c in m]
    return m.drop(columns=drop)


def session_returns(df: pd.DataFrame, price_col: str, freq: str = "1min") -> pd.Series:
    """Log returns between consecutive grid points of the same session; NaN otherwise."""
    d = df.sort_values("ts_utc")
    p = d[price_col]
    prev_p = p.shift(1)
    same_session = d["session_id"].eq(d["session_id"].shift(1))
    consecutive = (d["ts_utc"] - d["ts_utc"].shift(1)) == pd.Timedelta(freq)
    ok = same_session & consecutive & p.notna() & prev_p.notna() & (p > 0) & (prev_p > 0)
    r = np.log(p / prev_p).where(ok)
    return r.reindex(df.index)


def align_pair(grid: pd.DataFrame, ref: pd.DataFrame, perp: pd.DataFrame, max_age: str = "60s",
               ref_exact: bool | None = None, perp_exact: bool | None = None,
               freq: str = "1min") -> pd.DataFrame:
    """One aligned table per market pair: prices, ages, staleness, session-safe returns.

    Trade-bar series default to exact bar-end matching; quote series to as-of with max_age.
    """
    def is_bars(s):
        return s is not None and not s.empty and (s["price_type"] == "trade").all()

    ref_exact = is_bars(ref) if ref_exact is None else ref_exact
    perp_exact = is_bars(perp) if perp_exact is None else perp_exact
    out = asof_join(grid, ref, "ref", max_age, exact=ref_exact)
    out = asof_join(out, perp, "perp", max_age, exact=perp_exact)
    out["ref_ret"] = session_returns(out, "ref_price", freq)
    out["perp_ret"] = session_returns(out, "perp_price", freq)
    out["both_valid"] = out["ref_ret"].notna() & out["perp_ret"].notna()
    return out


def coverage_table(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per date/session: grid points, valid prices, valid paired returns, median ages."""
    g = aligned.groupby(["et_date", "session"])
    return pd.DataFrame({
        "grid_points": g.size(),
        "ref_valid": g["ref_price"].count(),
        "perp_valid": g["perp_price"].count(),
        "paired_returns": g["both_valid"].sum(),
        "ref_age_med_s": g["ref_age_s"].median().round(2),
        "perp_age_med_s": g["perp_age_s"].median().round(2),
    }).reset_index()


def to_et(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_convert(ET)


def attach_control(aligned: pd.DataFrame, series: pd.DataFrame, name: str = "es",
                   max_age: str = "60s", exact: bool | None = None, freq: str = "1min") -> pd.DataFrame:
    """Add a broad-market control (e.g. ES) on the same grid: {name}_price, ages, {name}_ret.

    Same timestamp convention as the pair: available at or before each grid time.
    """
    if exact is None:
        exact = series is not None and not series.empty and (series["price_type"] == "trade").all()
    out = asof_join(aligned, series, name, max_age, exact=exact)
    out[f"{name}_ret"] = session_returns(out, f"{name}_price", freq)
    return out


def build_for_study(cfg, study: str | None, instrument: str, ref: str | None = None,
                    ref_type: str = "trade", ref_kind: str | None = None, perp_source: str = "dwellir",
                    interval: str = "1m", freq: str = "1min", sessions: Sequence[str] = ("regular",),
                    max_age: str = "60s", which: str = "scored", control: str | None = None,
                    start=None, end=None, dates=None) -> pd.DataFrame:
    """Load, normalize and align one perp-vs-reference pair for a study's dates.

    `control` (e.g. "ES") adds lagged-control returns as `es_ret` (name lower-cased).
    """
    from . import normalize as N

    if cfg.is_deferred(instrument):
        raise ValueError(f"{instrument} is deferred")
    et_dates = cfg.study_dates(study, start, end, dates, which=which)
    trading = cfg.trading_days(et_dates)
    utc_days = cfg.utc_days(et_dates)
    grid = session_grid(trading, sessions, freq, cfg.extra_closures)

    coin = cfg.instruments[instrument]["perp"]["coin"]
    perp = (N.dwellir_bars(cfg, coin, utc_days, interval) if perp_source == "dwellir"
            else N.hl_live_bbo(cfg, coin, utc_days))

    def load_ref(name: str, kind: str | None, rtype: str) -> pd.DataFrame:
        inst = cfg.instruments[name]
        if "equity" in inst and kind != "futures":
            sym = inst["equity"]["taq_sym_root"]
            return N.taq_trade_bars(cfg, sym, trading, freq) if rtype == "trade" else N.taq_mid(cfg, sym, trading)
        sym = inst["futures"]["symbol"]
        return N.databento_bars(cfg, sym, utc_days) if rtype == "trade" else N.databento_bbo(cfg, sym, utc_days)

    ref_name = ref or instrument
    out = align_pair(grid, load_ref(ref_name, ref_kind, ref_type), perp, max_age=max_age, freq=freq)
    out["pair"] = f"{coin} vs {ref_name}"
    out["period"] = study or "custom"
    if study:
        s = cfg.study(study)
        for block in ("development", "evaluation"):
            if s.get(block):
                from .config import parse_blocks
                out.loc[out["et_date"].isin(set(parse_blocks(s[block]))), "block"] = block
    if control:
        out = attach_control(out, load_ref(control, "futures", ref_type), control.lower(), max_age, freq=freq)
    return out
