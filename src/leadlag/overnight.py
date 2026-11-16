"""Stage 5 / Module B: overnight opening-price and closing-auction forecasts.

Opening forecast (docs/project_plan.md):
    C          previous official primary-exchange close (auctions.py)
    P0         perp price immediately before that close: last completed bar
               ending at or before the scheduled close (trades strictly before
               the closing auction)
    P(t)       perp price at cutoff t (08:00, 09:00, 09:25, 09:29 ET): last
               completed bar ending at or before t
    implied_open(t)   = C * P(t) / P0                 (constant-basis forecast)
    O          next official primary-exchange open (target, at its own timestamp)
    actual_gap        = O / C - 1
    predicted_gap(t)  = P(t) / P0 - 1
    forecast_error_bps(t) = 1e4 * (implied_open(t) / O - 1)
Benchmarks at the same cutoff: previous close / no change (C), and the latest
eligible cash premarket NBBO midpoint. The raw perp-to-open premium is reported
separately as a descriptive measure.

Closing forecast: anchor at 15:00 ET (cash midpoint A_cash, perp A_perp),
    implied_close(t) = A_cash * P(t) / A_perp   for t = 15:30, 15:50, 15:59
target = official close; benchmarks: contemporaneous cash midpoint at t and
no change from the anchor. The target never enters its own forecast.

Leakage guard: every feature's availability time must be <= its cutoff, and
the cutoff must be strictly before the target auction's own timestamp;
violating rows are dropped and counted. Missing or stale inputs give a missing
forecast (never a fabricated price). Perp prices here are trade-based
(Dwellir candles); this is labelled in every output.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import normalize as N
from .auctions import official_prices
from .config import Config
from .market_calendar import ET, previous_trading_day, session_times

OPEN_CUTOFFS = ("08:00", "09:00", "09:25", "09:29")
CLOSE_CUTOFFS = ("15:30", "15:50", "15:59")
CLOSE_ANCHOR = "15:00"
SMALL_SAMPLE = 20


# ============================================================== lookups
def asof_lookup(series: pd.DataFrame, targets: pd.Series, max_age: str | None, label: str) -> pd.DataFrame:
    """For each target time, the latest observation AVAILABLE at or before it.

    Returns {label}, {label}_avail_ts, {label}_age_s, {label}_stale aligned to targets.index.
    Age uses the source timestamp when known, else the availability time.
    """
    out = pd.DataFrame(index=targets.index)
    if series is None or series.empty:
        out[label], out[f"{label}_avail_ts"] = np.nan, pd.NaT
        out[f"{label}_age_s"], out[f"{label}_stale"] = np.nan, True
        return out
    s = series.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    s = s[["ts_utc", "price"] + (["source_ts_utc"] if "source_ts_utc" in s else [])]
    left = pd.DataFrame({"_t": pd.to_datetime(targets, utc=True), "_i": targets.index}).dropna(subset=["_t"])
    m = pd.merge_asof(left.sort_values("_t"), s, left_on="_t", right_on="ts_utc", direction="backward",
                      allow_exact_matches=True).set_index("_i")
    ref = m["source_ts_utc"].fillna(m["ts_utc"]) if "source_ts_utc" in m else m["ts_utc"]
    age = (m["_t"] - ref).dt.total_seconds()
    stale = m["price"].isna() | (age > pd.Timedelta(max_age).total_seconds() if max_age else False)
    out[label] = m["price"].where(~stale)
    out[f"{label}_avail_ts"] = m["ts_utc"]
    out[f"{label}_age_s"] = age
    out[f"{label}_stale"] = stale
    out[f"{label}_stale"] = out[f"{label}_stale"].fillna(True).astype(bool)
    return out


def _at(d: dt.date, hhmm: str) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return pd.Timestamp(dt.datetime.combine(d, dt.time(h, m), ET)).tz_convert("UTC")


def _bps(a, b):
    return 1e4 * (a / b - 1)


def _load(cfg: Config, instrument: str, et_days: list[dt.date], interval: str):
    inst = cfg.instruments[instrument]
    coin, sym = inst["perp"]["coin"], inst["equity"]["taq_sym_root"]
    # every calendar day in the span, so overnight/weekend periods between a close
    # and the next open are loaded too (not just the trading days themselves)
    span = pd.date_range(min(et_days), max(et_days), freq="D").date
    perp = N.dwellir_bars(cfg, coin, cfg.utc_days(span), interval)
    mid = N.taq_mid(cfg, sym, cfg.trading_days(et_days))
    return perp, mid


# ============================================================== opening forecasts
def opening_forecasts(cfg: Config, instrument: str, scored: Iterable[dt.date],
                      cutoffs: Sequence[str] = OPEN_CUTOFFS, perp_max_age: str = "5min",
                      cash_max_age: str = "60s", interval: str = "1m") -> tuple[pd.DataFrame, dict]:
    scored = cfg.trading_days(sorted(set(scored)))
    prev = {d: previous_trading_day(d, cfg.extra_closures) for d in scored}
    days = sorted(set(scored) | set(prev.values()))
    auc = official_prices(cfg, instrument, days).set_index("et_date")
    perp, mid = _load(cfg, instrument, days, interval)

    rows = []
    for d in scored:
        p = prev[d]
        for c in cutoffs:
            rows.append({"et_date": d, "prev_date": p, "cutoff": c, "cutoff_ts": _at(d, c),
                         "anchor_ts": pd.Timestamp(session_times(p)["close"]).tz_convert("UTC"),
                         "C": auc.at[p, "close_price"] if p in auc.index else np.nan,
                         "close_ts": auc.at[p, "close_ts_utc"] if p in auc.index else pd.NaT,
                         "O": auc.at[d, "open_price"] if d in auc.index else np.nan,
                         "open_ts": auc.at[d, "open_ts_utc"] if d in auc.index else pd.NaT,
                         "open_source": auc.at[d, "open_source"] if d in auc.index else "missing",
                         "close_source": auc.at[p, "close_source"] if p in auc.index else "missing"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"rows": 0}
    df = df.join(asof_lookup(perp, df["anchor_ts"], perp_max_age, "P0"))
    df = df.join(asof_lookup(perp, df["cutoff_ts"], perp_max_age, "Pt"))
    df = df.join(asof_lookup(mid, df["cutoff_ts"], cash_max_age, "mid_t"))

    # leakage guard
    ok_target = df["cutoff_ts"] < df["open_ts"]
    ok_p0 = (df["P0_avail_ts"] <= df["anchor_ts"]) | df["P0"].isna()
    ok_close = (df["anchor_ts"] <= df["close_ts"]) | df["close_ts"].isna()
    ok_feat = ((df["Pt_avail_ts"] <= df["cutoff_ts"]) | df["Pt"].isna()) & \
              ((df["mid_t_avail_ts"] <= df["cutoff_ts"]) | df["mid_t"].isna())
    valid = (ok_target | df["open_ts"].isna()) & ok_p0 & ok_close & ok_feat
    info = {"rows": len(df), "dropped_for_leakage": int((~valid).sum())}
    df = df[valid].copy()

    df["actual_gap"] = df["O"] / df["C"] - 1
    df["pred_gap_perp"] = df["Pt"] / df["P0"] - 1
    df["pred_gap_mid"] = df["mid_t"] / df["C"] - 1
    df["implied_open_perp"] = df["C"] * df["Pt"] / df["P0"]
    df["err_bps_perp"] = _bps(df["implied_open_perp"], df["O"])
    df["err_bps_nochange"] = _bps(df["C"], df["O"])
    df["err_bps_premkt_mid"] = _bps(df["mid_t"], df["O"])
    df["perp_open_premium_bps"] = _bps(df["Pt"], df["O"])       # raw level, descriptive only
    for f in ("perp", "mid"):
        hit = np.sign(df[f"pred_gap_{f}"]) == np.sign(df["actual_gap"])
        df[f"dir_hit_{f}"] = hit.where(df[f"pred_gap_{f}"].notna() & df["actual_gap"].notna())
    df["perp_price_type"] = "trade (Dwellir 1m candle close)"
    df["cash_price_type"] = "mid (TAQ NBBO)"
    df["interval"] = "premarket"
    return df.reset_index(drop=True), info


# ============================================================== closing forecasts
def closing_forecasts(cfg: Config, instrument: str, scored: Iterable[dt.date],
                      cutoffs: Sequence[str] = CLOSE_CUTOFFS, anchor: str = CLOSE_ANCHOR,
                      perp_max_age: str = "5min", cash_max_age: str = "60s",
                      interval: str = "1m") -> tuple[pd.DataFrame, dict]:
    scored = cfg.trading_days(sorted(set(scored)))
    auc = official_prices(cfg, instrument, scored).set_index("et_date")
    perp, mid = _load(cfg, instrument, scored, interval)
    rows = []
    for d in scored:
        for c in cutoffs:
            rows.append({"et_date": d, "cutoff": c, "cutoff_ts": _at(d, c), "anchor_ts": _at(d, anchor),
                         "target_close": auc.at[d, "close_price"] if d in auc.index else np.nan,
                         "close_ts": auc.at[d, "close_ts_utc"] if d in auc.index else pd.NaT,
                         "close_source": auc.at[d, "close_source"] if d in auc.index else "missing"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"rows": 0}
    df = df.join(asof_lookup(mid, df["anchor_ts"], cash_max_age, "A_cash"))
    df = df.join(asof_lookup(perp, df["anchor_ts"], perp_max_age, "A_perp"))
    df = df.join(asof_lookup(perp, df["cutoff_ts"], perp_max_age, "Pt"))
    df = df.join(asof_lookup(mid, df["cutoff_ts"], cash_max_age, "mid_t"))
    valid = ((df["cutoff_ts"] < df["close_ts"]) | df["close_ts"].isna()) & (df["anchor_ts"] < df["cutoff_ts"])
    for lab, t in (("A_cash", "anchor_ts"), ("A_perp", "anchor_ts"), ("Pt", "cutoff_ts"), ("mid_t", "cutoff_ts")):
        valid &= (df[f"{lab}_avail_ts"] <= df[t]) | df[lab].isna()
    info = {"rows": len(df), "dropped_for_leakage_or_early_close": int((~valid).sum())}
    df = df[valid].copy()
    df["actual_move"] = df["target_close"] / df["A_cash"] - 1
    df["pred_move_perp"] = df["Pt"] / df["A_perp"] - 1
    df["pred_move_mid"] = df["mid_t"] / df["A_cash"] - 1
    df["implied_close_perp"] = df["A_cash"] * df["Pt"] / df["A_perp"]
    df["err_bps_perp"] = _bps(df["implied_close_perp"], df["target_close"])
    df["err_bps_cash_mid"] = _bps(df["mid_t"], df["target_close"])
    df["err_bps_anchor_nochange"] = _bps(df["A_cash"], df["target_close"])
    for f in ("perp", "mid"):
        hit = np.sign(df[f"pred_move_{f}"]) == np.sign(df["actual_move"])
        df[f"dir_hit_{f}"] = hit.where(df[f"pred_move_{f}"].notna() & df["actual_move"].notna())
    df["perp_price_type"] = "trade (Dwellir 1m candle close)"
    return df.reset_index(drop=True), info


# ============================================================== metrics
def forecast_metrics(df: pd.DataFrame, errors: dict[str, str], directions: dict[str, str] | None = None,
                     base: str = "perp") -> pd.DataFrame:
    """Per cutoff: availability, then MAE/RMSE/bias/direction on rows where ALL forecasts exist.

    Paired comparisons (perp vs each benchmark) use identical eligible dates.
    """
    directions = directions or {}
    rows = []
    for cutoff, g in df.groupby("cutoff", sort=False):
        matched = g[list(errors.values())].notna().all(axis=1) & g.get("O", g.get("target_close")).notna()
        gm = g[matched]
        for name, col in errors.items():
            e = gm[col]
            r = {"cutoff": cutoff, "forecast": name, "n_available": int(g[col].notna().sum()),
                 "n_missing_or_stale": int(g[col].isna().sum()), "n_matched": int(len(gm)),
                 "MAE_bps": e.abs().mean(), "RMSE_bps": float(np.sqrt((e ** 2).mean())) if len(e) else np.nan,
                 "bias_bps": e.mean()}
            if name in directions:
                r["direction_accuracy"] = gm[directions[name]].mean()
            if name != base and len(gm):
                diff = gm[errors[base]].abs() - e.abs()
                r[f"mean_abs_err_diff_{base}_minus_this_bps"] = diff.mean()
                r[f"share_{base}_better"] = float((diff < 0).mean())
            r["descriptive_only"] = len(gm) < SMALL_SAMPLE
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================== adjustment after the open
def post_open_adjustment(cfg: Config, instrument: str, scored: Iterable[dt.date],
                         minutes: Sequence[int] = (1, 5, 15), pre_cutoff: str = "09:29",
                         perp_max_age: str = "5min", cash_max_age: str = "60s",
                         interval: str = "1m") -> pd.DataFrame:
    """Cash and perp moves 1/5/15 minutes after the official open, and the basis change.

    cash_ret_h  = mid(open + h) / O - 1
    perp_ret_h  = P(open + h) / P(pre_cutoff) - 1
    basis_h     = P(open + h) / mid(open + h) - 1, vs basis_pre = P(pre_cutoff) / mid(pre_cutoff) - 1
    """
    scored = cfg.trading_days(sorted(set(scored)))
    auc = official_prices(cfg, instrument, scored).set_index("et_date")
    perp, mid = _load(cfg, instrument, scored, interval)
    base = pd.DataFrame({"et_date": scored})
    base["O"] = base["et_date"].map(auc["open_price"])
    base["open_ts"] = base["et_date"].map(auc["open_ts_utc"])
    base["pre_ts"] = [_at(d, pre_cutoff) for d in scored]
    base = base.join(asof_lookup(perp, base["pre_ts"], perp_max_age, "P_pre"))
    base = base.join(asof_lookup(mid, base["pre_ts"], cash_max_age, "mid_pre"))
    base["basis_pre_bps"] = _bps(base["P_pre"], base["mid_pre"])
    for h in minutes:
        t = base["open_ts"] + pd.Timedelta(minutes=h)
        p = asof_lookup(perp, t, perp_max_age, "P")
        m = asof_lookup(mid, t, cash_max_age, "M")
        base[f"cash_ret_{h}m_bps"] = _bps(m["M"], base["O"])
        base[f"perp_ret_{h}m_bps"] = _bps(p["P"], base["P_pre"])
        base[f"basis_{h}m_bps"] = _bps(p["P"], m["M"])
        base[f"basis_change_{h}m_bps"] = base[f"basis_{h}m_bps"] - base["basis_pre_bps"]
    keep = [c for c in base.columns if not c.endswith(("_avail_ts", "_stale", "_age_s"))]
    return base[keep]


# ============================================================== forecast paths
def forecast_paths(cfg: Config, instrument: str, scored: Iterable[dt.date], perp_max_age: str = "5min",
                   cash_max_age: str = "60s", interval: str = "1m") -> pd.DataFrame:
    """Minute-by-minute implied open from the previous close to 09:29, plus the
    premarket cash midpoint, for daily forecast-path plots (results.ipynb)."""
    scored = cfg.trading_days(sorted(set(scored)))
    prev = {d: previous_trading_day(d, cfg.extra_closures) for d in scored}
    days = sorted(set(scored) | set(prev.values()))
    auc = official_prices(cfg, instrument, days).set_index("et_date")
    perp, mid = _load(cfg, instrument, days, interval)
    parts = []
    for d in scored:
        p = prev[d]
        close_t = pd.Timestamp(session_times(p)["close"]).tz_convert("UTC")
        grid = pd.date_range(close_t + pd.Timedelta("1min"), _at(d, "09:29"), freq="1min")
        g = pd.DataFrame({"et_date": d, "ts_utc": grid})
        g["ts_et"] = g["ts_utc"].dt.tz_convert(ET)
        tt = g["ts_et"].dt.time
        g["interval"] = np.where(g["ts_et"].dt.date == p, np.where(tt < dt.time(20), "postmarket", "overnight"),
                                 np.where(tt < dt.time(4), "overnight", "premarket"))
        C, O = auc.at[p, "close_price"], auc.at[d, "open_price"]
        P0 = asof_lookup(perp, pd.Series([close_t]), perp_max_age, "P0")["P0"].iloc[0]
        g = g.join(asof_lookup(perp, g["ts_utc"], perp_max_age, "Pt")[["Pt", "Pt_age_s"]])
        g = g.join(asof_lookup(mid, g["ts_utc"], cash_max_age, "mid")[["mid", "mid_age_s"]])
        g["C"], g["O"], g["P0"] = C, O, P0
        g["implied_open_perp"] = C * g["Pt"] / P0
        g["err_bps_perp"] = _bps(g["implied_open_perp"], O)
        g["err_bps_mid"] = _bps(g["mid"], O)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
