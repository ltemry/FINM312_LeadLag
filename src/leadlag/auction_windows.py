"""Stage 6 / Module C: opening and closing boundary windows.

Windows (ET, relative to each day's scheduled boundary, so early closes work):
    open : boundary 09:30, from -30 to +30 minutes   (09:00-10:00)
    close: boundary 16:00, from -30 to +30 minutes   (15:30-16:30; 13:00 on early-close days)
For stocks the boundary is the primary-exchange auction; the official auction
price is a separate event, never treated as a continuous-trading quote. For the
S&P perp vs ES these are cash-session boundary comparisons (SPX is not
auctioned), and the 16:00 ET boundary coincides with CME's daily settlement
time, which is not the same event as an equity close.

Outputs:
    frame          1-minute grid over the windows: perp (trade bars) and reference
                   (quote midpoints by default) with ages, spreads, volume, and
                   session-safe returns. Pre/post sides are separate sessions, so
                   no return or lag ever crosses the auction itself.
    window_leadlag core lead-lag tests inside each (window, side) group
    event_profile  event-time profile per minute offset: normalized paths,
                   basis, spreads, quote ages, perp activity, missing shares
    event_changes  per day: pre-event discrepancy (basis at offset 0) and changes
                   at +1/+5/+15 minutes; auction print vs pre-event quote
    adjustment     median share of the 15-minute move completed by +1/+5
                   minutes, per venue (descriptive: which venue adjusts first)
Static discrepancies and convergence are descriptive evidence, not proof of leadership.
"""
from __future__ import annotations

import datetime as dt
from typing import Sequence

import numpy as np
import pandas as pd

from . import normalize as N
from .align import asof_join, session_grid, session_returns
from .auctions import official_prices
from .config import Config
from .lead_lag import run_analysis
from .market_calendar import ET, session_times

HALF_WIDTH_MIN = 30
HORIZONS = (1, 5, 15)
MIN_PAIRED_FOR_TESTS = 300


def window_frame(cfg: Config, instrument: str, et_dates, ref: str | None = None, ref_type: str = "mid",
                 max_age: str = "60s", interval: str = "1m", half_width: int = HALF_WIDTH_MIN) -> pd.DataFrame:
    """Aligned 1-minute grid restricted to the open/close windows of each trading day."""
    trading = cfg.trading_days(et_dates)
    utc_days = cfg.utc_days(et_dates)
    grid = session_grid(trading, ("pre", "regular", "post"), "1min", cfg.extra_closures)

    # window labels relative to each day's scheduled boundaries
    rows = []
    for d in trading:
        st = session_times(d)
        for name, b in (("open", st["open"]), ("close", st["close"])):
            rows.append((d, name, pd.Timestamp(b).tz_convert("UTC")))
    bounds = pd.DataFrame(rows, columns=["et_date", "window", "boundary_utc"])

    coin = cfg.instruments[instrument]["perp"]["coin"]
    perp = N.dwellir_bars(cfg, coin, utc_days, interval)
    ref_name = ref or instrument
    rinst = cfg.instruments[ref_name]
    if "equity" in rinst:
        sym = rinst["equity"]["taq_sym_root"]
        r = N.taq_mid(cfg, sym, trading) if ref_type == "mid" else N.taq_trade_bars(cfg, sym, trading)
    else:
        sym = rinst["futures"]["symbol"]
        r = N.databento_bbo(cfg, sym, utc_days) if ref_type == "mid" else N.databento_bars(cfg, sym, utc_days)
    extra_ref = ["spread_bps"] if ref_type == "mid" else []
    g = grid.sort_values("ts_utc").reset_index(drop=True)
    g = asof_join(g, r, "ref", max_age, exact=(ref_type == "trade"), extra_cols=extra_ref)
    g = asof_join(g, perp, "perp", max_age, exact=True, extra_cols=["volume", "n_trades"])
    # returns on the full-day grid; pre/regular/post are separate sessions, so no
    # return or lag ever crosses the auction itself
    g["ref_ret"] = session_returns(g, "ref_price")
    g["perp_ret"] = session_returns(g, "perp_price")
    g = g.merge(bounds, on="et_date")
    g["offset_min"] = ((g["ts_utc"] - g["boundary_utc"]) / pd.Timedelta("1min")).round().astype(int)
    g = g[g["offset_min"].abs() <= half_width].copy()
    g["side"] = np.where(g["offset_min"] <= 0, "pre", "post")      # offset 0 = last pre-boundary minute
    g["basis_bps"] = 1e4 * (g["perp_price"] / g["ref_price"] - 1)
    g["pair"] = f"{coin} vs {ref_name}"
    g["ref_name"] = ref_name
    return g


def window_leadlag(frame: pd.DataFrame, n_boot: int = 1000, seed: int = 0, ps=range(1, 4)) -> dict[str, pd.DataFrame]:
    """Core lead-lag tests within each (window, side). K = 5 lags, regressions p = 1..3."""
    res = run_analysis(frame, ("window", "side"), tuple(range(-5, 6)), ps, (), n_boot, seed)
    d = res["describe"]
    d["sufficiently_populated"] = (d["paired_returns"] >= MIN_PAIRED_FOR_TESTS) & (d["n_sessions"] >= 10)
    return res


def event_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Per window x minute offset: normalized paths (offset 0 = 0 bps), basis, spreads, ages, activity."""
    f = frame.copy()
    base = f[f["offset_min"] == 0].set_index(["et_date", "window"])[["perp_price", "ref_price"]]
    f = f.join(base.rename(columns=lambda c: c + "_0"), on=["et_date", "window"])
    f["perp_path_bps"] = 1e4 * (f["perp_price"] / f["perp_price_0"] - 1)
    f["ref_path_bps"] = 1e4 * (f["ref_price"] / f["ref_price_0"] - 1)
    f["abs_perp_path"] = f["perp_path_bps"].abs()
    f["abs_ref_path"] = f["ref_path_bps"].abs()
    f["abs_basis_bps"] = f["basis_bps"].abs()
    agg = {"days": ("et_date", "nunique"),
           "abs_perp_path_med_bps": ("abs_perp_path", "median"), "abs_ref_path_med_bps": ("abs_ref_path", "median"),
           "basis_med_bps": ("basis_bps", "median"), "abs_basis_med_bps": ("abs_basis_bps", "median"),
           "perp_missing_share": ("perp_price", lambda s: s.isna().mean()),
           "ref_missing_share": ("ref_price", lambda s: s.isna().mean()),
           "ref_age_med_s": ("ref_age_s", "median"), "perp_volume_mean": ("perp_volume", "mean"),
           "perp_trades_mean": ("perp_n_trades", "mean")}
    if "ref_spread_bps" in f:
        agg["ref_spread_med_bps"] = ("ref_spread_bps", "median")
    return f.groupby(["window", "offset_min"]).agg(**agg).reset_index()


def event_changes(frame: pd.DataFrame, auctions: pd.DataFrame | None = None,
                  horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    """Per day and window: basis at offset 0 and moves to +h minutes; auction vs pre-event quote."""
    f = frame.set_index(["et_date", "window", "offset_min"])
    rows = []
    for (d, w), _ in frame.groupby(["et_date", "window"]):
        def get(col, k):
            try:
                return f.at[(d, w, k), col]
            except KeyError:
                return np.nan
        p0, r0 = get("perp_price", 0), get("ref_price", 0)
        rec = {"et_date": d, "window": w, "basis_pre_bps": 1e4 * (p0 / r0 - 1)}
        if auctions is not None and d in auctions.index:
            a = auctions.loc[d]
            price = a["open_price"] if w == "open" else a["close_price"]
            rec["auction_price"] = price
            rec["auction_vs_pre_ref_bps"] = 1e4 * (price / r0 - 1)       # what the auction added
        for h in horizons:
            ph, rh = get("perp_price", h), get("ref_price", h)
            rec[f"perp_ret_{h}m_bps"] = 1e4 * (ph / p0 - 1)
            rec[f"ref_ret_{h}m_bps"] = 1e4 * (rh / r0 - 1)
            rec[f"basis_change_{h}m_bps"] = 1e4 * (ph / rh - 1) - rec["basis_pre_bps"]
        rows.append(rec)
    return pd.DataFrame(rows)


def adjustment_shares(changes: pd.DataFrame, min_move_bps: float = 5.0, horizons=(1, 5), total: int = 15
                      ) -> pd.DataFrame:
    """Median share of each venue's +15-minute move completed by +1 and +5 minutes.

    Only days where that venue moved at least `min_move_bps` by +15 enter (shares are
    unstable for tiny moves). Descriptive: which venue tends to adjust first.
    """
    rows = []
    for w, g in changes.groupby("window"):
        for venue in ("perp", "ref"):
            tot = g[f"{venue}_ret_{total}m_bps"]
            ok = tot.abs() >= min_move_bps
            rec = {"window": w, "venue": venue, "days_used": int(ok.sum())}
            for h in horizons:
                rec[f"share_done_by_{h}m"] = float((g.loc[ok, f"{venue}_ret_{h}m_bps"] / tot[ok]).median()) \
                    if ok.any() else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def run_windows(cfg: Config, instrument: str, studies: Sequence[str], ref: str | None = None,
                ref_type: str = "mid", max_age: str = "60s", n_boot: int = 1000, seed: int = 0
                ) -> dict[str, pd.DataFrame]:
    days: list[dt.date] = []
    for s in studies:
        days += cfg.study_dates(s, which="scored")
    days = sorted(set(cfg.trading_days(days)))
    frame = window_frame(cfg, instrument, days, ref, ref_type, max_age)
    frame["period"] = "+".join(studies)
    auctions = None
    if "equity" in cfg.instruments[ref or instrument]:
        auctions = official_prices(cfg, ref or instrument, days).set_index("et_date")
    out = window_leadlag(frame, n_boot, seed)
    changes = event_changes(frame, auctions)
    out.update(frame=frame, event_profile=event_profile(frame), event_changes=changes,
               adjustment=adjustment_shares(changes))
    return out
