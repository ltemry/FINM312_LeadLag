"""Session, calendar and as-of alignment tests (synthetic data, no network)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from leadlag.align import align_pair, asof_join, session_grid, session_returns
from leadlag.config import Config
from leadlag.market_calendar import nyse_early_closes, nyse_holidays, previous_trading_day

D = dt.date


def series(times_utc, prices, price_type="mid", source_ts=None):
    df = pd.DataFrame({"ts_utc": pd.to_datetime(times_utc, utc=True), "price": prices})
    df["source_ts_utc"] = pd.to_datetime(source_ts, utc=True) if source_ts is not None else df["ts_utc"]
    df["price_type"], df["source"] = price_type, "test"
    return df


# ---------------------------------------------------------------- calendar / DST
def test_september_and_november_utc_opens_differ():
    g = session_grid([D(2026, 9, 22), D(2026, 11, 10)])
    first = g.groupby("et_date")["ts_utc"].min()
    assert first[D(2026, 9, 22)] == pd.Timestamp("2026-09-22 13:31Z")   # EDT, UTC-4
    assert first[D(2026, 11, 10)] == pd.Timestamp("2026-11-10 14:31Z")  # EST, UTC-5
    assert (g.groupby("et_date").size() == 390).all()


def test_early_close_and_holidays():
    assert D(2026, 11, 26) in nyse_holidays(2026)            # Thanksgiving
    assert D(2026, 7, 3) in nyse_holidays(2026)              # Jul 4 is a Saturday
    assert nyse_early_closes(2026) == {D(2026, 11, 27), D(2026, 12, 24)}
    g = session_grid([D(2026, 11, 26), D(2026, 11, 27)])
    assert set(g.et_date) == {D(2026, 11, 27)}
    assert g.ts_utc.max() == pd.Timestamp("2026-11-27 18:00Z")         # 13:00 ET
    assert previous_trading_day(D(2026, 9, 21)) == D(2026, 9, 18)


def test_utc_days_cover_post_market():
    cfg = Config()
    # ET Sep 24 runs to 00:00 ET Sep 25 = 04:00 UTC Sep 25
    assert cfg.utc_days([D(2026, 9, 24)]) == [D(2026, 9, 24), D(2026, 9, 25)]


def test_panel_has_about_30_sessions_and_ant_is_deferred():
    cfg = Config()
    scored = cfg.trading_days(cfg.study_dates("panel", which="scored"))
    assert len(scored) == 30
    with pytest.raises(ValueError):
        cfg.build_tasks("panel", instruments=["ANT"])
    tasks, notes = cfg.build_tasks("panel")
    assert not any(t.instrument == "ANT" for t in tasks)
    assert any("TSLA futures symbol not resolved" in n for n in notes)


# ---------------------------------------------------------------- as-of joins
def test_no_future_quote_enters_grid():
    g = session_grid([D(2026, 11, 10)])
    t10 = pd.Timestamp("2026-11-10 15:00Z")                  # 10:00 ET
    s = series([t10 - pd.Timedelta("10s"), t10 + pd.Timedelta("30s")], [100.0, 999.0])
    out = asof_join(g, s, "x").set_index("ts_utc")
    assert out.loc[t10, "x_price"] == 100.0
    assert out.loc[t10, "x_age_s"] == 10
    assert out.loc[t10 + pd.Timedelta("1min"), "x_price"] == 999.0
    assert (out.loc[:t10, "x_price"] != 999.0).all()


def test_quote_available_exactly_at_grid_time_is_allowed():
    g = session_grid([D(2026, 11, 10)])
    t = pd.Timestamp("2026-11-10 15:00Z")
    out = asof_join(g, series([t], [5.0]), "x").set_index("ts_utc")
    assert out.loc[t, "x_price"] == 5.0


def test_stale_quotes_become_missing_with_age_kept():
    g = session_grid([D(2026, 11, 10)])
    t = pd.Timestamp("2026-11-10 15:00Z")
    out = asof_join(g, series([t], [5.0]), "x", max_age="60s").set_index("ts_utc")
    assert out.loc[t + pd.Timedelta("1min"), "x_price"] == 5.0          # age 60s: ok
    row = out.loc[t + pd.Timedelta("2min")]
    assert np.isnan(row["x_price"]) and row["x_stale"] and row["x_age_s"] == 120


def test_age_uses_source_timestamp():
    g = session_grid([D(2026, 11, 10)])
    t = pd.Timestamp("2026-11-10 15:00Z")
    s = series([t], [5.0], source_ts=[t - pd.Timedelta("45s")])
    out = asof_join(g, s, "x", max_age="30s").set_index("ts_utc")
    assert out.loc[t, "x_age_s"] == 45 and np.isnan(out.loc[t, "x_price"])


def test_trade_bars_require_exact_bar_end():
    g = session_grid([D(2026, 11, 10)])
    t = pd.Timestamp("2026-11-10 15:00Z")
    s = series([t], [5.0], price_type="trade")
    out = asof_join(g, s, "x", exact=True).set_index("ts_utc")
    assert out.loc[t, "x_price"] == 5.0
    assert np.isnan(out.loc[t + pd.Timedelta("1min"), "x_price"])     # missing bar, not filled


# ---------------------------------------------------------------- returns
def test_returns_never_cross_sessions_or_gaps():
    g = session_grid([D(2026, 11, 9), D(2026, 11, 10)], sessions=("pre", "regular"))
    g["p"] = 100.0 + np.arange(len(g)) * 0.01
    gap = g.index[(g.ts_utc == pd.Timestamp("2026-11-10 16:00Z"))][0]
    g.loc[gap, "p"] = np.nan
    r = session_returns(g, "p")
    first_of_session = g.groupby("session_id")["ts_utc"].idxmin()
    assert r.loc[first_of_session].isna().all()          # no lag across session boundaries
    assert np.isnan(r.loc[gap]) and np.isnan(r.loc[gap + 1])
    assert r.drop(index=list(first_of_session) + [gap, gap + 1]).notna().all()


def test_align_pair_with_synthetic_lag_keeps_order():
    """Perp leads by one minute: ref return at t equals perp return at t-1."""
    g = session_grid([D(2026, 11, 10)])
    rng = np.random.default_rng(0)
    perp_r = rng.normal(0, 1e-3, len(g))
    perp_p = 100 * np.exp(np.cumsum(perp_r))
    ref_p = np.r_[100.0, perp_p[:-1]]
    perp = series(g.ts_utc, perp_p, "trade")
    ref = series(g.ts_utc - pd.Timedelta("1s"), ref_p, "mid")
    a = align_pair(g, ref, perp)
    corr_lead = a["perp_ret"].corr(a["ref_ret"].shift(-1))   # corr(perp_t, ref_{t+1})
    assert corr_lead > 0.99
    assert a["perp_ret"].corr(a["ref_ret"].shift(1)) < 0.2
