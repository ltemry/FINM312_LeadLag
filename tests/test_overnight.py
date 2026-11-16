"""Stage 5 tests on synthetic files: formulas, no leakage, missing/stale handling, DST, fallbacks."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from leadlag.auctions import official_prices
from leadlag.config import Config
from leadlag.market_calendar import ET
from leadlag.overnight import (asof_lookup, closing_forecasts, forecast_metrics, opening_forecasts,
                               post_open_adjustment)

PREV, DAY = dt.date(2026, 11, 6), dt.date(2026, 11, 9)       # Fri -> Mon (EST)
C, O, CLOSE_MON = 50.0, 56.0, 57.0
BASE = pd.Timestamp("2026-11-06 00:00Z")


def et(d, hh, mm, ss=0):
    return pd.Timestamp(dt.datetime.combine(d, dt.time(hh, mm, ss), ET)).tz_convert("UTC")


def perp_price(bar_end: pd.Timestamp) -> float:
    """Price encodes the bar end, so the test can tell exactly which bar was used."""
    return 100.0 + (bar_end - BASE) / pd.Timedelta("1min") * 0.001


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = Config()
    monkeypatch.setattr(c, "data_dir", tmp_path)
    root = tmp_path / "raw"
    # auctions: Fri close, Mon open (09:30:05) and close
    for d, recs in {PREV: [("Q", "@6 X", C, et(PREV, 16, 0, 0)), ("Q", "@  M", C, et(PREV, 16, 0, 0))],
                    DAY: [("Q", "@O X", O, et(DAY, 9, 30, 5)), ("Q", "@  Q", O, et(DAY, 9, 30, 5)),
                          ("P", "@  Q", 1.0, et(DAY, 9, 30, 0)),            # other exchange: ignored
                          ("Q", "@6 X", CLOSE_MON, et(DAY, 16, 0, 0)), ("Q", "@  M", CLOSE_MON, et(DAY, 16, 0, 0))]
                    }.items():
        p = root / "taq/auction_trades/NVDA"
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs, columns=["ex", "tr_scond", "price", "trade_ts_utc"]).assign(
            tr_corr="00", size=100).to_parquet(p / f"{d}.parquet")
    # perp bars every minute, Fri 00:00 UTC .. Tue 00:00 UTC
    ends = pd.date_range("2026-11-06 00:01Z", "2026-11-10 00:00Z", freq="1min")
    bars = pd.DataFrame({"bar_end_utc": ends, "close": [perp_price(t) for t in ends],
                         "volume": 1.0, "notional": 1.0, "n_trades": 1, "finalized": True})
    p = root / "dwellir/candles_1m/xyz_NVDA"
    p.mkdir(parents=True, exist_ok=True)
    for day, g in bars.groupby((bars.bar_end_utc - pd.Timedelta("1ns")).dt.date):
        g.to_parquet(p / f"{day}.parquet")
    # cash quotes every second 04:00-20:00 ET; mid = 60 + seconds/1e5
    for d in (PREV, DAY):
        t = pd.date_range(et(d, 4, 0, 1), et(d, 20, 0), freq="1s")
        mid = 60 + (t - BASE).total_seconds() / 1e5
        p = root / "taq/nbbo_1s/NVDA"
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"bucket_end_utc": t, "quote_ts_utc": t - pd.Timedelta("100ms"), "bid": mid - 0.01,
                      "ask": mid + 0.01, "mid": mid, "bid_size": 1, "ask_size": 1}).to_parquet(p / f"{d}.parquet")
    return c


def test_official_prices_use_primary_exchange_and_own_timestamp(cfg):
    t = official_prices(cfg, "NVDA", [PREV, DAY]).set_index("et_date")
    assert t.at[DAY, "open_price"] == O and t.at[DAY, "open_source"] == "official_open"
    assert t.at[DAY, "open_ts_utc"] == et(DAY, 9, 30, 5)
    assert t.at[PREV, "close_price"] == C


def test_fallback_to_cross_print_is_flagged(cfg):
    f = cfg.data_dir / "raw/taq/auction_trades/NVDA" / f"{DAY}.parquet"
    a = pd.read_parquet(f)
    a[a.tr_scond != "@  Q"].to_parquet(f)
    t = official_prices(cfg, "NVDA", [DAY]).set_index("et_date")
    assert t.at[DAY, "open_price"] == O and t.at[DAY, "open_source"] == "open_print_fallback"


def test_opening_forecast_formula_and_no_future_bars(cfg):
    df, info = opening_forecasts(cfg, "NVDA", [DAY])
    assert info["dropped_for_leakage"] == 0
    r = df.set_index("cutoff")
    p0 = perp_price(et(PREV, 16, 0))                        # bar ending exactly at the scheduled close
    for c in ("08:00", "09:00", "09:25", "09:29"):
        h, m = map(int, c.split(":"))
        pt = perp_price(et(DAY, h, m))                      # bar ending exactly at the cutoff, never later
        assert r.at[c, "Pt"] == pytest.approx(pt)
        assert r.at[c, "P0"] == pytest.approx(p0)
        implied = C * pt / p0
        assert r.at[c, "implied_open_perp"] == pytest.approx(implied)
        assert r.at[c, "err_bps_perp"] == pytest.approx(1e4 * (implied / O - 1))
        assert r.at[c, "err_bps_nochange"] == pytest.approx(1e4 * (C / O - 1))
        assert (r.at[c, "Pt_avail_ts"] <= r.at[c, "cutoff_ts"]) and (r.at[c, "cutoff_ts"] < r.at[c, "open_ts"])
    assert r.at["09:29", "mid_t_avail_ts"] <= et(DAY, 9, 29)


def test_cutoffs_respect_dst():
    from leadlag.overnight import _at
    assert _at(dt.date(2026, 9, 22), "09:29") == pd.Timestamp("2026-09-22 13:29Z")
    assert _at(dt.date(2026, 11, 10), "09:29") == pd.Timestamp("2026-11-10 14:29Z")


def test_target_cannot_enter_forecast(cfg):
    """A cutoff at/after the actual opening cross is dropped, not scored."""
    df, info = opening_forecasts(cfg, "NVDA", [DAY], cutoffs=("09:29", "09:31"))
    assert list(df["cutoff"]) == ["09:29"] and info["dropped_for_leakage"] == 1


def test_stale_perp_gives_missing_forecast_not_fabricated(cfg):
    f = cfg.data_dir / "raw/dwellir/candles_1m/xyz_NVDA" / f"{DAY}.parquet"
    b = pd.read_parquet(f)
    gap = (b.bar_end_utc > et(DAY, 8, 50)) & (b.bar_end_utc <= et(DAY, 9, 26))
    b[~gap].to_parquet(f)
    df, _ = opening_forecasts(cfg, "NVDA", [DAY], perp_max_age="5min")
    r = df.set_index("cutoff")
    assert np.isnan(r.at["09:00", "err_bps_perp"]) and np.isnan(r.at["09:25", "err_bps_perp"])
    assert r.at["09:00", "Pt_age_s"] == 600                  # age kept for reporting
    assert not np.isnan(r.at["09:29", "err_bps_perp"])
    m = forecast_metrics(df, {"perp": "err_bps_perp", "no_change": "err_bps_nochange"}).set_index(["cutoff", "forecast"])
    assert m.at[("09:25", "perp"), "n_missing_or_stale"] == 1 and m.at[("09:25", "perp"), "n_matched"] == 0


def test_closing_forecast_formula_and_anchor(cfg):
    df, info = closing_forecasts(cfg, "NVDA", [DAY])
    r = df.set_index("cutoff")
    a_perp = perp_price(et(DAY, 15, 0))
    for c in ("15:30", "15:50", "15:59"):
        h, m = map(int, c.split(":"))
        pt = perp_price(et(DAY, h, m))
        implied = r.at[c, "A_cash"] * pt / a_perp
        assert r.at[c, "implied_close_perp"] == pytest.approx(implied)
        assert r.at[c, "err_bps_perp"] == pytest.approx(1e4 * (implied / CLOSE_MON - 1))
        assert r.at[c, "mid_t_avail_ts"] <= r.at[c, "cutoff_ts"] < r.at[c, "close_ts"]


def test_paired_metrics_use_identical_dates(cfg):
    df, _ = opening_forecasts(cfg, "NVDA", [DAY])
    df.loc[df.cutoff == "09:00", "err_bps_premkt_mid"] = np.nan          # benchmark missing on that row
    m = forecast_metrics(df, {"perp": "err_bps_perp", "premarket_mid": "err_bps_premkt_mid"}).set_index(
        ["cutoff", "forecast"])
    assert m.at[("09:00", "perp"), "n_matched"] == 0                     # perp not scored alone there
    assert m.at[("09:00", "perp"), "n_available"] == 1


def test_post_open_adjustment_uses_actual_open_time(cfg):
    adj = post_open_adjustment(cfg, "NVDA", [DAY], minutes=(1,)).iloc[0]
    # open at 09:30:05 -> +1 min = 09:31:05 -> last perp bar ends 09:31:00
    assert adj["perp_ret_1m_bps"] == pytest.approx(1e4 * (perp_price(et(DAY, 9, 31)) / perp_price(et(DAY, 9, 29)) - 1))


def test_asof_lookup_never_uses_future_rows():
    s = pd.DataFrame({"ts_utc": pd.to_datetime(["2026-11-09 14:00Z", "2026-11-09 14:02Z"]), "price": [1.0, 2.0]})
    out = asof_lookup(s, pd.Series(pd.to_datetime(["2026-11-09 14:01Z", "2026-11-09 13:59Z"])), "5min", "x")
    assert out["x"].iloc[0] == 1.0 and np.isnan(out["x"].iloc[1])
