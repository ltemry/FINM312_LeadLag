"""Stage 6 tests: window labels, no returns across the auction, event changes, early closes."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from leadlag.auction_windows import adjustment_shares, event_changes, event_profile, window_frame
from leadlag.auctions import official_prices
from test_overnight import DAY, O, cfg, et, perp_price  # noqa: F401  (cfg is a fixture)


def test_window_offsets_sides_and_boundaries(cfg):
    f = window_frame(cfg, "NVDA", [DAY])
    for w, hh in (("open", 9), ("close", 16)):
        g = f[f.window == w]
        assert g.offset_min.min() == -30 and g.offset_min.max() == 30 and len(g) == 61
        b = et(DAY, hh, 30 if w == "open" else 0)
        assert g.loc[g.offset_min == 0, "ts_utc"].iloc[0] == b
        assert set(g.loc[g.offset_min <= 0, "side"]) == {"pre"} and set(g.loc[g.offset_min > 0, "side"]) == {"post"}


def test_no_return_crosses_the_auction(cfg):
    f = window_frame(cfg, "NVDA", [DAY]).set_index(["window", "offset_min"])
    for w in ("open", "close"):
        assert np.isnan(f.at[(w, 1), "perp_ret"]) and np.isnan(f.at[(w, 1), "ref_ret"])
        assert not np.isnan(f.at[(w, 2), "perp_ret"]) and not np.isnan(f.at[(w, 0), "perp_ret"])
    # first window minute still has a return (computed on the full-day grid)
    assert not np.isnan(f.at[("open", -30), "perp_ret"])


def test_event_changes_match_prices_and_use_official_auction(cfg):
    f = window_frame(cfg, "NVDA", [DAY])
    auc = official_prices(cfg, "NVDA", [DAY]).set_index("et_date")
    ch = event_changes(f, auc).set_index("window")
    p0, p5 = perp_price(et(DAY, 9, 30)), perp_price(et(DAY, 9, 35))
    assert ch.at["open", "perp_ret_5m_bps"] == pytest.approx(1e4 * (p5 / p0 - 1))
    r0 = f[(f.window == "open") & (f.offset_min == 0)]["ref_price"].iloc[0]
    assert ch.at["open", "auction_price"] == O
    assert ch.at["open", "auction_vs_pre_ref_bps"] == pytest.approx(1e4 * (O / r0 - 1))


def test_event_profile_paths_zero_at_boundary(cfg):
    prof = event_profile(window_frame(cfg, "NVDA", [DAY])).set_index(["window", "offset_min"])
    assert prof.at[("open", 0), "abs_perp_path_med_bps"] == 0
    assert prof.at[("close", 0), "abs_ref_path_med_bps"] == 0


def test_early_close_boundary(cfg):
    f = window_frame(cfg, "NVDA", [dt.date(2026, 11, 27)])          # 13:00 ET close, no data on disk
    b = f[(f.window == "close") & (f.offset_min == 0)]["ts_utc"].iloc[0]
    assert b == pd.Timestamp("2026-11-27 18:00Z")
    assert f["perp_price"].isna().all()                             # missing data stays missing


def test_adjustment_shares_ignore_tiny_moves():
    ch = pd.DataFrame({"window": "open", "perp_ret_1m_bps": [5.0, 0.1], "perp_ret_5m_bps": [8.0, 0.2],
                       "perp_ret_15m_bps": [10.0, 0.3], "ref_ret_1m_bps": [2.0, 1.0],
                       "ref_ret_5m_bps": [6.0, 1.0], "ref_ret_15m_bps": [10.0, 1.0]})
    a = adjustment_shares(ch).set_index("venue")
    assert a.at["perp", "days_used"] == 1 and a.at["perp", "share_done_by_1m"] == pytest.approx(0.5)
    assert a.at["ref", "share_done_by_5m"] == pytest.approx(0.6)
