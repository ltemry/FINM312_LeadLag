"""Significance battery on synthetic data with a known lead (no network, no files)."""
import numpy as np
import pandas as pd
import pytest

from leadlag.lead_lag import boot_p_value
from leadlag.significance import activity_check, correlation_tests, holm, regression_tests


def frame(lead: str = "ref", n_sessions: int = 20, n_min: int = 390, seed: int = 1, stale_share: float = 0.0):
    """ref leads: perp_t = 0.6 ref_t + 0.4 ref_{t-1} + noise; perp leads: the mirror image;
    none: synchronous. stale_share: share of minutes where the perp is a stale copy of
    ref_{t-1} (low trade count) while active minutes are synchronous."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        ts = pd.date_range(f"2026-10-{s + 1:02d} 13:31Z", periods=n_min, freq="1min")
        x = rng.normal(0, 1e-3, n_min + 1)
        e = rng.normal(0, 3e-4, n_min)
        if lead == "ref":
            perp, ref = 0.6 * x[1:] + 0.4 * x[:-1] + e, x[1:]
        elif lead == "perp":
            perp, ref = x[1:], 0.6 * x[1:] + 0.4 * x[:-1] + e
        else:
            perp, ref = x[1:] + e, x[1:]
        trades = np.full(n_min, 100.0)
        if stale_share:
            low = rng.random(n_min) < stale_share
            perp = np.where(low, x[:-1], x[1:] + e)
            trades[low] = 1.0
        rows.append(pd.DataFrame({"ts_utc": ts, "session_id": f"2026-10-{s + 1:02d}_regular", "session": "regular",
                                  "perp_ret": perp, "ref_ret": ref, "perp_n_trades": trades}))
    return pd.concat(rows, ignore_index=True)


def by_test(rows):
    return {r["test"]: r for r in rows}


def test_reference_lead_is_detected_by_every_test():
    df = frame("ref")
    t = by_test(correlation_tests(df, n_boot=1000) + regression_tests(df, []))
    assert t["asymmetry_A"]["estimate"] < 0 and t["asymmetry_A"]["ci_hi"] < 0
    assert t["asymmetry_A"]["p_two_sided"] <= 0.002
    assert t["lag1_difference"]["estimate"] < 0
    assert t["session_sign_test"]["estimate"] == 1.0 and t["session_sign_test"]["p_two_sided"] < 1e-4
    assert t["session_t_test"]["p_one_sided_ref_leads"] < 1e-4
    assert t["granger_ref_to_perp_p1"]["p_two_sided"] < 1e-4
    assert t["granger_perp_to_ref_p1"]["p_two_sided"] > 0.01
    assert t["catchup_beta_l1"]["estimate"] > 0 and t["catchup_beta_l1"]["p_two_sided"] < 1e-4


def test_perp_lead_flips_the_sign():
    t = by_test(correlation_tests(frame("perp"), n_boot=1000))
    assert t["asymmetry_A"]["estimate"] > 0 and t["asymmetry_A"]["ci_lo"] > 0
    assert t["asymmetry_A"]["p_one_sided_ref_leads"] > 0.99
    assert t["session_sign_test"]["estimate"] == 0.0


def test_no_lead_is_not_significant():
    df = frame("none", seed=7)
    t = by_test(correlation_tests(df, n_boot=1000) + regression_tests(df, []))
    assert t["asymmetry_A"]["p_two_sided"] > 0.05
    assert t["granger_ref_to_perp_p1"]["p_two_sided"] > 0.01


def test_activity_filter_removes_a_stale_price_artifact():
    """A lead caused only by stale low-activity perp bars vanishes among active bars."""
    rows = activity_check(frame(stale_share=0.3), quantiles=(0.5,), n_boot=500)
    a = {r["filter"]: r for r in rows if r["test"] == "asymmetry_A"}
    assert a["all minutes"]["estimate"] < 0 and a["all minutes"]["ci_hi"] < 0
    active = a["≥ q50 trades"]
    assert active["min_trades_per_bar"] == 100 and active["ci_lo"] < 0 < active["ci_hi"]


def test_boot_p_value():
    rng = np.random.default_rng(0)
    assert boot_p_value(rng.normal(0, 1, 4000)) > 0.8
    assert boot_p_value(np.full(1000, 2.0)) == pytest.approx(1 / 1000)          # floored at 1/B
    assert boot_p_value(np.r_[np.full(975, 1.0), np.full(25, -1.0)]) == pytest.approx(0.05)


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03, np.nan])
    assert holm(p).round(6).tolist()[:3] == [0.03, 0.06, 0.06] and np.isnan(holm(p).iloc[3])
