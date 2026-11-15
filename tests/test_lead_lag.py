"""Synthetic known-lag tests for lead_lag.py (sign convention, no cross-session lags, inference)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from leadlag.align import session_grid
from leadlag.lead_lag import (_sorted, cross_correlation, holdout_forecast, mask_windows,
                              predictive_regression, regressions_both_ways, run_analysis,
                              shift_in_session, summary_table)

TRADING = [d for d in pd.bdate_range("2026-09-21", "2026-10-30").date]   # no holidays in range


def synthetic(lead: int = 2, beta: float = 0.8, n_days: int = 20, seed: int = 1, noise: float = 0.6,
              control_beta: float = 0.0) -> pd.DataFrame:
    """perp leads ref by `lead` minutes (lead < 0: ref leads perp), within sessions only."""
    g = session_grid(TRADING[:n_days]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    g["es_ret"] = rng.normal(0, 1e-3, len(g))
    driver = rng.normal(0, 1e-3, len(g))
    g["perp_ret"] = driver
    g["ref_ret"] = rng.normal(0, noise * 1e-3, len(g))
    d = _sorted(g)
    src, dst = ("perp_ret", "ref_ret") if lead >= 0 else ("ref_ret", "perp_ret")
    if lead < 0:
        d["ref_ret"] = driver
        d["perp_ret"] = rng.normal(0, noise * 1e-3, len(d))
    past = shift_in_session(d, src, -abs(lead)).fillna(0.0)
    d[dst] = d[dst] + beta * past
    if control_beta:
        d["ref_ret"] += control_beta * shift_in_session(d, "es_ret", -1).fillna(0.0)
    d["period"] = "synthetic"
    return d


def test_shift_never_crosses_sessions():
    d = _sorted(synthetic(n_days=2))
    lagged = shift_in_session(d, "perp_ret", -1)
    first = d.groupby("session_id").head(1).index
    assert lagged.loc[first].isna().all()
    lead = shift_in_session(d, "perp_ret", 1)
    last = d.groupby("session_id").tail(1).index
    assert lead.loc[last].isna().all()


def test_shift_does_not_bridge_dropped_rows():
    d = _sorted(synthetic(n_days=1)).drop(index=[50, 51])
    lagged = shift_in_session(d, "perp_ret", -1)
    assert np.isnan(lagged.loc[52])                      # row 49 is 3 minutes away, not 1


def test_sign_convention_perp_leads_gives_positive_k():
    t, a = cross_correlation(synthetic(lead=2), n_boot=200)
    rho = t.set_index("lag")["rho"]
    assert rho.idxmax() == 2 and rho[2] > 0.6
    assert abs(rho[-2]) < 0.05
    assert a["A"] > 0.5 and a["ci_lo"] > 0
    assert a["exploratory_peak_lag"] == 2


def test_sign_convention_ref_leads_gives_negative_k():
    t, a = cross_correlation(synthetic(lead=-1), n_boot=200)
    rho = t.set_index("lag")["rho"]
    assert rho.idxmax() == -1 and a["A"] < 0 and a["ci_hi"] < 0


def test_no_relation_gives_ci_covering_zero():
    t, a = cross_correlation(synthetic(beta=0.0), n_boot=500)
    assert a["ci_lo"] < 0 < a["ci_hi"]
    assert (t["ci_lo"] < 0).all() and (t["ci_hi"] > 0).all()


def test_sufficient_stat_rho_matches_pandas():
    d = _sorted(synthetic(lead=1, n_days=3))
    t, _ = cross_correlation(d, lags=(-1, 0, 1), n_boot=0)
    for k in (-1, 0, 1):
        expected = d["perp_ret"].corr(shift_in_session(d, "ref_ret", k))
        assert t.set_index("lag").loc[k, "rho"] == pytest.approx(expected, rel=1e-9)


def test_bootstrap_reproducible_and_small_sample_flagged():
    a1 = cross_correlation(synthetic(n_days=4), n_boot=100, seed=7)[1]
    a2 = cross_correlation(synthetic(n_days=4), n_boot=100, seed=7)[1]
    assert a1["ci_lo"] == a2["ci_lo"]
    assert a1["descriptive_only"] is True
    assert cross_correlation(synthetic(n_days=12), n_boot=0)[1]["descriptive_only"] is False


def test_regression_detects_direction_and_coefficient():
    r = regressions_both_ways(synthetic(lead=2, beta=0.5), ps=[3])
    fwd = r[r.direction == "perp -> ref"].iloc[0]
    rev = r[r.direction == "ref -> perp"].iloc[0]
    assert fwd["p_F"] < 1e-6 and fwd["b_perp_ret_l2"] == pytest.approx(0.5, abs=0.05)
    assert rev["p_F"] > 0.01
    assert fwd["delta_r2"] > 0.1 and rev["delta_r2"] >= 0


def test_restricted_and_unrestricted_use_same_rows():
    row = predictive_regression(synthetic(), "ref_ret", "perp_ret", 2)
    assert row["delta_r2"] >= 0 and row["n"] > 0 and row["status"] == "ok"


def test_controls_enter_regression():
    d = synthetic(lead=2, beta=0.5, control_beta=0.7)
    row = predictive_regression(d, "ref_ret", "perp_ret", 2, controls=["es_ret"])
    assert row["b_es_ret_l1"] == pytest.approx(0.7, abs=0.07)
    assert row["controls"] == "es_ret"


def test_holdout_positive_when_perp_leads_and_not_otherwise():
    d = synthetic(lead=1, beta=0.5, n_days=20)
    dates = sorted(d.et_date.unique())
    dev, ev = d.et_date.isin(dates[:14]), d.et_date.isin(dates[14:])
    h = holdout_forecast(d, dev, ev, "ref_ret", "perp_ret", 2, n_boot=200)
    assert h["oos_r2_vs_base"] > 0.2 and h["loss_diff_ci_lo_bps2"] > 0
    n = synthetic(beta=0.0, n_days=20)
    h0 = holdout_forecast(n, dev, ev, "ref_ret", "perp_ret", 2, n_boot=200)
    assert h0["oos_r2_vs_base"] < 0.01


def test_mask_windows_blanks_returns_but_keeps_rows():
    d = synthetic(n_days=2)
    m = mask_windows(d, ["perp_ret", "ref_ret"], exclude_open=True)
    assert len(m) == len(d)
    assert m.loc[m.open_window, "perp_ret"].isna().all()
    assert m.loc[~m.open_window, "perp_ret"].notna().all()


def test_run_analysis_groups_and_summary():
    a = synthetic(lead=1, n_days=12).assign(period="baseline")
    b = synthetic(lead=1, n_days=12, seed=5).assign(period="crisis")
    b["session_id"] = "c_" + b["session_id"]
    res = run_analysis(pd.concat([a, b], ignore_index=True), ps=[1, 2], n_boot=50)
    assert set(res["asymmetry"]["period"]) == {"baseline", "crisis", "pooled"}
    s = summary_table(res, p_focus=2)
    assert (s["asym_A"] > 0).all() and len(s) == 3


def test_wald_undefined_with_too_few_sessions():
    """3 sessions cannot support a 5-restriction clustered test: no spurious 0/1 p-values."""
    row = predictive_regression(synthetic(lead=1, n_days=3), "ref_ret", "perp_ret", 5)
    assert np.isnan(row["p_F"]) and np.isnan(row["p_chi2"]) and "undefined" in row["wald_note"]
    ok = predictive_regression(synthetic(lead=1, n_days=12), "ref_ret", "perp_ret", 5)
    assert ok["p_F"] < 1e-6 and ok["wald_note"] == ""
