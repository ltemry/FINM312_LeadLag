"""Formal significance tests for the direction of lead-lag at the 1-minute horizon.

Question: does the reference market (cash stock, or ES for the S&P perp) lead the
perpetual by a statistically significant and economically material amount?

Conventions follow lead_lag.py: rho(k) = Corr(perp_t, ref_{t+k}); positive k means the
perp moves first. "Reference leads" therefore means NEGATIVE asymmetry
A = sum_{k=1..5} rho(k) - sum_{k=1..5} rho(-k), and a NEGATIVE lag-1 difference
D1 = rho(+1) - rho(-1). Every test is reported two-sided; the one-sided p-value is for
the alternative "reference leads". The session (trading day) is the unit of
independence throughout: minutes within a day are never treated as independent draws.

Test battery, per pair and session type (regular / pre / post):
 1. asymmetry_A         A, whole-session bootstrap (B draws), percentile p-value.
 2. lag1_difference     D1 = rho(+1) - rho(-1), same bootstrap.
 3. session_sign_test   share of days with A_day < 0; exact binomial sign test.
 4. session_wilcoxon    Wilcoxon signed-rank test on the per-day A_day.
 5. session_t_test      t-test on the mean per-day A_day (G - 1 df).
 6. granger_*           Wald F tests (session-clustered, F(q, G-1)) that lagged ref
                        returns predict the perp (ref -> perp) and the reverse, at
                        p = 1 and p = 5 lags, with lagged ES as market control where used.
 7. catchup_beta_l1     coefficient on ref_{t-1} in the p = 1 perp regression (which also
                        holds perp_{t-1} and ES fixed). Empirically the perp's own-lag
                        coefficient is close to minus this one, so it measures gap closing:
                        the share of last minute's ref-minus-perp return gap that the perp
                        makes up in the next minute. reverse_beta_l1 is the same for the
                        reference. This pair is the "materiality" number (clustered t, G-1 df).
 8. holdout_*           panel only: out-of-sample loss reduction from adding lagged ref
                        returns (fit Sep-Oct, evaluate Nov), Diebold-Mariano-style
                        test on per-session mean loss differences.
Holm adjustment across pairs is applied within each (session type, test) family.

Staleness check (activity.csv): perp candles are last-trade prices, so a thinly traded
perp can look "late" mechanically (a stale close catches up next minute). The asymmetry
test is repeated keeping only perp returns whose two bars both had at least N trades,
for N at the pair's median and 75th percentile of trades per minute.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .align import build_for_study
from .config import Config
from .lead_lag import (MIN_SESSIONS_FOR_INFERENCE, _boot_counts, _rho, _session_stats, _sorted,
                       boot_p_value, holdout_forecast, regressions_both_ways, shift_in_session)
from .robustness import PAIRS

LAGS = tuple(range(-5, 6))


# --------------------------------------------------------------------------- data
def pair_frame(cfg: Config, pair: dict, sessions: Sequence[str] = ("regular",), ref_type: str = "trade",
               max_age: str = "60s") -> pd.DataFrame:
    parts = [build_for_study(cfg, s, pair["instrument"], pair["ref"], ref_type, sessions=tuple(sessions),
                             max_age=max_age, control=pair["control"]) for s in pair["studies"]]
    df = pd.concat(parts, ignore_index=True)
    # NVDA studies do not overlap, but never count a session twice
    return df.drop_duplicates(["session_id", "ts_utc"]).reset_index(drop=True)


# --------------------------------------------------------------------------- correlations
def _lag_stats(df: pd.DataFrame, lags=LAGS, freq: str = "1min"):
    d = _sorted(df)
    sessions = pd.Index(d["session_id"].unique())
    st = np.stack([_session_stats(d["perp_ret"], shift_in_session(d, "ref_ret", k, freq), d["session_id"], sessions)
                   for k in lags])                                    # (L, S, 6)
    keep = st[:, :, 0].min(axis=0) >= 10                              # >= 10 pairs at every lag
    return st[:, keep, :], sessions[keep]


def _stats_from(rho: np.ndarray, lags=LAGS) -> tuple[np.ndarray, np.ndarray]:
    """A and D1 from rho along the last axis."""
    lags = list(lags)
    pos = [lags.index(k) for k in lags if k > 0]
    neg = [lags.index(-k) for k in lags if k > 0]
    A = np.nansum(rho[..., pos], axis=-1) - np.nansum(rho[..., neg], axis=-1)
    D1 = rho[..., lags.index(1)] - rho[..., lags.index(-1)]
    return A, D1


def per_session(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """One row per trading session: rho(-1), rho(0), rho(+1), A_day, D1_day, activity."""
    st, sessions = _lag_stats(df, freq=freq)
    if not len(sessions):
        return pd.DataFrame()
    rho = _rho(np.moveaxis(st, 0, 1))                                 # (S, L)
    A, D1 = _stats_from(rho)
    L = list(LAGS)
    out = pd.DataFrame({"session_id": sessions, "rho_m1": rho[:, L.index(-1)], "rho_0": rho[:, L.index(0)],
                        "rho_p1": rho[:, L.index(1)], "A_day": A, "D1_day": D1,
                        "n_pairs": st[L.index(0), :, 0].astype(int)})
    if "perp_n_trades" in df:
        act = df.groupby("session_id")["perp_n_trades"].median()
        out["perp_trades_per_min_median"] = out["session_id"].map(act)
    out["et_date"] = out["session_id"].str.split("_").str[0]
    out["session"] = out["session_id"].str.split("_").str[1]
    return out


def _row(test, stat, est, lo=np.nan, hi=np.nan, p=np.nan, p_one=np.nan, n=np.nan, **kw):
    return {"test": test, "statistic": stat, "estimate": est, "ci_lo": lo, "ci_hi": hi,
            "p_two_sided": p, "p_one_sided_ref_leads": p_one, "n_sessions": n, **kw}


def correlation_tests(df: pd.DataFrame, n_boot: int = 10000, seed: int = 0, freq: str = "1min") -> list[dict]:
    st, sessions = _lag_stats(df, freq=freq)
    S = len(sessions)
    if S < 2:
        return []
    rho = _rho(st.sum(axis=1))                                        # pooled (L,)
    A, D1 = _stats_from(rho)
    w = _boot_counts(S, n_boot, seed)
    boot = _rho(np.einsum("bs,lsk->blk", w, st))                      # (B, L)
    bA, bD1 = _stats_from(boot)
    rows = []
    for name, stat, est, b in (("asymmetry_A", "A = Σρ(k>0) − Σρ(k<0), k ≤ 5", A, bA),
                               ("lag1_difference", "D1 = ρ(+1) − ρ(−1)", D1, bD1)):
        rows.append(_row(name, stat, float(est), *np.nanquantile(b, [0.025, 0.975]), p=boot_p_value(b),
                         p_one=float(max((b >= 0).mean(), 1 / len(b))), n=S, method=f"session bootstrap, B={n_boot}"))

    ps = per_session(df, freq)
    a = ps["A_day"].dropna().to_numpy()
    nz = a[a != 0]
    k = int((nz < 0).sum())
    bt = stats.binomtest(k, len(nz), 0.5)
    rows.append(_row("session_sign_test", "share of days with A_day < 0", k / len(nz), p=bt.pvalue,
                     p_one=stats.binomtest(k, len(nz), 0.5, alternative="greater").pvalue, n=len(nz),
                     method=f"exact binomial: {k} of {len(nz)} days ref-led"))
    if len(nz) >= 6:
        wt = stats.wilcoxon(nz)
        rows.append(_row("session_wilcoxon", "median A_day", float(np.median(nz)), p=float(wt.pvalue),
                         p_one=float(stats.wilcoxon(nz, alternative="less").pvalue), n=len(nz),
                         method="Wilcoxon signed-rank on per-day A"))
    se = a.std(ddof=1) / np.sqrt(len(a))
    t = a.mean() / se
    q = stats.t.ppf(0.975, len(a) - 1)
    rows.append(_row("session_t_test", "mean A_day", float(a.mean()), a.mean() - q * se, a.mean() + q * se,
                     p=float(2 * stats.t.sf(abs(t), len(a) - 1)), p_one=float(stats.t.cdf(t, len(a) - 1)),
                     n=len(a), method=f"t = {t:.2f}, {len(a) - 1} df"))
    return rows


# --------------------------------------------------------------------------- regressions
def regression_tests(df: pd.DataFrame, controls: Sequence[str], freq: str = "1min") -> list[dict]:
    reg = regressions_both_ways(df, [1, 5], controls, freq=freq)
    rows = []
    for _, r in reg.iterrows():
        if r.get("status") != "ok":
            continue
        tag = "granger_ref_to_perp" if r["direction"] == "ref -> perp" else "granger_perp_to_ref"
        q = int(r["p"])
        F = r["wald_chi2"] / q if np.isfinite(r["wald_chi2"]) else np.nan
        rows.append(_row(f"{tag}_p{q}", f"Wald F({q}, {int(r['n_sessions']) - 1}), ΔR²", float(r["delta_r2"]),
                         p=r["p_F"], n=int(r["n_sessions"]), F=F, n_obs=int(r["n"]), controls=r["controls"],
                         method="session-clustered CR1; ΔR² = incremental R² of the other venue's lags"))
    # catch-up coefficient: perp_t on ref_{t-1} (p = 1, with own lag and controls)
    r = reg[(reg["direction"] == "ref -> perp") & (reg["p"] == 1)]
    if len(r) and r.iloc[0].get("status") == "ok":
        r = r.iloc[0]
        b, se, G = r["b_ref_ret_l1"], r["se_ref_ret_l1"], int(r["n_sessions"])
        t = b / se
        q = stats.t.ppf(0.975, G - 1)
        rows.append(_row("catchup_beta_l1", "share of last minute's gap the perp closes", float(b), b - q * se,
                         b + q * se, p=float(2 * stats.t.sf(abs(t), G - 1)), p_one=float(stats.t.sf(t, G - 1)),
                         n=G, method=f"clustered t = {t:.2f}; coef on ref_(t-1) in perp_t (p=1, own lag and controls held fixed)"))
        rb = reg[(reg["direction"] == "perp -> ref") & (reg["p"] == 1)].iloc[0]
        b2, se2 = rb["b_perp_ret_l1"], rb["se_perp_ret_l1"]
        t2 = b2 / se2
        rows.append(_row("reverse_beta_l1", "share of last minute's gap the reference closes", float(b2), b2 - q * se2,
                         b2 + q * se2, p=float(2 * stats.t.sf(abs(t2), G - 1)), n=G,
                         method=f"clustered t = {t2:.2f}; coef on perp_(t-1) in ref_t (p=1, own lag and controls held fixed)"))
    return rows


def holdout_tests(df: pd.DataFrame, controls: Sequence[str], freq: str = "1min", n_boot: int = 10000,
                  seed: int = 0) -> list[dict]:
    if "block" not in df or df["block"].isna().all():
        return []
    rows = []
    for p in (1, 5):
        for direction, target, other in (("ref_to_perp", "perp_ret", "ref_ret"), ("perp_to_ref", "ref_ret", "perp_ret")):
            h = holdout_forecast(df, df["block"] == "development", df["block"] == "evaluation", target, other, p,
                                 controls, freq, n_boot, seed)
            if h.get("status") != "ok":
                continue
            p_one = (stats.t.sf(h["dm_t"], h["eval_sessions"] - 1) if direction == "ref_to_perp"
                     else np.nan)
            rows.append(_row(f"holdout_{direction}_p{p}", "mean loss reduction (bps²/min), out of sample",
                             h["mean_loss_diff_bps2"], h.get("loss_diff_ci_lo_bps2"), h.get("loss_diff_ci_hi_bps2"),
                             p=h.get("p_dm"), p_one=p_one, n=h["eval_sessions"], oos_r2=h["oos_r2_vs_base"],
                             p_boot=h.get("p_boot"),
                             method=f"fit {h['dev_sessions']} dev sessions, evaluate {h['eval_sessions']}; DM t = {h['dm_t']:.2f}"))
    return rows


# --------------------------------------------------------------------------- staleness check
def activity_check(df: pd.DataFrame, quantiles=(0.5, 0.75), n_boot: int = 10000, seed: int = 0,
                   freq: str = "1min") -> list[dict]:
    """Asymmetry when the perp return's two bars both had >= N trades (fresher closes)."""
    if "perp_n_trades" not in df or df["perp_n_trades"].notna().sum() == 0:
        return []
    d = _sorted(df)
    prev = shift_in_session(d, "perp_n_trades", -1, freq)
    active_min = np.fmin(d["perp_n_trades"], prev)
    rows = []
    base = d["perp_ret"].notna() & d["ref_ret"].notna()
    for qname, N in [("all minutes", 0)] + [(f"≥ q{int(q * 100)} trades", float(d["perp_n_trades"].quantile(q)))
                                            for q in quantiles]:
        sub = d.copy()
        if N > 0:
            sub.loc[~(active_min >= N), "perp_ret"] = np.nan
        kept = int((sub["perp_ret"].notna() & sub["ref_ret"].notna()).sum())
        t = [r for r in correlation_tests(sub, n_boot, seed, freq) if r["test"] in ("asymmetry_A", "lag1_difference",
                                                                                     "session_sign_test")]
        for r in t:
            rows.append({"filter": qname, "min_trades_per_bar": N, "paired_returns_kept": kept,
                         "share_kept": kept / max(int(base.sum()), 1), **r})
    return rows


# --------------------------------------------------------------------------- driver
def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (NaN-safe)."""
    v = p.dropna().sort_values()
    m = len(v)
    adj = np.maximum.accumulate([(m - i) * x for i, x in enumerate(v.to_numpy())]) if m else []
    out = pd.Series(np.nan, index=p.index)
    out.loc[v.index] = np.minimum(adj, 1.0)
    return out


def run_significance(cfg: Config, pairs: Sequence[str] | None = None,
                     session_sets: Sequence[Sequence[str]] = (("regular",), ("pre", "post")),
                     n_boot: int = 10000, seed: int = 0) -> dict[str, pd.DataFrame]:
    tests, per, act = [], [], []
    for pname in pairs or PAIRS:
        pair = PAIRS[pname]
        for sess in session_sets:
            df = pair_frame(cfg, pair, sess)
            for s, sub in df.groupby("session"):
                sub = sub.reset_index(drop=True)
                if (sub["perp_ret"].notna() & sub["ref_ret"].notna()).sum() < 100:
                    continue
                controls = ["es_ret"] if "es_ret" in sub and sub["es_ret"].notna().any() else []
                label = {"pair": pname, "session": s}
                rows = correlation_tests(sub, n_boot, seed) + regression_tests(sub, controls)
                if s == "regular":
                    rows += holdout_tests(sub, controls, n_boot=n_boot, seed=seed)
                    act += [{**label, **r} for r in activity_check(sub, n_boot=n_boot, seed=seed)]
                tests += [{**label, **r} for r in rows]
                ps = per_session(sub)
                if len(ps):
                    per.append(ps.assign(pair=pname))
    t = pd.DataFrame(tests)
    t["descriptive_only"] = t["n_sessions"] < MIN_SESSIONS_FOR_INFERENCE
    t["p_holm"] = t.groupby(["session", "test"])["p_two_sided"].transform(holm)
    t["significant_5pct_holm"] = t["p_holm"] < 0.05
    return {"tests": t, "per_session": pd.concat(per, ignore_index=True) if per else pd.DataFrame(),
            "activity": pd.DataFrame(act)}


HEADLINE = [("asymmetry_A", "A (bootstrap)"), ("lag1_difference", "ρ(+1) − ρ(−1)"),
            ("session_sign_test", "days ref-led"), ("granger_ref_to_perp_p1", "ref → perp, F p=1"),
            ("granger_perp_to_ref_p1", "perp → ref, F p=1"), ("catchup_beta_l1", "gap closed by perp"), ("reverse_beta_l1", "gap closed by ref"),
            ("holdout_ref_to_perp_p1", "OOS gain, ref → perp")]


def headline(tests: pd.DataFrame, session: str = "regular") -> pd.DataFrame:
    """One row per pair: estimate and p-value of the main tests (regular session)."""
    t = tests[tests["session"] == session]
    rows = []
    for pair, g in t.groupby("pair", sort=False):
        g = g.set_index("test")
        row = {"pair": pair, "sessions": int(g["n_sessions"].max())}
        for test, lab in HEADLINE:
            if test in g.index:
                row[lab] = g.at[test, "estimate"]
                row[f"p: {lab}"] = g.at[test, "p_two_sided"]
        rows.append(row)
    return pd.DataFrame(rows)

