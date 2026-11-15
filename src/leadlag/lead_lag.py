"""Stage 4 / Module A: lead-lag correlations and predictive regressions.

Input: an aligned table from align.align_pair (one row per grid point) with
    ts_utc, session_id, perp_ret, ref_ret   [+ optional control returns, e.g. es_ret]

Conventions (docs/project_plan.md):
* rho(k) = Corr(perp_ret[t], ref_ret[t+k]) for predefined k in -5..+5 grid steps.
  POSITIVE k means the perpetual comes first.
* Lags never cross sessions or gaps: a value k steps away is used only if it
  is in the same session_id AND exactly k * freq away in time.
* Uncertainty: whole-session bootstrap (resample sessions with replacement)
  for correlations; session-clustered standard errors (plus a session
  bootstrap) for regressions. With few sessions results are flagged
  `descriptive_only` and should be reported descriptively.
* No lag is selected as "best". The headline single statistic is the
  predefined lead asymmetry  A = sum_{k=1..K} rho(k) - sum_{k=1..K} rho(-k).
  A correlation peak is exploratory and is labelled as such.
* Predictive regressions in both directions with 1..p lags of each venue:
      ref_t  = a + sum b_j ref_{t-j}  + sum c_j perp_{t-j} [+ controls] + e   ("perp -> ref")
      perp_t = a + sum b_j perp_{t-j} + sum c_j ref_{t-j}  [+ controls] + e   ("ref -> perp")
  The test is the joint Wald test c_1..c_p = 0 (incremental prediction beyond own lags).
* Forecast claims use a chronological holdout: fit on development dates,
  compare held-out errors with the own-return baseline.
Predictive evidence is not causality and says nothing about sub-grid timing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_LAGS = tuple(range(-5, 6))
MIN_SESSIONS_FOR_INFERENCE = 10


# ============================================================== helpers
def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["session_id", "ts_utc"], kind="mergesort")


def shift_in_session(df: pd.DataFrame, col: str, k: int, freq: str = "1min") -> pd.Series:
    """Value of `col` at t + k grid steps (k < 0: the past), same session only.

    NaN when t + k falls outside the session or the row k positions away is not
    exactly k * freq away in time (so dropped rows/gaps never get bridged).
    `df` must be sorted by session_id, ts_utc.
    """
    if k == 0:
        return df[col]
    g = df.groupby("session_id", sort=False)
    val = g[col].shift(-k)
    ts = g["ts_utc"].shift(-k)
    ok = (ts - df["ts_utc"]) == k * pd.Timedelta(freq)
    return val.where(ok)


def _n_sessions(df: pd.DataFrame, mask: pd.Series | None = None) -> int:
    return int(df.loc[mask, "session_id"].nunique()) if mask is not None else int(df["session_id"].nunique())


# ============================================================== correlations
_STAT_COLS = ("n", "sx", "sy", "sxx", "syy", "sxy")


def _session_stats(x: pd.Series, y: pd.Series, sess: pd.Series, sessions: pd.Index) -> np.ndarray:
    """Per-session sufficient statistics over rows where both x and y exist -> (S, 6)."""
    ok = x.notna() & y.notna()
    d = pd.DataFrame({"s": sess[ok].values, "x": x[ok].values, "y": y[ok].values})
    d["n"] = 1.0
    d["sxx"], d["syy"], d["sxy"] = d.x * d.x, d.y * d.y, d.x * d.y
    agg = d.groupby("s")[["n", "x", "y", "sxx", "syy", "sxy"]].sum().reindex(sessions, fill_value=0.0)
    return agg.to_numpy(dtype=float)


def _rho(st: np.ndarray) -> np.ndarray:
    """Correlation from summed sufficient statistics (last axis = 6 stats)."""
    n, sx, sy, sxx, syy, sxy = np.moveaxis(st, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        return cov / np.sqrt(vx * vy)


def _boot_counts(n_sessions: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_sessions, size=(n_boot, n_sessions))
    return np.stack([np.bincount(r, minlength=n_sessions) for r in draws]).astype(float)


def cross_correlation(df: pd.DataFrame, x: str = "perp_ret", y: str = "ref_ret",
                      lags: Sequence[int] = DEFAULT_LAGS, n_boot: int = 1000, seed: int = 0,
                      ci: float = 0.95, freq: str = "1min") -> tuple[pd.DataFrame, dict]:
    """rho(k) = Corr(x_t, y_{t+k}) with whole-session bootstrap CIs.

    Returns (table, asymmetry) where table has lag, rho, n_pairs, ci_lo, ci_hi,
    and asymmetry = {A, ci_lo, ci_hi, ...} for A = sum_{k>0} rho(k) - sum_{k>0} rho(-k).
    """
    d = _sorted(df)
    sessions = pd.Index(d["session_id"].unique())
    st = np.stack([_session_stats(d[x], shift_in_session(d, y, k, freq), d["session_id"], sessions)
                   for k in lags])                                    # (L, S, 6)
    has_data = st[:, :, 0].sum(axis=0) > 0                            # sessions with any pairs
    st = st[:, has_data, :]
    S = int(has_data.sum())
    rho = _rho(st.sum(axis=1))
    out = pd.DataFrame({"lag": list(lags), "rho": rho, "n_pairs": st[:, :, 0].sum(axis=1).astype(int)})
    out["n_sessions"] = S
    out["descriptive_only"] = S < MIN_SESSIONS_FOR_INFERENCE

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    pos = [i for i, k in enumerate(lags) if k > 0 and -k in lags]
    neg = [list(lags).index(-lags[i]) for i in pos]
    asym = {"A": float(np.nansum(rho[pos]) - np.nansum(rho[neg])) if pos else np.nan,
            "K": len(pos), "n_sessions": S, "descriptive_only": S < MIN_SESSIONS_FOR_INFERENCE}
    if n_boot and S >= 2:
        w = _boot_counts(S, n_boot, seed)                             # (B, S)
        boot = _rho(np.einsum("bs,lsk->blk", w, st))                  # (B, L)
        out["ci_lo"] = np.nanquantile(boot, lo_q, axis=0)
        out["ci_hi"] = np.nanquantile(boot, hi_q, axis=0)
        if pos:
            a_boot = np.nansum(boot[:, pos], axis=1) - np.nansum(boot[:, neg], axis=1)
            asym.update(ci_lo=float(np.nanquantile(a_boot, lo_q)), ci_hi=float(np.nanquantile(a_boot, hi_q)),
                        share_boot_positive=float((a_boot > 0).mean()))
    else:
        out["ci_lo"] = out["ci_hi"] = np.nan
    peak = out.loc[out["rho"].abs().idxmax()] if out["rho"].notna().any() else None
    asym["exploratory_peak_lag"] = int(peak["lag"]) if peak is not None else None
    asym["exploratory_peak_rho"] = float(peak["rho"]) if peak is not None else np.nan
    return out, asym


# ============================================================== regressions
@dataclass
class OLSResult:
    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    n: int
    n_clusters: int
    r2: float
    resid: np.ndarray


def ols_cluster(y: np.ndarray, X: np.ndarray, clusters: np.ndarray, names: list[str]) -> OLSResult:
    """OLS with CR1 session-clustered covariance. X must include the intercept column."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    codes, uniq = pd.factorize(clusters)
    G = len(uniq)
    scores = np.zeros((G, k))
    np.add.at(scores, codes, X * u[:, None])
    meat = scores.T @ scores
    adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    cov = adj * XtX_inv @ meat @ XtX_inv
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (u @ u) / tss if tss > 0 else np.nan
    return OLSResult(names, beta, cov, n, G, float(r2), u)


def _design(df: pd.DataFrame, target: str, regressors: dict[str, int], freq: str
            ) -> tuple[pd.Series, pd.DataFrame]:
    """target_t and lagged regressors {col: p} -> columns col_l1..col_lp (same session only)."""
    X = {}
    for col, p in regressors.items():
        for j in range(1, p + 1):
            X[f"{col}_l{j}"] = shift_in_session(df, col, -j, freq)
    return df[target], pd.DataFrame(X, index=df.index)


def _wald(res: OLSResult, idx: list[int]) -> dict:
    b = res.beta[idx]
    V = res.cov[np.ix_(idx, idx)]
    q = len(idx)
    try:
        W = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        W = float(b @ np.linalg.pinv(V) @ b)
    # The clustered covariance has rank <= G - 1. With G - 1 < q restrictions the
    # joint Wald test is undefined, so report no p-value rather than a spurious 0 or 1.
    if res.n_clusters - 1 < q:
        out = {"wald_chi2": np.nan, "p_chi2": np.nan, "p_F": np.nan,
               "wald_note": f"undefined: {res.n_clusters} sessions for {q} restrictions"}
    else:
        out = {"wald_chi2": W, "p_chi2": float(stats.chi2.sf(W, q)),
               "p_F": float(stats.f.sf(W / q, q, res.n_clusters - 1)),   # small-cluster F(q, G-1)
               "wald_note": ""}
    a = np.ones(q)
    out["sum_coef"] = float(a @ b)
    out["sum_coef_se"] = float(np.sqrt(a @ V @ a))
    return out


def predictive_regression(df: pd.DataFrame, target: str, other: str, p: int,
                          controls: Sequence[str] = (), control_p: int | None = None,
                          freq: str = "1min", n_boot: int = 0, seed: int = 0) -> dict:
    """target_t on its own p lags, p lags of `other`, and lags of controls.

    Restricted (no `other`) and unrestricted models use the SAME rows.
    """
    d = _sorted(df)
    regs = {target: p, other: p, **{c: control_p or p for c in controls}}
    y, X = _design(d, target, regs, freq)
    ok = y.notna() & X.notna().all(axis=1)
    yv = y[ok].to_numpy(float)
    Xu = X[ok]
    cl = d.loc[ok, "session_id"].to_numpy()
    row = {"target": target, "other": other, "p": p, "controls": ",".join(controls) or "none",
           "n": int(ok.sum()), "n_sessions": int(pd.unique(cl).size)}
    row["descriptive_only"] = row["n_sessions"] < MIN_SESSIONS_FOR_INFERENCE
    if row["n"] <= Xu.shape[1] + 2:
        return {**row, "status": "too few observations"}
    names = ["const", *Xu.columns]
    Xmat = np.column_stack([np.ones(len(yv)), Xu.to_numpy(float)])
    res = ols_cluster(yv, Xmat, cl, names)
    other_idx = [i for i, nm in enumerate(names) if nm.startswith(f"{other}_l")]
    keep = [i for i in range(len(names)) if i not in other_idx]
    res_r = ols_cluster(yv, Xmat[:, keep], cl, [names[i] for i in keep])
    row.update(_wald(res, other_idx))
    row.update(r2_unrestricted=res.r2, r2_restricted=res_r.r2, delta_r2=res.r2 - res_r.r2, status="ok")
    for nm, b, se in zip(names, res.beta, np.sqrt(np.diag(res.cov))):
        row[f"b_{nm}"], row[f"se_{nm}"] = float(b), float(se)
    if n_boot and row["n_sessions"] >= 2:
        sess_codes, uniq = pd.factorize(cl)
        rows_by = [np.flatnonzero(sess_codes == g) for g in range(len(uniq))]
        rng = np.random.default_rng(seed)
        sums = []
        for _ in range(n_boot):
            pick = np.concatenate([rows_by[g] for g in rng.integers(0, len(uniq), len(uniq))])
            bb = np.linalg.lstsq(Xmat[pick], yv[pick], rcond=None)[0]
            sums.append(bb[other_idx].sum())
        row["sum_coef_boot_lo"], row["sum_coef_boot_hi"] = (float(v) for v in np.quantile(sums, [0.025, 0.975]))
    return row


def regressions_both_ways(df: pd.DataFrame, ps: Iterable[int] = range(1, 6), controls: Sequence[str] = (),
                          perp: str = "perp_ret", ref: str = "ref_ret", freq: str = "1min",
                          n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    rows = []
    for p in ps:
        for direction, target, other in (("perp -> ref", ref, perp), ("ref -> perp", perp, ref)):
            r = predictive_regression(df, target, other, p, controls, freq=freq, n_boot=n_boot, seed=seed)
            rows.append({"direction": direction, **r})
    return pd.DataFrame(rows)


# ============================================================== holdout
def holdout_forecast(df: pd.DataFrame, dev_mask: pd.Series, eval_mask: pd.Series, target: str, other: str,
                     p: int, controls: Sequence[str] = (), freq: str = "1min", n_boot: int = 1000,
                     seed: int = 0) -> dict:
    """Fit on development rows, forecast evaluation rows; compare with the own-lag baseline.

    oos_r2 = 1 - MSE(with other venue) / MSE(own lags only), on identical eval rows.
    Paired loss differential d_t = e_base^2 - e_full^2, session-bootstrap CI of mean(d).
    """
    d = _sorted(df)
    dev_mask, eval_mask = dev_mask.reindex(d.index).fillna(False), eval_mask.reindex(d.index).fillna(False)
    regs = {target: p, other: p, **{c: p for c in controls}}
    y, X = _design(d, target, regs, freq)
    ok = y.notna() & X.notna().all(axis=1)
    other_cols = [c for c in X.columns if c.startswith(f"{other}_l")]
    base_cols = [c for c in X.columns if c not in other_cols]

    def fit(cols, m):
        A = np.column_stack([np.ones(m.sum()), X.loc[m, cols].to_numpy(float)])
        return np.linalg.lstsq(A, y[m].to_numpy(float), rcond=None)[0]

    def pred(beta, cols, m):
        return np.column_stack([np.ones(m.sum()), X.loc[m, cols].to_numpy(float)]) @ beta

    dm, em = ok & dev_mask, ok & eval_mask
    out = {"target": target, "other": other, "p": p, "controls": ",".join(controls) or "none",
           "n_dev": int(dm.sum()), "n_eval": int(em.sum()),
           "dev_sessions": int(d.loc[dm, "session_id"].nunique()),
           "eval_sessions": int(d.loc[em, "session_id"].nunique())}
    if out["n_dev"] <= len(X.columns) + 2 or out["n_eval"] == 0:
        return {**out, "status": "too few observations"}
    yv = y[em].to_numpy(float)
    e_full = yv - pred(fit(list(X.columns), dm), list(X.columns), em)
    e_base = yv - pred(fit(base_cols, dm), base_cols, em)
    mse_full, mse_base, mse_zero = (e_full ** 2).mean(), (e_base ** 2).mean(), (yv ** 2).mean()
    out.update(mse_full=mse_full, mse_base=mse_base, mse_zero=mse_zero,
               oos_r2_vs_base=1 - mse_full / mse_base, oos_r2_vs_zero=1 - mse_full / mse_zero,
               sign_hit_full=float((np.sign(yv) == np.sign(yv - e_full))[yv != 0].mean()),
               status="ok")
    diff = (e_base ** 2 - e_full ** 2) * 1e8          # squared basis points
    sess = d.loc[em, "session_id"].to_numpy()
    codes, uniq = pd.factorize(sess)
    sums = np.bincount(codes, weights=diff, minlength=len(uniq))
    cnts = np.bincount(codes, minlength=len(uniq)).astype(float)
    out["mean_loss_diff_bps2"] = float(diff.mean())
    if n_boot and len(uniq) >= 2:
        w = _boot_counts(len(uniq), n_boot, seed)
        boot = (w @ sums) / (w @ cnts)
        out["loss_diff_ci_lo_bps2"], out["loss_diff_ci_hi_bps2"] = (
            float(v) for v in np.quantile(boot, [0.025, 0.975]))
    out["descriptive_only"] = out["eval_sessions"] < MIN_SESSIONS_FOR_INFERENCE
    return out


# ============================================================== descriptives
def describe(df: pd.DataFrame, perp: str = "perp_ret", ref: str = "ref_ret", freq: str = "1min") -> dict:
    """Coverage, volatility, first-order autocorrelation (bid-ask bounce check), staleness."""
    d = _sorted(df)
    paired = d[perp].notna() & d[ref].notna()
    out = {"n_sessions": int(d.loc[paired, "session_id"].nunique()), "grid_points": len(d),
           "paired_returns": int((d[perp].notna() & d[ref].notna()).sum())}
    out["paired_share"] = out["paired_returns"] / max(len(d), 1)
    for name, col in (("perp", perp), ("ref", ref)):
        r = d[col]
        out[f"{name}_vol_bps"] = float(r.std() * 1e4)
        out[f"{name}_ac1"] = float(r.corr(shift_in_session(d, col, -1, freq)))
        for extra in (f"{name}_stale", f"{name}_age_s"):
            if extra in d:
                out[f"{extra}_{'share' if extra.endswith('stale') else 'median'}"] = (
                    float(d[extra].mean()) if extra.endswith("stale") else float(d[extra].median()))
    out["contemporaneous_corr"] = float(d[perp].corr(d[ref]))
    return out


# ============================================================== orchestration
def mask_windows(df: pd.DataFrame, cols: Sequence[str], exclude_open: bool = False,
                 exclude_close: bool = False) -> pd.DataFrame:
    """Blank returns inside opening/closing windows (rows kept, so lags never bridge them)."""
    out = df.copy()
    m = pd.Series(False, index=out.index)
    if exclude_open and "open_window" in out:
        m |= out["open_window"].astype(bool)
    if exclude_close and "close_window" in out:
        m |= out["close_window"].astype(bool)
    out.loc[m, list(cols)] = np.nan
    return out


def run_analysis(df: pd.DataFrame, group_cols: Sequence[str] = ("period", "session"),
                 lags: Sequence[int] = DEFAULT_LAGS, ps: Iterable[int] = range(1, 6),
                 controls: Sequence[str] = (), n_boot: int = 1000, seed: int = 0,
                 freq: str = "1min", pooled_label: str = "pooled") -> dict[str, pd.DataFrame]:
    """Run descriptives, correlations and regressions for each group and for the pool.

    Groups are formed by `group_cols` (e.g. period x session). A "pooled" period
    combines all periods within each session.
    """
    ps = list(ps)
    groups: list[tuple[dict, pd.DataFrame]] = []
    gcols = [c for c in group_cols if c in df]
    if gcols:
        for key, sub in df.groupby(gcols, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            groups.append((dict(zip(gcols, key)), sub))
        if "period" in gcols and df["period"].nunique() > 1:
            others = [c for c in gcols if c != "period"]
            pooled = df.groupby(others) if others else [((), df)]
            for key, sub in pooled:
                key = key if isinstance(key, tuple) else (key,)
                groups.append(({"period": pooled_label, **dict(zip(others, key))}, sub))
    else:
        groups.append(({}, df))

    desc, xc, asym, reg = [], [], [], []
    for label, sub in groups:
        desc.append({**label, **describe(sub, freq=freq)})
        t, a = cross_correlation(sub, lags=lags, n_boot=n_boot, seed=seed, freq=freq)
        xc.append(t.assign(**label))
        asym.append({**label, **a})
        use_controls = [c for c in controls if c in sub and sub[c].notna().any()]
        r = regressions_both_ways(sub, ps, use_controls, freq=freq, n_boot=min(n_boot, 500), seed=seed)
        reg.append(r.assign(**label))
    order = list(groups[0][0].keys())
    return {
        "describe": pd.DataFrame(desc),
        "xcorr": pd.concat(xc, ignore_index=True)[order + [c for c in xc[0].columns if c not in order]],
        "asymmetry": pd.DataFrame(asym),
        "regressions": pd.concat(reg, ignore_index=True),
    }


def summary_table(results: dict[str, pd.DataFrame], p_focus: int = 5) -> pd.DataFrame:
    """One line per group: direction/horizon/uncertainty, for the report.

    Uses the predefined asymmetry statistic and the p = p_focus Wald tests;
    the correlation peak is shown but labelled exploratory.
    """
    a = results["asymmetry"]
    r = results["regressions"]
    keys = [c for c in ("period", "session") if c in a]
    rows = []
    for _, row in a.iterrows():
        sel = np.logical_and.reduce([r[k] == row[k] for k in keys]) if keys else np.ones(len(r), bool)
        rr = r[sel & (r["p"] == p_focus)]
        get = lambda d, col: (rr.loc[rr.direction == d, col].iloc[0]
                              if col in rr and (rr.direction == d).any() else np.nan)
        rows.append({**{k: row[k] for k in keys}, "sessions": row["n_sessions"],
                     "asym_A": row["A"], "asym_ci": f"[{row.get('ci_lo', np.nan):.3f}, {row.get('ci_hi', np.nan):.3f}]",
                     "peak_lag (exploratory)": row["exploratory_peak_lag"],
                     f"p(perp->ref) p={p_focus}": get("perp -> ref", "p_F"),
                     f"p(ref->perp) p={p_focus}": get("ref -> perp", "p_F"),
                     "dR2 perp->ref": get("perp -> ref", "delta_r2"),
                     "dR2 ref->perp": get("ref -> perp", "delta_r2"),
                     "descriptive_only": row["descriptive_only"]})
    return pd.DataFrame(rows)
