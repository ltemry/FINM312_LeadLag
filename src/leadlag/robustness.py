"""Robustness grid for the core lead-lag result (docs/project_plan.md, "ES and robustness").

For each pair, re-run the predefined statistics under fixed alternative choices:
    trade bars vs quote midpoints; quote-age limits 10/30/60 s; excluding the first
    and last 15 regular minutes; a 5-minute sampling grid.
One row per (pair, variant): asymmetry A with its session-bootstrap CI, and the
p = 3 regressions in both directions (session-clustered F p-value, incremental R2).
The variants are fixed in advance; none is chosen by its result.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .align import build_for_study
from .config import Config
from .lead_lag import cross_correlation, mask_windows, regressions_both_ways

PAIRS = {
    "TSLA perp vs TSLA": dict(instrument="TSLA", ref=None, studies=["panel"], control="ES"),
    "S&P perp vs ES": dict(instrument="SP500", ref="ES", studies=["panel"], control=None),
    "NVDA perp vs NVDA (pooled)": dict(instrument="NVDA", ref=None, control="ES",
                                       studies=["nvda_baseline", "nvda_comparison", "nvda_crisis"]),
}
VARIANTS = {
    "trade bars (main)": dict(ref_type="trade"),
    "midpoints, max age 60s": dict(ref_type="mid", max_age="60s"),
    "midpoints, max age 30s": dict(ref_type="mid", max_age="30s"),
    "midpoints, max age 10s": dict(ref_type="mid", max_age="10s"),
    "trade bars, excl. open/close 15 min": dict(ref_type="trade", exclude=True),
    "trade bars, 5-minute grid": dict(ref_type="trade", freq="5min"),
}


def _frame(cfg: Config, pair: dict, v: dict) -> pd.DataFrame:
    parts = [build_for_study(cfg, s, pair["instrument"], pair["ref"], v.get("ref_type", "trade"),
                             freq=v.get("freq", "1min"), max_age=v.get("max_age", "60s"),
                             control=pair["control"]) for s in pair["studies"]]
    df = pd.concat(parts, ignore_index=True)
    if v.get("exclude"):
        cols = ["perp_ret", "ref_ret"] + (["es_ret"] if "es_ret" in df else [])
        df = mask_windows(df, cols, exclude_open=True, exclude_close=True)
    return df


def run_grid(cfg: Config, pairs: Sequence[str] | None = None, variants: Sequence[str] | None = None,
             n_boot: int = 500, seed: int = 0, p: int = 3) -> pd.DataFrame:
    rows = []
    for pname in pairs or PAIRS:
        pair = PAIRS[pname]
        for vname in variants or VARIANTS:
            v = VARIANTS[vname]
            freq = v.get("freq", "1min")
            df = _frame(cfg, pair, v)
            controls = ["es_ret"] if "es_ret" in df and df["es_ret"].notna().any() else []
            _, a = cross_correlation(df, n_boot=n_boot, seed=seed, freq=freq)
            reg = regressions_both_ways(df, [p], controls, freq=freq).set_index("direction")
            rows.append({
                "pair": pname, "variant": vname, "sessions": a["n_sessions"],
                "paired_returns": int((df["perp_ret"].notna() & df["ref_ret"].notna()).sum()),
                "asym_A": a["A"], "A_ci_lo": a.get("ci_lo"), "A_ci_hi": a.get("ci_hi"),
                "p_ref_to_perp": reg.at["ref -> perp", "p_F"], "dR2_ref_to_perp": reg.at["ref -> perp", "delta_r2"],
                "p_perp_to_ref": reg.at["perp -> ref", "p_F"], "dR2_perp_to_ref": reg.at["perp -> ref", "delta_r2"],
                "control": ",".join(controls) or "none", "freq": freq,
            })
    return pd.DataFrame(rows)
