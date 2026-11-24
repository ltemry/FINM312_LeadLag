"""Static figures for results.ipynb (matplotlib).

Design rules (dataviz reference palette): fixed categorical order blue, orange,
aqua (never cycled); one y-axis per chart; 2px lines, >= 8px markers; solid
hairline grid; text in ink colors, never series colors; a legend whenever there
are two or more series, plus selective direct labels. Every figure in the
notebook is followed by its table.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1-3 (validated all-pairs)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
LW = 1.5                                              # ~2px at 100 dpi
MS = 6                                                # ~8px markers


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "lines.linewidth": LW,
        "lines.markersize": MS, "lines.solid_capstyle": "round",
    })


def _zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=AXIS, lw=0.9, zorder=1)


def _end_labels(ax, items, min_gap_pt: float = 11.0):
    """Direct labels at each line's last finite point, nudged apart vertically so they never
    overlap. `items` = [(x, y, text)]. Text is ink; the line carries the color."""
    pts = []
    for x, y, text in items:
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        if ok.any():
            i = np.flatnonzero(ok)[-1]
            pts.append([np.asarray(x)[i], y[i], text])
    if not pts:
        return
    fig = ax.figure
    fig.canvas.draw_idle()
    to_disp = ax.transData.transform
    disp = [to_disp((px, py))[1] for px, py, _ in pts]
    order = np.argsort(disp)
    gap = min_gap_pt * fig.dpi / 72
    placed = []
    for j in order:
        d = disp[j]
        if placed and d - placed[-1] < gap:
            d = placed[-1] + gap
        placed.append(d)
        offset_pt = (d - disp[j]) * 72 / fig.dpi
        px, py, text = pts[j]
        ax.annotate(text, (px, py), xytext=(6, offset_pt), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)


def xcorr_panels(panels: dict[str, pd.DataFrame], suptitle: str):
    """Small multiples of rho(k) with bootstrap CIs. k = 0 is omitted from the plot
    (it is ~0.96 and would flatten everything) and printed in each panel title."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, t) in zip(axes, panels.items()):
        t = t.sort_values("lag")
        r0 = t.loc[t.lag == 0, "rho"].iloc[0]
        t = t[t.lag != 0]
        ax.errorbar(t.lag, t.rho, yerr=[t.rho - t.ci_lo, t.ci_hi - t.rho], fmt="o", color=SERIES[0],
                    ecolor=SERIES[0], elinewidth=1.2, capsize=0, ms=MS, zorder=3)
        _zero(ax)
        ax.set_title(f"{title}\nρ(0) = {r0:.3f}", fontsize=10)
        ax.set_xticks(range(-5, 6))
        ax.set_xlabel("k (minutes)   ← reference first · perp first →")
    axes[0].set_ylabel("ρ(k) = Corr(perp$_t$, ref$_{t+k}$)")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig


def forest(df: pd.DataFrame, label: str, est: str, lo: str, hi: str, xlabel: str, title: str,
           panel: str | None = None, ref_line: float = 0.0):
    """Estimate + CI per row (one series), optionally faceted by `panel`."""
    groups = list(df.groupby(panel, sort=False)) if panel else [(None, df)]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 0.45 * max(len(g) for _, g in groups) + 1.4),
                             sharex=True, squeeze=False)
    for ax, (name, g) in zip(axes[0], groups):
        y = np.arange(len(g))[::-1]
        ax.errorbar(g[est], y, xerr=[g[est] - g[lo], g[hi] - g[est]], fmt="o", color=SERIES[0],
                    elinewidth=1.2, capsize=0, ms=MS, zorder=3)
        ax.axvline(ref_line, color=AXIS, lw=0.9, zorder=1)
        ax.set_yticks(y, g[label])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel)
        if name is not None:
            ax.set_title(str(name), fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig


def line_panels(panels: dict[str, pd.DataFrame], x: str, series: dict[str, str], xlabel: str, ylabel: str,
                title: str, markers: bool = True, xticks=None, zero: bool = False, ncols: int | None = None,
                figsize_per=(3.8, 3.0)):
    """Small multiples, up to 3 series each, fixed colors per series name, legend + end labels."""
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             sharey=False, squeeze=False)
    names = list(series)
    for ax, (ptitle, t) in zip(axes.flat, panels.items()):
        labels = []
        for i, (lab, col) in enumerate(series.items()):
            if col not in t:
                continue
            ax.plot(t[x], t[col], color=SERIES[i], marker="o" if markers else None, ms=MS - 1, label=lab, zorder=3)
            labels.append((t[x], t[col], lab))
        if n <= 4:
            ax._pending_labels = labels
        if zero:
            _zero(ax)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.set_title(ptitle, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.margins(x=0.18)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ax in axes.flat[:n]:                      # label after layout so nudges use final scales
        if getattr(ax, "_pending_labels", None):
            _end_labels(ax, ax._pending_labels)
    axes.flat[0].set_ylabel(ylabel)
    handles = [mpl.lines.Line2D([], [], color=SERIES[i], marker="o" if markers else None, lw=LW) for i in range(len(names))]
    fig.legend(handles, names, loc="upper right", ncol=len(names), bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def event_path(perp: pd.Series, cash: pd.Series, marks: dict[str, tuple], title: str, base_label: str):
    """Two series in bps vs a common base, ET clock on x; auction prints as separate marks."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(perp.index, perp.values, color=SERIES[0], lw=1.2, label="Perp (Dwellir 1m trade close)", zorder=3)
    ax.plot(cash.index, cash.values, color=SERIES[1], lw=0, marker="o", ms=2.5,
            label="Cash (TAQ 1m last trade, all sessions)", zorder=4)
    for lab, (t, v) in marks.items():
        ax.plot([t], [v], marker="D", ms=MS + 1, color=INK, zorder=5)
        ax.annotate(lab, (t, v), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK)
    _zero(ax)
    ax.set_ylabel(f"bps vs {base_label}")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b %d\n%H:%M", tz="America/New_York"))
    ax.set_title(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def session_dots(panels: dict[str, pd.DataFrame], y: str, title: str, ylabel: str):
    """One dot per trading day (chronological), small multiples by pair, with the count of
    days on each side of zero in each panel title. One series per panel, so no legend."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 3.2), sharey=True, squeeze=False)
    for ax, (name, t) in zip(axes[0], panels.items()):
        t = t.sort_values("et_date").reset_index(drop=True)
        v = t[y].to_numpy(float)
        ax.plot(np.arange(len(v)), v, "o", color=SERIES[0], ms=MS, zorder=3)
        _zero(ax)
        m = np.nanmean(v)
        ax.axhline(m, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)        # dashed = mean (named in title)
        below = int((v < 0).sum())
        ax.set_title(f"{name}\n{below} of {np.isfinite(v).sum()} days below zero · mean (dashed) {m:+.3f}",
                     fontsize=10)
        ticks = np.unique(np.linspace(0, len(v) - 1, min(4, len(v))).round().astype(int))
        ax.set_xticks(ticks, [pd.Timestamp(t.at[i, "et_date"]).strftime("%b %d") for i in ticks])
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig
