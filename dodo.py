"""Run the whole pipeline with doit:  uv run doit

    uv run doit                 # tests -> collect -> lead-lag -> significance -> overnight -> windows
                                #   -> robustness -> notebook
    uv run doit collect=0       # skip downloading; analyse the data already on disk
    uv run doit list            # show the tasks
    uv run doit lead_lag        # run one stage (plus whatever it depends on)
    uv run doit forget && uv run doit   # force every analysis step to run again

Collecting is always safe to repeat: each collector skips files that already exist and
only connects (WRDS, Databento) when something is missing. Dwellir candles are committed
in data/raw/dwellir/, so without `entitlement_verified: true` the Dwellir API is never
called. Analysis steps re-run only when the code, config.yaml or the raw data changed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from doit import get_var

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = DATA / "results"
CODE = sorted(str(p) for p in (ROOT / "src" / "leadlag").glob("*.py")) + [str(ROOT / "config" / "config.yaml")]

COLLECT = get_var("collect", "1") not in ("0", "false", "no")
COLLECT_STUDIES = ["nvda_crisis", "nvda_comparison", "nvda_baseline", "panel"]
NVDA_STUDIES = "nvda_baseline nvda_comparison nvda_crisis"

DOIT_CONFIG = {
    "default_tasks": ["test", "collect", "lead_lag", "significance", "overnight", "windows", "robustness",
                      "notebook"],
    "verbosity": 2,              # stream output live (and let WRDS ask for a password)
    "dep_file": str(ROOT / ".doit.db"),
    "backend": "json",
}


def _leadlag(args: str) -> str:
    return f"leadlag {args}"


# --------------------------------------------------------------------------- data signature
def _raw_signature() -> str:
    """Hash of (name, size, mtime) for every raw parquet the analysis reads. Hyperliquid
    metadata is excluded: it is snapshotted on every collect run and is not analysed."""
    h = hashlib.sha1()
    if RAW.exists():
        for p in sorted(RAW.rglob("*.parquet")):
            rel = p.relative_to(RAW)
            if rel.parts[0] == "hyperliquid":
                continue
            st = p.stat()
            h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def raw_data_unchanged(task, values):
    """doit `uptodate` check run when the task is reached (after collect has finished),
    so data downloaded earlier in the same run still triggers the analysis."""
    sig = _raw_signature()
    task.value_savers.append(lambda: {"raw_sig": sig})
    return values.get("raw_sig") == sig


def _analysis(name: str, actions, targets, extra_deps=()):
    return {
        "basename": name,
        "actions": actions,
        "file_dep": CODE + list(extra_deps),
        "targets": [str(t) for t in targets],
        "uptodate": [raw_data_unchanged],
        "task_dep": ["collect"] if COLLECT else [],
        "clean": True,
    }


# --------------------------------------------------------------------------- tasks
def task_test():
    """Unit tests (no network)."""
    tests = sorted(str(p) for p in (ROOT / "tests").glob("*.py"))
    return {"actions": ["pytest -q"], "file_dep": CODE + tests}


def task_collect():
    """Download TAQ, Databento ES and Hyperliquid funding for every study (skips existing files)."""
    if not COLLECT:
        return {"actions": [], "doc": "skipped (collect=0)", "uptodate": [True]}
    return {
        "actions": [_leadlag(f"collect --study {s}") for s in COLLECT_STUDIES] + [_leadlag("coverage")],
        "uptodate": [False],     # always check; the collectors themselves skip finished files
        "verbosity": 2,
    }


LEAD_LAG = {
    # name: (args, results dir)
    "TSLA_trade": ("--studies panel --instrument TSLA --holdout", "TSLA_vs_TSLA_trade_1min"),
    "TSLA_mid": ("--studies panel --instrument TSLA --ref-type mid", "TSLA_vs_TSLA_mid_1min"),
    "SP500_trade": ("--studies panel --instrument SP500 --ref ES --holdout", "SP500_vs_ES_trade_1min"),
    "SP500_mid": ("--studies panel --instrument SP500 --ref ES --ref-type mid", "SP500_vs_ES_mid_1min"),
    "NVDA_trade": (f"--studies {NVDA_STUDIES} --instrument NVDA", "NVDA_vs_NVDA_trade_1min"),
    "NVDA_mid": (f"--studies {NVDA_STUDIES} --instrument NVDA --ref-type mid", "NVDA_vs_NVDA_mid_1min"),
}


def task_lead_lag():
    """Stage 4: correlations, asymmetry, regressions (and holdout for the panel)."""
    for name, (args, out) in LEAD_LAG.items():
        t = _analysis("lead_lag", [_leadlag(f"lead-lag {args}")],
                      [RESULTS / out / f for f in ("xcorr.csv", "asymmetry.csv", "summary.csv")])
        t.update(name=name)
        yield t


OVERNIGHT = {"NVDA": NVDA_STUDIES, "TSLA": "panel"}


def task_overnight():
    """Stage 5: official auction prices and perp-implied opening/closing forecasts."""
    for inst, studies in OVERNIGHT.items():
        t = _analysis("overnight", [_leadlag(f"overnight --studies {studies} --instrument {inst} --paths")],
                      [RESULTS / f"overnight_{inst}" / f for f in ("open_metrics.csv", "open_paths.csv")])
        t.update(name=inst)
        yield t


WINDOWS = {
    "TSLA": ("--studies panel --instrument TSLA", "windows_TSLA_vs_TSLA_mid"),
    "SP500": ("--studies panel --instrument SP500 --ref ES", "windows_SP500_vs_ES_mid"),
    "NVDA": (f"--studies {NVDA_STUDIES} --instrument NVDA", "windows_NVDA_vs_NVDA_mid"),
}


def task_windows():
    """Stage 6: opening/closing boundary windows (+/- 30 min)."""
    for name, (args, out) in WINDOWS.items():
        t = _analysis("windows", [_leadlag(f"windows {args}")],
                      [RESULTS / out / f for f in ("asymmetry.csv", "event_profile.csv")])
        t.update(name=name)
        yield t


def task_significance():
    """Formal tests that the reference leads the perp at 1 minute (regular + extended hours)."""
    return _analysis("significance", [_leadlag("significance")],
                     [RESULTS / "significance" / f for f in ("tests.csv", "headline.csv", "per_session.csv",
                                                             "activity.csv")])


def task_robustness():
    """Stage 7: fixed robustness grid for the core lead-lag result."""
    return _analysis("robustness", [_leadlag("robustness")], [RESULTS / "robustness.csv"])


def _execute_notebook(path: str) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, path)
    print(f"Executed {Path(path).name}")


def task_notebook():
    """Re-execute results.ipynb in place against the fresh results."""
    nb = ROOT / "results.ipynb"
    deps = [str(RESULTS / out / "summary.csv") for _, out in LEAD_LAG.values()]
    deps += [str(RESULTS / f"overnight_{i}" / "open_metrics.csv") for i in OVERNIGHT]
    deps += [str(RESULTS / out / "event_profile.csv") for _, out in WINDOWS.values()]
    deps += [str(RESULTS / "robustness.csv"), str(RESULTS / "significance" / "tests.csv")]
    return {
        "actions": [(_execute_notebook, [str(nb)])],
        "file_dep": deps + [str(ROOT / "src" / "leadlag" / "plots.py")],
        "task_dep": ["lead_lag", "significance", "overnight", "windows", "robustness"],
        "verbosity": 2,
    }


if __name__ == "__main__":     # `python dodo.py` works too
    import doit

    os.chdir(ROOT)
    doit.run(globals())
