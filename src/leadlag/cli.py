"""Command-line entry point.

Stage 1: inspect before downloading
    leadlag plan            --study panel             # tasks + max Dwellir credits
    leadlag coverage-report --study panel [--check-wrds] [--check-databento]
    leadlag dwellir-probe   --instrument TSLA --date 2026-11-13   # <= 5 candles, tests access
    leadlag dwellir-usage                              # today's ledger usage vs. allowance
    leadlag databento-resolve --symbols ESZ6           # free symbology lookup
    leadlag hl-contracts    --match TSLA NVDA 500

Stage 2: collect (skips finished files; every attempt goes to data/manifest.csv)
    leadlag collect --study pilot
    leadlag collect --study panel --sources taq dwellir
    leadlag collect --study nvda_crisis --sources dwellir --interval 1s --dates 2026-11-12
    leadlag collect --study panel --sources databento --dry-run    # quotes only

Stage 3: align
    leadlag align --study pilot --instrument TSLA      # perp vs TAQ, 1-minute grid
    leadlag align --study panel --instrument SP500 --ref ES

Stage 4: lead-lag
    leadlag lead-lag --studies nvda_baseline nvda_comparison nvda_crisis --instrument NVDA
    leadlag lead-lag --studies panel --instrument TSLA --holdout
    leadlag lead-lag --studies panel --instrument SP500 --ref ES --control none

Stage 5: auctions and overnight forecasts
    leadlag auctions --study nvda_crisis --instrument NVDA
    leadlag overnight --studies nvda_baseline nvda_comparison nvda_crisis --instrument NVDA --paths
    leadlag overnight --studies panel --instrument TSLA

Stage 6: opening/closing windows
    leadlag windows --studies panel --instrument TSLA
    leadlag windows --studies panel --instrument SP500 --ref ES

Optional forward sample (not running by default):
    leadlag hl-record --instruments TSLA NVDA SP500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict

from .config import REPO_ROOT, SOURCES, Config, to_date
from .storage import Manifest

log = logging.getLogger("leadlag")


def _load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--study", help="study name from config.yaml")
    p.add_argument("--sources", nargs="+", choices=SOURCES, help="override the study's sources")
    p.add_argument("--instruments", nargs="+", help="override the study's instruments")
    p.add_argument("--start", help="YYYY-MM-DD, ET (overrides the study's retrieval dates)")
    p.add_argument("--end", help="YYYY-MM-DD, ET, inclusive")
    p.add_argument("--dates", nargs="+", help="explicit ET dates")
    p.add_argument("--interval", choices=["1s", "1m", "5m"], help="Dwellir candle interval")


def _tasks(cfg: Config, a):
    tasks, notes = cfg.build_tasks(a.study, a.sources, a.instruments, a.start, a.end, a.dates)
    for n in notes:
        log.warning(n)
    return tasks


# ---------------------------------------------------------------- stage 1
def cmd_plan(cfg: Config, a) -> None:
    from .collect_dwellir import max_credits

    tasks = _tasks(cfg, a)
    by = defaultdict(list)
    for t in tasks:
        by[(t.source, t.instrument)].append(t.date)
    print(f"{len(tasks)} source-instrument-days (TAQ: ET trading days; others: UTC days)")
    for (src, inst), ds in by.items():
        print(f"  {src:<12} {inst:<6} {len(ds):>3} days  {ds[0]} .. {ds[-1]}")
    dw = [t for t in tasks if t.source == "dwellir"]
    if dw:
        interval = a.interval or cfg.source_cfg("dwellir").get("interval", "1m")
        c = cfg.source_cfg("dwellir")
        usable = int(c.get("daily_allowance", 100000) * (1 - c.get("headroom_fraction", 0.2)))
        n = max_credits(dw, interval)
        print(f"\nDwellir {interval}: at most {n:,} candle credits "
              f"(~{-(-n // usable)} day(s) at the {usable:,}/day guarded allowance; sparse minutes lower this)")
    if a.study:
        s = cfg.study(a.study)
        scored = cfg.trading_days(cfg.study_dates(a.study, which="scored")) if s.get("scored") else []
        if scored:
            print(f"Scored cash sessions: {len(scored)} ({scored[0]} .. {scored[-1]})")


def cmd_coverage_report(cfg: Config, a) -> None:
    from .coverage_report import build_report, summarize, write_report

    tasks = _tasks(cfg, a)
    df = build_report(cfg, tasks, check_wrds=a.check_wrds, check_databento=a.check_databento,
                      dwellir_interval=a.interval)
    print(summarize(df))
    print(f"\nRequest manifest: {write_report(cfg, df, a.study)}")


def cmd_dwellir_probe(cfg: Config, a) -> None:
    from .collect_dwellir import probe

    if a.instrument and cfg.is_deferred(a.instrument):
        sys.exit(f"{a.instrument} is deferred")
    coin = a.coin or cfg.instruments[a.instrument]["perp"]["coin"]
    res = probe(cfg, coin, to_date(a.date), a.interval or "1m")
    print(json.dumps(res, indent=1, default=str))


def cmd_dwellir_usage(cfg: Config, a) -> None:
    from .collect_dwellir import make_budget

    b = make_budget(cfg)
    print(f"quota day {b.ledger.today()} ({b.ledger.tz}): used {b.ledger.used():,} | "
          f"usable {b.usable:,} of {b.daily_allowance:,} | remaining {b.remaining():,}")
    print(f"entitlement_verified = {cfg.source_cfg('dwellir').get('entitlement_verified')}")


def cmd_databento_resolve(cfg: Config, a) -> None:
    from .collect_databento import resolve_symbols

    res = resolve_symbols(a.symbols, a.stype_in, a.start, a.end, a.dataset, a.stype_out)
    print(json.dumps(res, indent=1, default=str))


def cmd_hl_contracts(cfg: Config, a) -> None:
    from .collect_hyperliquid import HyperliquidAPI

    api = HyperliquidAPI(cfg.source_cfg("hyperliquid").get("info_url", "https://api.hyperliquid.xyz/info"))
    for dex in api.perp_dexs():
        if not dex:
            continue
        meta, ctxs = api.meta_and_ctxs(dex["name"])
        for u, c in zip(meta["universe"], ctxs):
            name = u["name"]
            if a.match and not any(m.upper() in name.upper() for m in a.match):
                continue
            if u.get("isDelisted") and not a.all:
                continue
            print(f"{name:<18} dex={dex['name']:<6} deployer={dex['deployer']}  "
                  f"mark={c.get('markPx'):<10} oracle={c.get('oraclePx'):<10} "
                  f"24h_ntl=${float(c.get('dayNtlVlm') or 0):>14,.0f}  maxLev={u.get('maxLeverage')}"
                  + ("  DELISTED" if u.get("isDelisted") else ""))


# ---------------------------------------------------------------- stage 2
def cmd_collect(cfg: Config, a) -> None:
    from .collect_databento import DatabentoCollector
    from .collect_dwellir import DwellirCollector
    from .collect_hyperliquid import HyperliquidCollector
    from .collect_taq import TAQCollector

    tasks = _tasks(cfg, a)
    manifest = Manifest(cfg.data_dir)
    make = {"taq": lambda: TAQCollector(cfg, manifest, a.force, a.dry_run),
            "dwellir": lambda: DwellirCollector(cfg, manifest, a.force, a.dry_run, interval=a.interval),
            "hyperliquid": lambda: HyperliquidCollector(cfg, manifest, a.force, a.dry_run),
            "databento": lambda: DatabentoCollector(cfg, manifest, a.force, a.dry_run)}
    collectors = {}
    for t in tasks:
        if t.source not in collectors:
            collectors[t.source] = make[t.source]()
        try:
            collectors[t.source].run(t)
        except Exception as e:  # e.g. missing credentials: keep going with other sources
            manifest.record(t.source, "-", t.instrument, t.date, "error", message=str(e)[:200])
    for c in collectors.values():
        if hasattr(c, "close"):
            c.close()
        if getattr(c, "stopped", None):
            log.warning("dwellir stopped: %s", c.stopped)
    print(f"\nDone. Log of every attempt: {manifest.path}")


def cmd_coverage(cfg: Config, a) -> None:
    import pandas as pd

    root = cfg.data_dir / "raw"
    rows = []
    for f in root.glob("*/*/*/*.parquet"):
        src, dataset, inst = f.parts[-4:-1]
        rows.append({"source": src, "dataset": dataset, "instrument": inst, "date": f.stem})
    for f in root.glob("hyperliquid/live_*/*/*/"):
        rows.append({"source": "hyperliquid", "dataset": f.parts[-3], "instrument": f.parts[-2],
                     "date": f.name})
    if not rows:
        print(f"No data under {root}")
        return
    df = pd.DataFrame(rows)
    print(df.groupby(["source", "dataset", "instrument"])
            .agg(days=("date", "nunique"), first=("date", "min"), last=("date", "max"))
            .reset_index().to_string(index=False))


# ---------------------------------------------------------------- stage 3
def cmd_align(cfg: Config, a) -> None:
    from .align import build_for_study, coverage_table

    aligned = build_for_study(cfg, a.study, a.instrument, a.ref, a.ref_type, a.ref_kind, a.perp_source,
                              a.interval, a.freq, a.sessions, a.max_age,
                              which="scored" if a.scored else "retrieval",
                              start=a.start, end=a.end, dates=a.dates)
    out = cfg.data_dir / "aligned" / (a.study or "custom")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.instrument}_vs_{a.ref or a.instrument}_{a.ref_type}_{a.freq}_{'-'.join(a.sessions)}"
    aligned.to_parquet(out / f"{tag}.parquet", index=False)
    cov = coverage_table(aligned)
    cov.to_csv(out / f"{tag}_coverage.csv", index=False)
    print(cov.to_string(index=False))
    print(f"\nperp price type: {aligned['perp_price_type'].dropna().unique().tolist()} | "
          f"ref price type: {aligned['ref_price_type'].dropna().unique().tolist()}")
    print(f"Wrote {out / (tag + '.parquet')}")


# ---------------------------------------------------------------- stage 4
def cmd_lead_lag(cfg: Config, a) -> None:
    import datetime as dt

    import pandas as pd

    from .align import build_for_study
    from .lead_lag import holdout_forecast, mask_windows, run_analysis, summary_table

    control = None if (a.control or "").lower() == "none" else a.control
    if control and control == (a.ref or a.instrument):
        control = None  # the reference itself (e.g. SP500 vs ES): no separate ES control
    frames = []
    for study in a.studies:
        df = build_for_study(cfg, study, a.instrument, a.ref, a.ref_type, None, a.perp_source, a.interval,
                             a.freq, a.sessions, a.max_age, which="retrieval" if a.retrieval else "scored",
                             control=control)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    ctrl_cols = [f"{control.lower()}_ret"] if control else []
    if control and (ctrl_cols[0] not in df or df[ctrl_cols[0]].notna().sum() == 0):
        log.warning("no %s data on these dates: running WITHOUT the market control (download it first)", control)
        ctrl_cols = []
    df = mask_windows(df, ["perp_ret", "ref_ret", *ctrl_cols], a.exclude_open, a.exclude_close)
    if df[["perp_ret", "ref_ret"]].notna().all(axis=1).sum() == 0:
        sys.exit("No paired returns: collect the data first (see `leadlag coverage`).")

    lags = tuple(range(-a.lags, a.lags + 1))
    res = run_analysis(df, ("period", "session"), lags, range(1, a.max_p + 1), ctrl_cols, a.n_boot, a.seed, a.freq)
    if a.holdout:
        if "block" not in df or df["block"].isna().all():
            sys.exit("--holdout needs a study with development/evaluation blocks (e.g. panel)")
        rows = []
        for sess in a.sessions:
            sub = df[df["session"] == sess]
            for p in range(1, a.max_p + 1):
                for direction, target, other in (("perp -> ref", "ref_ret", "perp_ret"),
                                                 ("ref -> perp", "perp_ret", "ref_ret")):
                    r = holdout_forecast(sub, sub["block"] == "development", sub["block"] == "evaluation",
                                         target, other, p, ctrl_cols, a.freq, a.n_boot, a.seed)
                    rows.append({"session": sess, "direction": direction, **r})
        res["holdout"] = pd.DataFrame(rows)

    tag = f"{a.instrument}_vs_{a.ref or a.instrument}_{a.ref_type}_{a.freq}"
    out = cfg.data_dir / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    for name, t in res.items():
        t.to_csv(out / f"{name}.csv", index=False)
    summ = summary_table(res, p_focus=min(5, a.max_p))
    summ.to_csv(out / "summary.csv", index=False)
    settings = {k: v for k, v in vars(a).items() if k != "fn"}
    settings.update(run_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    control_used=ctrl_cols or None,
                    convention="rho(k) = Corr(perp_t, ref_{t+k}); k > 0 means the perp comes first")
    (out / "settings.json").write_text(json.dumps(settings, indent=1, default=str))

    pd.set_option("display.width", 200)
    print("rho(k) = Corr(perp_t, ref_t+k); positive k = perpetual first\n")
    piv = res["xcorr"].pivot_table(index="lag", columns=["period", "session"], values="rho")
    print(piv.round(3).to_string())
    print("\nSummary (asymmetry A = sum rho(k>0) - sum rho(k<0); p-values are session-clustered F tests):")
    print(summ.round(4).to_string(index=False))
    if "holdout" in res:
        h = res["holdout"][["session", "direction", "p", "n_eval", "eval_sessions", "oos_r2_vs_base",
                            "mean_loss_diff_bps2", "loss_diff_ci_lo_bps2", "loss_diff_ci_hi_bps2"]]
        print("\nChronological holdout (fit Sep-Oct, evaluate Nov; loss difference in bps^2, > 0 = other venue helps):")
        print(h.round(4).to_string(index=False))
    if summ["descriptive_only"].any():
        print("\nNote: groups with < 10 sessions are descriptive only; do not read their CIs/p-values as tests.")
    print(f"\nWrote {out}/ (describe, xcorr, asymmetry, regressions, summary, settings)")


# ---------------------------------------------------------------- stage 5
def cmd_auctions(cfg: Config, a) -> None:
    import pandas as pd

    from .auctions import official_prices

    days = cfg.trading_days(cfg.study_dates(a.study, a.start, a.end, a.dates, which="retrieval"))
    t = official_prices(cfg, a.instrument, days)
    pd.set_option("display.width", 200)
    t["open_et"] = t["open_ts_utc"].dt.tz_convert("America/New_York").dt.strftime("%H:%M:%S.%f").str[:12]
    t["close_et"] = t["close_ts_utc"].dt.tz_convert("America/New_York").dt.strftime("%H:%M:%S.%f").str[:12]
    print(t[["et_date", "open_price", "open_et", "open_source", "close_price", "close_et", "close_source",
             "open_vs_cross_bps", "close_vs_cross_bps", "halt_reopen_prints"]].to_string(index=False))


def cmd_overnight(cfg: Config, a) -> None:
    import datetime as dt

    import pandas as pd

    from .auctions import official_prices
    from .overnight import (closing_forecasts, forecast_metrics, forecast_paths, opening_forecasts,
                            post_open_adjustment)

    if cfg.is_deferred(a.instrument):
        sys.exit(f"{a.instrument} is deferred")
    o = cfg.source_cfg("overnight")
    kw = dict(perp_max_age=a.perp_max_age or o.get("perp_max_age", "5min"),
              cash_max_age=a.cash_max_age or o.get("cash_max_age", "60s"))
    frames_o, frames_c, frames_a, frames_p, info = [], [], [], [], {}
    for study in a.studies:
        scored = cfg.trading_days(cfg.study_dates(study, which="scored"))
        fo, io = opening_forecasts(cfg, a.instrument, scored, o.get("open_cutoffs"), **kw)
        fc, ic = closing_forecasts(cfg, a.instrument, scored, o.get("close_cutoffs"),
                                   o.get("close_anchor", "15:00"), **kw)
        adj = post_open_adjustment(cfg, a.instrument, scored, o.get("adjustment_minutes", [1, 5, 15]), **kw)
        frames_o.append(fo.assign(period=study))
        frames_c.append(fc.assign(period=study))
        frames_a.append(adj.assign(period=study))
        if a.paths:
            frames_p.append(forecast_paths(cfg, a.instrument, scored, **kw).assign(period=study))
        info[study] = {"open": io, "close": ic}
    fo, fc, adj = (pd.concat(f, ignore_index=True) for f in (frames_o, frames_c, frames_a))
    # one row per open target even if two studies share a date
    fo = fo.drop_duplicates(["et_date", "cutoff"])
    fc = fc.drop_duplicates(["et_date", "cutoff"])
    adj = adj.drop_duplicates(["et_date"])
    mo = forecast_metrics(fo, {"perp": "err_bps_perp", "no_change": "err_bps_nochange",
                               "premarket_mid": "err_bps_premkt_mid"},
                          {"perp": "dir_hit_perp", "premarket_mid": "dir_hit_mid"})
    mc = forecast_metrics(fc, {"perp": "err_bps_perp", "cash_mid": "err_bps_cash_mid",
                               "anchor_no_change": "err_bps_anchor_nochange"},
                          {"perp": "dir_hit_perp", "cash_mid": "dir_hit_mid"})
    out = cfg.data_dir / "results" / f"overnight_{a.instrument}"
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(set(fo["et_date"]) | set(fo["prev_date"]))
    official_prices(cfg, a.instrument, days).to_csv(out / "official_prices.csv", index=False)
    for name, t in (("open_forecasts", fo), ("close_forecasts", fc), ("open_metrics", mo),
                    ("close_metrics", mc), ("post_open_adjustment", adj)):
        t.to_csv(out / f"{name}.csv", index=False)
    if frames_p:
        pd.concat(frames_p, ignore_index=True).drop_duplicates(["et_date", "ts_utc"]).to_csv(
            out / "open_paths.csv", index=False)
    (out / "settings.json").write_text(json.dumps(
        {"studies": a.studies, "instrument": a.instrument, **kw, "config": o, "info": info,
         "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "perp_price_type": "trade (Dwellir 1m candle close)"}, indent=1, default=str))

    pd.set_option("display.width", 220)
    cols = ["cutoff", "forecast", "n_matched", "n_missing_or_stale", "MAE_bps", "RMSE_bps", "bias_bps",
            "direction_accuracy", "mean_abs_err_diff_perp_minus_this_bps", "share_perp_better"]
    print(f"Opening forecasts: {fo['et_date'].nunique()} overnight targets (official {a.instrument} opens)")
    print(mo[[c for c in cols if c in mo]].round(3).to_string(index=False))
    print(f"\nClosing forecasts: {fc['et_date'].nunique()} closes, anchored at {o.get('close_anchor', '15:00')} ET")
    print(mc[[c for c in cols if c in mc]].round(3).to_string(index=False))
    a_cols = [c for c in adj.columns if c.startswith(("cash_ret_", "perp_ret_", "basis_change_"))]
    print("\nAfter the open (median across days, bps):")
    print(adj[a_cols].median().round(2).to_string())
    if (mo["descriptive_only"]).any():
        print(f"\nNote: fewer than 20 matched dates per cutoff -> descriptive only, not a forecasting test.")
    print(f"\nWrote {out}/")


# ---------------------------------------------------------------- stage 6
def cmd_windows(cfg: Config, a) -> None:
    import datetime as dt

    import pandas as pd

    from .auction_windows import run_windows

    if cfg.is_deferred(a.instrument):
        sys.exit(f"{a.instrument} is deferred")
    res = run_windows(cfg, a.instrument, a.studies, a.ref, a.ref_type, a.max_age, a.n_boot, a.seed)
    tag = f"windows_{a.instrument}_vs_{a.ref or a.instrument}_{a.ref_type}"
    out = cfg.data_dir / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    res["frame"].to_parquet(out / "frame.parquet", index=False)
    for k, t in res.items():
        if k != "frame":
            t.to_csv(out / f"{k}.csv", index=False)
    (out / "settings.json").write_text(json.dumps(
        {k: v for k, v in vars(a).items() if k != "fn"} |
        {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "windows": "open 09:00-10:00, close 15:30-16:30 ET (boundary +/- 30 min)"}, indent=1, default=str))
    pd.set_option("display.width", 220)
    d = res["describe"][["window", "side", "n_sessions", "paired_returns", "contemporaneous_corr",
                         "sufficiently_populated"]]
    a_ = res["asymmetry"][["window", "side", "A", "ci_lo", "ci_hi"]]
    print("Lead-lag inside each window (A = sum rho(k>0) - sum rho(k<0); positive = perp first):")
    print(d.merge(a_, on=["window", "side"]).round(3).to_string(index=False))
    ch = res["event_changes"]
    cols = [c for c in ch.columns if c.endswith("_bps")]
    print("\nAround each boundary (median across days, bps; offset 0 = last minute before the boundary):")
    print(ch.groupby("window")[cols].median().round(2).T.to_string())
    print("\nShare of the +15 min move done by +1 / +5 min (median; days with >= 5 bps move):")
    print(res["adjustment"].round(2).to_string(index=False))
    print(f"\nWrote {out}/")


def cmd_robustness(cfg: Config, a) -> None:
    import pandas as pd

    from .robustness import run_grid

    t = run_grid(cfg, a.pairs, None, a.n_boot, a.seed)
    out = cfg.data_dir / "results" / "robustness.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(t.drop(columns=["control", "freq"]).round(4).to_string(index=False))
    print(f"\nWrote {out}")


# ---------------------------------------------------------------- optional
def cmd_hl_record(cfg: Config, a) -> None:
    from .hl_live import LiveRecorder

    hc = cfg.source_cfg("hyperliquid")
    coins = list(a.coins or [])
    for inst in a.instruments or []:
        if cfg.is_deferred(inst):
            sys.exit(f"{inst} is deferred in config.yaml; not recording it")
        coins.append(cfg.instruments[inst]["perp"]["coin"])
    if not coins:
        sys.exit("Give --instruments and/or --coins")
    live = hc.get("live", {})
    rec = LiveRecorder(hc.get("ws_url", "wss://api.hyperliquid.xyz/ws"), coins,
                       a.channels or live.get("channels", ["bbo", "trades"]),
                       cfg.data_dir, flush_seconds=a.flush or int(live.get("flush_seconds", 60)))
    asyncio.run(rec.run(duration=a.duration))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="leadlag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: config/config.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="list tasks and maximum Dwellir usage for a selection")
    _add_selection(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("coverage-report", help="free coverage checks + request manifest CSV")
    _add_selection(p)
    p.add_argument("--check-wrds", action="store_true", help="check TAQ tables exist on WRDS")
    p.add_argument("--check-databento", action="store_true", help="quote each Databento request (free)")
    p.set_defaults(fn=cmd_coverage_report)

    p = sub.add_parser("dwellir-probe", help="fetch <= probe_limit candles to test access")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--instrument")
    g.add_argument("--coin")
    p.add_argument("--date", required=True, help="UTC day, YYYY-MM-DD")
    p.add_argument("--interval", choices=["1s", "1m", "5m"])
    p.set_defaults(fn=cmd_dwellir_probe)

    p = sub.add_parser("dwellir-usage", help="show today's Dwellir credit usage from the ledger")
    p.set_defaults(fn=cmd_dwellir_usage)

    p = sub.add_parser("databento-resolve", help="free Databento symbology lookup")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--stype-in", default="raw_symbol")
    p.add_argument("--stype-out", default="instrument_id")
    p.add_argument("--dataset", default="GLBX.MDP3")
    p.add_argument("--start", default="2026-09-18")
    p.add_argument("--end", default="2026-11-14")
    p.set_defaults(fn=cmd_databento_resolve)

    p = sub.add_parser("hl-contracts", help="list Hyperliquid perps across HIP-3 deployers")
    p.add_argument("--match", nargs="*")
    p.add_argument("--all", action="store_true", help="include delisted contracts")
    p.set_defaults(fn=cmd_hl_contracts)

    p = sub.add_parser("collect", help="download data")
    _add_selection(p)
    p.add_argument("--force", action="store_true", help="re-download even if a file exists")
    p.add_argument("--dry-run", action="store_true", help="show requests / quotes without downloading")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("coverage", help="summarise downloaded files")
    p.set_defaults(fn=cmd_coverage)

    p = sub.add_parser("align", help="build an aligned perp-vs-reference table")
    p.add_argument("--study")
    p.add_argument("--instrument", required=True, help="instrument whose perp is compared")
    p.add_argument("--ref", help="reference instrument (default: same; e.g. ES for SP500)")
    p.add_argument("--ref-kind", choices=["equity", "futures"], help="force the reference leg")
    p.add_argument("--ref-type", choices=["trade", "mid"], default="trade",
                   help="trade = bars built like the perp candles (fallback); mid = TAQ/CME quotes")
    p.add_argument("--perp-source", choices=["dwellir", "hl_live"], default="dwellir")
    p.add_argument("--interval", default="1m", choices=["1s", "1m", "5m"])
    p.add_argument("--freq", default="1min")
    p.add_argument("--sessions", nargs="+", default=["regular"], choices=["pre", "regular", "post"])
    p.add_argument("--max-age", default="60s")
    p.add_argument("--scored", action="store_true", help="use scored dates instead of retrieval dates")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--dates", nargs="+")
    p.set_defaults(fn=cmd_align)

    p = sub.add_parser("lead-lag", help="lead-lag correlations + predictive regressions (Stage 4)")
    p.add_argument("--studies", nargs="+", required=True, help="e.g. nvda_baseline nvda_comparison nvda_crisis")
    p.add_argument("--instrument", required=True)
    p.add_argument("--ref", help="reference instrument (default: same; ES for SP500)")
    p.add_argument("--ref-type", choices=["trade", "mid"], default="trade")
    p.add_argument("--control", default="ES", help="market control instrument, or 'none'")
    p.add_argument("--perp-source", choices=["dwellir", "hl_live"], default="dwellir")
    p.add_argument("--interval", default="1m", choices=["1s", "1m", "5m"])
    p.add_argument("--freq", default="1min")
    p.add_argument("--sessions", nargs="+", default=["regular"], choices=["pre", "regular", "post"])
    p.add_argument("--max-age", default="60s")
    p.add_argument("--lags", type=int, default=5, help="K: correlations for k = -K..K")
    p.add_argument("--max-p", type=int, default=5, help="regressions with p = 1..max_p lags")
    p.add_argument("--exclude-open", action="store_true", help="blank the first 15 regular minutes")
    p.add_argument("--exclude-close", action="store_true", help="blank the last 15 regular minutes")
    p.add_argument("--holdout", action="store_true", help="fit development block, evaluate evaluation block")
    p.add_argument("--retrieval", action="store_true", help="use retrieval dates instead of scored dates")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_lead_lag)

    p = sub.add_parser("auctions", help="official primary-exchange open/close prices (TAQ)")
    p.add_argument("--study")
    p.add_argument("--instrument", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--dates", nargs="+")
    p.set_defaults(fn=cmd_auctions)

    p = sub.add_parser("overnight", help="opening/closing auction forecasts (Stage 5, Module B)")
    p.add_argument("--studies", nargs="+", required=True)
    p.add_argument("--instrument", required=True)
    p.add_argument("--perp-max-age")
    p.add_argument("--cash-max-age")
    p.add_argument("--paths", action="store_true", help="also write minute-by-minute forecast paths")
    p.set_defaults(fn=cmd_overnight)

    p = sub.add_parser("windows", help="opening/closing boundary windows (Stage 6, Module C)")
    p.add_argument("--studies", nargs="+", required=True)
    p.add_argument("--instrument", required=True)
    p.add_argument("--ref", help="reference instrument (default: same; ES for SP500)")
    p.add_argument("--ref-type", choices=["mid", "trade"], default="mid")
    p.add_argument("--max-age", default="60s")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_windows)

    p = sub.add_parser("robustness", help="fixed robustness grid for the core lead-lag result")
    p.add_argument("--pairs", nargs="+", help="subset of pair names (default: all)")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_robustness)

    p = sub.add_parser("hl-record", help="optional forward collector (bbo/trades)")
    p.add_argument("--instruments", nargs="+")
    p.add_argument("--coins", nargs="+")
    p.add_argument("--channels", nargs="+", choices=["bbo", "trades", "activeAssetCtx"])
    p.add_argument("--duration", type=float)
    p.add_argument("--flush", type=int)
    p.set_defaults(fn=cmd_hl_record)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "databento"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _load_dotenv()
    try:
        a.fn(Config(a.config), a)
    except (ValueError, KeyError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
