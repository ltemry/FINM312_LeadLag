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
    from . import normalize as N
    from .align import align_pair, coverage_table, session_grid

    if cfg.is_deferred(a.instrument):
        sys.exit(f"{a.instrument} is deferred")
    inst = cfg.instruments[a.instrument]
    et_dates = cfg.study_dates(a.study, a.start, a.end, a.dates,
                               which="scored" if a.scored else "retrieval")
    trading = cfg.trading_days(et_dates)
    utc_days = cfg.utc_days(et_dates)
    grid = session_grid(trading, a.sessions, a.freq, cfg.extra_closures)

    coin = inst["perp"]["coin"]
    perp = (N.dwellir_bars(cfg, coin, utc_days, a.interval) if a.perp_source == "dwellir"
            else N.hl_live_bbo(cfg, coin, utc_days))
    ref_name = a.ref or a.instrument
    ref_inst = cfg.instruments[ref_name]
    if "equity" in ref_inst and a.ref_kind != "futures":
        sym = ref_inst["equity"]["taq_sym_root"]
        ref = (N.taq_trade_bars(cfg, sym, trading, a.freq) if a.ref_type == "trade"
               else N.taq_mid(cfg, sym, trading))
    else:
        sym = ref_inst["futures"]["symbol"]
        ref = (N.databento_bars(cfg, sym, utc_days) if a.ref_type == "trade"
               else N.databento_bbo(cfg, sym, utc_days))
    aligned = align_pair(grid, ref, perp, max_age=a.max_age, freq=a.freq)
    aligned["pair"] = f"{coin} vs {ref_name}"
    out = cfg.data_dir / "aligned" / (a.study or "custom")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.instrument}_vs_{ref_name}_{a.ref_type}_{a.freq}_{'-'.join(a.sessions)}"
    aligned.to_parquet(out / f"{tag}.parquet", index=False)
    cov = coverage_table(aligned)
    cov.to_csv(out / f"{tag}_coverage.csv", index=False)
    print(cov.to_string(index=False))
    print(f"\nperp price type: {aligned['perp_price_type'].dropna().unique().tolist()} | "
          f"ref price type: {aligned['ref_price_type'].dropna().unique().tolist()}")
    print(f"Wrote {out / (tag + '.parquet')}")


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
