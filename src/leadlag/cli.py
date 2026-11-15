"""Command-line entry point.

    leadlag plan     --study pilot                 # show what would be collected
    leadlag collect  --study pilot                 # collect it (skips finished files)
    leadlag collect  --study baseline --sources hyperliquid --start 2026-11-10 --end 2026-11-13
    leadlag coverage                               # what's on disk, per source/day
    leadlag hl-contracts --match TSLA NVDA ANT 500 # find HIP-3 contracts across deployers
    leadlag hl-record --instruments TSLA NVDA ANT SP500   # live recorder (Ctrl-C to stop)

Run `python -m leadlag.cli <command> -h` for options.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict

from .config import REPO_ROOT, SOURCES, Config
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
    p.add_argument("--study", help="study name from config.yaml (pilot, baseline, events, ant_ipo, sp_es)")
    p.add_argument("--sources", nargs="+", choices=SOURCES, help="override the study's sources")
    p.add_argument("--instruments", nargs="+", help="override the study's instruments")
    p.add_argument("--start", help="YYYY-MM-DD (overrides the study's dates)")
    p.add_argument("--end", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--dates", nargs="+", help="explicit list of YYYY-MM-DD dates")


def _tasks(cfg: Config, a: argparse.Namespace):
    if not a.study and not (a.start or a.dates):
        sys.exit("Give --study, or --start/--end, or --dates.")
    tasks, notes = cfg.build_tasks(a.study, a.sources, a.instruments, a.start, a.end, a.dates)
    for n in notes:
        log.warning(n)
    return tasks


def cmd_plan(cfg: Config, a) -> None:
    tasks = _tasks(cfg, a)
    by = defaultdict(list)
    for t in tasks:
        by[(t.source, t.instrument)].append(t.date)
    print(f"{len(tasks)} source-instrument-days")
    for (src, inst), ds in by.items():
        print(f"  {src:<12} {inst:<6} {len(ds):>3} days  {ds[0]} .. {ds[-1]}")


def cmd_collect(cfg: Config, a) -> None:
    from .collect_databento import DatabentoCollector
    from .collect_hyperliquid import HyperliquidCollector
    from .collect_taq import TAQCollector

    tasks = _tasks(cfg, a)
    manifest = Manifest(cfg.data_dir)
    classes = {"taq": TAQCollector, "hyperliquid": HyperliquidCollector, "databento": DatabentoCollector}
    collectors = {}
    for t in tasks:
        if t.source not in collectors:
            collectors[t.source] = classes[t.source](cfg, manifest, force=a.force, dry_run=a.dry_run)
        try:
            collectors[t.source].run(t)
        except Exception as e:  # one bad source (e.g. missing credentials) shouldn't stop the others
            manifest.record(t.source, "-", t.instrument, t.date, "error", message=str(e)[:200])
    for c in collectors.values():
        if hasattr(c, "close"):
            c.close()
    print(f"\nDone. Log of every attempt: {manifest.path}")


def cmd_coverage(cfg: Config, a) -> None:
    import pandas as pd

    root = cfg.data_dir / "raw"
    rows = []
    for f in root.glob("*/*/*/*.parquet"):
        src, dataset, inst = f.parts[-4:-1]
        rows.append({"source": src, "dataset": dataset, "instrument": inst, "date": f.stem,
                     "rows": pd.read_parquet(f, columns=[]).shape[0] if a.rows else None})
    for f in root.glob("hyperliquid/live_*/*/*/"):
        rows.append({"source": "hyperliquid", "dataset": f.parts[-3], "instrument": f.parts[-2],
                     "date": f.name, "rows": None})
    if not rows:
        print(f"No data under {root}")
        return
    df = pd.DataFrame(rows)
    summary = (df.groupby(["source", "dataset", "instrument"])
                 .agg(days=("date", "nunique"), first=("date", "min"), last=("date", "max"))
                 .reset_index())
    print(summary.to_string(index=False))
    if a.rows:
        print("\nRows per file:\n" + df.sort_values(["source", "dataset", "instrument", "date"])
              .to_string(index=False))


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
                  f"24h_ntl=${float(c.get('dayNtlVlm') or 0):>14,.0f}  maxLev={u.get('maxLeverage')}")


def cmd_hl_record(cfg: Config, a) -> None:
    from .hl_live import LiveRecorder

    hc = cfg.source_cfg("hyperliquid")
    coins = list(a.coins or [])
    for inst in a.instruments or []:
        perp = cfg.instruments[inst].get("perp") or {}
        coins.append(perp["coin"])
        if a.alt_coins:
            coins += perp.get("alt_coins") or []
    if not coins:
        sys.exit("Give --instruments and/or --coins")
    live = hc.get("live", {})
    rec = LiveRecorder(hc.get("ws_url", "wss://api.hyperliquid.xyz/ws"), coins,
                       a.channels or live.get("channels", ["bbo", "trades", "activeAssetCtx"]),
                       cfg.data_dir, flush_seconds=a.flush or int(live.get("flush_seconds", 60)))
    asyncio.run(rec.run(duration=a.duration))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="leadlag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: config/config.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="list the source-instrument-days a selection resolves to")
    _add_selection(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("collect", help="download data")
    _add_selection(p)
    p.add_argument("--force", action="store_true", help="re-download even if a file exists")
    p.add_argument("--dry-run", action="store_true", help="show queries / costs without downloading")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("coverage", help="summarise downloaded files")
    p.add_argument("--rows", action="store_true", help="also count rows per file")
    p.set_defaults(fn=cmd_coverage)

    p = sub.add_parser("hl-contracts", help="list Hyperliquid perps across all HIP-3 deployers")
    p.add_argument("--match", nargs="*", help="substrings to filter names, e.g. TSLA NVDA ANT 500")
    p.add_argument("--all", action="store_true", help="include delisted contracts")
    p.set_defaults(fn=cmd_hl_contracts)

    p = sub.add_parser("hl-record", help="record live Hyperliquid bbo/trades/mark-oracle")
    p.add_argument("--instruments", nargs="+", help="instrument names from config.yaml")
    p.add_argument("--coins", nargs="+", help="extra raw coin names, e.g. km:TSLA")
    p.add_argument("--alt-coins", action="store_true", help="also record each instrument's alt_coins")
    p.add_argument("--channels", nargs="+", choices=["bbo", "trades", "activeAssetCtx"])
    p.add_argument("--duration", type=float, help="stop after N seconds (default: run until Ctrl-C)")
    p.add_argument("--flush", type=int, help="seconds between writes to disk")
    p.set_defaults(fn=cmd_hl_record)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    _load_dotenv()
    try:
        a.fn(Config(a.config), a)
    except (ValueError, KeyError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
