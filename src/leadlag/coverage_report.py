"""Free coverage / symbology inspection and a request manifest, before downloading.

For a study, lists every planned request with what is known about it, using
only free checks:
  * on disk:      file present / empty marker / missing
  * TAQ:          whether the WRDS daily tables exist (--check-wrds; a free
                  catalogue lookup, may trigger a Duo prompt on first connect)
  * Hyperliquid:  contract listed / delisted now, and the first funding record
                  (a proxy for when the contract started trading)
  * Dwellir:      max candle credits per day, entitlement flag, budget left
  * Databento:    zero-cost quote per request (--check-databento; quotes are
                  free metadata calls and are logged in the request ledger)
Writes data/reports/request_manifest_<study>_<UTC timestamp>.csv.
"""
from __future__ import annotations

import datetime as dt
import logging
import os

import pandas as pd

from .config import Config, Task
from .storage import raw_path

log = logging.getLogger("leadlag")


def _disk_status(cfg: Config, source: str, dataset: str, inst: str, day: dt.date) -> str:
    p = raw_path(cfg.data_dir, source, dataset, inst, day)
    if p.exists():
        return "present"
    if p.with_suffix(".empty").exists():
        return "empty"
    return "missing"


def _hl_contracts(cfg: Config, coins: set[str]) -> dict[str, dict]:
    from .collect_hyperliquid import HyperliquidAPI

    api = HyperliquidAPI(cfg.source_cfg("hyperliquid").get("info_url", "https://api.hyperliquid.xyz/info"))
    info: dict[str, dict] = {}
    for dex in {c.split(":")[0] for c in coins if ":" in c}:
        meta, _ = api.meta_and_ctxs(dex)
        for u in meta["universe"]:
            if u["name"] in coins:
                info[u["name"]] = {"listed": True, "delisted": bool(u.get("isDelisted"))}
    for c in coins:
        rec = info.setdefault(c, {"listed": False, "delisted": None})
        first = api.info({"type": "fundingHistory", "coin": c, "startTime": 0})[:1]  # one page
        rec["first_funding_utc"] = (pd.Timestamp(first[0]["time"], unit="ms", tz="UTC").date()
                                    if first else None)
    return info


def build_report(cfg: Config, tasks: list[Task], check_wrds: bool = False,
                 check_databento: bool = False, dwellir_interval: str | None = None) -> pd.DataFrame:
    rows = []
    coins = {t.leg["coin"] for t in tasks if t.source in ("dwellir", "hyperliquid")}
    hl = {}
    if coins:
        try:
            hl = _hl_contracts(cfg, coins)
        except Exception as e:
            log.warning("Hyperliquid contract check failed: %s", e)

    taq_cfg = cfg.source_cfg("taq")
    db = None
    if check_wrds and any(t.source == "taq" for t in tasks):
        import wrds
        db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))

    guard = None
    if check_databento and any(t.source == "databento" for t in tasks):
        from .collect_databento import DatabentoCollector
        from .storage import Manifest
        guard = DatabentoCollector(cfg, Manifest(cfg.data_dir)).guard

    dw = cfg.source_cfg("dwellir")
    interval = dwellir_interval or dw.get("interval", "1m")
    from .dwellir_budget import INTERVAL_MS
    per_day = 86_400_000 // INTERVAL_MS[interval]

    for t in tasks:
        base = {"source": t.source, "instrument": t.instrument, "date": t.date}
        if t.source == "taq":
            sec = int(taq_cfg.get("sample_seconds", 1))
            tag = "raw" if sec <= 0 else f"{sec}s"
            for ds in (f"nbbo_{tag}", f"trades_{tag}", "auction_trades"):
                r = dict(base, dataset=ds, request=f"{t.leg['taq_sym_root']} {t.date}",
                         on_disk=_disk_status(cfg, "taq", ds, t.instrument, t.date))
                if db is not None:
                    key = "nbbo_table_pattern" if ds.startswith("nbbo") else "trades_table_pattern"
                    table = (taq_cfg.get("library_pattern", "taqm_{year}").format(year=t.date.year) + "." +
                             taq_cfg[key].format(yyyymmdd=t.date.strftime("%Y%m%d"), year=t.date.year))
                    ok = db.raw_sql("SELECT to_regclass(%(t)s) IS NOT NULL AS ok", params={"t": table})
                    r["available"] = bool(ok["ok"].iloc[0])
                    r["note"] = table
                rows.append(r)
        elif t.source == "dwellir":
            coin = t.leg["coin"]
            c = hl.get(coin, {})
            first = c.get("first_funding_utc")
            rows.append(dict(base, dataset=f"candles_{interval}", request=f"{coin} {interval} {t.date}",
                             on_disk=_disk_status(cfg, "dwellir", f"candles_{interval}", coin, t.date),
                             max_credits=per_day,
                             available=(first is not None and first <= t.date) if c else None,
                             note=f"listed={c.get('listed')} delisted={c.get('delisted')} "
                                  f"first_funding={first} entitlement_verified={dw.get('entitlement_verified')}"))
        elif t.source == "hyperliquid":
            coin = t.leg["coin"]
            rows.append(dict(base, dataset="funding", request=f"{coin} funding {t.date}",
                             on_disk=_disk_status(cfg, "hyperliquid", "funding", coin, t.date)))
        elif t.source == "databento":
            from .collect_databento import day_request
            for schema in cfg.source_cfg("databento").get("schemas", ["bbo-1s"]):
                ds = schema.replace("-", "_")
                r = dict(base, dataset=ds, request=f"{t.leg['symbol']} {schema} {t.date}",
                         on_disk=_disk_status(cfg, "databento", ds, t.leg["symbol"], t.date))
                if guard is not None and r["on_disk"] == "missing":
                    try:
                        a = guard.quote(day_request(t.leg, schema, t.date))
                        r["quoted_usd"], r["available"] = str(a.cost), True
                    except Exception as e:
                        r["quoted_usd"], r["available"], r["note"] = None, False, str(e)[:150]
                rows.append(r)
    if db is not None:
        db.close()
    return pd.DataFrame(rows)


def write_report(cfg: Config, df: pd.DataFrame, study: str | None) -> str:
    out = cfg.data_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"request_manifest_{study or 'custom'}_{ts}.csv"
    df.to_csv(path, index=False)
    return str(path)


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "no planned requests"
    agg = {"date": ["count", "min", "max"]}
    s = df.groupby(["source", "instrument", "dataset", "on_disk"]).agg(agg)
    s.columns = ["days", "first", "last"]
    lines = [s.reset_index().to_string(index=False)]
    if "max_credits" in df:
        need = df.loc[(df.source == "dwellir") & (df.on_disk == "missing"), "max_credits"].sum()
        lines.append(f"\nDwellir: up to {int(need):,} candle credits still to download "
                     "(sparse minutes lower the real count)")
    if "available" in df:
        bad = df[df["available"] == False]  # noqa: E712
        if not bad.empty:
            lines.append(f"\n{len(bad)} request(s) flagged unavailable/blocked:\n"
                         + bad[["source", "instrument", "dataset", "date", "note"]].head(20).to_string(index=False))
    return "\n".join(lines)
