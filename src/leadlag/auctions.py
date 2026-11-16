"""Official primary-exchange opening and closing prices from TAQ auction prints.

Uses the records collected in data/raw/taq/auction_trades (all exchanges,
unfiltered). For each stock-day, on the PRIMARY exchange only (config
`primary_exchange`, Q = Nasdaq for NVDA/TSLA):

  official open : tr_scond contains 'Q' (market center official open)
                  fallback 'O' (opening print), flagged
  official close: tr_scond contains 'M' (market center official close)
                  fallback '6' (closing print), flagged

Never substitutes first/last consolidated trades or minute-bar OHLC. Each
price keeps its own exchange timestamp, which is the target time: the opening
cross can print after 09:30:00 (e.g. 09:30:08 on 2026-11-12), and features
must be strictly earlier than it. Corrected/cancelled records (tr_corr != '00')
are ignored and counted; a reopening print ('5') flags a halt.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd

from .config import Config
from .storage import safe_name

OPEN_PRIMARY, OPEN_FALLBACK = "Q", "O"
CLOSE_PRIMARY, CLOSE_FALLBACK = "M", "6"


def _pick(rows: pd.DataFrame, primary: str, fallback: str, which: str):
    """Return (price, ts, source) from primary-exchange rows, preferring the official record."""
    for code, label in ((primary, f"official_{which}"), (fallback, f"{which}_print_fallback")):
        hit = rows[rows["tr_scond"].fillna("").str.contains(code, regex=False)]
        if not hit.empty:
            r = hit.iloc[0] if which == "open" else hit.iloc[-1]
            return float(r["price"]), r["trade_ts_utc"], label
    return None, pd.NaT, "missing"


def official_prices(cfg: Config, instrument: str, et_dates: Iterable[dt.date]) -> pd.DataFrame:
    leg = cfg.instruments[instrument]["equity"]
    sym, ex = leg["taq_sym_root"], leg.get("primary_exchange", "Q")
    base = cfg.data_dir / "raw" / "taq" / "auction_trades" / safe_name(sym)
    rows = []
    for d in sorted(set(et_dates)):
        f = base / f"{d.isoformat()}.parquet"
        rec = {"et_date": d, "symbol": sym, "primary_exchange": ex}
        if not f.exists():
            rows.append({**rec, "open_source": "no_file", "close_source": "no_file"})
            continue
        a = pd.read_parquet(f).sort_values("trade_ts_utc")
        rec["corrected_records_ignored"] = int((a["tr_corr"] != "00").sum())
        rec["halt_reopen_prints"] = int(a["tr_scond"].fillna("").str.contains("5", regex=False).sum())
        p = a[(a["ex"] == ex) & (a["tr_corr"] == "00")]
        rec["open_price"], rec["open_ts_utc"], rec["open_source"] = _pick(p, OPEN_PRIMARY, OPEN_FALLBACK, "open")
        rec["close_price"], rec["close_ts_utc"], rec["close_source"] = _pick(p, CLOSE_PRIMARY, CLOSE_FALLBACK, "close")
        # consistency: official record vs. the cross print on the same exchange
        o_print = p[p["tr_scond"].fillna("").str.contains(OPEN_FALLBACK, regex=False)]
        c_print = p[p["tr_scond"].fillna("").str.contains(CLOSE_FALLBACK, regex=False)]
        rec["open_vs_cross_bps"] = (1e4 * (rec["open_price"] / o_print["price"].iloc[0] - 1)
                                    if rec["open_price"] and not o_print.empty else None)
        rec["close_vs_cross_bps"] = (1e4 * (rec["close_price"] / c_print["price"].iloc[-1] - 1)
                                     if rec["close_price"] and not c_print.empty else None)
        rows.append(rec)
    out = pd.DataFrame(rows)
    for c in ("open_ts_utc", "close_ts_utc"):
        if c in out:
            out[c] = pd.to_datetime(out[c], utc=True)
    return out
