"""WRDS Daily TAQ collector: NBBO quotes (and optional trades) per stock-day.

Sampling happens on the WRDS server: for each `sample_seconds` bucket we keep
the LAST valid NBBO update, together with its own timestamp, so quote age can
be computed later. Buckets with no update are simply absent; the alignment
step carries the previous quote forward (as-of join) and records its age.

Credentials: the `wrds` package reads ~/.pgpass, or prompts once and offers to
create it. Set WRDS_USERNAME to skip the username prompt.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re

import pandas as pd

from .config import Config, Task
from .storage import Manifest, already_done, mark_empty, raw_path, retry, write_parquet

log = logging.getLogger("leadlag")
_IDENT = re.compile(r"^[a-z0-9_]+$")


def _is_sql_error(e: Exception) -> bool:
    """Bad SQL (missing column/table) won't fix itself; don't retry it."""
    return type(e).__name__ == "ProgrammingError"


class TAQCollector:
    source = "taq"

    def __init__(self, cfg: Config, manifest: Manifest, force: bool = False, dry_run: bool = False):
        self.cfg = cfg
        self.c = cfg.source_cfg("taq")
        self.manifest = manifest
        self.force = force
        self.dry_run = dry_run
        self.sec = int(self.c.get("sample_seconds", 1))
        self._db = None

    # ---------------------------------------------------------------- helpers
    @property
    def db(self):
        if self._db is None:
            import wrds  # imported lazily so other collectors work without it

            self._db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()

    def _table(self, pattern_key: str, d: dt.date) -> str:
        lib = self.c.get("library_pattern", "taqm_{year}").format(year=d.year)
        tbl = self.c[pattern_key].format(yyyymmdd=d.strftime("%Y%m%d"), year=d.year)
        if not (_IDENT.match(lib) and _IDENT.match(tbl)):
            raise ValueError(f"Suspicious table name {lib}.{tbl}")
        return f"{lib}.{tbl}"

    @retry(tries=3, base_delay=5, giveup=_is_sql_error)
    def _sql(self, sql: str, params: dict) -> pd.DataFrame:
        return self.db.raw_sql(sql, params=params)

    def _table_exists(self, table: str) -> bool:
        df = self._sql("SELECT to_regclass(%(t)s) IS NOT NULL AS ok", {"t": table})
        return bool(df["ok"].iloc[0])

    def _params(self, task: Task) -> dict:
        return {
            "sym": task.leg["taq_sym_root"],
            "suffix": task.leg.get("taq_sym_suffix") or "",
            "t0": self.c.get("session_start", "04:00:00"),
            "t1": self.c.get("session_end", "20:00:00"),
        }

    # ---------------------------------------------------------------- queries
    def nbbo_sql(self, table: str) -> str:
        valid = "AND best_bid > 0 AND best_ask > 0 AND best_ask >= best_bid" if self.c.get(
            "drop_invalid_on_server", True) else ""
        if self.sec <= 0:
            return f"""
                SELECT * FROM {table}
                WHERE sym_root = %(sym)s AND coalesce(sym_suffix, '') = %(suffix)s
                  AND time_m BETWEEN %(t0)s AND %(t1)s {valid}
                ORDER BY time_m"""
        return f"""
            SELECT DISTINCT ON (bucket)
                   floor(extract(epoch FROM time_m) / {self.sec})::bigint AS bucket,
                   date, time_m, time_m_nano, sym_root, sym_suffix,
                   best_bid, best_bidsizeshares, best_ask, best_asksizeshares,
                   qu_cond, natbbo_ind
            FROM {table}
            WHERE sym_root = %(sym)s AND coalesce(sym_suffix, '') = %(suffix)s
              AND time_m BETWEEN %(t0)s AND %(t1)s {valid}
            ORDER BY bucket, time_m DESC, time_m_nano DESC"""

    def trades_sql(self, table: str) -> str:
        if self.sec <= 0:
            return f"""
                SELECT * FROM {table}
                WHERE sym_root = %(sym)s AND coalesce(sym_suffix, '') = %(suffix)s
                  AND time_m BETWEEN %(t0)s AND %(t1)s
                ORDER BY time_m"""
        return f"""
            SELECT floor(extract(epoch FROM time_m) / {self.sec})::bigint AS bucket,
                   min(date) AS date,
                   count(*) AS n_trades,
                   sum(size) AS volume,
                   sum(price * size) / nullif(sum(size), 0) AS vwap,
                   (array_agg(price ORDER BY time_m DESC))[1] AS last_price,
                   max(time_m) AS last_trade_time
            FROM {table}
            WHERE sym_root = %(sym)s AND coalesce(sym_suffix, '') = %(suffix)s
              AND time_m BETWEEN %(t0)s AND %(t1)s
              AND tr_corr = '00' AND price > 0
              {self._condition_filter()}
            GROUP BY 1 ORDER BY 1"""

    def _condition_filter(self) -> str:
        """Drop prints that do not set the last-sale price (late / out-of-sequence,
        average-price, odd-lot, prior-reference, derivatively priced, Form T...) and
        auction/official prints (kept separately in auction_trades)."""
        excl = self.c.get("trade_exclude_conditions", "")
        if not excl:
            return ""
        if not re.fullmatch(r"[A-Z0-9]+", excl):
            raise ValueError(f"bad trade_exclude_conditions {excl!r}")
        return f"AND coalesce(tr_scond, '') !~ '[{excl}]'"

    def auction_sql(self, table: str) -> str:
        """Opening/closing/official/reopening prints, all exchanges, NOT filtered by
        tr_corr or session, so official-price fallbacks and corrections stay visible."""
        conds = self.c.get("auction_conditions", "OQ6M5")
        if not re.fullmatch(r"[A-Z0-9]+", conds):
            raise ValueError(f"bad auction_conditions {conds!r}")
        return f"""
            SELECT date, time_m, time_m_nano, ex, sym_root, sym_suffix, tr_scond, size, price,
                   tr_corr, tr_seqnum, tr_id, tr_source, tr_rf, part_time, part_time_nano
            FROM {table}
            WHERE sym_root = %(sym)s AND coalesce(sym_suffix, '') = %(suffix)s
              AND tr_scond ~ '[{conds}]'
            ORDER BY time_m, time_m_nano"""

    # ---------------------------------------------------------------- tidy
    def _to_utc(self, date_s: pd.Series, time_s: pd.Series) -> pd.Series:
        """TAQ stores ET wall-clock date + time; return tz-aware UTC timestamps."""
        if pd.api.types.is_timedelta64_dtype(time_s):
            td = time_s
        else:
            td = pd.to_timedelta(time_s.astype(str))
        local = pd.to_datetime(date_s) + td
        return local.dt.tz_localize(self.cfg.market_tz).dt.tz_convert("UTC")

    def _bucket_end_utc(self, d: dt.date, bucket: pd.Series) -> pd.Series:
        midnight = pd.Timestamp(d).tz_localize(self.cfg.market_tz)
        return (midnight + pd.to_timedelta((bucket + 1) * self.sec, unit="s")).dt.tz_convert("UTC")

    def _tidy_nbbo(self, df: pd.DataFrame, d: dt.date) -> pd.DataFrame:
        if self.sec <= 0:
            nano = df.get("time_m_nano")
            df["quote_ts_utc"] = self._to_utc(df["date"], df["time_m"])
            if nano is not None:
                df["quote_ts_utc"] += pd.to_timedelta(nano.fillna(0).astype("int64"), unit="ns")
            return df
        out = pd.DataFrame({
            "bucket_end_utc": self._bucket_end_utc(d, df["bucket"]),
            "quote_ts_utc": self._to_utc(df["date"], df["time_m"])
            + pd.to_timedelta(df["time_m_nano"].fillna(0).astype("int64"), unit="ns"),
            "symbol": df["sym_root"],
            "bid": df["best_bid"].astype(float),
            "ask": df["best_ask"].astype(float),
            "bid_size": df["best_bidsizeshares"].astype("Int64"),
            "ask_size": df["best_asksizeshares"].astype("Int64"),
            "qu_cond": df["qu_cond"],
            "natbbo_ind": df["natbbo_ind"],
        })
        out["mid"] = (out["bid"] + out["ask"]) / 2
        return out

    def _tidy_auction(self, df: pd.DataFrame, d: dt.date) -> pd.DataFrame:
        df["trade_ts_utc"] = self._to_utc(df["date"], df["time_m"]) + pd.to_timedelta(
            df["time_m_nano"].fillna(0).astype("int64"), unit="ns")
        df["price"] = df["price"].astype(float)
        df["size"] = df["size"].astype("int64")
        for col in ("part_time", "time_m"):
            df[col] = df[col].astype(str)
        return df

    def _tidy_trades(self, df: pd.DataFrame, d: dt.date) -> pd.DataFrame:
        if self.sec <= 0:
            df["trade_ts_utc"] = self._to_utc(df["date"], df["time_m"])
            return df
        return pd.DataFrame({
            "bucket_end_utc": self._bucket_end_utc(d, df["bucket"]),
            "last_trade_ts_utc": self._to_utc(df["date"], df["last_trade_time"]),
            "n_trades": df["n_trades"].astype("int64"),
            "volume": df["volume"].astype("int64"),
            "vwap": df["vwap"].astype(float),
            "last_price": df["last_price"].astype(float),
        })

    # ---------------------------------------------------------------- run
    def run(self, task: Task) -> None:
        tag = "raw" if self.sec <= 0 else f"{self.sec}s"
        jobs = [("nbbo", "nbbo_table_pattern", self.nbbo_sql, self._tidy_nbbo)]
        if self.c.get("collect_trades", True):
            jobs.append(("trades", "trades_table_pattern", self.trades_sql, self._tidy_trades))
        if self.c.get("collect_auction_trades", True):
            jobs.append(("auction", "trades_table_pattern", self.auction_sql, self._tidy_auction))

        for kind, pattern_key, sql_fn, tidy in jobs:
            dataset = "auction_trades" if kind == "auction" else f"{kind}_{tag}"
            path = raw_path(self.cfg.data_dir, self.source, dataset, task.instrument, task.date)
            if already_done(path, self.force):
                self.manifest.record(self.source, dataset, task.instrument, task.date, "skipped", path=path)
                continue
            table = self._table(pattern_key, task.date)
            sql = sql_fn(table)
            if self.dry_run:
                log.info("[dry-run] %s %s %s\n%s\nparams=%s", dataset, task.instrument, task.date,
                         sql.strip(), self._params(task))
                continue
            try:
                if not self._table_exists(table):
                    self.manifest.record(self.source, dataset, task.instrument, task.date, "unavailable",
                                         message=f"{table} not on WRDS (yet?)")
                    continue
                df = self._sql(sql, self._params(task))
                if df.empty:
                    mark_empty(path, "query returned no rows")
                    self.manifest.record(self.source, dataset, task.instrument, task.date, "empty", path=path)
                    continue
                df = tidy(df, task.date)
                write_parquet(df, path)
                self.manifest.record(self.source, dataset, task.instrument, task.date, "ok", len(df), path)
            except Exception as e:  # keep going with other days
                self.manifest.record(self.source, dataset, task.instrument, task.date, "error",
                                     message=str(e).splitlines()[0][:200])
