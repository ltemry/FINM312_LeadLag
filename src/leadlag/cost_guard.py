"""Databento zero-incremental-cost guard.

Policy (docs/project_plan.md): authorized additional data spend is exactly USD 0.

* Every Databento data request is an immutable `DatabentoRequest` with explicit
  symbols, schema, stype_in and a finite [start, end). ALL_SYMBOLS, parent /
  continuous symbology and open-ended ranges are rejected.
* `CostGuard.quote` calls metadata.get_cost for that identical request and
  returns an `Approval` only if the parsed quote is a finite number exactly
  equal to zero. Anything else (positive, negative, NaN, inf, None, garbage,
  or an exception) blocks.
* `CostGuard.download` refuses to run without an Approval whose digest matches
  the request, so a changed request needs a new quote.
* `CostGuard.fetch` is the one entry point collectors use: cached file -> no
  network; otherwise quote, then download, and every retry re-quotes.
* Every quote and outcome is appended to data/ledger/databento_requests.jsonl.

This module is the ONLY place in the package that calls timeseries.get_range
(tests/test_cost_guard.py checks that).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("leadlag")

DEFAULT_ALLOWED_STYPE_IN = ("raw_symbol", "instrument_id")


class CostBlocked(Exception):
    """The request was not quoted at exactly USD 0 (or could not be quoted)."""


class InvalidRequest(ValueError):
    pass


def _parse_ts(x: str) -> dt.datetime:
    ts = dt.datetime.fromisoformat(str(x))
    if ts.tzinfo is None:
        raise InvalidRequest(f"timestamp must include a timezone: {x}")
    return ts


@dataclass(frozen=True)
class DatabentoRequest:
    dataset: str
    symbols: tuple[str, ...]
    schema: str
    stype_in: str
    start: str
    end: str
    limit: int | None = None

    def validate(self, allowed_stype_in=DEFAULT_ALLOWED_STYPE_IN) -> None:
        if not self.dataset:
            raise InvalidRequest("dataset is required")
        if not isinstance(self.symbols, tuple) or not self.symbols:
            raise InvalidRequest("symbols must be a non-empty tuple of explicit symbols")
        for s in self.symbols:
            if not isinstance(s, str) or not s.strip():
                raise InvalidRequest(f"bad symbol {s!r}")
            if s.strip().upper() == "ALL_SYMBOLS":
                raise InvalidRequest("ALL_SYMBOLS is not allowed")
        if self.stype_in not in allowed_stype_in:
            raise InvalidRequest(f"stype_in {self.stype_in!r} not allowed; use one of {allowed_stype_in}")
        if not self.schema:
            raise InvalidRequest("schema is required")
        if self.start in (None, "") or self.end in (None, ""):
            raise InvalidRequest("finite start and end are required")
        a, b = _parse_ts(self.start), _parse_ts(self.end)
        if b <= a:
            raise InvalidRequest("end must be after start")
        if self.limit is not None and (not isinstance(self.limit, int) or self.limit <= 0):
            raise InvalidRequest("limit must be a positive int")

    def kwargs(self) -> dict[str, Any]:
        kw = dict(dataset=self.dataset, symbols=list(self.symbols), schema=self.schema,
                  stype_in=self.stype_in, start=self.start, end=self.end)
        if self.limit is not None:
            kw["limit"] = self.limit
        return kw

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Approval:
    digest: str
    cost: Decimal
    quoted_at: str


def parse_zero_cost(value: Any) -> Decimal:
    """Return Decimal(0) if `value` is a finite number exactly equal to 0, else raise CostBlocked."""
    if value is None or isinstance(value, bool):
        raise CostBlocked(f"missing or non-numeric quote: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise CostBlocked(f"non-finite quote: {value!r}")
    try:
        d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        raise CostBlocked(f"unparseable quote: {value!r}") from None
    if not d.is_finite():
        raise CostBlocked(f"non-finite quote: {value!r}")
    if d < 0:
        raise CostBlocked(f"negative quote: {value!r}")
    if d != 0:
        raise CostBlocked(f"quoted cost {value!r} is not exactly 0")
    return d


class CostGuard:
    def __init__(self, client, ledger_path: Path,
                 allowed_stype_in=DEFAULT_ALLOWED_STYPE_IN,
                 now: Callable[[], dt.datetime] | None = None):
        self.client = client
        self.ledger_path = Path(ledger_path)
        self.allowed_stype_in = tuple(allowed_stype_in)
        self._now = now or (lambda: dt.datetime.now(dt.timezone.utc))

    # ------------------------------------------------------------ ledger
    def _log(self, req: DatabentoRequest, outcome: str, cost: Any = None, message: str = "") -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts_utc": self._now().isoformat(timespec="seconds"), "digest": req.digest()[:16],
               "request": asdict(req), "quoted_cost": None if cost is None else str(cost),
               "outcome": outcome, "message": message}
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------ guard
    def quote(self, req: DatabentoRequest) -> Approval:
        req.validate(self.allowed_stype_in)
        try:
            raw = self.client.metadata.get_cost(**req.kwargs())
        except Exception as e:
            self._log(req, "block", message=f"quote failed: {type(e).__name__}: {e}"[:300])
            raise CostBlocked(f"cost quote failed: {e}") from e
        try:
            cost = parse_zero_cost(raw)
        except CostBlocked as e:
            self._log(req, "block", cost=raw, message=str(e))
            raise
        self._log(req, "allow", cost=cost)
        return Approval(req.digest(), cost, self._now().isoformat(timespec="seconds"))

    def download(self, req: DatabentoRequest, approval: Approval):
        if not isinstance(approval, Approval) or approval.digest != req.digest():
            raise CostBlocked("approval does not match this exact request; re-quote")
        if approval.cost != 0:
            raise CostBlocked("approval cost is not zero")
        return self.client.timeseries.get_range(**req.kwargs())

    def fetch(self, req: DatabentoRequest, path: Path, save: Callable[[Any, Path], int],
              retries: int = 3) -> tuple[str, int]:
        """Cached -> ('cached', 0). Else quote + download + save -> ('ok', rows).

        Each attempt re-quotes. CostBlocked is never retried.
        """
        if Path(path).exists():
            return "cached", 0
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            approval = self.quote(req)              # raises CostBlocked -> stop
            try:
                store = self.download(req, approval)
            except CostBlocked:
                raise
            except Exception as e:
                last = e
                self._log(req, "download_error", cost=approval.cost,
                          message=f"attempt {attempt}: {type(e).__name__}: {e}"[:300])
                continue
            rows = save(store, Path(path))
            self._log(req, "downloaded", cost=approval.cost, message=f"rows={rows}")
            return "ok", rows
        raise RuntimeError(f"download failed after {retries} attempts: {last}")
