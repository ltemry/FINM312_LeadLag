"""Quota-aware usage budget for Dwellir's free plan.

Dwellir bills candle-range pages per returned candle (X-Iris-Billing-Weight),
the free plan allows 100,000 responses per day and offers no overages.

Rules implemented here (docs/project_plan.md, "Dwellir free-usage guard"):
* A persistent, append-only ledger (JSONL) records a `reserve` line before
  every request and a `settle` line after it.
* Before a request we reserve a conservative upper bound (page limit capped by
  the interval buckets left in the range). If used + bound would exceed the
  usable allowance (allowance x (1 - headroom) - other usage today), we stop.
* A reservation with no settle line (crash, lost response) counts in full.
* A request whose response never arrived is settled at its full reservation,
  because the server may still have billed it.
* Days are bucketed in the configured quota-reset timezone.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo


class QuotaExhausted(Exception):
    """Stop: the next request could exceed the usable daily allowance (or it is uncertain)."""


class EntitlementNotVerified(Exception):
    """Bulk retrieval refused until `dwellir.entitlement_verified: true` in config."""


INTERVAL_MS = {"1s": 1_000, "1m": 60_000, "5m": 300_000}


def page_upper_bound(start_ms: int, end_ms: int, interval: str, limit: int) -> int:
    """Max candles a page can return: page limit, capped by buckets in [start, end)."""
    buckets = max(0, math.ceil((end_ms - start_ms) / INTERVAL_MS[interval]))
    return min(int(limit), buckets)


class UsageLedger:
    def __init__(self, path: Path, tz: str = "UTC", now: Callable[[], dt.datetime] | None = None):
        self.path = Path(path)
        self.tz = ZoneInfo(tz)
        self._now = now or (lambda: dt.datetime.now(dt.timezone.utc))

    def today(self) -> str:
        return self._now().astimezone(self.tz).date().isoformat()

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _records(self):
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def used(self, day: str | None = None) -> int:
        """Credits charged on `day`: settled amounts, plus full reservations never settled."""
        day = day or self.today()
        reserved: dict[str, int] = {}
        settled: dict[str, int] = {}
        for r in self._records():
            if r.get("day") != day:
                continue
            if r["kind"] == "reserve":
                reserved[r["id"]] = int(r["credits"])
            elif r["kind"] == "settle":
                settled[r["id"]] = int(r["credits"])
        return sum(settled.get(i, c) for i, c in reserved.items())

    def reserve(self, credits: int, meta: dict) -> tuple[str, str]:
        rid = uuid.uuid4().hex[:12]
        day = self.today()
        self._append({"kind": "reserve", "id": rid, "day": day, "credits": int(credits),
                      "ts_utc": self._now().isoformat(timespec="seconds"), **meta})
        return rid, day

    def settle(self, rid: str, day: str, credits: int, status: str) -> None:
        self._append({"kind": "settle", "id": rid, "day": day, "credits": int(credits),
                      "status": status, "ts_utc": self._now().isoformat(timespec="seconds")})


class Budget:
    def __init__(self, ledger: UsageLedger, daily_allowance: int, headroom_fraction: float = 0.2,
                 other_usage_today: int = 0):
        if not 0 <= headroom_fraction < 1:
            raise ValueError("headroom_fraction must be in [0, 1)")
        self.ledger = ledger
        self.daily_allowance = int(daily_allowance)
        self.headroom_fraction = float(headroom_fraction)
        self.other_usage_today = int(other_usage_today)

    @property
    def usable(self) -> int:
        return math.floor(self.daily_allowance * (1 - self.headroom_fraction)) - self.other_usage_today

    def remaining(self) -> int:
        return self.usable - self.ledger.used()

    def reserve(self, upper_bound: int, meta: dict) -> tuple[str, str]:
        if upper_bound < 0:
            raise ValueError("upper_bound must be >= 0")
        rem = self.remaining()
        if upper_bound > rem:
            raise QuotaExhausted(
                f"next page could use {upper_bound} credits but only {rem} remain today "
                f"(usable {self.usable} of {self.daily_allowance}); resume after the "
                f"{self.ledger.tz} day resets")
        return self.ledger.reserve(upper_bound, meta)

    def settle(self, rid: str, day: str, credits: int, status: str) -> None:
        self.ledger.settle(rid, day, credits, status)
