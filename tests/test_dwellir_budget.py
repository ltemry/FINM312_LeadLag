"""Mocked tests for the Dwellir quota guard and client (no network)."""
import datetime as dt
import json

import pytest
import requests

from leadlag.collect_dwellir import DwellirClient, DwellirCollector, day_bounds_ms
from leadlag.config import Config, Task
from leadlag.dwellir_budget import Budget, QuotaExhausted, UsageLedger, page_upper_bound
from leadlag.storage import Manifest

KEY = "SECRET-KEY-123"
DAY = dt.date(2026, 11, 13)


class Resp:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text

    def json(self):
        return self._body


def candle(t, interval_ms=60_000):
    return {"s": "xyz:TSLA", "i": "1m", "t": t, "T": t + interval_ms - 1, "o": "1", "h": "1",
            "l": "1", "c": "1", "v": "1", "q": "1", "n": 1, "x": True}


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock(dt.datetime(2026, 11, 15, 12, tzinfo=dt.timezone.utc))


def make(tmp_path, responses, clock, allowance=100_000, headroom=0.2, other=0):
    ledger = UsageLedger(tmp_path / "usage.jsonl", "UTC", now=clock)
    budget = Budget(ledger, allowance, headroom, other)
    sess = FakeSession(responses)
    client = DwellirClient("https://example.test", KEY, budget, max_rps=0, session=sess,
                           sleep=lambda s: None)
    return client, budget, sess, ledger


def one_day_page(n=3, more=False, cursor=None):
    start, _ = day_bounds_ms(DAY)
    body = {"data": [candle(start + i * 60_000) for i in range(n)], "hasMore": more}
    if cursor:
        body["nextCursor"] = cursor
    return Resp(200, body, {"X-Iris-Billing-Weight": str(n)})


def test_page_upper_bound():
    assert page_upper_bound(0, 86_400_000, "1m", 5000) == 1440
    assert page_upper_bound(0, 86_400_000, "1s", 5000) == 5000
    assert page_upper_bound(0, 90_000, "1m", 5000) == 2   # partial bucket rounds up
    assert page_upper_bound(10, 10, "1m", 5000) == 0


def test_quota_exhaustion_stops_before_network(tmp_path, clock):
    client, budget, sess, ledger = make(tmp_path, [one_day_page()], clock, allowance=10_000)
    ledger.reserve(7_000, {})            # unsettled -> counts in full; usable = 8000
    start, end = day_bounds_ms(DAY)
    with pytest.raises(QuotaExhausted):
        list(client.candle_pages("xyz:TSLA", "1m", start, end))   # needs 1440 > 1000
    assert sess.calls == []


def test_insufficient_page_budget_counts_other_usage(tmp_path, clock):
    client, budget, sess, _ = make(tmp_path, [one_day_page()], clock, allowance=2_000, other=500)
    assert budget.usable == 1100
    start, end = day_bounds_ms(DAY)
    with pytest.raises(QuotaExhausted):
        list(client.candle_pages("xyz:TSLA", "1m", start, end))
    assert sess.calls == []


def test_success_settles_actual_billing(tmp_path, clock):
    client, budget, sess, ledger = make(tmp_path, [one_day_page(3)], clock)
    start, end = day_bounds_ms(DAY)
    pages = list(client.candle_pages("xyz:TSLA", "1m", start, end))
    assert len(pages) == 1 and pages[0]["_billed"] == 3
    assert ledger.used() == 3            # reservation of 1440 replaced by actual 3


def test_billing_never_below_returned_rows(tmp_path, clock):
    r = one_day_page(4)
    r.headers = {"X-Iris-Billing-Weight": "1"}
    client, _, _, ledger = make(tmp_path, [r], clock)
    start, end = day_bounds_ms(DAY)
    list(client.candle_pages("xyz:TSLA", "1m", start, end))
    assert ledger.used() == 4


def test_retry_after_lost_response_charges_full_reservation(tmp_path, clock):
    client, _, sess, ledger = make(tmp_path, [requests.ConnectionError("boom"), one_day_page(3)], clock)
    start, end = day_bounds_ms(DAY)
    list(client.candle_pages("xyz:TSLA", "1m", start, end))
    assert len(sess.calls) == 2
    assert ledger.used() == 1440 + 3     # lost attempt charged at its full upper bound


def test_retries_stop_when_budget_runs_out(tmp_path, clock):
    client, _, sess, _ = make(tmp_path, [requests.Timeout("t1"), one_day_page(3)], clock,
                              allowance=2_500)  # usable 2000: one lost 1440 leaves 560 < 1440
    start, end = day_bounds_ms(DAY)
    with pytest.raises(QuotaExhausted):
        list(client.candle_pages("xyz:TSLA", "1m", start, end))
    assert len(sess.calls) == 1


def test_crash_without_settle_counts_in_full(tmp_path, clock):
    ledger = UsageLedger(tmp_path / "usage.jsonl", "UTC", now=clock)
    ledger.reserve(1440, {})
    rid, day = ledger.reserve(1440, {})
    ledger.settle(rid, day, 10, "ok")
    assert ledger.used() == 1450


def test_usage_resets_by_quota_day(tmp_path, clock):
    ledger = UsageLedger(tmp_path / "usage.jsonl", "UTC", now=clock)
    rid, day = ledger.reserve(5000, {})
    ledger.settle(rid, day, 5000, "ok")
    clock.t += dt.timedelta(days=1)
    assert ledger.used() == 0


def test_429_backoff_then_success_and_repeated_429_stops(tmp_path, clock):
    client, _, sess, ledger = make(tmp_path, [Resp(429), one_day_page(2)], clock)
    start, end = day_bounds_ms(DAY)
    list(client.candle_pages("xyz:TSLA", "1m", start, end))
    assert ledger.used() == 2
    client2, _, _, _ = make(tmp_path / "b", [Resp(429)] * 10, clock)
    with pytest.raises(QuotaExhausted):
        list(client2.candle_pages("xyz:TSLA", "1m", start, end))


def test_pagination_follows_cursor(tmp_path, clock):
    start, end = day_bounds_ms(DAY)
    p1 = one_day_page(2, more=True, cursor=str(start + 120_000))
    p2 = Resp(200, {"data": [candle(start + 120_000)], "hasMore": False}, {"X-Iris-Billing-Weight": "1"})
    client, _, sess, ledger = make(tmp_path, [p1, p2], clock)
    pages = list(client.candle_pages("xyz:TSLA", "1m", start, end, limit=2))
    assert len(pages) == 2
    assert "start" in sess.calls[0][1] and sess.calls[1][1]["cursor"] == str(start + 120_000)
    assert ledger.used() == 3


def test_api_key_never_logged_or_stored(tmp_path, clock, caplog):
    err = requests.ConnectionError(f"failed https://example.test/{KEY}/v1/candles/range")
    client, _, _, ledger = make(tmp_path, [err, err, err, Resp(500, text=f"bad {KEY}")], clock)
    start, end = day_bounds_ms(DAY)
    with pytest.raises(Exception) as ei:
        list(client.candle_pages("xyz:TSLA", "1m", start, end))
    assert KEY not in str(ei.value)
    assert KEY not in caplog.text
    assert KEY not in ledger.path.read_text()


def _collector(tmp_path, monkeypatch, verified, responses, clock):
    cfg = Config()
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    cfg.raw["dwellir"] = dict(cfg.raw["dwellir"], entitlement_verified=verified)
    client, _, sess, ledger = make(tmp_path, responses, clock)
    col = DwellirCollector(cfg, Manifest(tmp_path), client=client)
    task = Task("dwellir", "TSLA", DAY, cfg.instruments["TSLA"]["perp"])
    return col, task, sess


def test_bulk_blocked_until_entitlement_verified(tmp_path, monkeypatch, clock):
    col, task, sess = _collector(tmp_path, monkeypatch, False, [one_day_page()], clock)
    col.run(task)
    assert sess.calls == []
    assert "blocked" in (tmp_path / "manifest.csv").read_text()


def test_collect_writes_then_skips_cached(tmp_path, monkeypatch, clock):
    col, task, sess = _collector(tmp_path, monkeypatch, True, [one_day_page(3)], clock)
    col.run(task)
    col.run(task)                         # second run: cached, no request
    assert len(sess.calls) == 1
    import pandas as pd
    df = pd.read_parquet(tmp_path / "raw/dwellir/candles_1m/xyz_TSLA/2026-11-13.parquet")
    assert len(df) == 3 and (df.bar_end_utc - df.open_ts_utc == pd.Timedelta("1min")).all()
    assert set(df.price_type) == {"trade"}
