"""Mocked safety tests for the Databento zero-cost guard (no network)."""
import dataclasses
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from leadlag.cost_guard import (Approval, CostBlocked, CostGuard, DatabentoRequest,
                                InvalidRequest, parse_zero_cost)

SRC = Path(__file__).resolve().parents[1] / "src" / "leadlag"


class FakeClient:
    def __init__(self, costs, download_failures=0):
        self.costs = list(costs) if isinstance(costs, list) else None
        self.cost = costs
        self.quote_calls = []
        self.download_calls = []
        self.download_failures = download_failures
        outer = self

        class Meta:
            def get_cost(self, **kw):
                outer.quote_calls.append(kw)
                c = outer.costs.pop(0) if outer.costs is not None else outer.cost
                if isinstance(c, Exception):
                    raise c
                return c

        class TS:
            def get_range(self, **kw):
                outer.download_calls.append(kw)
                if outer.download_failures > 0:
                    outer.download_failures -= 1
                    raise ConnectionError("network blip")
                return "STORE"

        self.metadata, self.timeseries = Meta(), TS()


def req(**over):
    base = dict(dataset="GLBX.MDP3", symbols=("ESZ6",), schema="bbo-1s", stype_in="raw_symbol",
                start="2026-11-13T00:00:00+00:00", end="2026-11-14T00:00:00+00:00")
    base.update(over)
    return DatabentoRequest(**base)


def save(store, path):
    path.write_text(store)
    return 7


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "ledger.jsonl"


def test_zero_allows_exactly_one_matching_request(tmp_path, ledger):
    c = FakeClient(0)
    g = CostGuard(c, ledger)
    status, rows = g.fetch(req(), tmp_path / "out.parquet", save)
    assert (status, rows) == ("ok", 7)
    assert len(c.quote_calls) == 1 and len(c.download_calls) == 1
    assert c.quote_calls[0] == c.download_calls[0] == req().kwargs()


@pytest.mark.parametrize("cost", [0.0, "0", "0.000", Decimal("0"), 0])
def test_zero_forms_accepted(cost):
    assert parse_zero_cost(cost) == 0


@pytest.mark.parametrize("cost", [0.01, 1e-12, "0.0000001", -0.0001, float("nan"), float("inf"),
                                  float("-inf"), None, "", "free", True, False, [0], {"cost": 0},
                                  "NaN", "Infinity"])
def test_every_nonzero_or_nonfinite_quote_blocks_without_download(tmp_path, ledger, cost):
    c = FakeClient([cost])  # one-element quote sequence
    with pytest.raises(CostBlocked):
        CostGuard(c, ledger).fetch(req(), tmp_path / "out.parquet", save)
    assert c.download_calls == []
    assert not (tmp_path / "out.parquet").exists()
    assert json.loads(ledger.read_text().splitlines()[-1])["outcome"] == "block"


def test_quote_exception_blocks_without_download(tmp_path, ledger):
    c = FakeClient(TimeoutError("quote service down"))
    with pytest.raises(CostBlocked):
        CostGuard(c, ledger).fetch(req(), tmp_path / "out.parquet", save)
    assert c.download_calls == []


def test_mutated_request_invalidates_approval(ledger):
    c = FakeClient(0)
    g = CostGuard(c, ledger)
    approval = g.quote(req())
    for changed in (req(symbols=("ESH7",)), req(schema="mbp-1"), req(end="2026-11-15T00:00:00+00:00"),
                    req(limit=10)):
        with pytest.raises(CostBlocked):
            g.download(changed, approval)
    with pytest.raises(CostBlocked):
        g.download(req(), Approval("not-the-digest", Decimal(0), "x"))
    with pytest.raises(CostBlocked):
        g.download(req(), None)
    assert c.download_calls == []
    g.download(req(), approval)  # the identical request is fine
    assert len(c.download_calls) == 1


def test_retries_requote_every_attempt(tmp_path, ledger):
    c = FakeClient(0, download_failures=2)
    status, _ = CostGuard(c, ledger).fetch(req(), tmp_path / "out.parquet", save, retries=3)
    assert status == "ok"
    assert len(c.quote_calls) == 3 and len(c.download_calls) == 3


def test_retry_stops_if_requote_becomes_nonzero(tmp_path, ledger):
    c = FakeClient([0, 0.5], download_failures=1)
    with pytest.raises(CostBlocked):
        CostGuard(c, ledger).fetch(req(), tmp_path / "out.parquet", save, retries=3)
    assert len(c.download_calls) == 1  # second attempt blocked before downloading


def test_cached_file_skips_quote_and_download(tmp_path, ledger):
    out = tmp_path / "out.parquet"
    out.write_text("already here")
    c = FakeClient(0)
    assert CostGuard(c, ledger).fetch(req(), out, save) == ("cached", 0)
    assert c.quote_calls == [] and c.download_calls == []


@pytest.mark.parametrize("bad", [
    dict(symbols=("ALL_SYMBOLS",)), dict(symbols=("all_symbols",)), dict(symbols=()),
    dict(symbols=["ESZ6"]), dict(stype_in="parent"), dict(stype_in="continuous"),
    dict(start=""), dict(end=None), dict(start="2026-11-13T00:00:00"),
    dict(end="2026-11-12T00:00:00+00:00"), dict(limit=0),
])
def test_invalid_requests_rejected_before_quote(ledger, bad):
    c = FakeClient(0)
    with pytest.raises(InvalidRequest):
        CostGuard(c, ledger).quote(req(**bad))
    assert c.quote_calls == []


def test_request_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        req().symbols = ("ESH7",)


def test_only_cost_guard_calls_get_range():
    offenders = [p.name for p in SRC.glob("*.py")
                 if p.name != "cost_guard.py" and re.search(r"get_range\s*\(", p.read_text())]
    assert offenders == []


def test_collector_routes_through_guard(tmp_path, monkeypatch):
    """DatabentoCollector with a nonzero quote must not download."""
    import datetime as dt

    from leadlag.collect_databento import DatabentoCollector
    from leadlag.config import Config, Task
    from leadlag.storage import Manifest

    cfg = Config()
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    c = FakeClient(2.5)
    col = DatabentoCollector(cfg, Manifest(tmp_path), client=c)
    col.run(Task("databento", "ES", dt.date(2026, 11, 13), cfg.instruments["ES"]["futures"]))
    assert c.download_calls == []
    assert "blocked" in (tmp_path / "manifest.csv").read_text()
