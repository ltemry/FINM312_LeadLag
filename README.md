# FINM312 Lead-Lag

Do US equity perpetuals (Hyperliquid HIP-3: `xyz:NVDA`, `xyz:TSLA`, `xyz:SP500`)
lead or lag their US exchange prices, and does that change during major news?
The full specification is [docs/project_plan.md](docs/project_plan.md).

**Status:** Stages 1–3 of the plan are built: config and guards, collectors, and
normalize/align. Next up are Stage 4 (`lead_lag.py`) and Stage 5 (`auctions.py`, `overnight.py`).
Anthropic (ANT) is deferred: it is marked `deferred: true` in the config, and every collector refuses it.

## Layout

```
config/config.yaml          instruments, studies (scored + retrieval dates), source settings
config/events.csv           verified event timeline (verify before labelling observations)
docs/project_plan.md        implementation plan this code follows
src/leadlag/
  config.py                 studies -> (source, instrument, day) tasks; ET -> UTC day mapping
  market_calendar.py        NYSE holidays, early closes, session times (DST-aware)
  storage.py                Parquet layout, skip-if-done, retries, manifest
  cost_guard.py             Databento USD 0 guard (the only code that downloads from Databento)
  dwellir_budget.py         Dwellir free-plan quota ledger and reservations
  collect_taq.py            WRDS TAQ: 1s NBBO, 1s trade buckets, auction/official prints
  collect_dwellir.py        Dwellir historical Hyperliquid candles (1s/1m/5m), quota-guarded
  collect_hyperliquid.py    Hyperliquid public API: contract metadata snapshots, funding
  collect_databento.py      CME futures through the cost guard
  coverage_report.py        free coverage checks + request manifest
  normalize.py              common schema: ts_utc (available), source_ts_utc, price, price_type
  align.py                  session grids, as-of joins, quote age, session-safe returns
  hl_live.py                optional forward collector (not running by default)
  cli.py                    `leadlag` command
tests/                      mocked guard tests + alignment/calendar tests (pytest)
data/                       downloads, ledgers, reports (git-ignored)
```

## Setup

```bash
uv venv .venv && uv pip install -e ".[dev]"
source .venv/bin/activate
cp .env.example .env        # DWELLIR_API_KEY, DATABENTO_API_KEY, WRDS_USERNAME
pytest                      # 66 tests, no network
```

## Studies and dates

Dates are US/Eastern. Each study has `scored` dates, which count in the analysis,
and `retrieval` dates, which are downloaded and include buffers.

| Study | Scored | Retrieval | Instruments |
|---|---|---|---|
| `pilot` | Nov 13 | Nov 13 | TSLA |
| `nvda_baseline` | Sep 22–24 | Sep 21–24 | NVDA, ES |
| `nvda_comparison` | Nov 3–5 | Nov 2–5 | NVDA, ES |
| `nvda_crisis` | Nov 9–13 (focus Nov 11–12) | Nov 6–13 | NVDA, ES |
| `panel` | Sep 21–Oct 16, Nov 2–13 (30 sessions) | Sep 18–Oct 16, Oct 30–Nov 13 | TSLA, SP500, ES |

Each source gets the days it needs from the retrieval range:
- **TAQ:** NYSE trading days only.
- **Dwellir and Hyperliquid:** every UTC day that overlaps the ET range, so overnights and weekends are kept.
- **Databento:** the same UTC days, minus Saturdays.

## Workflow

```bash
# Stage 1: inspect (free)
leadlag plan --study panel                                  # tasks + max Dwellir credits
leadlag coverage-report --study panel --check-wrds          # request manifest CSV in data/reports/
leadlag dwellir-probe --instrument TSLA --date 2026-11-13   # <= 5 candles: tests your key/plan
leadlag dwellir-usage
leadlag databento-resolve --symbols ESZ6

# Stage 2: collect (re-runs skip finished files; see data/manifest.csv)
leadlag collect --study pilot
leadlag collect --study panel --sources taq hyperliquid
leadlag collect --study panel --sources dwellir             # needs entitlement_verified: true
leadlag collect --study panel --sources databento --dry-run # quotes only

# Stage 3: align (1-minute grid, regular session by default)
leadlag align --study pilot --instrument TSLA               # perp candles vs TAQ trade bars
leadlag align --study pilot --instrument TSLA --ref-type mid
leadlag align --study panel --instrument SP500 --ref ES
```

## Spending guards

**Databento: authorized spend is exactly USD 0.**
- Every request is an immutable `DatabentoRequest`: explicit symbols, `raw_symbol`/`instrument_id` only, and a finite `[start, end)`. `ALL_SYMBOLS` and parent or continuous symbology are rejected.
- Each request is quoted with `metadata.get_cost` immediately before download. Only a finite quote exactly equal to 0 is allowed; positive, negative, NaN, missing, or failed quotes all block.
- An approval is tied to the request's hash, so any change to the request needs a new quote. Every retry re-quotes, and cached files skip the network entirely.
- All quotes and outcomes go to `data/ledger/databento_requests.jsonl`.
- There is no config option to allow a cost.

**Dwellir: free plan only.** The free plan has 100k responses/day and no overages; candle ranges bill per returned candle.
- Bulk collection is refused until you set `dwellir.entitlement_verified: true`. Do that only after confirming on your account that the free plan includes the candle-range endpoint and has no paid-overage route.
- Before every page, a conservative upper bound is reserved in `data/ledger/dwellir_usage.jsonl`. The run stops if it would exceed 80% of the daily allowance minus `other_usage_today`.
- A request whose response never arrives is charged in full. Repeated 429s stop the run rather than switching plans or endpoints.
- The API key is part of Dwellir's URL, so URLs are never logged and error text is redacted.

**Hyperliquid:** only the free public API is used. The requester-pays S3 archive is not used.

## Price types and timing

- Every normalized series records `price_type` (`mid` or `trade`) and `source`. They are never silently mixed.
- `ts_utc` is when an observation is available. A candle's close is usable at its bar end, not its start.
- `source_ts_utc` is the underlying quote or trade time. It is NaT for candles, because a bar's timestamp is not the time of its last trade.
- Alignment uses backward as-of joins only. Quotes older than `--max-age` (default 60s) become NaN, with their age kept. Trade bars must end exactly at the grid point, so a missing bar stays missing.
- Returns are only computed between consecutive grid points in the same session.
- Historical Hyperliquid quotes aren't available yet, so the first version is trade-based. Dwellir 1m candles are compared with TAQ 1m last-trade bars built the same way. TAQ midpoints are kept for robustness (`--ref-type mid`).

## Known data gaps (checked 2026-11-15)

- **WRDS has no `complete_nbbo` table for 2026-11-11 yet** (raw `cqm`/`nbbom` exist; `wct` is also missing). NBBO for that day is recorded as `unavailable`, but trades and auction prints are still collected. Re-run `leadlag collect` once WRDS builds the table.
- **TSLA single-stock futures:** the Databento symbol is unresolved (`symbol: null`). Use `databento-resolve`, then pin one expiry.
- **ES:** fixed at `ESZ6` for all windows. ESU6 expired on 2026-09-18.
