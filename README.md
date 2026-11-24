# FINM312 Lead-Lag

Do US equity perpetuals (Hyperliquid HIP-3: `xyz:NVDA`, `xyz:TSLA`, `xyz:SP500`)
lead or lag their US exchange prices, and does that change during major news?
The full specification is [docs/project_plan.md](docs/project_plan.md).

**Status:** all seven stages of the plan are built and run on the full sample. Results, figures
and conclusions are in [`results.ipynb`](results.ipynb).
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
  align.py                  session grids, as-of joins, quote age, session-safe returns, ES control
  lead_lag.py               lead-lag correlations, predictive regressions, holdout, bootstrap
  auctions.py               official primary-exchange open/close from TAQ auction prints
  overnight.py              opening/closing forecasts vs benchmarks, post-open adjustment, paths
  auction_windows.py        opening/closing boundary windows: in-window lead-lag, event profiles
  robustness.py             fixed robustness grid (midpoints, quote age, excl. open/close, 5-min)
  plots.py                  notebook figures
  hl_live.py                optional forward collector (not running by default)
  cli.py                    `leadlag` command
tests/                      mocked guard tests + alignment/calendar tests (pytest)
data/                       downloads, ledgers, reports (git-ignored)
```

## Run everything (doit)

```bash
cp .env.example .env        # WRDS_USERNAME, DATABENTO_API_KEY (Dwellir key optional, see below)
uv run doit                 # tests -> collect -> lead-lag -> significance -> overnight -> windows -> robustness -> notebook
```

`uv run` creates `.venv` from `uv.lock` (including `doit`, `pytest` and the notebook packages) and `dodo.py` runs every stage in order. Collection is always safe to repeat: finished files are skipped, so a second run downloads nothing. The analysis stages only re-run when the code, `config/config.yaml` or the raw data changed. Then it re-executes `results.ipynb` in place.

```bash
uv run doit list            # the stages
uv run doit collect=0       # analyse what is on disk; no WRDS/Databento connection
uv run doit lead_lag        # one stage (sub-tasks: lead_lag:TSLA_trade, windows:SP500, ...)
uv run doit forget && uv run doit   # force all stages to re-run
```

## Setup (manual)

```bash
uv venv .venv && uv pip install -e ".[dev]"
source .venv/bin/activate
cp .env.example .env        # DWELLIR_API_KEY, DATABENTO_API_KEY, WRDS_USERNAME
pytest                      # 103 tests, no network
```

## For teammates: running without a Dwellir key

The Dwellir perp candles for every study window are **committed** in `data/raw/dwellir/` (1-minute candles for `xyz:NVDA`, `xyz:TSLA` and `xyz:SP500`, Sep 18 – Nov 14, 2026). The collector skips any file that already exists, so after cloning you never call Dwellir's API. Leave `entitlement_verified: false` and no `DWELLIR_API_KEY` is needed.

The other sources are **not** committed and must be pulled with your own access:
- **TAQ** (WRDS-licensed): needs your WRDS account. Set `WRDS_USERNAME` in `.env`.
- **ES futures** (Databento): needs `DATABENTO_API_KEY`. Every request is quoted at $0 before downloading.

```bash
git clone https://github.com/ltemry/FINM312_LeadLag.git && cd FINM312_LeadLag
cp .env.example .env            # add WRDS_USERNAME and DATABENTO_API_KEY (no Dwellir key needed)
uv run doit                     # collect (Dwellir is skipped) + every analysis stage + notebook
```

Dwellir lines in the collect output should all say `skipped`. To run step by step instead, see the commands below.

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

# Stage 4: lead-lag (scored dates; results in data/results/<pair>/)
leadlag lead-lag --studies nvda_baseline nvda_comparison nvda_crisis --instrument NVDA
leadlag lead-lag --studies panel --instrument TSLA --holdout
leadlag lead-lag --studies panel --instrument SP500 --ref ES
leadlag lead-lag --studies panel --instrument TSLA --ref-type mid --exclude-open --exclude-close
```

## Opening/closing windows, robustness, and the results notebook (Stages 6–7)

```bash
leadlag windows --studies panel --instrument TSLA                 # 09:00-10:00 and 15:30-16:30 ET
leadlag windows --studies panel --instrument SP500 --ref ES
leadlag significance                                             # formal tests that the reference leads (data/results/significance/)
leadlag robustness                                               # writes data/results/robustness.csv
uv pip install -e ".[notebook]"                                  # matplotlib + Jupyter kernel
jupyter nbconvert --to notebook --execute --inplace results.ipynb   # or open it in Jupyter / VS Code
```

- **Windows** run from 30 minutes before to 30 minutes after each boundary. Pre and post sides are separate sessions, so no return or lag crosses the auction, and the auction print is a separate event.
- **Window outputs:** in-window lead-lag tests, minute-by-minute event profiles (moves, basis, spreads, quote ages, perp activity), and moves at +1/+5/+15 minutes.
- **The notebook only presents results;** it reads `data/results/` and needs the data to re-run. The committed copy includes its outputs.

## Auction and overnight forecasts (Stage 5)

```bash
leadlag auctions --study nvda_crisis --instrument NVDA          # official opens/closes, with timestamps
leadlag overnight --studies nvda_baseline nvda_comparison nvda_crisis --instrument NVDA --paths
leadlag overnight --studies panel --instrument TSLA --paths
```

- **Targets** are the official Nasdaq open/close records (`Q` / `M`) on the primary exchange; the cross prints (`O` / `6`) are the flagged fallback. Never first/last trades or bar OHLC.
- **Each target keeps its own timestamp** (the opening cross can print at 09:30:08). Every feature must be available at or before its cutoff, and the cutoff must be strictly before the target; violating rows are dropped and counted.
- **Opening:** `implied_open(t) = C × P(t) / P0` at 08:00, 09:00, 09:25 and 09:29 ET, compared with no change (C) and the TAQ premarket midpoint at the same cutoff.
- **Closing:** anchored at 15:00 ET, `implied_close(t) = A_cash × P(t) / A_perp` at 15:30, 15:50 and 15:59, compared with the contemporaneous cash midpoint and no change.
- **Metrics:** MAE, RMSE, bias and direction accuracy on matched dates only (all forecasts present), plus paired error differences. Missing or stale cases are counted separately. Fewer than 20 matched dates is flagged descriptive-only (each NVDA window).
- **Also written:** post-open adjustment at 1/5/15 minutes (cash return, perp return, basis change), and minute-by-minute forecast paths (`--paths`) for the plots in `results.ipynb`.
- **Price types:** perp prices are trade-based (Dwellir candle closes); cash benchmarks are NBBO midpoints. Outputs are in `data/results/overnight_<instrument>/`.

## Lead-lag analysis (Stage 4)

- **Correlation convention:** `rho(k) = Corr(perp_t, ref_{t+k})` for k = −5..+5 minutes. **Positive k means the perp moves first.**
- **No best-lag picking:** the headline statistic is the predefined asymmetry `A = Σρ(k>0) − Σρ(k<0)`, with a whole-session bootstrap CI. The peak lag is reported, but labelled exploratory.
- **Predictive regressions in both directions,** with p = 1..5 lags of each venue plus lagged ES returns (`--control ES`, the default for stocks). The test is a session-clustered Wald test that the other venue's lags add nothing beyond the target's own lags. `delta_r2` is the incremental R².
- **Lags never cross sessions or gaps.** `--exclude-open` / `--exclude-close` blank the first/last 15 regular-session minutes without bridging them.
- **`--holdout`** (panel): fits on Sep–Oct, evaluates on Nov, and compares against the own-lag baseline (`oos_r2_vs_base` plus a session-bootstrap CI of the loss difference).
- **Groups with fewer than 10 sessions** (each NVDA window) are flagged `descriptive_only`: report those descriptively, not as tests.
- **Outputs** go to `data/results/<pair>/`: `describe`, `xcorr`, `asymmetry`, `regressions`, `summary`, `holdout`, and `settings.json` (the exact options used).

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

- **TAQ trade buckets** exclude prints that don't set the last-sale price (late or out-of-sequence, average-price, odd-lot and similar). Form T extended-hours trades are kept.
- **WRDS has no `complete_nbbo` table for 2026-11-11 yet** (raw `cqm`/`nbbom` exist; `wct` is also missing). NBBO for that day is recorded as `unavailable`, but trades and auction prints are still collected. Re-run `leadlag collect` once WRDS builds the table.
- **TSLA single-stock futures:** the Databento symbol is unresolved (`symbol: null`). Use `databento-resolve`, then pin one expiry.
- **ES:** fixed at `ESZ6` for all windows. ESU6 expired on 2026-09-18.
