# FINM312 Lead-Lag

Do Hyperliquid equity perpetuals lead or lag the underlying stocks (TSLA, NVDA),
the ANT IPO, and ES futures? This repo starts with the **data pipeline**. Cleaning,
alignment, lead-lag regressions, and event studies come next.

## Layout

```
config/config.yaml          instruments, contracts, study dates, source settings
config/events.csv           verified event timeline (fill in by hand)
src/leadlag/
  config.py                 turns studies + dates into (source, instrument, day) tasks
  market_calendar.py        NYSE trading days
  storage.py                Parquet layout, skip-if-done, retries, manifest
  collect_taq.py            WRDS Daily TAQ NBBO + trades (sampled on WRDS)
  collect_hyperliquid.py    Hyperliquid candles, funding, metadata, optional S3 L2 book
  hl_live.py                live Hyperliquid recorder (bbo, trades, mark/oracle)
  collect_databento.py      CME ES via Databento (cost-checked)
  cli.py                    `leadlag` command
data/                       downloaded data (git-ignored)
```

Coming next: `clean_align.py`, `lead_lag.py`, `event_study.py`, `results.ipynb`.

## Setup

```bash
uv venv .venv && uv pip install -e ".[s3]"    # or: python -m venv .venv && pip install -e ".[s3]"
source .venv/bin/activate
cp .env.example .env                           # add DATABENTO_API_KEY (and WRDS_USERNAME)
```

WRDS uses `~/.pgpass`. The first connection may prompt for your password and
send a Duo push.

## Choosing dates

Each part of the project is a **study** in `config/config.yaml`:

| Study | Instruments | Sources |
|---|---|---|
| `pilot` | TSLA | taq, hyperliquid |
| `baseline` | TSLA, NVDA, SP500 (for the ES control) | taq, hyperliquid, databento |
| `events` | TSLA, NVDA | taq, hyperliquid |
| `ant_ipo` | ANT | taq, hyperliquid |
| `sp_es` | SP500 | hyperliquid, databento |

Give each study either `dates: [...]` or `start:` / `end:` (inclusive), plus
optional `exclude_dates`. The pipeline then picks valid days for each source:

- **TAQ:** NYSE trading days only.
- **Hyperliquid:** every calendar day (the perps trade 24/7, so weekends are kept for closure analysis).
- **Databento:** every day except Saturday.

You can override dates from the command line without editing the config:

```bash
leadlag plan    --study baseline                                  # preview tasks
leadlag collect --study pilot                                     # first milestone
leadlag collect --study baseline --sources hyperliquid --start 2026-11-10 --end 2026-11-13
leadlag collect --study events --dates 2026-10-22 2026-10-23
leadlag collect --study baseline --sources taq --dry-run          # print the SQL, download nothing
leadlag collect --study sp_es --sources databento --dry-run       # print Databento cost estimates
leadlag coverage                                                  # what's on disk
```

Finished files are skipped on re-runs; use `--force` to re-download.
Every attempt is logged to `data/manifest.csv` with one of these statuses:
`ok`, `empty`, `unavailable`, `incomplete`, `skipped`, or `error`.

## Output

`data/raw/<source>/<dataset>/<instrument>/<YYYY-MM-DD>.parquet`, all timestamps tz-aware UTC.

| Source | Dataset | Contents |
|---|---|---|
| taq | `nbbo_1s` | Last valid NBBO in each 1s bucket: `bucket_end_utc`, `quote_ts_utc` (for quote age), bid/ask/sizes, mid |
| taq | `trades_1s` | Per-bucket trade count, volume, VWAP, last price (`tr_corr='00'` only) |
| hyperliquid | `candles_{1m,5m,15m,1h}` | Trade-based OHLCV bars. Minutes with no trades are absent |
| hyperliquid | `funding` | Hourly funding rate and premium |
| hyperliquid | `meta/<dex>/<ts>.json` | Contract specs and mark/oracle snapshot from each run (tracks rule changes) |
| hyperliquid | `live_bbo`, `live_trades`, `live_activeAssetCtx` | From `leadlag hl-record`: quotes, trades, and mark/oracle (kept separate from quotes) |
| databento | `bbo_1s` | ES best bid/offer each second (schema set in config) |

Set `taq.sample_seconds: 0` to download every NBBO update and trade, without sampling (for validating the pilot day).

## Data limitations

- **Hyperliquid's API serves only the last 5,000 candles per interval:** about 3.5 days of 1m bars, 17 days of 5m, 52 days of 15m, and 208 days of 1h. Older days get only the coarser bars, and the manifest marks the rest `unavailable`.
- **Historical perp quotes and mark/oracle series are not available from the API.** For sample periods that are still ahead, keep the live recorder running:
  ```bash
  leadlag hl-record --instruments TSLA NVDA ANT SP500
  ```
  Also run `leadlag collect` daily to archive 1m candles before they roll off.
- **Past dates fall back to trade-based perp prices** (1m candle closes) rather than quote midpoints. Document this as a limitation of the analysis.
- **The S3 archive is optional:** set `hyperliquid.s3_l2book.enabled: true` to try the requester-pays archive (`s3://hyperliquid-archive`). It needs AWS credentials and you pay transfer costs. It is updated roughly monthly, and its HIP-3 coverage is undocumented.
- **Most alternate deployers' TSLA, NVDA, and S&P perps are delisted.** `xyz` (trade.xyz) is the primary venue. Run `leadlag hl-contracts --match TSLA NVDA ANT 500` to re-check.
- **WRDS Daily TAQ lags real time**, so very recent days show `unavailable` until WRDS loads them.
