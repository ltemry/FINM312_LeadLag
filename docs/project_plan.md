# Equity perpetuals versus US exchanges: implementation plan
Prepared November 15, 2026. Planning specification only; no market data downloaded and no account entitlements checked.

## Objective and scope
Primary research question: Do US equity perpetuals lead or lag their corresponding US exchange prices, and does that relationship change during major news?

The core is simultaneous-trading lead–lag analysis for NVDA and TSLA versus their TAQ cash-equity prices, comparing the selected news and baseline windows and the broader sample. The S&P perpetual/ES comparison and eligible TSLA futures add cross-market evidence. They complement the cash-equity comparisons required by the assignment.

Opening-price prediction and opening/closing adjustment remain supporting extensions. They ask whether earlier perpetual moves anticipate cash auction prices and persist when cash trading resumes. Forecast accuracy or a price difference alone is not evidence of lead–lag. “Accurate” means smaller measured forecast errors against specified benchmarks, not a presumption that an exchange price is fundamental value.

Suggested emphasis: approximately two-thirds of the analysis and results discussion on direct lead–lag, and one-third on overnight forecasts and auction behavior. Complete the core results before expanding the extensions. The abstract and conclusion should answer which venue leads, at what sampled horizon, and under which conditions, including weak or inconclusive evidence.

Active: NVDA, TSLA, S&P 500 perpetual/ES. Deferred: all Anthropic collection and analysis until the user resumes it. No automatic future task.

## Corrections and instrument map
- NVIDIA's CEO-transition release is dated November 12, 2026. Its SEC filing says the board accepted the resignation November 11. The public information timestamp, not an internal decision date, defines an event study. November 3–5 is retained as the user's earlier comparison window; news chronology must be verified before labeling individual observations.
- September 22–24 is a candidate baseline, not an established quiet period. Report realized volatility, spreads, volume and relevant announcements before characterizing it.
- CME lists TSLA single-stock futures. Resolve actual Databento symbols, historical listing coverage, expiry, contract units and liquidity; do not assume the cash ticker is the futures symbol.
- For NVDA: perpetual versus TAQ cash equity. For TSLA: perpetual versus TAQ cash equity and, conditional on coverage and liquidity, CME single-stock futures. For S&P: perpetual versus ES futures; SPY via TAQ is optional for an actual ETF auction comparison.
- SPX is an index, not an individually auctioned security. ES observations at the equity open/close are cash-session boundary comparisons, not Nasdaq opening/closing auctions. SPY is a proxy with different units and tracking characteristics, not SPX itself.
- Futures levels include basis/carry. Compare returns and anchored implied cash prices; report raw level differences separately without calling all basis pricing error.

## Proposed date windows (2026; inclusive cash trading dates)
| Purpose | Scored dates | Retrieval buffer |
|---|---|---|
| NVDA candidate baseline | Sep 22–24 | From Sep 21 before the close through Sep 24 after the close |
| NVDA earlier comparison | Nov 3–5 | From Nov 2 before the close through Nov 5 after the close |
| NVDA main crisis case study | Nov 9–13; focus Nov 11–12 | From Nov 6 before the close, including the weekend, through Nov 13 after the close |
| Broader TSLA/S&P and optional NVDA panel | Sep 21–Oct 16 and Nov 2–13 | Previous trading-day close before each block; include intervening overnight periods |

The broader panel supplies roughly 30 cash sessions; coverage and calendars determine the exact count. Keep date selection fixed before inspecting lead–lag results. Use September/October for development and November as a chronological evaluation block where fitting is needed. The November regime differs, which is a limitation to report. Six or nine overnight outcomes alone cannot establish robust forecasting performance, regardless of the number of intraday rows.

All session logic uses America/New_York and an exchange calendar, then converts to UTC. September and November have different daylight-saving offsets. Do not hardcode a single UTC open.

## Module A: core analysis — simultaneous-trading lead–lag
Run the same specification for NVDA and TSLA perpetuals versus cash equity; compare crisis and baseline periods. Then apply it to S&P perpetual/ES and eligible TSLA futures.

Prefer synchronous one-minute midpoint log returns. If quote history is unavailable, use the explicit, consistently trade-based candle fallback below for the first version and report its limitations. Use 5-second/1-second data only where both sources support it.
- Correlation convention: corr(perp_return[t], reference_return[t+k]); positive k means the perpetual comes first. Predefine k from -5 to +5 minutes.
- Predictive regressions in both directions, initially with 1–5 lagged returns of each venue.
- Add lagged ES returns as a broad-market control in equity comparisons, using a common timestamp convention.
- Fit on development dates only; compare held-out forecast errors with an own-return baseline where forecasting is claimed.
- Separate regular sessions, extended overlaps and opening/closing windows. Never construct intraday lags across closures or missing stretches.
- Summarize dependence-aware uncertainty using whole-session resampling on the broader sample; treat tiny event samples descriptively. Avoid treating minute observations as independent overnight forecasts.
- Predictive evidence does not establish causality or microsecond reaction priority.

Core deliverables: lead–lag correlation panels, predictive regressions in both directions, event price paths, and a summary of direction/horizon/uncertainty by market and period. A correlation peak is exploratory; assess incremental prediction, timing quality and robustness before describing a market as leading.

## Module B: supporting extension — overnight and auction-price forecasts
Let C be the previous official cash closing price, P0 the perpetual midpoint immediately before that close, and P(t) the eligible perpetual midpoint at a forecast cutoff t. Main forecast:
    implied_open(t) = C * P(t) / P0
Target O is the next official primary-exchange opening price.
    actual_gap = O / C - 1
    predicted_gap(t) = P(t) / P0 - 1
    forecast_error_bps(t) = 10000 * (implied_open(t) / O - 1)

Anchor P0 strictly before the reference auction; C is known by the next morning. This is a transparent constant-basis forecast, not a full fair-value model. Report the raw perpetual-to-open premium as a separate descriptive measure.

Predefine opening cutoffs at 08:00, 09:00, 09:25 and 09:29 ET. Use only information available by each cutoff and document timestamp provenance. Report:
- Mean absolute error, RMSE, signed bias and gap-direction accuracy, including the sample count.
- Comparison with previous-close/no-change and latest eligible cash premarket midpoint at the same cutoff.
- Equivalent anchored futures forecasts for TSLA where valid futures quotes exist.
- Paired forecast-error comparisons on identical eligible dates. Display missing/stale cases separately.
- Daily forecast-path plots to the opening auction, followed by cash/perpetual adjustment over 1, 5 and 15 minutes.

Distinguish postmarket, overnight and premarket intervals. Cash-market activity outside regular hours means pre-open accuracy alone does not establish independent perpetual price discovery. Compare with available cash quotes and futures at matching cutoffs.

For closing forecasts use 15:30, 15:50 and 15:59 ET and the official cash closing auction target. Anchor implied cash forecasts at a common earlier intraday timestamp, e.g. 15:00 ET, using information available then; never use the target closing price as an input. Include the contemporaneous cash midpoint benchmark.

## Module C: supporting extension — opening/closing adjustment
For TSLA use the actual primary-exchange official open/close records from TAQ trade conditions and exchange identifiers. Do not substitute first/last consolidated trades or minute-bar OHLC. Retain official-price fallbacks where the exchange specifies them, flagging absence of an actual auction.

For S&P perpetual/ES, analyze 09:00–10:00 and 15:30–16:30 ET as equity-session boundary windows. Optional SPY auctions require TAQ SPY data; compare normalized returns or anchored implied SPY prices, not raw SPY and ES levels.

Apply the core lead–lag tests within sufficiently populated overlapping pre/post-auction windows, and use event paths to examine which venue adjusts first. Treat the auction print itself as a distinct event, not a continuous-trading quote. Static discrepancies and subsequent convergence are descriptive evidence, not standalone proof of leadership.

Plot normalized prices, basis changes, spreads, quote ages and volume. Quantify pre-event discrepancies and post-event changes at 1, 5 and 15 minutes. Do not mistake futures settlement, contract-expiration settlement or a cash boundary timestamp for the same event.

## Historical retrieval and optional live recording
Start with historical retrieval; no live collector is required to begin the first version. Historical Hyperliquid bid/ask coverage remains unresolved. The source descriptions below reflect documentation checked November 15, 2026, not authenticated tests or verified entitlements.

| Data | Intended source | Recording decision |
|---|---|---|
| Stock trades, quotes, official open/close | WRDS TAQ | Retrieve history, subject to current subscription/date coverage |
| ES and eligible TSLA futures | Databento | Retrieve only after the exact-request USD 0 check |
| Hyperliquid historical trade candles | Dwellir Hyperliquid Index | First source to test for the selected contracts and dates |
| Hyperliquid historical bid/ask quotes | A suitable archive, if available at no cost | Verify separately; candle access does not establish quote access |
| News timestamps and contract rules | Verified public sources | Maintain an event/metadata table with sources and verification notes |

### Dwellir historical candle route
Dwellir documents 1-second, 1-minute and 5-minute trade-derived OHLCV candles, HIP-3 symbols in deployer:ASSET form, archive history from July 27, 2025, and paginated ranges of up to 5,000 candles. An archive start date is not a guarantee for every contract: verify market existence, first/last observations, and completeness for each chosen window. Empty intervals are omitted.

Use the paginated candle-range API with a privately configured API key. Do not assume dashboard CSV/Parquet full-history exports are free: Dwellir describes those as requiring separate export credits. Do not log API keys or full authenticated URLs.

First retrieval steps:
1. Confirm free-plan access to the Index candle-range endpoint and account usage/overage controls.
2. Resolve exact NVDA, TSLA and S&P perpetual symbols; keep deployers separate.
3. Retrieve a bounded small sample from each intended September/November window, then run the matching one-day TSLA pilot.
4. Check UTC interval boundaries, finalized status, nonempty coverage, volume/trade counts, pagination completeness and duplicate bars.
5. Estimate total returned-candle usage, then retrieve/cache one-minute history. Reserve second-level requests for selected event windows.

### Dwellir free-usage guard
Published free-plan limits are 100,000 daily responses and 20 responses per second. Historical range billing counts one credit per returned candle, not one credit per HTTP call. Verify actual account rules and reset timing before bulk retrieval; do not assume this is Databento-style get_cost quoting.

For 30 calendar days of continuous trading, calculated maximum candle counts are:
| Scope | Maximum candle credits |
|---|---:|
| One market, one-minute bars | 43,200 |
| Three markets, one-minute bars | 129,600 |
| One market, one-second bars | 2,592,000 |

These are calendar-day illustrations, not estimates for 30 trading sessions or the exact configured sample. Sparse intervals reduce usage. Three markets at minute frequency could fit across two daily allowances if endpoint access is included and other account usage leaves sufficient quota.

Implement a quota-aware downloader alongside the Databento cost guard:
- Require verified free entitlement and a hard usage cap/no paid-overage route before bulk requests; otherwise stop.
- Compute a conservative upper bound per page from remaining interval buckets and the requested page limit; reserve it before requesting.
- Maintain a persistent account-wide usage ledger, account for other key activity where available, and leave headroom below the verified daily allowance.
- Limit request rate and page size under verified service semantics; returned-object billing does not mean a large page is necessarily one unit of rate-limit capacity.
- Stop on quota exhaustion or uncertain allowance. Back off on rate-limit errors without switching plans or endpoints.
- Cache completed pages/partitions, resume safely, and budget retries conservatively because a failed local receipt may still have consumed usage.
- Log sanitized request parameters, returned candle count, estimated credits and coverage. Never buy export credits, enable paid overages, or upgrade.
- Mock-test quota exhaustion, insufficient page budget, retry accounting and cached-request skipping.

### Quote-based preference and explicit candle fallback
Preferred analysis remains midpoint-versus-midpoint. Dwellir's queryable historical order-book product was labeled “coming soon” when checked; do not infer free historical quotes from its OHLCV endpoint. Search for a usable zero-cost quote archive separately. Hyperliquid's own historical archive may have gaps/delays and requester-pays transfers, so it is not automatically a zero-cost fallback.

If historical quotes are unavailable, explicitly select a trade-based first version: align one-minute last-trade prices from Dwellir with equivalently constructed TAQ/Databento trade bars. Record price_type and source per series; never silently mix a trade close with a midpoint. Keep TAQ quote/auction extracts for benchmarks and robustness.

Use completed bars only. A candle's close is usable at its interval end, not its start; verify each source's endpoint convention. A bar timestamp is not necessarily the timestamp of its final trade, so do not claim exact trade age from candles alone. Require activity in eligible intervals, flag missing bars, and test sensitivity to sparse trading. Do not fill missing bars as fresh observations or create returns/lags across missing stretches. Trade counts, volume and spread checks where available help assess apparent lead–lag caused by illiquidity or bid–ask bounce.

For opening forecasts under this fallback, use only the last completed eligible bar before each cutoff and label the result trade-based. Retain official cash auction targets. Report the possible within-bar staleness and avoid sub-minute leadership claims from minute candles.

### Optional forward collector
Add only if historical quotes are unavailable and a new forward sample would strengthen the project. This is a planned option, not a running process. It cannot reconstruct September or earlier November quotes, and must remain a separately labeled sample.

Use the public Hyperliquid WebSocket best-bid/ask or order-book stream, plus trades as needed. Save instrument/deployer, bid/ask prices and sizes, source timestamps, local UTC receipt timestamps, trade identifiers/prices/sizes, and disconnect/reconnect/gap records. Keep optional mark/oracle context separate. Persist incrementally; resume and flag gaps. Record feed cadence and timestamp limitations rather than treating receipt latency as market leadership. Full order-level reconstruction is unnecessary for the initial project.

Anthropic collection and analysis remain deferred; do not start an ANT collector or scheduled task without a new instruction.

## Data processing requirements
1. Free coverage/symbology inspection and a request manifest before market-data downloads.
2. Resolve Hyperliquid deployer, instrument, historical source, price units, oracle rules and any rule changes. Historical availability in September is a gating item. No paid archive, requester-pays transfer or vendor subscription is authorized.
3. Query TAQ by symbol/date and select essential fields. Keep auction-trade records separate from cleaned quote data. Use WRDS-side cleaning/sampling when possible, retaining provenance and filters.
4. Default to quote midpoints; retain last trades, mark prices and oracle prices as distinct fields. No silent substitutions.
5. Align using backward/as-of joins only. Store source timestamp and age at every sampled observation; never interpolate from future values.
6. Flag duplicates, cancellations/corrections, invalid bid/ask, crossed quotes, locked quotes, halts and trading-state anomalies under documented rules. Do not simply delete legitimate auction records through a continuous-trading filter.
7. Start with a 60-second maximum quote age at minute frequency; report sensitivity to 10/30/60 seconds. Fix final rules using coverage diagnostics, not which lead–lag result is strongest.
8. Compare all competing forecasts on matched eligible observations. Produce exclusions and coverage tables.
9. Resolve a fixed liquid futures expiry per window using prior available liquidity information; exclude ambiguous rolls from high-frequency event windows. No back-adjusted series for auction-level price forecasts.
10. Store raw extracts immutably and normalized/derived outputs in Parquet, partitioned by date/instrument. Keep API credentials outside files/notebooks.

## Databento zero-incremental-cost policy
The authorized additional data spend is exactly USD 0.
- Route every data request through one guarded downloader; no alternate unguarded download path.
- Build an immutable request with dataset, explicit resolved symbols, schema, stype_in, finite start/end and any record limit. Disallow ALL_SYMBOLS and open-ended ranges.
- Call metadata.get_cost for that identical request immediately before retrieval. It respects flat-rate discounts. Metadata and symbology calls are documented as free; definition/status data streams still need the guard.
- Accept only a successfully parsed, finite numeric cost exactly equal to zero. Never round a small positive amount to zero.
- Positive, missing, negative, NaN, infinite, stale or failed quotes block the download. Do not assume free credits, a subscription or a cached quote makes a different request free.
- Check every chunk, retry and changed schema separately. Cache completed files to avoid repeated retrieval.
- Log request parameters, hash, quote time, quoted cost and allow/block outcome. Dry-run mode lists the planned requests without downloading.
- Never activate a subscription, upgrade or paid fallback. A quote is a provider estimate, not an account-level spending cap; verify billing controls if available without changing paid plans.

Required mocked safety tests: zero allows one matching request; every nonzero/nonfinite/error case prevents network retrieval; request mutation invalidates approval; retries re-quote; cached files avoid retrieval; all download entry points use the guard.

## Implementation order and acceptance checks
| Stage | Modules/output | Completion criterion |
|---|---|---|
| 1 | config.yaml, coverage report, cost_guard.py, dwellir_budget.py | Contracts/dates resolved; Databento zero-cost and Dwellir free entitlement/quota controls verified or blocked explicitly |
| 2 | collect_taq.py, collect_dwellir.py, collect_databento.py; one-day pilot | Dwellir small-window coverage checks, then matching TSLA day; price type explicit; optional zero-cost futures sample |
| 3 | normalize.py, align.py | UTC/session mapping, no future joins and quote-age flags verified |
| 4 | lead_lag.py | Core NVDA/TSLA cash-equity comparisons, crisis/baseline results and robustness; S&P/ES and eligible TSLA futures comparisons |
| 5 | auctions.py, overnight.py | Official auction targets verified; opening forecast tables and benchmark errors |
| 6 | auction_windows.py | Supporting opening/closing adjustment plots and within-window lead–lag where feasible |
| 7 | results.ipynb | Reproducible tables, figures, sample counts and limitations |

Meaningful analysis tests: synthetic known-lag data verifies sign convention; future-dated quotes never enter features; September/November UTC conversions differ correctly; lags do not cross sessions; auction targets cannot enter their forecasts; missing/stale quotes yield missing forecasts rather than fabricated prices.

Final report order: research question and data/coverage; core lead–lag correlation panels and predictive regressions; crisis-versus-baseline comparison and robustness; supporting overnight forecasts and opening/closing adjustment; limitations and conclusion. Core results lead the presentation. Retain NVDA forecast paths, opening/closing error tables versus benchmarks, and TSLA/S&P boundary plots as extensions. Anthropic remains disabled.

Core completion criterion: the project can answer the assignment using NVDA/TSLA perpetual-versus-cash evidence even if an extension lacks usable data. Report null findings honestly. Missing futures or auction data must not prevent completion of the cash-equity lead–lag analysis.

## Sources checked
- NVIDIA CEO transition, Nov 12: https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition
- NVIDIA SEC filing (board action Nov 11): https://www.sec.gov/Archives/edgar/data/1045810/000104581026000082/nvda-20261111.htm
- NVIDIA statement chronology: https://nvidianews.nvidia.com/news/?c=21926
- CME single-stock futures specifications: https://www.cmegroup.com/markets/equities/single-stock-futures.quotes.html
- CME launch announcement: https://www.cmegroup.com/media-room/press-releases/2026/6/30/cme_group_to_launchsinglestockfuturesonjuly27.html
- Nasdaq opening/closing crosses: https://nasdaqtrader.com/Trader.aspx?id=OpenClose
- Databento cost quote: https://databento.com/docs/api-historical/metadata/metadata-get-cost
- Databento free metadata/symbology: https://databento.com/docs/api-historical/basics?lang=python
- Hyperliquid historical archive limitations: https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data
- Dwellir candle coverage, schema, sparse intervals and pagination: https://www.dwellir.com/docs/hyperliquid/hyperliquid-index/ohlcv-data
- Dwellir Index products and historical-book availability: https://www.dwellir.com/hyperliquid-index
- Dwellir free allowance and usage controls: https://www.dwellir.com/pricing
- Hyperliquid WebSocket subscriptions: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions

