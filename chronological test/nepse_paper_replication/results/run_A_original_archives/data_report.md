# NEPSE data cleaning, panel construction and leakage audit

Generated 2026-11-08 09:52 from the stage 01-05 reports.

## 1. Data validation and cleaning (stages 01–02, 01b)

### 1.1 Floorsheet trades

| File | Size (MB) | CRC ok | SHA-256 | Rows | Dates |
|---|---|---|---|---|---|
| floorsheet_data_part1.zip | 393 | True | `2f0d17037a60c3a0…` | 41,169,555 | 2018-10-01 → 2024-01-31 |
| floorsheet_data_part2.zip | 205 | True | `f1dc39c48ca63786…` | 19,364,427 | 2024-02-01 → 2025-01-28 |
| floorsheet_data_part3.zip | 357 | True | `25f403e6889a11c6…` | 30,514,666 | 2025-02-02 → 2026-11-05 |

Trades parsed: **91,048,648**; kept after cleaning and de-duplication: **91,009,140** (99.957%). Duplicate transaction numbers removed: **39,500** (exact copies 39,500, conflicting 0; by source {'part3->part3': 39500}; months {'202607': 39500}). De-duplication is global: trades are bucketed by the month encoded in the transaction number, so every copy of a transaction lands in the same bucket. Dropped: missing required field 0, non-positive quantity/rate/amount 8, unusable broker id 0, bad date 0. Kept and flagged: 33,758 broker fields with alphanumeric codes (D01; real dealer trades), 102,076 trades from 2018-10-01 to Dec 2018 whose transaction numbers use an older format, 710 rows where amount ≠ quantity × rate by more than NPR 1. `trade_time` is empty in 79.4% of rows, so intraday order is taken from the transaction number. OHLC violations after aggregation: 0; trade counts reconcile with daily aggregates: True. Result: 405,370 symbol-sessions, 655 symbols, 1840 sessions (2018-10-01 → 2026-11-05).

### 1.2 Auxiliary sources

- **Adjusted OHLCV** (`livedata_adjusted`): 540,011 rows, 479 symbols incl. the NEPSE index and 16 sub-indices, 1997-07-20 → 2026-11-05; no duplicate (symbol, date) keys; `sector_id` is empty; 22,628 bars with open/close outside high–low (87.0% dated before 2019; 6.8% on index rows) are repaired by widening high/low. Back-adjusted for corporate actions (price ratio to raw floorsheet close is 1 after the last action and steps at bonus/rights dates); volume is in raw shares. It lists **currently listed symbols only**, so it is used for listed stocks and the floorsheet supplies the delisted/merged ones.
- **Quarterly fundamentals** (`new_fundamentals`): 5,625 reports, 270 companies, FY 2070/2071 → 2082/2083. The table's `date` column lies 29–31 days *before* the quarter ends in 100.0% of rows, so it is not a publication date. Reports are used only from quarter end + 45 days. The table's P/E × EPS equals the raw closing price at quarter end within 2% for 86.8% of reports (vs 17.8% at the `date` column), so E/P and B/P are rebuilt from the quarter-end price and the adjusted return since.
- **Nine `fundamental_*` tables** (fundamental_growths, fundamental_risk_detections, fundamental_valuations, fundamental_profitabilities, fundamental_leverages, fundamental_market_performances, fundamental_liquidities, fundamental_efficiencies, fundamental_earning_qualities): each holds at most one fiscal period per company (latest quarter, created 2026-11-05). Applying them to earlier dates would leak future information, so they are **excluded** from modelling.
- **Feature store** (`nepse_prediction_featurestore`): 148,289 rows, 101 symbols, 2012-02-26 → 2026-11-04. Its target is the next-session return. Its fundamental columns switch to a new report on the table's `date` column — 99.7% of switches happen before the quarter even ends — so it carries look-ahead bias and is **not** used as model input.

## 2. Modeling panel and targets (stages 03–04)

Universe: {'equity': 369, 'promoter_share': 143, 'debenture_or_fund_series': 112, 'mutual_fund': 30} → ordinary shares only; price source for those: {'ohlcv_adjusted': 287, 'floorsheet': 82}. Bars: {'ohlcv_adjusted': 369216, 'floorsheet_chained': 1522, 'floorsheet_delisted': 34753}; 1,522 floorsheet bars chained onto adjusted series before a symbol's first OHLCV bar. Eligible samples: 360,943 (362 symbols, 1855 sessions, 2018-10-01 → 2026-11-05). Predictors: 53 core (price/return, volatility, candle, trend, oscillators, range, liquidity, floorsheet/broker, NEPSE market) + 7 point-in-time fundamentals; this run used feature set `all` (60 predictors). Floorsheet features are present for 88.5% of samples (by year {'2018': 0.976, '2019': 0.983, '2020': 0.981, '2021': 0.989, '2022': 0.292, '2023': 0.812, '2024': 0.963, '2025': 0.993, '2026': 0.995}) and are left missing, not imputed, where the floorsheet has gaps; fundamentals cover 72.4%. Residual price jumps > 11% (bad prints or unadjusted actions) neutralised on 225 symbol-sessions.

| Horizon | Labeled | Up-rate | Flat outcomes excluded | Mean fwd ret |
|---|---|---|---|---|
| 1d | 340,322 | 45.5% | 5.6% | 0.08% |
| 3d | 354,301 | 45.7% | 1.6% | 0.23% |
| 5d | 355,809 | 46.2% | 1.0% | 0.38% |
| 10d | 356,498 | 46.6% | 0.5% | 0.77% |

## 3. Leakage and split audit (stage 05)

All checks passed: **True**.

| Check | Passed | Detail |
|---|---|---|
| embargo_covers_label_horizon | True | min (val_start - (train_end + 10)) = 16 sessions; embargo=25 |
| validation_blocks_contiguous_non_overlapping | True | blocks tile the development period |
| first_train_window_min_sessions | True | 504 sessions |
| final_test_isolated | True | gap=25 eligible sessions, 26 calendar sessions; test=250 sessions |
| dev_test_row_disjoint | True | dev rows=290,409, test rows=64,387 |
| no_duplicate_symbol_session | True | unique (symbol, session) |
| targets_unavailable_in_last_h_sessions | True | labelled rows in last h sessions: {1: 0, 3: 0, 5: 0, 10: 0} |
| truncation_same_rows | True | 263,496 / 263,496 rows present in both builds |
| features_lookahead_free_truncation_test | True | max |diff|=0.00e+00 (None); NaN-pattern mismatches=0; 60 features x 263,496 rows |
| no_suspicious_feature_target_correlation | True | top |corr| with y_1: [('ret2', -0.041), ('buyer_top5', 0.039), ('close_vwap', -0.033), ('avg_trade_size', 0.033), ('volratio_1_20', 0.032)] |

| Fold | Train sessions | Val sessions | Train dates | Val dates | Train rows | Val rows |
|---|---|---|---|---|---|---|
| 1 | 504 | 131 | 2018-10-01→2021-01-21 | 2021-02-28→2021-09-09 | 72,100 | 23,150 |
| 2 | 635 | 131 | 2018-10-01→2021-08-02 | 2021-09-12→2022-04-03 | 94,897 | 23,258 |
| 3 | 766 | 131 | 2018-10-01→2022-02-21 | 2022-04-04→2022-09-29 | 118,468 | 21,146 |
| 4 | 897 | 131 | 2018-10-01→2022-08-26 | 2022-10-09→2023-05-08 | 139,758 | 23,332 |
| 5 | 1028 | 131 | 2018-10-01→2023-04-02 | 2023-05-09→2023-12-05 | 162,246 | 27,599 |
| 6 | 1159 | 131 | 2018-10-01→2023-10-16 | 2023-12-06→2024-06-26 | 189,215 | 31,235 |
| 7 | 1290 | 131 | 2018-10-01→2024-05-19 | 2024-06-27→2025-01-22 | 219,886 | 31,781 |
| 8 | 1421 | 134 | 2018-10-01→2024-12-12 | 2025-01-23→2025-08-17 | 251,706 | 32,629 |

Final test: 250 sessions, 2025-10-12 → 2026-11-05 (64,387 rows). Embargo: 25 sessions.
