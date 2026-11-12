# Multi-horizon calibrated ensemble forecasting on NEPSE floorsheet data

> **Model family: scikit-learn stand-ins.** CatBoost, LightGBM, XGBoost, the Transformer and the TFT were not run; every number below comes from the scikit-learn models listed in section 4. Do not report these results as the named algorithms. Run `FAMILY=native START=6 bash run_pipeline.sh` with the libraries installed to produce them.

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

## 4. Models (stages 06–08)

Model family `sklearn`; feature set `all` (60 predictors).

- `hgb`: HistGradientBoostingClassifier (31 leaves, lr 0.05, L2 5, early stopping on a time-ordered hold-out)
- `extratrees`: ExtraTreesClassifier (200 trees, min leaf 100, 30% features, 50% bootstrap)
- `logistic`: Logistic regression (L2, C=0.1) on quantile-normalised inputs
- `mlp`: Multilayer perceptron (128-64, ReLU, Adam, early stopping), one network for all four horizons
- `seq_mlp`: Windowed MLP: 12 key features over the last 20 sessions + the current 60-feature vector, one network for all horizons
- `ranker`: HistGradientBoostingRegressor on within-session return quintiles (pointwise ranker)

## 5. Out-of-fold results (nested region folds 3..8; stages 09–11)

### 1-session horizon (n=163,145, sessions=789)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 56.3% | 54.8% | 0.574 | 0.6817 | 0.2443 | 0.0242 | 68.2% [63.6%, 72.3%] | 6.01% |
| weighted | 57.1% | 54.8% | 0.579 | 0.6786 | 0.2428 | 0.0079 | 71.3% [62.3%, 78.3%] | 1.20% |
| stacking | 56.6% | 54.8% | 0.575 | 0.6856 | 0.2458 | 0.0385 | 61.3% [56.0%, 67.1%] | 8.90% |
| regime | 56.5% | 54.8% | 0.564 | 0.6999 | 0.2503 | 0.0464 | 60.0% [56.0%, 65.0%] | 11.65% |
| hgb | 56.8% | 54.8% | 0.576 | 0.6819 | 0.2443 | 0.0249 | 66.0% [61.3%, 70.3%] | 7.15% |
| extratrees | 56.8% | 54.8% | 0.573 | 0.6811 | 0.2440 | 0.0152 | 74.5% [64.7%, 91.5%] | 0.13% |
| logistic | 54.8% | 54.8% | 0.546 | 0.6871 | 0.2470 | 0.0211 | 68.7% [64.3%, 73.0%] | 1.75% |
| mlp | 54.7% | 54.8% | 0.552 | 0.7594 | 0.2713 | 0.1305 | 58.2% [56.3%, 60.1%] | 39.21% |
| seq_mlp | 53.6% | 54.8% | 0.541 | 0.8461 | 0.2946 | 0.1860 | 55.6% [53.7%, 57.4%] | 53.16% |
| avg__platt | 56.1% | 54.8% | 0.572 | 0.6842 | 0.2454 | 0.0381 | 65.1% [58.9%, 71.1%] | 6.90% |
| avg__iso | 56.0% | 54.8% | 0.569 | 0.6859 | 0.2458 | 0.0392 | 65.1% [58.3%, 71.5%] | 4.15% |
| weighted__platt | 56.6% | 54.8% | 0.578 | 0.6850 | 0.2455 | 0.0389 | 61.9% [55.8%, 68.9%] | 8.14% |
| weighted__iso | 56.6% | 54.8% | 0.572 | 0.6865 | 0.2460 | 0.0408 | 63.6% [56.7%, 71.2%] | 4.81% |
| stacking__platt | 56.1% | 54.8% | 0.568 | 0.6899 | 0.2476 | 0.0294 | 59.0% [52.8%, 64.6%] | 9.71% |
| stacking__iso | 56.3% | 54.8% | 0.564 | 0.7010 | 0.2480 | 0.0365 | 60.8% [54.8%, 66.8%] | 6.66% |
| regime__platt | 55.6% | 54.8% | 0.555 | 0.7036 | 0.2519 | 0.0354 | 58.0% [53.0%, 62.6%] | 10.82% |
| regime__iso | 56.2% | 54.8% | 0.550 | 0.7570 | 0.2547 | 0.0448 | 59.4% [54.8%, 65.3%] | 8.11% |

### 3-session horizon (n=166,486, sessions=789)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 54.2% | 54.5% | 0.540 | 0.6958 | 0.2510 | 0.0512 | 60.9% [54.3%, 67.6%] | 6.33% |
| weighted | 54.1% | 54.5% | 0.535 | 0.6888 | 0.2478 | 0.0155 | 68.5% [57.8%, 79.3%] | 0.37% |
| stacking | 54.6% | 54.5% | 0.531 | 0.6914 | 0.2490 | 0.0327 | 61.7% [51.4%, 71.9%] | 3.28% |
| regime | 54.0% | 54.5% | 0.511 | 0.7041 | 0.2547 | 0.0598 | 55.1% [48.5%, 61.6%] | 8.53% |
| hgb | 53.3% | 54.5% | 0.536 | 0.6924 | 0.2495 | 0.0361 | 58.6% [48.8%, 71.7%] | 1.30% |
| extratrees | 54.5% | 54.5% | 0.536 | 0.6883 | 0.2476 | 0.0162 | 55.1% [42.8%, 82.4%] | 0.22% |
| logistic | 53.3% | 54.5% | 0.526 | 0.6933 | 0.2500 | 0.0335 | 62.0% [55.1%, 67.7%] | 1.53% |
| mlp | 52.9% | 54.5% | 0.529 | 0.8587 | 0.2991 | 0.1919 | 54.6% [52.1%, 57.2%] | 52.90% |
| seq_mlp | 52.4% | 54.5% | 0.525 | 0.9291 | 0.3138 | 0.2201 | 53.4% [50.9%, 55.6%] | 58.68% |
| avg__platt | 54.5% | 54.5% | 0.541 | 0.6924 | 0.2494 | 0.0475 | 61.4% [52.5%, 70.2%] | 7.42% |
| avg__iso | 54.6% | 54.5% | 0.536 | 0.6937 | 0.2499 | 0.0500 | 62.8% [54.5%, 71.2%] | 6.34% |
| weighted__platt | 54.5% | 54.5% | 0.538 | 0.6930 | 0.2497 | 0.0499 | 62.6% [51.5%, 71.9%] | 5.72% |
| weighted__iso | 54.5% | 54.5% | 0.535 | 0.6930 | 0.2496 | 0.0485 | 62.8% [47.4%, 73.3%] | 4.12% |
| stacking__platt | 54.5% | 54.5% | 0.532 | 0.6950 | 0.2506 | 0.0536 | 62.9% [52.2%, 71.3%] | 9.04% |
| stacking__iso | 54.5% | 54.5% | 0.528 | 0.6948 | 0.2505 | 0.0510 | 63.0% [49.4%, 72.1%] | 7.56% |
| regime__platt | 54.5% | 54.5% | 0.525 | 0.6982 | 0.2519 | 0.0575 | 60.7% [51.0%, 69.5%] | 11.19% |
| regime__iso | 54.5% | 54.5% | 0.525 | 0.6967 | 0.2513 | 0.0572 | 63.1% [49.3%, 73.0%] | 7.76% |

### 5-session horizon (n=166,932, sessions=789)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 53.1% | 54.1% | 0.526 | 0.7035 | 0.2546 | 0.0625 | 56.5% [47.9%, 65.2%] | 6.74% |
| weighted | 52.8% | 54.1% | 0.525 | 0.6916 | 0.2492 | 0.0220 | 49.4% [37.9%, 68.3%] | 0.29% |
| stacking | 53.7% | 54.1% | 0.522 | 0.6955 | 0.2509 | 0.0464 | 59.6% [49.5%, 70.4%] | 4.78% |
| regime | 53.2% | 54.1% | 0.516 | 0.7107 | 0.2576 | 0.0827 | 56.7% [49.4%, 64.2%] | 14.00% |
| hgb | 51.9% | 54.1% | 0.525 | 0.7046 | 0.2553 | 0.0632 | 56.3% [47.6%, 63.8%] | 4.42% |
| extratrees | 53.5% | 54.1% | 0.536 | 0.6897 | 0.2483 | 0.0174 | 64.7% [49.4%, 77.3%] | 1.22% |
| logistic | 52.1% | 54.1% | 0.512 | 0.6977 | 0.2521 | 0.0429 | 59.6% [53.2%, 67.5%] | 1.24% |
| mlp | 52.9% | 54.1% | 0.527 | 0.9040 | 0.3085 | 0.2083 | 54.0% [51.3%, 57.4%] | 56.88% |
| seq_mlp | 50.8% | 54.1% | 0.504 | 0.9828 | 0.3281 | 0.2422 | 51.1% [48.3%, 53.7%] | 60.19% |
| avg__platt | 54.0% | 54.1% | 0.533 | 0.6978 | 0.2519 | 0.0657 | 62.2% [51.9%, 71.7%] | 11.55% |
| avg__iso | 54.1% | 54.1% | 0.532 | 0.7060 | 0.2527 | 0.0675 | 63.7% [50.9%, 73.6%] | 5.48% |
| weighted__platt | 54.1% | 54.1% | 0.534 | 0.6969 | 0.2516 | 0.0631 | 63.6% [46.1%, 74.3%] | 7.61% |
| weighted__iso | 54.1% | 54.1% | 0.537 | 0.6960 | 0.2510 | 0.0600 | 64.4% [47.7%, 74.2%] | 4.66% |
| stacking__platt | 54.1% | 54.1% | 0.535 | 0.6949 | 0.2508 | 0.0588 | 62.6% [38.6%, 78.4%] | 1.35% |
| stacking__iso | 54.1% | 54.1% | 0.534 | 0.6963 | 0.2512 | 0.0598 | 63.7% [46.8%, 75.1%] | 8.92% |
| regime__platt | 54.1% | 54.1% | 0.528 | 0.6960 | 0.2512 | 0.0597 | 61.0% [45.1%, 74.9%] | 6.66% |
| regime__iso | 54.1% | 54.1% | 0.531 | 0.6994 | 0.2523 | 0.0632 | 62.0% [47.1%, 74.0%] | 11.44% |

### 10-session horizon (n=167,291, sessions=789)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 51.8% | 53.8% | 0.514 | 0.7104 | 0.2579 | 0.0740 | 55.3% [46.0%, 64.6%] | 7.43% |
| weighted | 52.3% | 53.8% | 0.521 | 0.6956 | 0.2511 | 0.0322 | 53.3% [25.3%, 74.1%] | 1.47% |
| stacking | 53.5% | 53.8% | 0.510 | 0.7095 | 0.2569 | 0.0694 | 55.5% [42.4%, 69.3%] | 11.86% |
| regime | 52.3% | 53.8% | 0.503 | 0.7572 | 0.2741 | 0.1267 | 53.8% [46.3%, 61.9%] | 31.03% |
| hgb | 51.3% | 53.8% | 0.514 | 0.7200 | 0.2621 | 0.0790 | 42.6% [32.9%, 50.7%] | 8.59% |
| extratrees | 53.1% | 53.8% | 0.532 | 0.6934 | 0.2500 | 0.0245 | 55.2% [30.1%, 74.8%] | 2.71% |
| logistic | 51.7% | 53.8% | 0.510 | 0.7042 | 0.2551 | 0.0587 | 53.6% [41.8%, 64.2%] | 3.45% |
| mlp | 52.4% | 53.8% | 0.524 | 0.9292 | 0.3144 | 0.2197 | 53.3% [49.1%, 57.7%] | 58.49% |
| seq_mlp | 49.8% | 53.8% | 0.492 | 0.9855 | 0.3324 | 0.2477 | 49.6% [46.3%, 52.7%] | 59.08% |
| avg__platt | 53.8% | 53.8% | 0.532 | 0.7124 | 0.2575 | 0.0964 | 62.4% [50.7%, 74.0%] | 22.07% |
| avg__iso | 53.8% | 53.8% | 0.540 | 0.7127 | 0.2562 | 0.0949 | 63.3% [49.8%, 76.3%] | 15.41% |
| weighted__platt | 53.8% | 53.8% | 0.535 | 0.7117 | 0.2576 | 0.1000 | 62.3% [51.7%, 73.0%] | 24.48% |
| weighted__iso | 53.8% | 53.8% | 0.540 | 0.7113 | 0.2573 | 0.0978 | 62.3% [51.6%, 73.4%] | 21.90% |
| stacking__platt | 53.8% | 53.8% | 0.541 | 0.7084 | 0.2564 | 0.0953 | 62.2% [51.7%, 73.0%] | 25.53% |
| stacking__iso | 53.8% | 53.8% | 0.542 | 0.7082 | 0.2555 | 0.0901 | 68.5% [47.5%, 84.3%] | 12.54% |
| regime__platt | 53.8% | 53.8% | 0.540 | 0.7180 | 0.2575 | 0.0944 | 68.4% [47.6%, 84.1%] | 12.60% |
| regime__iso | 53.8% | 53.8% | 0.540 | 0.7822 | 0.2628 | 0.1099 | 68.4% [47.6%, 84.2%] | 12.60% |

## 6. Significance tests (block bootstrap over sessions; DM/HAC; Holm)

Primary candidate per horizon was **chosen on OOF log-loss before the test was touched**:

- **1D** `weighted`: accuracy 57.1% vs majority 54.8% (bootstrap p=0.000); AUC 0.579; ECE 0.0079. At confidence ≥0.70: accuracy 71.3%, coverage 1.20%, n=1,959; p(acc > majority)=0.000; **p(acc ≥ 0.70)=0.407**.
  - DM log-loss vs climatology: mean Δ=+0.01006, stat=4.87, one-sided p=0.0000, Holm p=0.0001
  - DM log-loss vs best_single: mean Δ=+0.00266, stat=2.32, one-sided p=0.0102, Holm p=0.7436
  - DM log-loss vs avg: mean Δ=+0.00275, stat=1.72, one-sided p=0.0429, Holm p=1.0000
  - Regime split of confident calls: market-up n=4776 acc=66.5%; market-down n=6087 acc=56.4%, share of down-calls 99.7%
  - Folds beating majority baseline: 6/6
- **3D** `weighted`: accuracy 54.1% vs majority 54.5% (bootstrap p=0.630); AUC 0.535; ECE 0.0155. At confidence ≥0.70: accuracy 68.5%, coverage 0.37%, n=623; p(acc > majority)=0.003; **p(acc ≥ 0.70)=0.563**.
  - DM log-loss vs climatology: mean Δ=+0.00103, stat=0.39, one-sided p=0.3499, Holm p=1.0000
  - DM log-loss vs best_single: mean Δ=-0.00027, stat=-0.15, one-sided p=0.5586, Holm p=1.0000
  - DM log-loss vs avg: mean Δ=+0.00798, stat=2.23, one-sided p=0.0130, Holm p=1.0000
  - Regime split of confident calls: market-up n=1506 acc=72.0%; market-down n=11085 acc=61.7%, share of down-calls 100.0%
  - Folds beating majority baseline: 1/6
- **5D** `weighted`: accuracy 52.8% vs majority 54.1% (bootstrap p=0.840); AUC 0.525; ECE 0.0220. At confidence ≥0.70: accuracy 49.4%, coverage 0.29%, n=488; p(acc > majority)=0.700; **p(acc ≥ 0.70)=0.980**.
  - DM log-loss vs climatology: mean Δ=-0.00082, stat=-0.24, one-sided p=0.5932, Holm p=1.0000
  - DM log-loss vs best_single: mean Δ=-0.00205, stat=-1.03, one-sided p=0.8492, Holm p=1.0000
  - DM log-loss vs avg: mean Δ=+0.01246, stat=2.68, one-sided p=0.0038, Holm p=0.2949
  - Regime split of confident calls: market-up n=1870 acc=86.4%; market-down n=13016 acc=60.4%, share of down-calls 100.0%
  - Folds beating majority baseline: 0/6
- **10D** `weighted`: accuracy 52.3% vs majority 53.8% (bootstrap p=0.837); AUC 0.521; ECE 0.0322. At confidence ≥0.70: accuracy 53.3%, coverage 1.47%, n=2,464; p(acc > majority)=0.477; **p(acc ≥ 0.70)=0.907**.
  - DM log-loss vs climatology: mean Δ=-0.00343, stat=-0.59, one-sided p=0.7223, Holm p=1.0000
  - DM log-loss vs best_single: mean Δ=-0.00256, stat=-0.96, one-sided p=0.8315, Holm p=1.0000
  - DM log-loss vs avg: mean Δ=+0.01560, stat=3.23, one-sided p=0.0006, Holm p=0.0494
  - Regime split of confident calls: market-up n=2593 acc=92.6%; market-down n=18384 acc=65.0%, share of down-calls 100.0%
  - Folds beating majority baseline: 0/6

### Ranker (pointwise HistGradientBoosting, scikit-learn stand-in) – cross-sectional experiment

| Horizon | Mean rank-IC | IC t (HAC) | Top-bottom quintile spread | Top-decile up-rate |
|---|---|---|---|---|
| 1D | 0.2063 | 33.62 | 1.04% | 55.9% |
| 3D | 0.1729 | 20.26 | 1.45% | 53.9% |
| 5D | 0.1416 | 14.00 | 1.47% | 51.7% |
| 10D | 0.1046 | 7.28 | 1.44% | 50.9% |

![reliability](figs/reliability.png)

## 7. Untouched final test (stage 12)

Dates 2025-10-12 → 2026-11-05. Forced re-runs recorded in lock: 0.

| Horizon | Primary | Acc [95% CI] | Majority | AUC | LogLoss | ECE | Acc@≥0.70 | Coverage | n@≥0.70 | p(acc≥0.70) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1D | weighted | 59.0% [57.0%, 60.9%] | 55.3% | 0.608 | 0.6720 | 0.0323 | 81.1% | 0.40% | 243 | 0.149 |
| 3D | weighted | 59.2% [55.7%, 62.6%] | 57.3% | 0.610 | 0.6732 | 0.0488 | 100.0% | 0.03% | 22 | 0.000 |
| 5D | weighted | 58.3% [53.7%, 62.5%] | 57.3% | 0.597 | 0.6801 | 0.0623 | n/a | 0.00% | 0 | n/a |
| 10D | weighted | 55.3% [49.1%, 60.9%] | 56.8% | 0.584 | 0.6863 | 0.0732 | 0.0% | 0.00% | 1 | 1.000 |

![accuracy](figs/accuracy.png)

## 8. Backtest (stage 13, final test, net of costs)

Costs/side: {'commission': 0.0036, 'sebon': 0.00015, 'slippage': 0.001}; CGT 7.5%; long-only; entry next-session open.

| Horizon | Thr | Signals | Filled | Trades | Mean net/trade [95% CI] | Hit rate | p(≤0) | Placebo mean | p vs placebo | Sharpe | MaxDD | Cum ret |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1D | 0.60 | 423 | 410 | 410 | -0.20% [-0.56%, 0.20%] | 41.2% | 0.832 | -0.65% | 0.000 | -1.31 | -29.8% | -27.0% |
| 1D | 0.70 | 7 | 7 | – | too few trades | | | | | | | |
| 3D | 0.60 | 252 | 244 | 244 | 0.55% [-0.25%, 1.41%] | 50.4% | 0.118 | -0.37% | 0.000 | 0.72 | -12.1% | 8.2% |
| 3D | 0.70 | 0 | 0 | – | too few trades | | | | | | | |
| 5D | 0.60 | 135 | 128 | 128 | -0.15% [-1.98%, 1.44%] | 36.7% | 0.580 | -0.43% | 0.170 | -0.55 | -5.7% | -2.1% |
| 5D | 0.70 | 0 | 0 | – | too few trades | | | | | | | |
| 10D | 0.60 | 453 | 409 | 409 | 0.52% [-1.01%, 2.21%] | 50.4% | 0.290 | -0.27% | 0.000 | 1.62 | -3.7% | 5.2% |
| 10D | 0.70 | 0 | 0 | – | too few trades | | | | | | | |

![equity](figs/equity_curves.png)

## 9. What the evidence supports

- **1D:** overall test accuracy 59.0% – statistically above the majority baseline (55.3%). At ≥0.70 confidence accuracy was 81.1% on 0.40% of cases (n=243); the 95% CI is [63.3%, 85.2%], so a true accuracy of 70% is NOT statistically supported (lower bound below 70%).
- **3D:** overall test accuracy 59.2% – NOT statistically distinguishable from the majority baseline (57.3%). At ≥0.70 confidence accuracy was 100.0% on 0.03% of cases (n=22); the 95% CI is [100.0%, 100.0%], so a true accuracy of 70% is not assessable: fewer than 100 calls.
- **5D:** overall test accuracy 58.3% – NOT statistically distinguishable from the majority baseline (57.3%). No predictions reached 0.70 confidence.
- **10D:** overall test accuracy 55.3% – NOT statistically distinguishable from the majority baseline (56.8%). At ≥0.70 confidence accuracy was 0.0% on 0.00% of cases (n=1); the 95% CI is [0.0%, 0.0%], so a true accuracy of 70% is not assessable: fewer than 100 calls.

70% is therefore a *selective-prediction operating point* (accuracy conditional on confidence ≥ 0.70 with the stated coverage), not a model-wide accuracy. Calibration (ECE) shows whether stated confidences are honest.

## 10. Limitations

- Daily aggregation hides intraday dynamics; no order-book data.
- The floorsheet is largely missing for Feb 2022 – Mar 2023 and Mar 2024; broker features are missing there (prices come from the adjusted OHLCV table).
- Fundamental publication dates are approximated as quarter end + 45 days; a sensitivity check with a longer lag is advisable.
- Overlapping labels (h>1) are handled by block bootstrap and HAC tests but reduce the effective sample size.
- Corporate-action adjustment is heuristic (jumps > 11%); dividends are not modelled.
- Long-only execution at the next open assumes fills; thin names may fill worse than the 10 bp slippage assumed.
- A single 250-session test window covers one market regime; repeated forward evaluation is recommended before any deployment claim.
- Results from a smoke-test run (stand-in models) must not be cited as results of the named algorithms.