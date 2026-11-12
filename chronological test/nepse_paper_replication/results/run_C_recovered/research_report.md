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

## 2. Modeling panel and targets (stages 03–04)

Universe: {'equity': 369, 'promoter_share': 143, 'debenture_or_fund_series': 112, 'mutual_fund': 30} → ordinary shares only; price source for those: {'ohlcv_adjusted': 287, 'floorsheet': 82}. Bars: {'ohlcv_adjusted': 369216, 'floorsheet_chained': 1522, 'floorsheet_delisted': 37294}; 1,522 floorsheet bars chained onto adjusted series before a symbol's first OHLCV bar. Eligible samples: 333,786 (362 symbols, 1682 sessions, 2018-10-01 → 2026-11-05). Predictors: 53 core (price/return, volatility, candle, trend, oscillators, range, liquidity, floorsheet/broker, NEPSE market) + 7 point-in-time fundamentals; this run used feature set `all` (60 predictors). Floorsheet features are present for 98.8% of samples (by year {'2018': 0.976, '2019': 0.983, '2020': 0.981, '2021': 0.989, '2022': 0.941, '2023': 0.998, '2024': 1.0, '2025': 0.993, '2026': 0.995}) and are left missing, not imputed, where the floorsheet has gaps; fundamentals cover 72.2%. Residual price jumps > 11% (bad prints or unadjusted actions) neutralised on 230 symbol-sessions.

**Thin-floorsheet sessions excluded.** A session is thin when fewer than 50% of its eligible stocks appear in the floorsheet. Runs of at least 5 consecutive thin sessions are dropped from the modelling panel, together with the 19 sessions that follow each run, so that no rolling or sequence window reads a thin session. 180 sessions are thin; 173 are excluded in total. Blocks: 2022-07-28 → 2023-03-30 (154 sessions, buffer to 2023-04-26).

| Horizon | Labeled | Up-rate | Flat outcomes excluded | Mean fwd ret |
|---|---|---|---|---|
| 1d | 314,131 | 45.7% | 5.8% | 0.09% |
| 3d | 327,477 | 45.9% | 1.6% | 0.26% |
| 5d | 328,881 | 46.7% | 1.1% | 0.44% |
| 10d | 329,467 | 47.4% | 0.5% | 0.89% |

## 3. Leakage and split audit (stage 05)

All checks passed: **True**.

| Check | Passed | Detail |
|---|---|---|
| embargo_covers_label_horizon | True | min (val_start - (train_end + 10)) = 16 sessions; embargo=25 |
| validation_blocks_contiguous_non_overlapping | True | blocks tile the development period |
| first_train_window_min_sessions | True | 504 sessions |
| final_test_isolated | True | gap=25 eligible sessions, 26 calendar sessions; test=250 sessions |
| dev_test_row_disjoint | True | dev rows=263,252, test rows=64,387 |
| no_duplicate_symbol_session | True | unique (symbol, session) |
| targets_unavailable_in_last_h_sessions | True | labelled rows in last h sessions: {1: 0, 3: 0, 5: 0, 10: 0} |
| truncation_same_rows | True | 278,781 / 278,781 rows present in both builds |
| features_lookahead_free_truncation_test | True | max |diff|=0.00e+00 (None); NaN-pattern mismatches=0; 60 features x 278,781 rows |
| no_suspicious_feature_target_correlation | True | top |corr| with y_1: [('ret2', -0.041), ('close_vwap', -0.035), ('volratio_1_20', 0.03), ('buyer_top5', 0.03), ('broker_imbalance', 0.029)] |

| Fold | Train sessions | Val sessions | Train dates | Val dates | Train rows | Val rows |
|---|---|---|---|---|---|---|
| 1 | 504 | 109 | 2018-10-01→2021-01-21 | 2021-02-28→2021-08-05 | 72,100 | 19,165 |
| 2 | 613 | 109 | 2018-10-01→2021-06-30 | 2021-08-08→2022-01-24 | 90,920 | 19,754 |
| 3 | 722 | 109 | 2018-10-01→2021-12-16 | 2022-01-25→2022-07-06 | 110,601 | 20,305 |
| 4 | 831 | 109 | 2018-10-01→2022-06-06 | 2022-07-07→2023-09-11 | 130,885 | 21,919 |
| 5 | 940 | 109 | 2018-10-01→2023-08-03 | 2023-09-12→2024-03-10 | 152,186 | 24,881 |
| 6 | 1049 | 109 | 2018-10-01→2024-02-01 | 2024-03-12→2024-08-28 | 176,323 | 26,486 |
| 7 | 1158 | 109 | 2018-10-01→2024-07-18 | 2024-08-29→2025-02-23 | 202,744 | 26,443 |
| 8 | 1267 | 115 | 2018-10-01→2025-01-13 | 2025-02-24→2025-08-17 | 229,166 | 28,020 |

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

### 1-session horizon (n=144,100, sessions=660)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 56.5% | 54.8% | 0.578 | 0.6806 | 0.2438 | 0.0201 | 69.2% [65.0%, 73.3%] | 5.14% |
| weighted | 57.7% | 54.8% | 0.590 | 0.6764 | 0.2417 | 0.0140 | 70.1% [59.1%, 79.4%] | 0.63% |
| stacking | 57.2% | 54.8% | 0.587 | 0.6794 | 0.2431 | 0.0369 | 68.2% [64.1%, 72.0%] | 7.59% |
| regime | 57.1% | 54.8% | 0.574 | 0.6885 | 0.2468 | 0.0455 | 62.3% [56.4%, 68.4%] | 13.35% |
| hgb | 56.5% | 54.8% | 0.577 | 0.6814 | 0.2441 | 0.0270 | 68.4% [63.2%, 73.6%] | 5.15% |
| extratrees | 57.3% | 54.8% | 0.586 | 0.6793 | 0.2431 | 0.0223 | n/a [None, None] | 0.00% |
| logistic | 54.8% | 54.8% | 0.548 | 0.6869 | 0.2469 | 0.0227 | 68.2% [62.9%, 73.0%] | 2.20% |
| mlp | 55.9% | 54.8% | 0.569 | 0.7367 | 0.2631 | 0.1106 | 60.1% [58.0%, 62.3%] | 36.84% |
| seq_mlp | 52.8% | 54.8% | 0.531 | 0.8703 | 0.3008 | 0.1967 | 54.5% [52.9%, 56.3%] | 53.85% |
| avg__platt | 56.2% | 54.8% | 0.577 | 0.6816 | 0.2443 | 0.0369 | 70.1% [64.8%, 74.9%] | 7.12% |
| avg__iso | 55.9% | 54.8% | 0.575 | 0.6826 | 0.2447 | 0.0374 | 71.0% [66.0%, 75.5%] | 6.67% |
| weighted__platt | 57.3% | 54.8% | 0.590 | 0.6785 | 0.2427 | 0.0354 | 68.5% [64.0%, 72.4%] | 8.62% |
| weighted__iso | 56.9% | 54.8% | 0.587 | 0.6799 | 0.2432 | 0.0364 | 69.2% [64.3%, 73.8%] | 5.79% |
| stacking__platt | 57.2% | 54.8% | 0.582 | 0.6780 | 0.2425 | 0.0091 | 70.5% [65.6%, 75.0%] | 4.55% |
| stacking__iso | 56.9% | 54.8% | 0.578 | 0.6789 | 0.2430 | 0.0131 | 75.4% [68.1%, 81.1%] | 0.90% |
| regime__platt | 56.6% | 54.8% | 0.568 | 0.6838 | 0.2452 | 0.0208 | 64.2% [56.0%, 71.4%] | 7.43% |
| regime__iso | 56.4% | 54.8% | 0.565 | 0.6838 | 0.2450 | 0.0237 | 61.5% [53.0%, 74.1%] | 2.97% |

### 3-session horizon (n=147,012, sessions=660)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 54.5% | 54.3% | 0.551 | 0.6920 | 0.2492 | 0.0442 | 62.7% [55.6%, 70.2%] | 4.93% |
| weighted | 54.7% | 54.3% | 0.547 | 0.6874 | 0.2471 | 0.0182 | 64.3% [55.8%, 73.8%] | 1.12% |
| stacking | 55.3% | 54.3% | 0.548 | 0.6912 | 0.2487 | 0.0318 | 61.3% [51.1%, 71.2%] | 7.82% |
| regime | 53.5% | 54.3% | 0.504 | 0.7165 | 0.2595 | 0.0685 | 51.3% [42.5%, 62.4%] | 11.82% |
| hgb | 51.9% | 54.3% | 0.524 | 0.7032 | 0.2546 | 0.0631 | 54.2% [45.5%, 61.8%] | 4.35% |
| extratrees | 54.6% | 54.3% | 0.543 | 0.6865 | 0.2467 | 0.0094 | 40.0% [20.0%, 100.0%] | 0.01% |
| logistic | 53.2% | 54.3% | 0.527 | 0.6945 | 0.2505 | 0.0372 | 61.4% [53.2%, 69.5%] | 2.31% |
| mlp | 54.2% | 54.3% | 0.548 | 0.8209 | 0.2877 | 0.1711 | 56.5% [53.4%, 59.4%] | 50.09% |
| seq_mlp | 52.4% | 54.3% | 0.527 | 0.9289 | 0.3142 | 0.2226 | 53.6% [51.1%, 55.9%] | 59.43% |
| avg__platt | 55.1% | 54.3% | 0.550 | 0.6926 | 0.2493 | 0.0480 | 62.1% [51.0%, 71.6%] | 11.12% |
| avg__iso | 54.8% | 54.3% | 0.544 | 0.6940 | 0.2499 | 0.0504 | 62.0% [50.6%, 72.4%] | 7.99% |
| weighted__platt | 54.7% | 54.3% | 0.545 | 0.6961 | 0.2506 | 0.0478 | 61.4% [50.6%, 70.4%] | 12.97% |
| weighted__iso | 54.6% | 54.3% | 0.537 | 0.6991 | 0.2515 | 0.0512 | 61.4% [50.6%, 70.3%] | 12.93% |
| stacking__platt | 54.9% | 54.3% | 0.541 | 0.6960 | 0.2506 | 0.0411 | 62.2% [52.0%, 70.2%] | 14.69% |
| stacking__iso | 54.8% | 54.3% | 0.535 | 0.7154 | 0.2530 | 0.0508 | 61.4% [51.5%, 69.8%] | 14.39% |
| regime__platt | 54.1% | 54.3% | 0.508 | 0.7056 | 0.2548 | 0.0551 | 55.8% [44.2%, 67.5%] | 13.68% |
| regime__iso | 54.3% | 54.3% | 0.508 | 0.7315 | 0.2600 | 0.0683 | 55.1% [43.7%, 66.4%] | 13.59% |

### 5-session horizon (n=147,390, sessions=660)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 52.3% | 53.9% | 0.523 | 0.7043 | 0.2552 | 0.0630 | 55.7% [47.0%, 65.3%] | 4.77% |
| weighted | 52.9% | 53.9% | 0.528 | 0.6932 | 0.2499 | 0.0271 | 56.4% [41.4%, 74.9%] | 0.83% |
| stacking | 54.0% | 53.9% | 0.531 | 0.7006 | 0.2528 | 0.0475 | 59.1% [47.6%, 70.1%] | 10.87% |
| regime | 52.4% | 53.9% | 0.495 | 0.7262 | 0.2637 | 0.0804 | 49.6% [39.9%, 60.0%] | 13.58% |
| hgb | 50.1% | 53.9% | 0.500 | 0.7221 | 0.2635 | 0.0958 | 50.4% [40.7%, 59.3%] | 7.38% |
| extratrees | 52.5% | 53.9% | 0.523 | 0.6909 | 0.2489 | 0.0171 | 73.3% [0.0%, 100.0%] | 0.01% |
| logistic | 51.4% | 53.9% | 0.506 | 0.7011 | 0.2538 | 0.0523 | 59.5% [47.8%, 70.2%] | 1.72% |
| mlp | 53.3% | 53.9% | 0.540 | 0.8714 | 0.3007 | 0.1966 | 54.9% [51.4%, 58.0%] | 55.10% |
| seq_mlp | 50.2% | 53.9% | 0.499 | 0.9876 | 0.3310 | 0.2483 | 50.6% [48.0%, 53.1%] | 60.38% |
| avg__platt | 53.9% | 53.9% | 0.528 | 0.7028 | 0.2539 | 0.0644 | 59.6% [47.6%, 70.7%] | 15.86% |
| avg__iso | 53.9% | 53.9% | 0.525 | 0.7029 | 0.2541 | 0.0675 | 60.8% [48.5%, 71.5%] | 14.57% |
| weighted__platt | 53.9% | 53.9% | 0.529 | 0.7037 | 0.2541 | 0.0654 | 59.1% [46.8%, 69.7%] | 17.22% |
| weighted__iso | 53.8% | 53.9% | 0.523 | 0.7045 | 0.2542 | 0.0679 | 57.7% [46.0%, 68.3%] | 18.03% |
| stacking__platt | 53.6% | 53.9% | 0.531 | 0.6998 | 0.2526 | 0.0587 | 62.3% [51.7%, 71.6%] | 18.57% |
| stacking__iso | 53.2% | 53.9% | 0.520 | 0.7070 | 0.2544 | 0.0661 | 59.8% [48.5%, 69.9%] | 18.35% |
| regime__platt | 53.8% | 53.9% | 0.496 | 0.7089 | 0.2564 | 0.0667 | 58.4% [46.1%, 69.4%] | 17.66% |
| regime__iso | 53.7% | 53.9% | 0.508 | 0.7089 | 0.2562 | 0.0682 | 56.1% [43.2%, 68.3%] | 16.43% |

### 10-session horizon (n=147,671, sessions=660)

| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |
|---|---|---|---|---|---|---|---|---|
| avg | 49.7% | 53.4% | 0.493 | 0.7206 | 0.2628 | 0.0910 | 47.2% [38.7%, 57.7%] | 6.31% |
| weighted | 49.1% | 53.4% | 0.478 | 0.7110 | 0.2584 | 0.0664 | 35.3% [17.2%, 64.9%] | 1.81% |
| stacking | 53.1% | 53.4% | 0.519 | 0.7380 | 0.2648 | 0.0694 | 53.6% [38.6%, 71.1%] | 17.30% |
| regime | 51.3% | 53.4% | 0.486 | 0.7826 | 0.2802 | 0.1169 | 49.0% [38.5%, 60.0%] | 24.23% |
| hgb | 48.4% | 53.4% | 0.467 | 0.7502 | 0.2759 | 0.1178 | 33.3% [24.9%, 43.5%] | 11.88% |
| extratrees | 50.2% | 53.4% | 0.499 | 0.6991 | 0.2529 | 0.0483 | 51.2% [28.8%, 82.7%] | 0.32% |
| logistic | 50.1% | 53.4% | 0.493 | 0.7116 | 0.2586 | 0.0727 | 46.7% [33.9%, 62.1%] | 3.32% |
| mlp | 51.9% | 53.4% | 0.523 | 0.9170 | 0.3137 | 0.2164 | 52.6% [48.7%, 56.4%] | 56.77% |
| seq_mlp | 48.8% | 53.4% | 0.481 | 1.0034 | 0.3377 | 0.2563 | 48.4% [45.7%, 51.4%] | 58.95% |
| avg__platt | 53.4% | 53.4% | 0.506 | 0.7496 | 0.2677 | 0.0988 | 59.4% [44.4%, 75.1%] | 24.56% |
| avg__iso | 53.4% | 53.4% | 0.519 | 0.7331 | 0.2650 | 0.1068 | 60.2% [45.5%, 75.4%] | 26.45% |
| weighted__platt | 53.4% | 53.4% | 0.504 | 0.7703 | 0.2712 | 0.1018 | 60.2% [45.5%, 75.2%] | 26.13% |
| weighted__iso | 53.4% | 53.4% | 0.519 | 0.7572 | 0.2669 | 0.1112 | 60.7% [45.7%, 75.5%] | 27.59% |
| stacking__platt | 53.4% | 53.4% | 0.511 | 0.7739 | 0.2731 | 0.1179 | 60.9% [45.9%, 75.6%] | 28.32% |
| stacking__iso | 53.4% | 53.4% | 0.533 | 0.7849 | 0.2713 | 0.1219 | 60.9% [45.9%, 75.6%] | 28.36% |
| regime__platt | 53.0% | 53.4% | 0.504 | 0.7734 | 0.2737 | 0.1178 | 58.9% [43.8%, 74.5%] | 26.23% |
| regime__iso | 52.6% | 53.4% | 0.502 | 0.8116 | 0.2755 | 0.1234 | 59.0% [43.9%, 74.6%] | 26.29% |

## 6. Significance tests (block bootstrap over sessions; DM/HAC; Holm)

Primary candidate per horizon was **chosen on OOF log-loss before the test was touched**:

- **1D** `weighted`: accuracy 57.7% vs majority 54.8% (bootstrap p=0.000); AUC 0.590; ECE 0.0140. At confidence ≥0.70: accuracy 70.1%, coverage 0.63%, n=913; p(acc > majority)=0.000; **p(acc ≥ 0.70)=0.495**.
  - DM log-loss vs climatology: mean Δ=+0.01356, stat=7.21, one-sided p=0.0000, Holm p=0.0000
  - DM log-loss vs best_single: mean Δ=+0.00293, stat=2.65, one-sided p=0.0041, Holm p=0.2692
  - DM log-loss vs avg: mean Δ=+0.00413, stat=2.66, one-sided p=0.0041, Holm p=0.2675
  - Regime split of confident calls: market-up n=525 acc=72.2%; market-down n=766 acc=77.7%, share of down-calls 97.3%
  - Folds beating majority baseline: 5/6
- **3D** `weighted`: accuracy 54.7% vs majority 54.3% (bootstrap p=0.392); AUC 0.547; ECE 0.0182. At confidence ≥0.70: accuracy 64.3%, coverage 1.12%, n=1,651; p(acc > majority)=0.017; **p(acc ≥ 0.70)=0.869**.
  - DM log-loss vs climatology: mean Δ=+0.00488, stat=1.33, one-sided p=0.0922, Holm p=1.0000
  - DM log-loss vs best_single: mean Δ=-0.00130, stat=-0.50, one-sided p=0.6912, Holm p=1.0000
  - DM log-loss vs avg: mean Δ=+0.00504, stat=1.76, one-sided p=0.0397, Holm p=1.0000
  - Regime split of confident calls: market-up n=7726 acc=66.1%; market-down n=13425 acc=58.8%, share of down-calls 99.7%
  - Folds beating majority baseline: 4/6
- **5D** `weighted`: accuracy 52.9% vs majority 53.9% (bootstrap p=0.770); AUC 0.528; ECE 0.0271. At confidence ≥0.70: accuracy 56.4%, coverage 0.83%, n=1,222; p(acc > majority)=0.338; **p(acc ≥ 0.70)=0.894**.
  - DM log-loss vs climatology: mean Δ=+0.00241, stat=0.54, one-sided p=0.2950, Holm p=1.0000
  - DM log-loss vs best_single: mean Δ=-0.00289, stat=-0.82, one-sided p=0.7925, Holm p=1.0000
  - DM log-loss vs avg: mean Δ=+0.01205, stat=3.28, one-sided p=0.0005, Holm p=0.0428
  - Regime split of confident calls: market-up n=10686 acc=66.8%; market-down n=16356 acc=55.2%, share of down-calls 100.0%
  - Folds beating majority baseline: 2/6
- **10D** `weighted`: accuracy 49.1% vs majority 53.4% (bootstrap p=0.998); AUC 0.478; ECE 0.0664. At confidence ≥0.70: accuracy 35.3%, coverage 1.81%, n=2,672; p(acc > majority)=0.896; **p(acc ≥ 0.70)=0.988**.
  - DM log-loss vs climatology: mean Δ=-0.01157, stat=-1.71, one-sided p=0.9563, Holm p=1.0000
  - DM log-loss vs best_single: mean Δ=-0.01425, stat=-2.27, one-sided p=0.9882, Holm p=1.0000
  - DM log-loss vs avg: mean Δ=+0.00989, stat=2.17, one-sided p=0.0151, Holm p=1.0000
  - Regime split of confident calls: market-up n=18052 acc=70.4%; market-down n=23826 acc=53.7%, share of down-calls 100.0%
  - Folds beating majority baseline: 1/6

### Ranker (pointwise HistGradientBoosting, scikit-learn stand-in) – cross-sectional experiment

| Horizon | Mean rank-IC | IC t (HAC) | Top-bottom quintile spread | Top-decile up-rate |
|---|---|---|---|---|
| 1D | 0.2192 | 37.14 | 1.13% | 56.8% |
| 3D | 0.1866 | 23.04 | 1.60% | 55.1% |
| 5D | 0.1529 | 16.84 | 1.61% | 53.1% |
| 10D | 0.1211 | 7.66 | 1.89% | 51.5% |

![reliability](figs/reliability.png)

## 7. Untouched final test (stage 12)

Dates 2025-10-12 → 2026-11-05. Forced re-runs recorded in lock: 0.

| Horizon | Primary | Acc [95% CI] | Majority | AUC | LogLoss | ECE | Acc@≥0.70 | Coverage | n@≥0.70 | p(acc≥0.70) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1D | weighted | 58.5% [56.4%, 60.5%] | 55.3% | 0.602 | 0.6730 | 0.0250 | 80.4% | 0.50% | 301 | 0.105 |
| 3D | weighted | 58.2% [54.9%, 61.6%] | 57.3% | 0.586 | 0.6767 | 0.0466 | 83.3% | 0.01% | 6 | 0.090 |
| 5D | weighted | 57.2% [54.1%, 60.6%] | 57.3% | 0.585 | 0.6803 | 0.0601 | n/a | 0.00% | 0 | n/a |
| 10D | weighted | 54.8% [49.9%, 59.5%] | 56.8% | 0.577 | 0.6854 | 0.0692 | n/a | 0.00% | 0 | n/a |

![accuracy](figs/accuracy.png)

## 8. Backtest (stage 13, final test, net of costs)

Costs/side: {'commission': 0.0036, 'sebon': 0.00015, 'slippage': 0.001}; CGT 7.5%; long-only; entry next-session open.

| Horizon | Thr | Signals | Filled | Trades | Mean net/trade [95% CI] | Hit rate | p(≤0) | Placebo mean | p vs placebo | Sharpe | MaxDD | Cum ret |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1D | 0.60 | 523 | 499 | 499 | -0.24% [-0.50%, 0.03%] | 40.9% | 0.962 | -0.78% | 0.000 | 0.64 | -18.2% | 17.2% |
| 1D | 0.70 | 11 | 11 | – | too few trades | | | | | | | |
| 3D | 0.60 | 243 | 223 | 223 | 0.10% [-0.64%, 1.37%] | 41.3% | 0.425 | -0.45% | 0.003 | 1.52 | -3.6% | 20.4% |
| 3D | 0.70 | 0 | 0 | – | too few trades | | | | | | | |
| 5D | 0.60 | 358 | 326 | 326 | -0.14% [-0.91%, 0.74%] | 38.3% | 0.609 | -0.44% | 0.053 | 1.20 | -2.9% | 8.0% |
| 5D | 0.70 | 0 | 0 | – | too few trades | | | | | | | |
| 10D | 0.60 | 651 | 561 | 561 | 0.60% [-0.90%, 1.91%] | 49.6% | 0.231 | -0.30% | 0.000 | 1.67 | -5.3% | 6.3% |
| 10D | 0.70 | 0 | 0 | – | too few trades | | | | | | | |

![equity](figs/equity_curves.png)

## 9. What the evidence supports

- **1D:** overall test accuracy 58.5% – statistically above the majority baseline (55.3%). At ≥0.70 confidence accuracy was 80.4% on 0.50% of cases (n=301); the 95% CI is [61.0%, 89.2%], so a true accuracy of 70% is NOT statistically supported (lower bound below 70%).
- **3D:** overall test accuracy 58.2% – NOT statistically distinguishable from the majority baseline (57.3%). At ≥0.70 confidence accuracy was 83.3% on 0.01% of cases (n=6); the 95% CI is [66.7%, 100.0%], so a true accuracy of 70% is not assessable: fewer than 100 calls.
- **5D:** overall test accuracy 57.2% – NOT statistically distinguishable from the majority baseline (57.3%). No predictions reached 0.70 confidence.
- **10D:** overall test accuracy 54.8% – NOT statistically distinguishable from the majority baseline (56.8%). No predictions reached 0.70 confidence.

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