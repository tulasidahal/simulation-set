# NEPSE forecasting: everything needed to train the models yourself

This file describes the data, features, targets, splits, model settings and metrics used in the study, so the same
walk-forward experiment can be rerun with XGBoost, LightGBM, CatBoost, a Transformer or a TFT in any Python environment
. Numbers produced by the study's scikit-learn stand-ins are at the end, as the reference to compare with.

## 1. Files

| File | What it is |
| --- | --- |
| `nepse_model_panel.csv.gz` | 360,943 stock-sessions x 71 columns (70 MB compressed): `symbol`, `date`, `sess`, 60 features, 4 labels `y_1 y_3 y_5 y_10`, 4 forward returns `fwd_ret_*` |
| `nepse_model_panel_lite.csv.gz` | Same columns for the 100 most liquid stocks only (113,050 rows, 22 MB) for sandboxes with little memory |
| `splits.json` | The 8 validation folds, the development period and the final-test period, as `sess` ranges |
| `train_nepse.py` | Complete walk-forward training script (XGBoost, LightGBM, CatBoost, HistGradientBoosting fallback, weighted ensemble, all metrics) |

`sess` is an integer trading-session index shared by all stocks (one value per trading day). Missing values are blank cells and must be left missing
for XGBoost, LightGBM and CatBoost (they handle them natively); for neural networks use median imputation plus a missing-flag.

## 2. Source data and cleaning (what produced the panel)

- 91,048,648 NEPSE floorsheet trades (Oct 2018 – 5 Nov 2026) were parsed from three zips; 39,500 exact duplicate transaction numbers were removed (99.957% kept). The `sn` transaction number encodes the date.
- Trades were aggregated to one row per stock-session: OHLC, VWAP, volume, amount, trade count, and broker concentration (buyer and seller HHI, top-1 and top-5 shares, broker counts, self-trade share).
- Prices come from the adjusted OHLCV table (back-adjusted for bonus and rights) for 287 stocks; 82 delisted stocks use floorsheet prices. Moves above 11% (the NEPSE limit is 10%) were neutralised on 225 symbol-sessions.
- Floorsheet coverage is thin (20–60 stocks a session) in 14 months (Feb 2022, Apr 2022 – Mar 2023, Mar 2024). Floorsheet and broker features are blank there; prices are complete.
- Fundamentals (7 features) are point-in-time: a quarterly report is used only from quarter end + 45 days. E/P and B/P are recomputed from the quarter-end price and the adjusted return since.
- Universe: ordinary shares only (promoter shares, debentures and mutual funds excluded); 362 symbols, 1,855 sessions.
- Excluded as leaky: the nine single-quarter `fundamental_*` tables and `nepse_prediction_featurestore` (its fundamentals switch before the quarter ends).

## 3. Features (60) — all use data available at the close of session t

| Group | Columns |
| --- | --- |
| Returns and momentum (11) | `ret1 ret2 ret3 ret5 ret10 ret20 ret60 rel_ret5 drawdown60 dist_high252 dist_low252` |
| Volatility and range (7) | `vol5 vol10 vol20 vol60 hl_range atr14_pct bb_pos` |
| Candle shape and gaps (4) | `oc_ret close_vwap gap close_pos` |
| Trend and oscillators (7) | `ma5_dist ma10_dist ma20_dist ma50_dist rsi14 macd_hist stoch_k14` |
| Liquidity and volume (9) | `volratio_1_20 volratio_5_20 log_amount log_trades amihud20 stale20 turnover_trend avg_trade_size trades_ratio20` |
| Broker concentration (13) | `buyer_hhi seller_hhi buyer_top1 seller_top1 buyer_top5 seller_top5 log_buyers log_sellers self_share large_share size_cv broker_imbalance buyer_hhi_chg5` |
| NEPSE market (2) | `mkt_ret1 mkt_vol20` |
| Fundamentals (7) | `fund_ep fund_bp fund_roe fund_de fund_rev_growth fund_eps_yoy fund_age` |

Exact formulas are in `s03_04_panel.py` of `nepse_pipeline.zip`. A leakage test (rebuilding every feature from data truncated at each cut-off) found zero differences.

## 4. Targets

`y_h` = 1 if the adjusted close h sessions ahead is above today's close, 0 if below, blank if unchanged or not yet known. h = 1, 3, 5, 10.
`fwd_ret_h` is the matching return. Up-rate in the labelled data: 45.5%, 45.7%, 46.2%, 46.6%. The majority class is therefore "down" and the baseline accuracy is 54–57%, not 50%.

## 5. Splits (walk-forward, expanding window)

Session indices (`sess`), as in `splits.json`. Each fold trains on everything from the first session to its train end, then skips a 25-session embargo before validating.

| Fold | Train `sess` | Validation `sess` | Train dates | Validation dates |
| --- | --- | --- | --- | --- |
| 1 | 312–815 | 841–971 | 2018-10-01→2021-01-21 | 2021-02-28→2021-09-09 |
| 2 | 312–946 | 972–1102 | 2018-10-01→2021-08-02 | 2021-09-12→2022-04-03 |
| 3 | 312–1077 | 1103–1233 | 2018-10-01→2022-02-21 | 2022-04-04→2022-09-29 |
| 4 | 312–1208 | 1234–1364 | 2018-10-01→2022-08-26 | 2022-10-09→2023-05-08 |
| 5 | 312–1339 | 1365–1495 | 2018-10-01→2023-04-02 | 2023-05-09→2023-12-05 |
| 6 | 312–1470 | 1496–1626 | 2018-10-01→2023-10-16 | 2023-12-06→2024-06-26 |
| 7 | 312–1601 | 1627–1757 | 2018-10-01→2024-05-19 | 2024-06-27→2025-01-22 |
| 8 | 312–1732 | 1758–1891 | 2018-10-01→2024-12-12 | 2025-01-23→2025-08-17 |

- Out-of-fold scoring uses folds 3–8 (789 sessions), the folds that have at least 766 training sessions.
- Development period: `sess` 312–1891. Final test: `sess` 1917–2168 (250 sessions, 2025-10-12 → 2026-11-05), separated from development by a 25-session embargo.
- Choose the ensemble and its weights on out-of-fold log-loss only, fit the final models once on all development sessions, and score the test once.
- Early stopping uses the last 12% of training sessions, purged by 15 sessions, never random rows.

## 6. Models and settings

| Model | Settings |
| --- | --- |
| XGBoost | `XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, tree_method="hist", early_stopping_rounds=100, eval_metric="logloss")` |
| LightGBM | `LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5)`, early stopping 100 |
| CatBoost | `CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=5, loss_function="Logloss")`, early stopping 100 |
| Logistic regression | L2, C=0.1, median imputation then a quantile transform to normal |
| Transformer | Encoder, 3 layers, d_model 64, 4 heads, feed-forward 128, dropout 0.2, pre-norm, learned positions; input = last 30 sessions of the 60 features per stock; one network with four sigmoid heads (one per horizon); AdamW lr 1e-3, weight decay 1e-2, gradient clip 1.0; early stopping on validation log-loss |
| TFT-style | Per-feature embeddings with variable selection (gated residual network), LSTM encoder (d=32), gated skip connections, 4-head attention from the last step over the window, four heads; same optimiser and stopping |
| LightGBM ranker | `LGBMRanker` (LambdaRank) on within-session return quintiles (relevance 0–4); group = session; evaluated by rank-IC and top-minus-bottom quintile return |
| Neural inputs | Robust scaling (median, IQR/1.349), clipped to ±5, missing set to 0 plus flags for `close_vwap` and `fund_ep` |

Seed 42 throughout. The study capped ExtraTrees and the MLPs at 120,000–150,000 training rows per fit; `train_nepse.py` caps every model at 250,000 randomly chosen training rows by default (`--row-cap`), which is the main speed setting to raise if memory allows.

## 7. Ensembles, calibration and metrics

- **Ensembles:** simple average; weighted average (convex weights minimising out-of-fold log-loss); logistic stacking on logits; regime-aware stacking (market trend x volatility regime).
- **Calibration:** Platt scaling and isotonic regression, fitted on earlier folds only (nested walk-forward).
- **Metrics:** accuracy against the majority-class baseline, AUC, log-loss, Brier, ECE (10 bins), and selective accuracy with coverage at confidence = max(p, 1 - p) >= 0.55, 0.60, 0.65, 0.70.
- **Tests:** session-block bootstrap (resample whole sessions), Diebold–Mariano with HAC variance and Holm correction, rank-IC for the ranker.
- **Backtest:** long-only, buy names whose calibrated P(up) >= threshold at the close, enter at the next open, exit at the close of t + h; costs per side 0.36% commission + 0.015% SEBON + 0.10% slippage; 7.5% capital-gains tax on profits.

## 8. Reference results from the study (scikit-learn stand-ins; compare your XGBoost/LightGBM run with these)

Final test, weighted ensemble:

| Horizon | Accuracy | Majority baseline | AUC | Log-loss | ECE |
| --- | --- | --- | --- | --- | --- |
| 1 session | 59.0% | 55.3% | 0.608 | 0.6720 | 0.032 |
| 3 sessions | 59.2% | 57.3% | 0.610 | 0.6732 | 0.049 |
| 5 sessions | 58.3% | 57.3% | 0.597 | 0.6801 | 0.062 |
| 10 sessions | 55.3% | 56.8% | 0.584 | 0.6863 | 0.073 |

Out-of-fold (folds 3–8) weighted ensemble: accuracy 57.1%, 54.1%, 52.8%, 52.3% against baselines 54.8%, 54.5%, 54.1%, 53.8%.
At confidence >= 0.70 the 1-session ensemble was right 81.1% of the time on 0.40% of stock-sessions (243 calls, 95% interval 63.3–85.2%).

## 9. How to run

```bash
pip install xgboost lightgbm catboost scikit-learn pandas scipy
python train_nepse.py --data nepse_model_panel.csv.gz --splits splits.json --models xgboost,lightgbm,catboost --horizons 1 3 5 10
# small sandbox:  --data nepse_model_panel_lite.csv.gz --row-cap 100000 --folds 8 --horizons 1
```

The script prints per-fold accuracy, writes `results.json` (out-of-fold and test metrics for every model and the weighted ensemble) and never touches the test period except for the final scoring step.

## 10. training set

> Attached are `nepse_model_panel_lite.csv.gz`, `splits.json`, `train_nepse.py` and `TRAINING_SPEC.md`. Read TRAINING_SPEC.md first. Run `train_nepse.py` with `--models xgboost,lightgbm --folds 8 --horizons 1 3` (install packages if the sandbox allows), then report accuracy against the majority baseline, AUC, log-loss, ECE and the selective-accuracy table for each model and the weighted ensemble. Do not tune on the test period, do not change the splits, and do not report accuracy conditional on confidence without its coverage and sample size.
