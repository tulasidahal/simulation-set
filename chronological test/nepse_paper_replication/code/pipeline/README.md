# NEPSE multi-horizon forecasting pipeline

Thirteen stages from raw floorsheet dumps to a research report. Run everything with `bash run_pipeline.sh`
(see the header of that file for folders and environment variables). Each stage writes a JSON report to
`$NEPSE_WORK/reports/`; `python s13_backtest_export.py --data-only` writes `data_report.md` from stages 01–05 alone.

| Stage | File | What it does |
|---|---|---|
| 01–02 | `s01_02_ingest.py` | Checksums/CRC/schema of the zips; streams 91M trades into compact monthly buckets (32 bytes/trade, resumable with checkpoints); global de-duplication on the transaction number; daily OHLCV + broker aggregates |
| 01b | `s01b_aux_sources.py` | Parses, validates and de-duplicates the OHLCV, fundamentals and feature-store SQL dumps; audits each for look-ahead |
| 03–04 | `s03_04_panel.py` | Universe, adjusted bars, 53 core + 7 point-in-time fundamental predictors, targets for 1/3/5/10 sessions |
| 05 | `s05_audit.py` | Leakage and split audit (504-session first window, 25-session embargo, 8 folds, 250-session final test, truncation test) |
| 06–08 | `s06_08_models.py` | CatBoost, LightGBM, XGBoost, logistic regression, Transformer; LightGBM ranker and TFT-style model; out-of-fold predictions |
| 09–10 | `s09_10_ensemble_calibration.py` | Average, weighted, stacking, regime-aware ensembles; Platt and isotonic calibration, nested walk-forward |
| 11 | `s11_stats.py` | Metrics, session block bootstrap, Diebold–Mariano (HAC) with Holm correction, rank-IC |
| 12 | `s12_final_test.py` | Selection locked on out-of-fold log-loss, single run on the untouched final test |
| 13 | `s13_backtest_export.py` | Long-only net-of-cost backtest vs random placebo, figures, model manifest, `research_report.md` |
| 14 | `s14_feature_group_importance.py` | Optional: grouped permutation importance on development folds 7–8 (never the final test) → `drivers.json` |

## Model families

* `FAMILY=native` (default): CatBoost, LightGBM, XGBoost, logistic regression, Transformer, TFT-style model, LightGBM LambdaRank. Needs the libraries in *Requirements*.
* `FAMILY=sklearn`: HistGradientBoosting, ExtraTrees, logistic regression, MLP, 20-session windowed MLP and a pointwise HistGradientBoosting ranker. Runs with scikit-learn only.
  The results in `research_report.md` shipped with this package come from this family; the report says so in its header. Rerun with
  `FAMILY=native START=6 bash run_pipeline.sh` to produce results for the named algorithms.
* `NEPSE_FEATURESET=core` drops the seven fundamentals (ablation); run it in its own `NEPSE_WORK` folder.

## Data decisions (details and numbers in `data_report.md`)

* **Duplicates** are removed on the transaction number across all three parts (39,500 exact copies, all in part 3, July 2026).
* **Dealer trades** with broker code `D01` are real trades and are kept (they were previously dropped as "invalid broker").
* **Floorsheet gaps**: Feb 2022 – Mar 2023 and Mar 2024 have only ~2–60 stocks per session. Prices for listed stocks come from
  the adjusted OHLCV table (complete); broker features stay missing in the gaps rather than being imputed.
* **Survivorship**: the OHLCV table lists only current symbols, so delisted/merged companies (e.g. NCCB, NBB, MEGA, CBL, BOKL) use floorsheet prices.
* **Universe**: ordinary shares only; promoter shares, debentures and closed-end mutual funds are excluded.
* **Fundamentals** are usable from quarter end + 45 days. The `date` column in `fundamental_data.sql` is ~30 days *before* quarter end, so it is not used.
  The nine single-quarter `fundamental_*` snapshot tables and `nepse_prediction_featurestore` (whose fundamentals switch before quarter end) are not model inputs.

## Requirements

`pandas numpy scipy scikit-learn joblib matplotlib lightgbm catboost xgboost torch`.
`--allow-fallback` (stage 06) swaps missing libraries for scikit-learn stand-ins **for smoke tests only**; the report then
carries a warning banner and those numbers must not be reported as the named models.
