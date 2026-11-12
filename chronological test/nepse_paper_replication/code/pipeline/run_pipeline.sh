#!/usr/bin/env bash
# Full NEPSE pipeline, stages 01-13. Every stage is resumable or idempotent; rerun from any stage with START=<n>.
#
#   pip install pandas numpy scipy scikit-learn joblib matplotlib lightgbm catboost xgboost torch
#   NEPSE_UPLOADS=/path/to/zips_and_sql NEPSE_AUX=/path/to/fundamental_sql NEPSE_WORK=./work bash run_pipeline.sh
#
# NEPSE_UPLOADS : folder with floorsheet_data_part{1,2,3}.zip, ohlcv_data.sql, nepse_prediction_featurestore.sql
# NEPSE_AUX     : folder with the fundamental_*.sql files (defaults to NEPSE_UPLOADS)
# NEPSE_WORK    : output folder (cleaned data, models, reports)
# FAMILY        : "native" (CatBoost, LightGBM, XGBoost, logistic, Transformer, TFT, LambdaRank; default)
#                 or "sklearn" (HistGradientBoosting, ExtraTrees, logistic, MLP, windowed MLP, HGB ranker)
# NEPSE_FEATURESET : "all" (53 core + 7 fundamentals, default) or "core" (53 core) - run both for the ablation,
#                    each in its own NEPSE_WORK folder (copy the stage 01-05 outputs across to skip re-cleaning).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export NEPSE_UPLOADS="${NEPSE_UPLOADS:-/mnt/user-data/uploads}"
export NEPSE_AUX="${NEPSE_AUX:-$NEPSE_UPLOADS}"
export NEPSE_WORK="${NEPSE_WORK:-./work}"
START="${START:-1}"
FAMILY="${FAMILY:-native}"
run() { echo "=== stage $1: $2"; shift 2; python "$@"; }
[ "$START" -le 1 ]  && run 1  "clean + de-duplicate floorsheets"      s01_02_ingest.py
[ "$START" -le 2 ]  && run 2  "validate auxiliary sources"            s01b_aux_sources.py
[ "$START" -le 3 ]  && run 3  "panel, features, targets"              s03_04_panel.py
[ "$START" -le 5 ]  && run 5  "leakage and split audit"               s05_audit.py
[ "$START" -le 6 ]  && run 6  "base models, ranker, TFT, OOF"         s06_08_models.py --family "$FAMILY"
[ "$START" -le 9 ]  && run 9  "ensembles + calibration"               s09_10_ensemble_calibration.py
[ "$START" -le 11 ] && run 11 "statistical evaluation"                s11_stats.py --bootstrap-B 1000
[ "$START" -le 12 ] && run 12 "untouched final test (runs once)"      s12_final_test.py --bootstrap-B 1000
[ "$START" -le 13 ] && run 13 "backtest, export, research report"     s13_backtest_export.py
[ "$START" -le 14 ] && run 14 "feature-group importance (dev folds only)" s14_feature_group_importance.py
echo "done -> $NEPSE_WORK/reports/research_report.md"
