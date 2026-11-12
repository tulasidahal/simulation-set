# NEPSE ICDE 2027 paper: artifact bundle (what is in this zip)

| folder | content |
|---|---|
| `code/` | full experiment harness `nepse_stack` (v3): config, data audits, splits, models (CatBoost, LightGBM, customised XGBoost, Transformer), pipeline, evaluation, reporting, `run_experiment.py`, `run_efficiency.py`, architecture-figure script, 18 unit tests, README; `legacy_nepse_xgb.py` is the early standalone XGBoost module |
| `results_run1/` | everything from the first full run on the real data: `SUMMARY.md`, `config.json`, `data_quality.json`, `leakage_audit.json`, `tables/` (per-model and per-seed metrics, ablations, significance, protocols, trading, efficiency, folds, regimes, stacker weights), `figures/` |
| `manuscript/` | paper (v8 .docx), Pandoc template and build scripts that regenerate every table from `results_run1/`, reference list, scope statement, submission checklist, Fig. 1 |
| `data/` | `SYNTHETIC_sample_panel.csv` only (synthetic, for smoke tests: `python run_experiment.py --demo --quick`) |

## NOT in this zip (never uploaded to this session; they are on your machine )
- cleaned floorsheet data
- OHLCV data
- the training panel (the parquet/CSV passed as `--data`)
- the database
- run-2 results and the efficiency run output
- per-model prediction files (run 1 saved none; run 2 writes `predictions_seed_avg.csv.gz`)

Add them under `data/` (e.g. `data/floorsheet_clean/`, `data/ohlcv/`, `data/training_panel.parquet`, `database/`) before publishing the repository. Licence: Laganisutra permits release of the data with the paper's artifacts.

## Reproduce
```
cd code && pip install xgboost lightgbm catboost torch scikit-learn pandas numpy scipy matplotlib tabulate pyarrow
python -m pytest tests -q
python run_experiment.py --data ../data/training_panel.parquet --strict --proposed stack_mean_prob --out ../results_run2
python run_efficiency.py --data ../data/training_panel.parquet --strict --out ../results_run2/eff
```
Library versions of run 1: XGBoost 3.1.3, LightGBM 4.6.0, CatBoost 1.2.8, PyTorch 2.10.0 (CPU).
