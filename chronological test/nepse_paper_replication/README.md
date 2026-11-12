# NEPSE Forecasting Study — Replication Package

Everything behind the paper: source databases, cleaned data, training sets, all code, every
stage report, per-model predictions and the final results, for all three experimental runs.

Built 2026-11-11. 118 files, 394 MiB uncompressed. `MANIFEST.csv` lists every file with its
size and SHA-256.

---

## The three runs

All three score an **identical, untouched final test**: the last 250 sessions, 2025-10-12 to
2026-11-05, 60,526 scored rows at the 1-session horizon. They differ only in which sessions
entered training.

| Run | Floorsheet | Thin sessions | Panel rows | Where |
|---|---|---|---|---|
| **A — original** | the three original archives | kept | 360,943 | `results/run_A_original_archives/` |
| **B — gap excluded** | the three original archives | excluded (317 sessions) | 305,474 | `results/run_B_gap_excluded/` |
| **C — recovered** *(primary)* | archives + 88 recovered day files | excluded (173 sessions) | 333,786 | `results/run_C_recovered/` |

Headline, 1-session horizon, against a 55.3% majority baseline:
**A 59.0% · B 58.6% · C 58.5%**, each with bootstrap p < 0.001.
Full comparison: `results/final_results_three_runs.csv`.

A session is **thin** when fewer than 50% of its eligible stocks appear in the floorsheet.
Runs of 5 or more consecutive thin sessions are dropped along with the 19 sessions that
follow each run, so no rolling or sequence window reads a thin session.

---

## Layout

### `code/`
- **`pipeline/`** — the pipeline, stage by stage. `s01_02_ingest.py` (floorsheet ingest),
  `s01b_aux_sources.py` (OHLCV, fundamentals, feature store), `s02b_supplement.py` (merge
  recovered day files), `s03_04_panel.py` (panel and targets, and the thin-session rule),
  `s05_audit.py` (leakage and split audit), `s06_08_models.py` (six model families, walk-forward
  out-of-fold), `s09_10_ensemble_calibration.py`, `s11_stats.py`, `s12_final_test.py`,
  `s13_backtest_export.py`, `s14_feature_group_importance.py`, plus `common.py`, `metrics.py`,
  `run_pipeline.sh`, `export_pack.py` and `make_training_set.py`.
- **`training_pack/`** — `TRAINING_SPEC.md`, `train_nepse.py`, `splits.json`: enough to retrain
  elsewhere without this repository.
- **`standalone/`** — `train_models.py`, `ensemble_multihorizon.py`: the first scripts written
  for this study, superseded by `pipeline/` and kept only for provenance.
- **`document_build/`** — scripts that generate the `.docx` reports in `documents/` directly
  from the JSON stage reports.

### `data/sql_sources/`
The databases as supplied, gzipped.
`ohlcv_data.sql.gz` (adjusted OHLCV, the price source),
`nepse_prediction_featurestore.sql.gz` (the pre-built feature store — **audited and not used
for modelling**: 99.7% of its fundamental switches occur before the quarter end it reports, so
it is not point-in-time), and `fundamentals/` (ten tables, used point-in-time with a 45-day
publication lag).

### `data/floorsheet/`
- `daily_floorsheet_v1_original_archives.csv.gz` — 91,009,140 trades aggregated to daily
  symbol-session bars from the three original archives.
- `daily_floorsheet_v2_supplemented.csv.gz` — 94,671,327 trades, after the recovered day files
  were merged. **This is the floorsheet the primary run uses.**
- `session_coverage_and_missing_dates.csv` — all 1,858 sessions with floorsheet coverage before
  and after the merge, status (complete / thin / absent), and whether the session is excluded
  from training.
- `recovered_day_files/` — the 88 recovered SQL dumps as supplied: `days_2022-02-10_to_2022-07-27.zip`
  (87 per-day files) and `floorsheet_missing_20240317_20240331.sql.gz`. 3,687,558 trades, parsed
  with no unparsed lines, no rejected rows and no duplicate transaction numbers; a strict superset
  of what the originals held for those dates.

**Still missing from the archive**: 154 sessions, 2022-07-28 to 2023-03-30; ten Fridays
(2022-05-20, 05-27, 06-03, 06-10, 06-17, 06-24, 07-01, 07-08, 07-15, 07-22); and nineteen
isolated days listed in the coverage register. The paper should state this rather than let a
reader infer continuous coverage from 2018 to 2026.

### `data/panel/`
- `panel_run_C_recovered.csv.gz` — the full panel for the primary run: 408,032 rows × 82 columns,
  including the 60 predictors, the four forward returns and their binary targets, the eligibility
  flag, prices and tradable returns. **This is the training data for the primary run.**
- `training_set_lite_100_stocks.csv.gz` — the 100-stock lite panel used by the separate native
  model runs.
- `instrument_meta.csv`, `market_index_by_session.csv`, and `splits_run_*.json` (fold manifests).

To build the model-ready training set for run C (eligible rows, 71 columns):
`python code/pipeline/make_training_set.py data/panel/panel_run_C_recovered.csv.gz out.csv.gz`

**Numeric precision.** Panels are written with `%.9g`, which round-trips `float32` exactly — the
dtype of 73 of the 82 columns — so the training data is bit-identical to what the models read.
The single `float64` column, `adj_close`, is written at full `repr`. Prediction files use `%.6g`:
six significant figures on a probability reproduces every metric in `results/` far beyond the
precision they are reported to, and the exact arrays can be regenerated by rerunning the pipeline.

### `results/`
- `final_results_three_runs.csv` — the comparison table: accuracy with confidence interval,
  AUC, log-loss, calibration error, significance, the 70%-confidence operating point and the
  cross-sectional ranker, for 4 horizons × 3 runs.
- `run_*/` — every stage report as JSON, the figures, and `research_report.md` for that run.
  `stage12_final_test.json` holds all 28 candidates per horizon; `stage13_backtest.json` holds
  the trading results; `drivers.json` holds the feature-group analysis.
- `predictions/` — per-model out-of-fold predictions (6 models × 4 horizons) and final-test
  predictions (28 candidates × 4 horizons) as CSV, for runs B and C.

### `documents/`
The technical documentation, the two comparison reports (thin-session exclusion, recovered days)
in both `.docx` and `.pdf`, two earlier ICDE manuscript versions, and the model execution register.

---

## Reproducing

```bash
export PYTHONPATH=$PWD/code/pipeline NEPSE_WORK=./work NEPSE_AUX=<fundamentals dir> NEPSE_UPLOADS=<sql dir>

python code/pipeline/s01b_aux_sources.py                    # parse OHLCV + fundamentals
python code/pipeline/s02b_supplement.py --src <recovered>   # merge recovered day files
NEPSE_EXCLUDE_GAP=1 FAMILY=sklearn START=3 bash code/pipeline/run_pipeline.sh
```

Stage 01–02 (the 91-million-trade ingest) can be skipped by placing
`daily_floorsheet_v2_supplemented.csv.gz` at `work/daily_floorsheet.csv.gz`. The rebuild was
verified this way: with the thin-session rule off it reproduced the original run exactly —
405,491 panel rows, 360,943 eligible, 362 symbols, 1,855 sessions and all four label counts.

The thin-session thresholds are `GAP_MIN_COVERAGE`, `GAP_MIN_BLOCK` and `GAP_BUFFER` in
`code/pipeline/common.py`.

---

## Two things to know

**The models are scikit-learn stand-ins.** CatBoost, LightGBM, XGBoost, the Transformer and the
TFT are implemented in `s06_08_models.py` under `--family native` but could not be installed in
the environment these runs were executed in, so no number in `results/` comes from them. The
families that did run are histogram gradient boosting, extremely randomised trees, logistic
regression, a multilayer perceptron, a sequence MLP and a gradient-boosted ranker
(`results/predictions/model_backends_*.csv`). The separate native lite-panel runs are recorded
in the model execution register in `documents/`, and none of their figures appear here.

**Two large files are referenced rather than included**, to keep the package under the 400 MiB
transfer limit. Both are files you already hold, and both are regenerable from what is here.

| File | Bytes | SHA-256 | How to regenerate |
|---|---|---|---|
| `floorsheet_data_part1.zip` | 393,370,161 | `2f0d17037a60c3a032ce7b572dbc6874dba23f2a4974c2ae9cacf37f42011595` | raw archive; `data/floorsheet/daily_floorsheet_v1_original_archives.csv.gz` is its cleaned form |
| `floorsheet_data_part2.zip` | 204,507,687 | `f1dc39c48ca63786b75ee6fedc594590315a5e0ffc1ea2ef54b1bec060039365` | — |
| `floorsheet_data_part3.zip` | 357,184,144 | `25f403e6889a11c6c19371a8fbf0d7a427cfab0ee38296b2edb130d7794583fc` | — |
| `nepse_model_panel.csv.gz` (training set for runs A and B) | 70,505,525 | `4c65c5a4412931e0be8f621839cee09ffc846b77f40ddceae5e339f2d08e9506` | run `s03_04_panel.py` on `daily_floorsheet_v1_original_archives.csv.gz`, then `make_training_set.py` |

Run A's per-model prediction files were not retained when its working directory was reclaimed;
its complete stage reports are in `results/run_A_original_archives/`. Predictions for runs B and
C are included in full.
