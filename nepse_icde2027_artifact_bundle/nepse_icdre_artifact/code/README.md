# NEPSE four-model stack: experiment harness for ICDE 2027

Leakage-safe walk-forward evaluation of a CatBoost / LightGBM / customised XGBoost /
Transformer ensemble with a chronological logistic stacker, for 3-class
(UP / FLAT / DOWN) direction prediction on the Nepal Stock Exchange.
One command produces every table and figure the paper needs.

```
nepse_stack/
  config.py        every experimental setting (reported in the paper's setup section)
  data.py          schema, point-in-time labels, rank normalisation, integrity audits
  splits.py        purged + embargoed walk-forward; stacker training windows
  common.py        sample weights, embargoed inner split, temperature scaling, sequences
  models.py        customised XGBoost, LightGBM, CatBoost, Transformer, baselines
  pipeline.py      out-of-fold predictions, stacker variants, leakage protocols
  evaluation.py    metrics, selective accuracy, backtest, DM / McNemar / bootstrap
  reporting.py     CSV + LaTeX tables, PDF figures
run_experiment.py  the full suite
make_architecture_figure.py   Fig. 1 (vector PDF)
tests/             17 unit tests (leakage detectors, splits, gradients, metrics, costs)
```


|---|---|
| Trading lost ~78 %/yr: a full round-trip cost was charged every day, even on unchanged holdings | costs on turnover only; gross and net returns; break-even cost in bps; top-20 %-confidence rule |
| 85 day-over-day moves beyond the ±10 % circuit band remained in the *adjusted* close (incomplete bonus/rights adjustment) and became wrong labels | labels with a forward move beyond the band are dropped and counted (`--keep-label-breaches` to disable) |
| Leakage canary missed (IC 0.24 < 0.25): heavy-tailed returns inflated the noise scale | robust (MAD) noise; threshold 0.15 (real features max 0.095) |
| 0.80 confidence flag never fired (max confidence ≈ 0.55) | accuracy on the top 10 / 20 / 30 % most confident predictions |
| Learned logistic stacker was worse than simple averaging | `--proposed` chooses the method all tests compare against; averaging ablation added |
| Configuration chosen on pooled test results would be test-set tuning | `dev_vs_holdout.csv`: block 2 = development, blocks 3–5 = holdout for every method |
| Time decay and symbol/sector features hurt XGBoost | `xgb_recommended` ablation row (both removed), to be confirmed on the holdout blocks |

## 1. Install

```bash
pip install "xgboost>=2.0" lightgbm catboost torch scikit-learn pandas numpy scipy \
            matplotlib tabulate pyarrow
pip install optuna            # optional, for tuning
python -m pytest tests -q     # 17 tests should pass
```

If a library is missing, the code substitutes a scikit-learn stand-in and prints
`[FALLBACK]`. Those numbers are for smoke tests only. **For the paper always run
with `--strict`**, which fails instead of substituting.

## 2. Input data

One row per (date, symbol), sorted or unsorted:

| column | meaning |
|---|---|
| `date` | trading date |
| `symbol`, `sector` | identifiers (used as native categoricals) |
| `close` | **corporate-action-adjusted** close (labels are built from this) |
| `close_raw` | unadjusted close (optional; only used by the corporate-action audit) |
| `volume` | optional |
| any other numeric column | a feature, which must be known at the close of `date` |

## 3. Run

```bash
python run_experiment.py --demo --quick                     # 2-minute smoke test
python run_experiment.py --data nepse.parquet --strict --proposed stack_mean_prob   # main run
python run_experiment.py --data nepse.parquet --strict --horizon 5 --out results/h5
python run_experiment.py --data nepse.parquet --strict --label-mode market_relative --out results/rel
python run_experiment.py --data nepse.parquet --strict --flat-mode warmup_quantile --out results/q
python run_experiment.py --data nepse.parquet --strict --no-rank --out results/norank
python run_experiment.py --data nepse.parquet --strict --cost-bps 80 --cgt-rate 0.075
```

Set `--cost-bps` and `--cgt-rate` from the **current** NEPSE broker commission,
SEBON fee, DP charge and capital-gains-tax schedule; the defaults are placeholders.

## 4. What each output is for

| file in `results/<run>/` | paper use |
|---|---|
| `tables/main.tex` | main comparison: baselines, each base model, 5 combiner variants |
| `tables/dev_vs_holdout.csv` | every method on the development block vs the holdout blocks |
| `tables/avg_ablation.tex` | leave-one-base-model-out for probability averaging |
| `tables/xgb_ablation.tex` | one-switch-at-a-time XGBoost customisation ablation |
| `tables/stack_ablation.tex` | leave-one-base-model-out stacker ablation |
| `tables/significance.csv` | DM (log loss), McNemar, block-bootstrap CIs, BH-adjusted |
| `tables/protocols.csv`, `figures/protocols.pdf` | leakage experiment: embargo sweep, no purge, random k-fold, shuffled labels |
| `tables/trading_*.tex`, `figures/equity.pdf` | long-only, net-of-cost economic evaluation |
| `figures/reliability.pdf`, `figures/risk_coverage.pdf`, `tables/selective_curve.csv` | calibration and the high-confidence flag |
| `tables/per_regime_block.csv`, `tables/per_sector.csv` | robustness across market regimes and sectors |
| `tables/efficiency.csv`, `tables/timings.csv` | training / inference cost per model |
| `tables/stacker_weights.csv` | how much the stacker relies on each base model, per block |
| `data_quality.json`, `leakage_audit.json` | data-integrity section; canary detection proves the audit works |
| `config.json`, `tables/folds.csv` | exact protocol, for reproducibility |
| `predictions_seed_avg.csv.gz` | released predictions |
| `SUMMARY.md` | one-page overview of the run |

## 5. XGBoost customisations (all switchable)

| switch | default | ablation row |
|---|---|---|
| class-balanced sample weights | on | `xgb_no_class_balance` |
| time-decay weights (half-life 500 days) | on | `xgb_no_time_decay` |
| native categorical symbol / sector | on | `xgb_no_symbol_sector` |
| objective | softmax | `xgb_focal_g1`, `xgb_focal_g2` (custom focal loss) |
| early-stopping metric | log loss | `xgb_stop_macro_f1`, `xgb_stop_mcc` |
| temperature scaling before the stacker | on | `xgb_no_calibration` |
| regularised hyperparameters + early stopping | on | `xgb_original_350x4` (your original model) |

Early stopping and temperature fitting both use the last 15 % of each training
window, separated by the 25-day embargo. The test block is never touched.

## 7. Reporting accuracy honestly

3-class daily direction on a real market is close to unpredictable; balanced accuracy a
few points above 0.333 is a genuine result, and reviewers treat very high accuracy as a
leakage signal. Lead with balanced accuracy, macro-F1 and MCC with confidence intervals,
show calibration, and present higher accuracy only together with its coverage
(e.g. "x % accuracy on the y % of days flagged high-confidence").

## 8. Efficiency experiments (`run_efficiency.py`)

```bash
python run_efficiency.py --data nepse.parquet --strict --out results/eff
```
Writes `EFFICIENCY.md`, `hardware.json` (CPU, RAM, OS, Python and library versions for the
setup section) and `tables/eff_prep_scale.csv` (E1, data-preparation scale-up),
`eff_incremental.csv` (E2, incremental daily update vs full rebuild with an equality check,
plus one-day inference latency) and `eff_model_scale.csv` (E3, fit time vs training rows).
Add `--with-transformer` to include the Transformer (slow on CPU). Run it on the same
machine as the main run so the hardware description matches.
