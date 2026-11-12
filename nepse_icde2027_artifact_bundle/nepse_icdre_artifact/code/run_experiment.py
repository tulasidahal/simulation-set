#!/usr/bin/env python3
"""
Run the full NEPSE experiment suite and write every table and figure the
paper needs into one results folder.

Examples
  python run_experiment.py --demo --quick                 # synthetic smoke test
  python run_experiment.py --data nepse.parquet --strict  # full run, 5 seeds
  python run_experiment.py --data nepse.parquet --strict --proposed stack_mean_prob
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nepse_stack import data as D
from nepse_stack import evaluation as E
from nepse_stack import reporting as R
from nepse_stack.config import ExperimentConfig
from nepse_stack.models import XGB_ABLATIONS, has
from nepse_stack.pipeline import (STACK_VARIANTS, block_of_rows, prepare, run_base_models,
                                  run_protocols, run_stacker)
from nepse_stack.splits import walk_forward

BASELINES = ["majority", "persistence", "momentum20", "logreg"]
MAIN_COLS = ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "kappa",
             "log_loss", "brier", "ece", "aurc"]
TRADE_COLS = ["ann_return", "sharpe", "max_drawdown", "gross_ann_return", "gross_sharpe",
              "breakeven_cost_bps",
              "avg_turnover", "hit_rate", "exposure"]


def parse():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    a.add_argument("--data", help="panel file (.csv or .parquet)")
    a.add_argument("--demo", action="store_true", help="use a synthetic panel")
    a.add_argument("--out", default="results/run")
    a.add_argument("--seeds", type=int, nargs="+")
    a.add_argument("--horizon", type=int)
    a.add_argument("--flat-threshold", type=float)
    a.add_argument("--flat-mode", choices=["fixed", "warmup_quantile"])
    a.add_argument("--label-mode", choices=["absolute", "market_relative"])
    a.add_argument("--n-blocks", type=int)
    a.add_argument("--embargo", type=int)
    a.add_argument("--no-rank", action="store_true", help="disable cross-sectional ranking")
    a.add_argument("--keep-label-breaches", action="store_true",
                   help="keep labels whose move exceeds the circuit band")
    a.add_argument("--models", nargs="+")
    a.add_argument("--proposed", help="method the significance tests compare against")
    a.add_argument("--cost-bps", type=float, help="round-trip cost in basis points")
    a.add_argument("--cgt-rate", type=float)
    a.add_argument("--skip-ablation", action="store_true")
    a.add_argument("--skip-protocols", action="store_true")
    a.add_argument("--strict", action="store_true", help="fail if any library is missing")
    a.add_argument("--quick", action="store_true", help="2 seeds, 4 blocks, 200 bootstraps")
    return a.parse_args()


def build_config(a) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for arg, field in [("horizon", "horizon"), ("flat_threshold", "flat_threshold"),
                       ("flat_mode", "flat_mode"), ("label_mode", "label_mode"),
                       ("n_blocks", "n_blocks"), ("embargo", "embargo_days"),
                       ("cost_bps", "cost_bps_roundtrip"), ("cgt_rate", "cgt_rate"),
                       ("proposed", "proposed")]:
        if getattr(a, arg) is not None:
            setattr(cfg, field, getattr(a, arg))
    if a.seeds:
        cfg.seeds = tuple(a.seeds)
    if a.models:
        cfg.models = tuple(a.models)
    if a.no_rank:
        cfg.cross_sectional_rank = False
    if a.keep_label_breaches:
        cfg.drop_label_breaches = False
    if a.strict:
        cfg.allow_fallbacks = False
    if a.quick:
        cfg.seeds = cfg.seeds[:2]
        cfg.n_blocks = min(cfg.n_blocks, 4)
        cfg.n_bootstrap = 200
        cfg.embargo_days = min(cfg.embargo_days, 10)
    return cfg


def evaluate_method(y, P, frame, cfg) -> tuple[dict, dict]:
    m = E.classification_metrics(y, P)
    m["aurc"] = E.aurc(y, P)
    for c in cfg.coverages:
        m[f"acc_top{int(c * 100)}"] = E.accuracy_at_coverage(y, P, c)
    m["max_confidence_p99"] = float(np.quantile(P.max(1), 0.99))
    trades = {}
    for rule in ("argmax", "top_conf"):
        bt, _ = E.backtest(frame, P, cfg, rule, cfg.trade_top_frac)
        trades[rule] = bt
    return m, trades


def main():
    a = parse()
    if not (a.data or a.demo):
        sys.exit("give --data PATH or --demo")
    cfg = build_config(a)
    PROPOSED = cfg.proposed
    COV_COLS = [f"acc_top{int(c * 100)}" for c in cfg.coverages]
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------- data
    raw = D.make_synthetic_panel() if a.demo else D.load_panel(a.data)
    prep = prepare(raw, cfg)
    df, dates, y = prep.df, prep.df[cfg.date_col], prep.y
    folds = walk_forward(dates, cfg)
    blocks = block_of_rows(dates, folds)
    ev = blocks >= cfg.min_stack_blocks
    frame_ev = df.loc[ev].reset_index(drop=True)
    y_ev, blk_ev = y[ev], blocks[ev]
    print(f"rows {len(df):,}  symbols {df[cfg.symbol_col].nunique()}  dates {dates.nunique()}"
          f"  features {len(prep.feats)}  tau {prep.tau:.4f}  eval rows {ev.sum():,}"
          f"  label breaches dropped {prep.quality.get('label_breaches_dropped', 0)}")

    fallback_libs = [lib for m, lib in [("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
                                        ("catboost", "catboost"), ("transformer", "torch")]
                     if m in cfg.models and not has(lib)]
    R.write_json({**cfg.to_dict(), "tau": prep.tau, "features": prep.feats,
                  "fallback_libraries": fallback_libs, "demo": a.demo}, out / "config.json")
    R.write_json(prep.quality, out / "data_quality.json")
    pd.DataFrame([f.describe() for f in folds]).assign(
        evaluated=lambda d: d.block >= cfg.min_stack_blocks).to_csv(out / "tables/folds.csv", index=False)
    canary = D.canary_validation(df, prep.feats, cfg)
    R.write_json({"real_features": prep.leakage, "detector_validation": canary},
                 out / "leakage_audit.json")

    # ------------------------------------------------------------- seeds
    base_specs = {m: (m, {}) for m in cfg.models}
    base_specs.update({b: (b, {}) for b in BASELINES})
    abl_specs = {} if a.skip_ablation else {k: ("xgboost", v) for k, v in XGB_ABLATIONS.items()
                                            if k != "xgb_full"}
    rows, trade_rows, block_rows, sector_rows, timings, coef_rows = [], [], [], [], [], []
    P_sum = {}
    for seed in cfg.seeds:
        print(f"\n== seed {seed}")
        oof, tm = run_base_models(prep, cfg, {**base_specs, **abl_specs}, seed, folds)
        timings += tm
        if "xgboost" in oof and abl_specs:
            oof["xgb_full"] = oof["xgboost"]
        names = list(cfg.models)
        preds = {k: oof[k] for k in oof}
        for v in STACK_VARIANTS:
            res = run_stacker(oof, names, prep, cfg, folds, v, return_coefs=(v == "lr_logits"))
            if v == "lr_logits":
                res, coefs = res
                coef_rows += [{"seed": seed, **c} for c in coefs]
            preds[f"stack_{v}"] = res
        for drop in names if len(names) > 1 else []:
            keep = [n for n in names if n != drop]
            preds[f"stack_without_{drop}"] = run_stacker(oof, keep, prep, cfg, folds, "lr_logits")
            preds[f"avg_without_{drop}"] = run_stacker(oof, keep, prep, cfg, folds, "mean_prob")

        for name, LP in preds.items():
            P = np.exp(LP[ev])
            P /= P.sum(1, keepdims=True)
            P_sum[name] = P_sum.get(name, 0) + P
            m, tr = evaluate_method(y_ev, P, frame_ev, cfg)
            rows.append({"seed": seed, "method": name, **m})
            for rule, bt in tr.items():
                trade_rows.append({"seed": seed, "method": name, "rule": rule,
                                   **bt["strategy"],
                                   **{f"bench_{k}": v for k, v in bt["benchmark"].items()}})
            # Per block for every method: block 2 is the development block,
            # blocks 3+ are the holdout used to confirm any configuration choice.
            for b in sorted(set(blk_ev)):
                mb = blk_ev == b
                block_rows.append({"seed": seed, "method": name, "block": b,
                                   **E.classification_metrics(y_ev[mb], P[mb])})
            if not name.startswith(("xgb_", "stack_without", "avg_without")):
                for s, ms in frame_ev.groupby(cfg.sector_col).indices.items():
                    sector_rows.append({"seed": seed, "method": name, "sector": s,
                                        **E.classification_metrics(y_ev[ms], P[ms])})

    res = pd.DataFrame(rows)
    res.to_csv(out / "tables/metrics_by_seed.csv", index=False)
    trades = pd.DataFrame(trade_rows)
    trades.to_csv(out / "tables/trading_by_seed.csv", index=False)
    pd.DataFrame(timings).to_csv(out / "tables/timings.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out / "tables/stacker_weights.csv", index=False)

    # ------------------------------------------------------------- main tables
    groups = {
        "main": ["majority", "persistence", "momentum20", "logreg", *cfg.models,
                 "stack_mean_prob", "stack_mean_logprob", "stack_lr_probs",
                 "stack_lr_logits_balanced", "stack_lr_logits"],
        "stack_ablation": ["stack_lr_logits"] + [f"stack_without_{m}" for m in cfg.models],
        "avg_ablation": ["stack_mean_prob"] + [f"avg_without_{m}" for m in cfg.models],
        "xgb_ablation": (["xgb_full"] + list(abl_specs)) if abl_specs else [],
    }
    if PROPOSED not in groups["main"]:
        groups["main"].append(PROPOSED)
    for gname, methods in groups.items():
        methods = [m for m in methods if m in set(res.method)]
        if not methods:
            continue
        sub = res[res.method.isin(methods)]
        tab = R.mean_std_table(sub, "method", MAIN_COLS + COV_COLS).loc[methods]
        tab.to_csv(out / f"tables/{gname}.csv")
        R.to_latex(tab[["accuracy", "balanced_accuracy", "macro_f1", "mcc", "log_loss", "ece",
                        COV_COLS[1]]],
                   out / f"tables/{gname}.tex",
                   caption=f"{gname.replace('_', ' ').capitalize()}: mean $\\pm$ std over "
                           f"{len(cfg.seeds)} seeds, walk-forward test blocks.",
                   label=f"tab:{gname}",
                   bold_best={"balanced_accuracy": "max", "macro_f1": "max", "mcc": "max",
                              "log_loss": "min", "ece": "min", COV_COLS[1]: "max"})

    tsub = trades[trades.method.isin(groups["main"])]
    for rule in ("argmax", "top_conf"):
        t = R.mean_std_table(tsub[tsub.rule == rule], "method", TRADE_COLS)
        t.to_csv(out / f"tables/trading_{rule}.csv")
        R.to_latex(t[["ann_return", "sharpe", "gross_ann_return", "breakeven_cost_bps"]],
                   out / f"tables/trading_{rule}.tex",
                   caption=f"Long-only trading, {rule} rule, net of "
                           f"{cfg.cost_bps_roundtrip:.0f} bps round-trip costs on turnover.",
                   label=f"tab:trading_{rule}", bold_best={"sharpe": "max"})
    bench = trades[trades.rule == "argmax"].filter(like="bench_").mean()
    bench.to_csv(out / "tables/trading_benchmark.csv")

    bdf = pd.DataFrame(block_rows)
    bdf.groupby(["method", "block"])[["accuracy", "balanced_accuracy", "macro_f1", "mcc",
                                      "log_loss"]].mean().round(4).to_csv(out / "tables/per_regime_block.csv")
    dev_block = min(blk_ev)
    dh = bdf.assign(split=np.where(bdf.block == dev_block, "dev", "holdout")) \
        .groupby(["method", "split"])[["balanced_accuracy", "macro_f1", "mcc", "log_loss"]].mean() \
        .unstack("split")
    dh.columns = [f"{m}_{s}" for m, s in dh.columns]
    dh.round(4).to_csv(out / "tables/dev_vs_holdout.csv")
    pd.DataFrame(sector_rows).groupby(["method", "sector"])[["n", "balanced_accuracy",
        "macro_f1", "mcc"]].mean().round(4).to_csv(out / "tables/per_sector.csv")

    # ------------------------------------------------------------- significance
    Pbar = {k: v / len(cfg.seeds) for k, v in P_sum.items()}
    codes, uniq = pd.factorize(frame_ev[cfg.date_col])
    sig = []
    Pp = Pbar[PROPOSED]
    cm_p = E.confusion_by_date(y_ev, Pp.argmax(1), codes, len(uniq))
    others = [m for m in groups["main"] + groups["stack_ablation"] + groups["avg_ablation"]
              if m != PROPOSED and m in Pbar]
    for other in dict.fromkeys(others):
        Po = Pbar[other]
        dm = E.diebold_mariano(E.per_row_log_loss(y_ev, Pp), E.per_row_log_loss(y_ev, Po),
                               frame_ev[cfg.date_col].to_numpy(), cfg.horizon)
        mc = E.mcnemar(Pp.argmax(1) == y_ev, Po.argmax(1) == y_ev)
        cm_o = E.confusion_by_date(y_ev, Po.argmax(1), codes, len(uniq))
        bm = E.block_bootstrap_diff(cm_p, cm_o, E.mcc_cm, cfg.n_bootstrap)
        bf = E.block_bootstrap_diff(cm_p, cm_o, E.macro_f1_cm, cfg.n_bootstrap)
        sig.append({"proposed": PROPOSED, "vs": other, "dm_stat_logloss": dm["dm_stat"],
                    "dm_p": dm["p_value"], "mcnemar_p": mc["p_value"], "mcc_diff": bm["diff"],
                    "mcc_ci_low": bm["ci_low"], "mcc_ci_high": bm["ci_high"],
                    "macro_f1_diff": bf["diff"], "f1_ci_low": bf["ci_low"],
                    "f1_ci_high": bf["ci_high"], "boot_p_mcc": bm["p_one_sided"]})
    sig = pd.DataFrame(sig)
    if len(sig):
        for c in ["dm_p", "mcnemar_p", "boot_p_mcc"]:
            sig[c + "_bh"] = E.benjamini_hochberg(sig[c].to_numpy())
    sig.round(4).to_csv(out / "tables/significance.csv", index=False)

    # ------------------------------------------------------------- protocols
    if not a.skip_protocols:
        print("\n== leakage / protocol experiments (seed 0, xgboost)")
        pr = run_protocols(prep, cfg, cfg.seeds[0], "xgboost",
                           embargo_sweep=(0, 5, cfg.embargo_days) if a.quick else (0, 5, 10, 25, 50))
        prow = []
        for k, LP in pr["preds"].items():
            mask = pr["eval_rows"] & ~np.isnan(LP).any(1)
            P = np.exp(LP[mask])
            prow.append({"protocol": k, "coverage_of_eval_rows": mask.sum() / pr["eval_rows"].sum(),
                         **E.classification_metrics(y[mask], P)})
        ptab = pd.DataFrame(prow).set_index("protocol")
        ptab.round(4).to_csv(out / "tables/protocols.csv")
        R.protocol_figure(ptab, "balanced_accuracy", out / "figures/protocols")

    # ------------------------------------------------------------- efficiency + figures
    tdf = pd.DataFrame(timings)
    eff = tdf.groupby("model").agg(fit_s_mean=("fit_s", "mean"), predict_s_mean=("predict_s", "mean"),
                                   n_train_mean=("n_train", "mean"), n_test_mean=("n_test", "mean"),
                                   iterations_mean=("n_iter", "mean"),
                                   temperature_mean=("temperature", "mean"),
                                   fallback=("fallback", "max"))
    eff["predict_rows_per_s"] = eff.n_test_mean / eff.predict_s_mean.replace(0, np.nan)
    eff.round(4).to_csv(out / "tables/efficiency.csv")

    fig_methods = list(dict.fromkeys([PROPOSED, "stack_mean_prob", "stack_lr_logits", "catboost",
                                      "logreg"]))
    curves = {k: (y_ev, Pbar[k]) for k in fig_methods if k in Pbar}
    R.reliability_figure(curves, out / "figures/reliability")
    R.risk_coverage_figure({k: E.selective_curve(y_ev, Pbar[k]) for k in curves},
                           out / "figures/risk_coverage")
    _, eq = E.backtest(frame_ev, Pbar[PROPOSED], cfg, "top_conf", cfg.trade_top_frac)
    R.equity_figure(eq, out / "figures/equity", f"{PROPOSED}, top {int(cfg.trade_top_frac * 100)}% confidence")
    pd.concat({k: E.selective_curve(y_ev, Pbar[k]) for k in curves}, names=["method"]) \
        .round(4).to_csv(out / "tables/selective_curve.csv")

    pred_out = frame_ev[[cfg.date_col, cfg.symbol_col, cfg.sector_col, "label", "fwd_ret_abs"]].copy()
    for k in dict.fromkeys([PROPOSED, "stack_mean_prob", *cfg.models, "logreg"]):
        if k in Pbar:
            for j, c in enumerate(D.CLASS_NAMES):
                pred_out[f"{k}_p_{c}"] = Pbar[k][:, j].round(5)
    pred_out.to_csv(out / "predictions_seed_avg.csv.gz", index=False)

    # ------------------------------------------------------------- summary
    main = pd.read_csv(out / "tables/main.csv", index_col=0)
    lines = [f"# Results summary ({time.strftime('%Y-%m-%d %H:%M')}, {time.time() - t_start:.0f}s)\n"]
    if a.demo:
        lines.append("> SYNTHETIC DATA. These numbers only show the pipeline runs.\n")
    if fallback_libs:
        lines.append(f"> FALLBACK MODELS USED (missing: {', '.join(fallback_libs)}). "
                     "Not valid for the paper. Install the libraries and rerun with --strict. "
                     "Stand-ins ignore XGBoost-specific switches (objective, stop metric, "
                     "params), so those ablation rows repeat xgb_full.\n")
    lines += [f"Rows evaluated: {ev.sum():,} | seeds: {list(cfg.seeds)} | horizon: {cfg.horizon} | "
              f"tau: {prep.tau:.4f} | embargo: {cfg.embargo_days} | proposed: {PROPOSED} | "
              f"label breaches dropped: {prep.quality.get('label_breaches_dropped', 0)}\n",
              "## Main comparison (mean ± std over seeds)\n",
              main[["accuracy", "balanced_accuracy", "macro_f1", "mcc", "log_loss", "ece"]
                   + COV_COLS].to_markdown()]
    if len(sig):
        lines += [f"\n## {PROPOSED} vs others (seed-averaged probabilities)\n",
                  sig[["vs", "dm_stat_logloss", "dm_p_bh", "mcnemar_p_bh", "mcc_diff",
                       "mcc_ci_low", "mcc_ci_high"]].round(4).to_markdown(index=False)]
    tconf = pd.read_csv(out / "tables/trading_top_conf.csv", index_col=0)
    lines += ["\n## Trading, top-confidence rule (net of costs on turnover)\n",
              tconf[["ann_return", "sharpe", "gross_ann_return", "avg_turnover", "breakeven_cost_bps"]].to_markdown(),
              f"\nBenchmark (equal-weight universe): ann. return {bench.get('bench_ann_return', np.nan):.3f}, "
              f"Sharpe {bench.get('bench_sharpe', np.nan):.2f}"]
    if not a.skip_protocols:
        lines += ["\n## Protocol / leakage experiment (xgboost, seed 0)\n",
                  ptab[["balanced_accuracy", "macro_f1", "mcc", "log_loss"]].round(4).to_markdown()]
    lines += ["\n## Data audits\n",
              f"- Corporate-action breaches: " + ", ".join(
                  f"{k}: {v['n_breaches']}" for k, v in prep.quality["corporate_actions"].items()),
              f"- Labels dropped (move beyond circuit band): {prep.quality.get('label_breaches_dropped', 0)}",
              f"- Features flagged by leakage IC audit (|IC| > {cfg.leak_ic_threshold}): "
              f"{list(prep.leakage['flagged']) or 'none'}",
              f"- Planted canary detected: {canary['canary_flagged']} (IC {canary['canary_ic']})"]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nDone in {time.time() - t_start:.0f}s -> {out}/SUMMARY.md")


if __name__ == "__main__":
    main()
