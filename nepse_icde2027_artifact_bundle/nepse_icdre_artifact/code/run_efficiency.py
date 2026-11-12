#!/usr/bin/env python3
"""
Data-engineering efficiency experiments for the paper (CPU only).

  E1  data-preparation scale-up: time of validation, labelling, rank normalisation and
      the integrity audits as the number of securities and dates grows;
  E2  incremental daily update: append one trading day with an incremental update and
      with a full rebuild, check the two give identical features and labels, and time
      one-day inference for the trained model;
  E3  model scale-up: fit time and inference throughput against training-set size.

Also records the hardware and library versions the paper's setup section asks for.

  python run_efficiency.py --demo --quick              # smoke test, synthetic panel
  python run_efficiency.py --data nepse.parquet --strict --out results/eff
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from nepse_stack import data as D
from nepse_stack.config import ExperimentConfig
from nepse_stack.models import has, make_model
from nepse_stack.pipeline import model_input, prepare


def med_time(fn, reps):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def hardware():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_model": platform.processor() or "n/a", "cpu_cores_logical": os.cpu_count()}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip(); break
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        pass
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "catboost", "torch"]:
        try: info[p] = metadata.version(p)
        except Exception: info[p] = "not installed"
    return info


# ----------------------------------------------------------------------------- E1
def prep_stages(raw, cfg, feats):
    t = {}
    t0 = time.perf_counter(); df = D.validate_schema(raw, cfg); t["validate"] = time.perf_counter() - t0
    t0 = time.perf_counter(); df, _ = D.add_returns_and_labels(df, cfg); t["labels"] = time.perf_counter() - t0
    t0 = time.perf_counter(); D.corporate_action_audit(df, cfg); t["breach_audit"] = time.perf_counter() - t0
    t0 = time.perf_counter(); D.leakage_ic_audit(df, feats, cfg); t["ic_scan"] = time.perf_counter() - t0
    t0 = time.perf_counter(); D.rank_normalise(df, feats, cfg); t["rank_norm"] = time.perf_counter() - t0
    t["total"] = sum(t.values()); t["rows"] = len(df)
    return t


def exp_prep_scale(raw, cfg, feats, fracs, reps):
    syms = np.array(sorted(raw[cfg.symbol_col].unique()))
    dates = np.array(sorted(raw[cfg.date_col].unique()))
    rows = []
    for fs in fracs:
        for fd in fracs:
            ss = syms[: max(5, int(len(syms) * fs))]
            dd = dates[-max(60, int(len(dates) * fd)):]
            sub = raw[raw[cfg.symbol_col].isin(ss) & raw[cfg.date_col].isin(dd)]
            runs = [prep_stages(sub, cfg, feats) for _ in range(reps)]
            r = {k: float(np.median([x[k] for x in runs])) for k in runs[0]}
            r.update(securities=len(ss), dates=len(dd), us_per_row=r["total"] / r["rows"] * 1e6)
            rows.append(r)
            print(f"  E1 {len(ss):>4} sec x {len(dd):>5} dates  {r['rows']:>8,} rows  {r['total']:.2f}s")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E2
def incremental_update(tail, new_day, cfg, feats):
    """
    Append one trading day. `tail` holds the last h+1 trading days of raw rows
    (all that the labels need). Returns (ranked features of the new day,
    labels that became available, new tail). Cost is independent of history length.
    """
    new_day = D.validate_schema(new_day, cfg)
    ranked = D.rank_normalise(new_day, feats, cfg)                       # same-day only
    both = pd.concat([tail, new_day[tail.columns]], ignore_index=True).sort_values(
        [cfg.symbol_col, cfg.date_col])
    g = both.groupby(cfg.symbol_col, sort=False)[cfg.price_col]
    both["fwd_ret"] = g.shift(-cfg.horizon) / both[cfg.price_col] - 1
    d_lab = sorted(both[cfg.date_col].unique())[-cfg.horizon - 1]
    lab = both[both[cfg.date_col] == d_lab][[cfg.symbol_col, cfg.date_col, "fwd_ret"]].dropna()
    keep = sorted(both[cfg.date_col].unique())[-(cfg.horizon + 1):]
    return ranked, lab, both[both[cfg.date_col].isin(keep)][tail.columns].reset_index(drop=True)


def exp_incremental(raw, cfg, feats, model_name, reps):
    raw = D.validate_schema(raw, cfg)
    last = raw[cfg.date_col].max()
    hist, new = raw[raw[cfg.date_col] < last], raw[raw[cfg.date_col] == last]
    # full rebuild on the extended panel (what a non-incremental store would do)
    def full():
        df = D.validate_schema(raw, cfg)
        lab, _ = D.add_returns_and_labels(df.copy(), cfg)
        rk = D.rank_normalise(df, feats, cfg)
        return lab, rk
    t_full, (lab_full, rk_full) = med_time(full, reps)
    d_all = sorted(hist[cfg.date_col].unique())
    tail = hist[hist[cfg.date_col].isin(d_all[-(cfg.horizon + 1):])][raw.columns].reset_index(drop=True)
    t_inc, (rk_inc, lab_inc, _) = med_time(lambda: incremental_update(tail, new, cfg, feats), reps)
    # equality with the full rebuild
    a = rk_full[rk_full[cfg.date_col] == last].set_index(cfg.symbol_col)[feats].sort_index()
    b = rk_inc.set_index(cfg.symbol_col)[feats].sort_index()
    feat_ok = bool(np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12))
    d_lab = lab_inc[cfg.date_col].iloc[0]
    la = lab_full[lab_full[cfg.date_col] == d_lab].set_index(cfg.symbol_col)["fwd_ret"].sort_index()
    lb = lab_inc.set_index(cfg.symbol_col)["fwd_ret"].sort_index()
    ix = la.index.intersection(lb.index)
    # the full pipeline drops breach labels; only compare rows it kept
    lab_ok = bool(np.allclose(la.loc[ix].to_numpy(), lb.loc[ix].to_numpy(), atol=1e-12)) and len(ix) > 0
    # inference latency for the new day with a model trained on the history
    prep = prepare(hist, cfg, feats)
    y = prep.y; dts = prep.df[cfg.date_col]
    m = make_model(model_name, cfg, 0)
    m.fit(model_input(prep, m.input_kind, cfg, None), y, dts)
    X = rk_inc.assign(**{cfg.sector_col: rk_inc[cfg.sector_col]})[feats + [cfg.symbol_col, cfg.sector_col]]
    t_inf, _ = med_time(lambda: m.predict_log_proba(X), reps)
    return {"securities": int(new[cfg.symbol_col].nunique()), "history_rows": int(len(hist)),
            "full_rebuild_s": t_full, "incremental_s": t_inc, "speedup": t_full / t_inc,
            "features_identical": feat_ok, "labels_identical": lab_ok,
            "inference_model": model_name + (" (fallback)" if m.is_fallback else ""),
            "one_day_inference_s": t_inf, "one_day_total_s": t_inc + t_inf}


# ----------------------------------------------------------------------------- E3
def exp_model_scale(raw, cfg, models, fracs, reps):
    prep = prepare(raw, cfg)
    df, y = prep.df, prep.y
    dts = df[cfg.date_col]
    ud = np.sort(dts.unique()); cut = ud[int(len(ud) * 0.8)]
    te = (dts >= cut).to_numpy(); tr_all = np.flatnonzero(~te & (dts < ud[int(len(ud) * 0.8) - 26]).to_numpy())
    rows = []
    for name in models:
        for f in fracs:
            tr = np.zeros(len(df), bool); tr[tr_all[-int(len(tr_all) * f):]] = True
            fit_t, pr_t, fb = [], [], False
            for r in range(reps):
                m = make_model(name, cfg, r); fb = m.is_fallback
                m.fit(model_input(prep, m.input_kind, cfg, tr), y[tr], dts[tr])
                m.predict_log_proba(model_input(prep, m.input_kind, cfg, te))
                fit_t.append(m.fit_seconds); pr_t.append(m.predict_seconds)
            rows.append({"model": name, "fallback": fb, "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                         "fit_s": float(np.median(fit_t)), "predict_s": float(np.median(pr_t)),
                         "infer_rows_per_s": float(te.sum() / np.median(pr_t))})
            print(f"  E3 {name:12s} {tr.sum():>8,} rows  fit {rows[-1]['fit_s']:.1f}s")
    return pd.DataFrame(rows)


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    a.add_argument("--data"); a.add_argument("--demo", action="store_true")
    a.add_argument("--out", default="results/eff"); a.add_argument("--quick", action="store_true")
    a.add_argument("--strict", action="store_true"); a.add_argument("--with-transformer", action="store_true")
    a.add_argument("--reps", type=int)
    a = a.parse_args()
    if not (a.data or a.demo): sys.exit("give --data PATH or --demo")
    cfg = ExperimentConfig(allow_fallbacks=not a.strict)
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    raw = D.make_synthetic_panel() if a.demo else D.load_panel(a.data)
    raw = D.validate_schema(raw, cfg)
    feats = D.feature_columns(raw, cfg)
    reps = a.reps or (2 if a.quick else 5)
    fracs = [0.5, 1.0] if a.quick else [0.25, 0.5, 0.75, 1.0]
    hw = hardware(); (out / "hardware.json").write_text(json.dumps(hw, indent=2))
    print("hardware:", hw)

    e1 = exp_prep_scale(raw, cfg, feats, fracs, reps); e1.to_csv(out / "tables/eff_prep_scale.csv", index=False)
    inc_model = "lightgbm"
    e2 = exp_incremental(raw, cfg, feats, inc_model, reps); pd.DataFrame([e2]).to_csv(out / "tables/eff_incremental.csv", index=False)
    models = ["lightgbm", "xgboost", "catboost"] + (["transformer"] if a.with_transformer else [])
    e3 = exp_model_scale(raw, cfg, models, fracs[1:] if a.quick else [0.25, 0.5, 1.0], max(1, reps // 2))
    e3.to_csv(out / "tables/eff_model_scale.csv", index=False)

    full = e1[(e1.securities == e1.securities.max())]
    sy = e1[(e1.dates == e1.dates.max())]
    lines = ["# Efficiency summary", "", f"CPU: {hw.get('cpu_model')} · logical cores {hw.get('cpu_cores_logical')} · RAM {hw.get('ram_gb','?')} GB",
             f"Python {hw['python']} · numpy {hw['numpy']} · pandas {hw['pandas']} · scikit-learn {hw['scikit-learn']}", "",
             "## E1 data preparation (median seconds, all stages)", e1.round(3).to_markdown(index=False), "",
             f"Log-log slope of total time vs rows: {slope(e1.rows, e1.total):.2f} (1.0 = linear); "
             f"vs dates at full securities: {slope(full.dates, full.total):.2f}; vs securities at full dates: {slope(sy.securities, sy.total):.2f}", "",
             "## E2 incremental daily update", pd.DataFrame([e2]).round(5).T.to_markdown(), "",
             "## E3 model scale-up", e3.round(3).to_markdown(index=False), ""]
    for m, g in e3.groupby("model"):
        if len(g) > 1: lines.append(f"Fit-time slope vs training rows, {m}: {slope(g.train_rows, g.fit_s):.2f}")
    (out / "EFFICIENCY.md").write_text("\n".join(lines)); print("\n".join(lines[-12:]))
    assert e2["features_identical"] and e2["labels_identical"], "incremental update differs from full rebuild"


if __name__ == "__main__":
    main()
