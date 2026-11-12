"""
Stage 12  Untouched final test (last FINAL_TEST sessions).
Protocol (fixed BEFORE any test prediction is evaluated):
  1. Select, per horizon, the primary candidate = lowest OOF log-loss among ensemble/calibration candidates (stage 11 results),
     and the best single model by OOF log-loss. Selection is written to selection.json and hashed into final_test.lock.
  2. Refit every base model on ALL development sessions (embargo already separates dev from test).
  3. Fit ensemble weights/stackers on all dev OOF predictions; fit calibrators on the second-level OOF outputs.
  4. Predict the test sessions once; evaluate with block bootstrap + DM tests. Re-running requires --force and is recorded.
"""
import argparse, hashlib, joblib
import numpy as np, pandas as pd
from common import *
from metrics import classification_metrics
import s05_audit as A, s06_08_models as M, s09_10_ensemble_calibration as EC, s11_stats as S

LOCK = WORK / "final_test.lock"

def select(stats):
    sel = {}
    for h in HORIZONS:
        C = stats["horizons"][f"{h}d"]["candidates"]
        ens = [k for k in C if k.split("__")[0] in EC.APPROACHES]; base = [k for k in C if "__" not in k and k not in EC.APPROACHES]
        sel[f"{h}d"] = {"primary": min(ens, key=lambda k: C[k]["logloss"]), "best_single": min(base, key=lambda k: C[k]["logloss"]),
                        "oof_logloss_primary": C[min(ens, key=lambda k: C[k]["logloss"])]["logloss"]}
    return sel

def run(B=1000, force=False):
    if LOCK.exists() and not force: raise SystemExit(f"{LOCK} exists: the final test was already run. Re-running contaminates it; use --force to override (recorded).")
    panel = pd.read_pickle(WORK / "panel.pkl"); splits = load_json(WORK / "splits.json"); market = pd.read_csv(WORK / "market.csv")
    E = joblib.load(WORK / "ens_oof.pkl"); O = joblib.load(WORK / "oof.pkl"); stats = load_json(REPORTS / "stage11_stats.json")
    models, cfg = E["models"], O["cfg"]; fold_id = E["fold_id"]
    sel = select(stats); sel_hash = hashlib.sha256(json.dumps(sel, sort_keys=True).encode()).hexdigest(); save_json(sel, WORK / "selection.json")
    prev = load_json(LOCK) if LOCK.exists() else {}
    save_json({"selection_sha256": sel_hash, "manifest_sha256": splits["manifest_sha256"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running",
               "forced_reruns": prev.get("forced_reruns", 0) + (1 if prev else 0)}, LOCK)
    log(f"selection locked: {sel}")
    X, Y, sess = M.prepare(panel); dev = A.dev_mask(panel, splits); test = A.test_mask(panel, splits)
    log(f"refit on {dev.sum():,} dev rows; predict {test.sum():,} test rows over {FINAL_TEST} sessions")
    res, be, tf = M.run_models(panel, X, Y, sess, dev, test, models + ["ranker"], cfg, cfg["threads"], save_dir=MODELS, tag="_final")
    for m in models: assert be[m] == O["backends"][m], f"backend mismatch for {m}: OOF used {O['backends'][m]}, final used {be[m]}"
    ti = np.where(test)[0]; U = np.asarray(splits["U"]); pos = np.searchsorted(U, sess)
    mret60 = np.nan_to_num(market.mkt_ret60.values[sess]); mvol = np.nan_to_num(market.mkt_vol20.values[sess]); n = len(panel)
    TP, report = {}, {"selection": sel, "backends": be, "horizons": {}, "test_dates": None, "forced_reruns": load_json(LOCK)["forced_reruns"]}
    for h in HORIZONS:
        y_all = panel[f"y_{h}"].values; oofp = E["preds"][h]
        P_dev = np.column_stack([E["preds"][h][m] for m in models]); avail = ~np.isnan(P_dev).any(1) & (fold_id > 0); lab = avail & ~np.isnan(y_all)
        ens = EC.Ensembler(models).fit(P_dev[lab], y_all[lab].astype(int), mret60[lab], mvol[lab])
        Pt = np.column_stack([res[m][h] for m in models]); raw = ens.predict(Pt, mret60[ti], mvol[ti])
        for m, j in zip(models, range(len(models))): raw[m] = Pt[:, j]
        out = dict(raw)
        for name in list(raw):
            first = 2 if name in EC.APPROACHES else 1
            tr = lab & (fold_id >= first) & ~np.isnan(oofp[name])
            for meth in ("platt", "iso"):
                out[f"{name}__{meth}"] = EC.Calibrator(meth).fit(oofp[name][tr].astype("float64"), y_all[tr].astype(int)).predict(raw[name].astype("float64"))
        TP[h] = {k: v.astype("float32") for k, v in out.items()}; TP[h]["ranker"] = res["ranker"][h]
        joblib.dump({"ensembler": ens, "models": models, "calibrators_note": "refit on OOF; see test_predictions for outputs"}, MODELS / f"second_level_h{h}.joblib")
        # ---- evaluation ----
        yt = y_all[ti]; ok = ~np.isnan(yt); y, s = yt[ok].astype(int), sess[ti][ok]
        clim = np.clip(y_all[lab].mean(), 1e-3, 1 - 1e-3); lag = max(h, 5) + 5
        H = {"n": int(ok.sum()), "sessions": int(len(np.unique(s))), "candidates": {}, "dm": {}}
        keep = [k for k in TP[h] if k != "ranker"]
        for k in keep: H["candidates"][k] = S.eval_block(k, y, TP[h][k][ok].astype("float64"), s, B, with_auc="__" not in k)
        pc = np.full(len(y), clim); _, l_clim = S.session_losses(y, pc, s); prim, bs = sel[f"{h}d"]["primary"], sel[f"{h}d"]["best_single"]
        fam, keys = [], []
        for nme in {prim, bs, "avg", "stacking"}:
            _, ln = S.session_losses(y, TP[h][nme][ok].astype("float64"), s)
            for ref, lr in (("climatology", l_clim), ("best_single_by_oof", S.session_losses(y, TP[h][bs][ok].astype("float64"), s)[1]), ("avg", S.session_losses(y, TP[h]["avg"][ok].astype("float64"), s)[1])):
                if nme == (bs if ref == "best_single_by_oof" else "avg" if ref == "avg" else None): continue
                r = S.dm_test(lr, ln, lag); H["dm"][f"{nme}_vs_{ref}"] = r; fam.append(r["p_one_sided_B_better"]); keys.append(f"{nme}_vs_{ref}")
        for kk, a in zip(keys, S.holm(fam)): H["dm"][kk]["p_holm"] = float(a)
        H["reliability_primary"] = S.reliability(y, TP[h][prim][ok].astype("float64"))
        H["ranker"] = S.rank_ic(panel.iloc[ti].reset_index(drop=True), res["ranker"][h], h, np.ones(len(ti), bool)) if len(ti) > 1000 else None
        report["horizons"][f"{h}d"] = H
        c = H["candidates"][prim]; log(f"[{h}d] primary={prim}: acc={c['accuracy']:.4f} (maj {c['majority_baseline_acc']:.4f}) auc={c['auc'] if 'auc' in c else float('nan'):.4f} ece={c['ece']:.4f} | sel70 acc={c['sel70']['accuracy']} cov={c['sel70']['coverage']:.4f} n={c['sel70']['n']}")
    d0, d1 = panel.loc[panel.sess == splits["test_sess"][0], "date"].iloc[0], panel.loc[panel.sess == splits["test_sess"][1], "date"].iloc[0]
    report["test_dates"] = [str(d0.date()), str(d1.date())]
    meta = panel.iloc[ti][["symbol", "date", "sess", "amount", "adj_close"] + [f"fwd_ret_{h}" for h in HORIZONS] + [f"y_{h}" for h in HORIZONS] + [f"trade_ret_{h}" for h in HORIZONS]].reset_index(drop=True)
    joblib.dump({"meta": meta, "preds": TP, "selection": sel, "ti": ti}, WORK / "test_predictions.pkl")
    save_json(report, REPORTS / "stage12_final_test.json")
    lk = load_json(LOCK); lk["status"] = "complete"; save_json(lk, LOCK); log("FINAL TEST complete -> stage12_final_test.json (lock written)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bootstrap-B", type=int, default=1000); ap.add_argument("--force", action="store_true"); a = ap.parse_args(); run(a.bootstrap_B, a.force)
