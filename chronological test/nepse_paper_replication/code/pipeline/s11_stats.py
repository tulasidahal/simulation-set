"""
Stage 11  Statistical evaluation (nested-OOF region: folds >= EVAL_FIRST_FOLD).
  * metrics: accuracy, balanced acc, MCC, AUC, log-loss, Brier, ECE, selective accuracy/coverage at confidence thresholds
  * circular moving-block bootstrap over sessions (95% CIs, one-sided p-values)
  * Diebold-Mariano (HAC / Newey-West, Harvey-Leybourne-Newbold small-sample correction) on session-level loss differentials
  * Holm step-down correction across comparison families
  * ranker experiments: daily rank-IC, quintile spread
  * regime breakdown and per-fold stability
"""
import argparse, joblib
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from common import *
from metrics import classification_metrics, block_bootstrap, ci, selective
import s05_audit as A

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def session_losses(y, p, sess, kind="logloss"):
    p = np.clip(p, 1e-6, 1 - 1e-6); l = -(y * np.log(p) + (1 - y) * np.log(1 - p)) if kind == "logloss" else (p - y) ** 2
    df = pd.DataFrame({"s": sess, "l": l}).groupby("s").l.mean(); return df.index.values, df.values

def dm_test(yA_loss, yB_loss, lag):
    """H0: equal expected loss; positive stat => B has lower loss than A (A worse). One-sided p for 'B better'."""
    d = np.asarray(yA_loss) - np.asarray(yB_loss); T = len(d); dm = d - d.mean()
    g0 = np.dot(dm, dm) / T; V = g0
    for j in range(1, lag + 1): V += 2 * (1 - j / (lag + 1)) * np.dot(dm[j:], dm[:-j]) / T
    V = max(V, 1e-18); stat = d.mean() / np.sqrt(V / T)
    hln = np.sqrt((T + 1 - 2 * lag + lag * (lag - 1) / T) / T) if T > 2 * lag else 1.0
    stat_c = stat * hln; p_one = 1 - stats.t.cdf(stat_c, df=T - 1); p_two = 2 * (1 - stats.t.cdf(abs(stat_c), df=T - 1))
    return {"mean_loss_diff": float(d.mean()), "dm_stat": float(stat_c), "p_one_sided_B_better": float(p_one), "p_two_sided": float(p_two), "T_sessions": int(T), "lag": int(lag)}

def climatology(y, fold_id, lab):
    c = np.full(len(y), np.nan)
    for k in np.unique(fold_id[fold_id > 0]):
        tr = lab & (fold_id < k)
        c[fold_id == k] = y[tr].mean() if tr.sum() > 0 else 0.5
    return c

def eval_block(name, y, p, sess, B, with_auc=True, thr=0.70):
    mm = classification_metrics(y, p); bb = block_bootstrap(y, p, sess, B=B, with_auc=with_auc, thr=thr)
    mm["ci"] = {k: ci(v) for k, v in bb.items()}
    base = mm["majority_baseline_acc"]
    mm["p_acc_gt_majority"] = float(np.mean(bb["accuracy"] <= base))                       # bootstrap one-sided
    sa = bb["sel_acc"][~np.isnan(bb["sel_acc"])]
    mm["sel70"] = {**mm[f"sel_{thr:.2f}"], "ci_acc": ci(bb["sel_acc"]), "p_acc_gt_majority": float(np.mean(sa <= base)) if len(sa) else None,
                   "p_acc_ge_0.70": float(np.mean(sa < 0.70)) if len(sa) else None, "ci_cov": ci(bb["sel_cov"])}
    return mm

def reliability(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); rows = []
    for b in range(bins):
        m = idx == b
        if m.sum(): rows.append({"bin": b, "mean_p": float(p[m].mean()), "freq": float(y[m].mean()), "n": int(m.sum())})
    return rows

def rank_ic(panel, scores, h, region):
    df = pd.DataFrame({"s": panel.sess.values[region], "sc": scores[region], "f": panel[f"fwd_ret_{h}"].values[region]}).dropna()
    df = df[df.groupby("s").sc.transform("size") >= 10]
    ic = df.groupby("s").apply(lambda g: spearmanr(g.sc, g.f)[0], include_groups=False).dropna()
    df["q"] = df.groupby("s").sc.rank(pct=True); top, bot = df[df.q > 0.8].f.mean(), df[df.q <= 0.2].f.mean()
    d10 = df[df.q > 0.9]; lag = max(h, 5); x = ic.values - ic.values.mean(); T = len(x); V = np.dot(x, x) / T
    for j in range(1, lag + 1): V += 2 * (1 - j / (lag + 1)) * np.dot(x[j:], x[:-j]) / T
    return {"mean_ic": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else None, "ic_t_hac": float(ic.mean() / np.sqrt(max(V, 1e-18) / T)), "sessions": int(T),
            "top_quintile_mean_fwd": float(top), "bottom_quintile_mean_fwd": float(bot), "spread": float(top - bot),
            "top_decile_up_rate": float((d10.f > 0).mean()), "pos_ic_share": float((ic > 0).mean())}

def run(B=300):
    panel = pd.read_pickle(WORK / "panel.pkl"); splits = load_json(WORK / "splits.json"); market = pd.read_csv(WORK / "market.csv")
    E = joblib.load(WORK / "ens_oof.pkl"); O = joblib.load(WORK / "oof.pkl"); fold_id = E["fold_id"]; first = E["eval_first_fold"]; models = E["models"]
    sess = panel.sess.values; out = {"region": f"folds {first}..{N_FOLDS}", "bootstrap_B": B, "horizons": {}}
    regime_up = (np.nan_to_num(market.mkt_ret60.values[sess]) > 0)
    all_p = []; keys = []
    for h in HORIZONS:
        y_all = panel[f"y_{h}"].values; preds = E["preds"][h]; lab = ~np.isnan(y_all) & (fold_id > 0)
        reg = lab & (fold_id >= first); clim = climatology(y_all, fold_id, lab)
        # common rows: every candidate present
        names = [k for k in preds if not np.isnan(preds[k][reg]).all()]; common = reg.copy()
        for k in names: common &= ~np.isnan(preds[k])
        y, s = y_all[common].astype(int), sess[common]; log(f"=== horizon {h}d: common rows {common.sum():,}, sessions {len(np.unique(s))}")
        H = {"n": int(common.sum()), "sessions": int(len(np.unique(s))), "candidates": {}, "dm": {}, "per_fold": {}, "regime": {}}
        pc = np.clip(clim[common], 1e-3, 1 - 1e-3); H["climatology_logloss"] = float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
        for nme in names:
            p = preds[nme][common].astype("float64"); raw = "__" not in nme
            H["candidates"][nme] = eval_block(nme, y, p, s, B, with_auc=raw)
            if nme in ("avg", "stacking", "regime", "stacking__iso", "stacking__platt"): H["candidates"][nme]["reliability"] = reliability(y, p)
        # ---- DM tests: B better than A on log-loss ----
        su, l_clim = session_losses(y, pc, s); lag = max(h, 5) + 5
        base_models = [m for m in models]; best_single = min(base_models, key=lambda m: H["candidates"][m]["logloss"]); H["best_single_by_logloss"] = best_single
        fam, famkeys = [], []
        for nme in names:
            _, l_n = session_losses(y, preds[nme][common].astype("float64"), s)
            for ref, lref in (("climatology", l_clim), ("best_single", session_losses(y, preds[best_single][common].astype("float64"), s)[1]), ("avg", session_losses(y, preds["avg"][common].astype("float64"), s)[1])):
                if ref == nme or (ref == "best_single" and nme == best_single) or (ref == "avg" and nme == "avg"): continue
                r = dm_test(lref, l_n, lag); H["dm"][f"{nme}_vs_{ref}"] = r; fam.append(r["p_one_sided_B_better"]); famkeys.append(f"{nme}_vs_{ref}")
        adj = holm(fam)
        for kname, a in zip(famkeys, adj): H["dm"][kname]["p_holm"] = float(a)
        # ---- per-fold stability and regime breakdown for the main approaches ----
        for nme in ["avg", "weighted", "stacking", "regime", "stacking__iso", "regime__iso"] + models:
            if nme not in preds: continue
            rows = []
            for k in range(first, N_FOLDS + 1):
                m = common & (fold_id == k)
                if m.sum() < 200: continue
                yy, pp = y_all[m].astype(int), preds[nme][m].astype("float64"); rows.append({"fold": k, "acc": float(((pp > 0.5) == yy).mean()), "maj": float(max(yy.mean(), 1 - yy.mean())), "n": int(m.sum())})
            H["per_fold"][nme] = {"folds": rows, "folds_beating_majority": int(sum(r["acc"] > r["maj"] for r in rows)), "n_folds": len(rows)}
        for nme in ["stacking__iso", "regime__iso", "stacking", "regime"]:
            if nme not in preds: continue
            p = preds[nme][common].astype("float64"); up = regime_up[common]; conf = np.maximum(p, 1 - p) >= 0.70; r = {}
            for lab_, msk in (("market_up", up), ("market_down", ~up)):
                m = conf & msk; r[lab_] = {"n_conf": int(m.sum()), "accuracy": float(((p[m] > .5) == y[m]).mean()) if m.sum() else None, "share_down_calls": float((p[m] <= .5).mean()) if m.sum() else None,
                                           "majority_acc_in_regime": float(max(y[msk].mean(), 1 - y[msk].mean())) if msk.any() else None}
            H["regime"][nme] = r
        # ---- ranker experiment ----
        if "ranker" in O["oof"]:
            sc = O["oof"]["ranker"][h]; rg = (O["fold_id"] >= first) & ~np.isnan(sc)
            if rg.sum() > 1000: H["ranker"] = rank_ic(panel, sc, h, rg)
        out["horizons"][f"{h}d"] = H
        best = max(names, key=lambda k: H["candidates"][k]["auc"] if "auc" in H["candidates"][k] and H["candidates"][k]["auc"] == H["candidates"][k]["auc"] else 0)
        log(f"  best AUC: {best} {H['candidates'][best]['auc']:.4f}; acc {H['candidates'][best]['accuracy']:.4f} vs majority {H['candidates'][best]['majority_baseline_acc']:.4f}")
    save_json(out, REPORTS / "stage11_stats.json"); log("stats done -> stage11_stats.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bootstrap-B", type=int, default=300); a = ap.parse_args(); run(a.bootstrap_B)
