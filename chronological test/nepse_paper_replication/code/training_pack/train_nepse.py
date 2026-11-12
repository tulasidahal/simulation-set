"""Walk-forward training of XGBoost / LightGBM / CatBoost (or a scikit-learn fallback) on the NEPSE panel.

Usage (in ChatGPT's Python sandbox or locally):
    python train_nepse.py --data nepse_model_panel_lite.csv.gz --splits splits.json --models xgboost,lightgbm --horizons 1 3 5 10
Add --folds 8 for a quick run (one fold); default scores folds 3-8 like the study.

Protocol (identical to the study): expanding training window, 25-session embargo already built into splits.json,
early stopping on the last 12% of training sessions (purged 15 sessions), one final fit on all development sessions,
scored once on the untouched final-test sessions. Reports accuracy vs majority baseline, AUC, log-loss, Brier, ECE
and selective accuracy at confidence >= 0.55/0.60/0.65/0.70 (confidence = max(p, 1-p)).
"""
import argparse, json, sys, time, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
SEED = 42
META = {"symbol", "date", "sess"}

def make_model(kind, n_jobs=-1):
    if kind == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5.0, tree_method="hist", early_stopping_rounds=100, eval_metric="logloss", n_jobs=n_jobs, random_state=SEED)
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=5.0, n_jobs=n_jobs, random_state=SEED, verbose=-1)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=5, loss_function="Logloss", eval_metric="Logloss",
                                  random_seed=SEED, verbose=0, thread_count=n_jobs if n_jobs > 0 else None, early_stopping_rounds=100)
    if kind == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=5.0,
                                              max_features=0.8, early_stopping=True, n_iter_no_change=50, random_state=SEED)
    raise ValueError(kind)

def fit_predict(kind, X, y, s, Xp):
    """Early-stop on the last 12% of training sessions, purged by 15 sessions; return P(up) for Xp."""
    cut = np.quantile(s, 0.88); a, e = s < cut - 15, s >= cut
    m = make_model(kind)
    if kind == "xgboost": m.fit(X[a], y[a], eval_set=[(X[e], y[e])], verbose=False)
    elif kind == "lightgbm":
        import lightgbm as lgb
        m.fit(X[a], y[a], eval_set=[(X[e], y[e])], callbacks=[lgb.early_stopping(100, verbose=False)])
    elif kind == "catboost": m.fit(X[a], y[a], eval_set=(X[e], y[e]), use_best_model=True)
    else: m.fit(X, y)                                   # hgb early-stops internally
    return m.predict_proba(Xp)[:, 1]

def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))

def metrics(y, p):
    maj = max(y.mean(), 1 - y.mean()); out = {"n": int(len(y)), "acc": float(((p > .5) == y).mean()), "majority": float(maj), "auc": float(roc_auc_score(y, p)),
                                             "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}
    conf = np.maximum(p, 1 - p)
    for t in (0.55, 0.60, 0.65, 0.70):
        m = conf >= t; out[f"acc@{t}"] = float(((p[m] > .5) == y[m]).mean()) if m.any() else None; out[f"cov@{t}"] = float(m.mean()); out[f"n@{t}"] = int(m.sum())
    return out

def weights(P, y):
    """Log-loss-optimal convex weights on out-of-fold probabilities (columns of P)."""
    k = P.shape[1]; f = lambda w: log_loss(y, np.clip(P @ (np.abs(w) / np.abs(w).sum()), 1e-6, 1 - 1e-6))
    w = minimize(f, np.ones(k) / k, method="Nelder-Mead", options={"maxiter": 400}).x; return np.abs(w) / np.abs(w).sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--models", default="xgboost,lightgbm"); ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--folds", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8]); ap.add_argument("--row-cap", type=int, default=250_000)
    ap.add_argument("--out", default="results.json"); a = ap.parse_args()
    df = pd.read_csv(a.data); sp = json.load(open(a.splits)); models = a.models.split(",")
    feats = [c for c in df.columns if c not in META and not c.startswith(("y_", "fwd_ret_"))]
    print(f"{len(df):,} rows, {len(feats)} features, models={models}"); rng = np.random.default_rng(SEED); res = {}
    s_all = df.sess.values; X_all = df[feats].values.astype("float32")
    for h in a.horizons:
        y_all = df[f"y_{h}"].values; lab = ~np.isnan(y_all); res[f"{h}d"] = {"oof": {}, "test": {}}
        def run(tr_rng, pr_rng):
            tr = lab & (s_all >= tr_rng[0]) & (s_all <= tr_rng[1]); pr = lab & (s_all >= pr_rng[0]) & (s_all <= pr_rng[1])
            idx = np.where(tr)[0]
            if len(idx) > a.row_cap: idx = np.sort(rng.choice(idx, a.row_cap, replace=False))
            return {m: fit_predict(m, X_all[idx], y_all[idx], s_all[idx], X_all[pr]) for m in models}, y_all[pr]
        oof = {m: [] for m in models}; yo = []
        for k in a.folds:                                   # out-of-fold scoring
            f = sp["folds"][k - 1]; t0 = time.time(); P, y = run(f["train_sess"], f["val_sess"]); yo.append(y)
            for m in models: oof[m].append(P[m])
            print(f"h={h} fold {k}: " + "  ".join(f"{m} acc={((P[m] > .5) == y).mean():.3f}" for m in models) + f"  maj={max(y.mean(), 1 - y.mean()):.3f}  ({time.time() - t0:.0f}s)")
        yo = np.concatenate(yo); O = {m: np.concatenate(v) for m, v in oof.items()}
        for m in models: res[f"{h}d"]["oof"][m] = metrics(yo, O[m])
        w = weights(np.column_stack([O[m] for m in models]), yo) if len(models) > 1 else np.ones(1)
        res[f"{h}d"]["weights"] = dict(zip(models, map(float, w)))
        res[f"{h}d"]["oof"]["weighted"] = metrics(yo, np.column_stack([O[m] for m in models]) @ w)
        P, y = run([sp["dev_sess"][0], sp["dev_sess"][1]], sp["test_sess"])            # final test, scored once
        for m in models: res[f"{h}d"]["test"][m] = metrics(y, P[m])
        res[f"{h}d"]["test"]["weighted"] = metrics(y, np.column_stack([P[m] for m in models]) @ w)
        print(f"== {h}d test:", {k: (round(v['acc'], 4), round(v['majority'], 4), round(v['auc'], 4)) for k, v in res[f'{h}d']['test'].items()})
    json.dump(res, open(a.out, "w"), indent=1); print("saved", a.out)

if __name__ == "__main__":
    main()
