"""
Multi-horizon (1,3,5,10 day) direction forecasting on NEPSE with a stacked, calibrated ensemble
and confidence-thresholded (selective) prediction. Reports honest out-of-sample metrics.

Base learners: Logistic, HistGradientBoosting, ExtraTrees, MLP  (+ XGBoost, Transformer if --full)
Stacking: logistic meta-learner on VALIDATION predictions; isotonic calibration on a held-out calibration slice.
Leakage control: purge/embargo of `h` trading days between train/val/cal/test; no random shuffling.
"""
import argparse, json, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
warnings.filterwarnings("ignore")
from train_models import build, FEATS

HORIZONS = [1, 3, 5, 10]
TRAIN_END, VAL_END, CAL_END = "2025-03-31", "2025-07-31", "2025-11-30"   # test = after CAL_END
CONF = 0.70

def ece(y, p, bins=10):
    idx = np.minimum((p*bins).astype(int), bins-1); e = 0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean()*abs(y[m].mean()-p[m].mean())
    return e

def label(d, h):
    g = d.groupby("symbol", sort=False)
    raw = g.close.pct_change().abs() > 0.11               # corporate-action jumps (10% circuit limit)
    d = d.assign(_bad=raw.astype(int))
    gb = d.groupby("symbol", sort=False)._bad
    fwd = sum(gb.shift(-k).fillna(0) for k in range(1, h+1))
    past = gb.transform(lambda s: s.rolling(20, min_periods=1).sum())
    ret = g.close.shift(-h) / d.close - 1
    out = d.assign(ret_h=ret, y=(ret > 0).astype(int))
    ok = ret.notna() & (fwd == 0) & (past == 0) & out[FEATS].notna().all(axis=1)
    return out[ok]

def split(d, h):
    dates = np.sort(d.trade_date.unique())
    def shift(cut):                                       # purge h trading days before each boundary
        i = np.searchsorted(dates, np.datetime64(cut), side="right")
        return dates[max(i-h-1, 0)]
    t = d.trade_date
    tr = t <= shift(TRAIN_END)
    va = (t > TRAIN_END) & (t <= shift(VAL_END))
    ca = (t > VAL_END) & (t <= shift(CAL_END))
    te = t > CAL_END
    return tr, va, ca, te

def base_models(full):
    ms = {
      "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)),
      "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                            l2_regularization=1.0, early_stopping=True, random_state=0),
      "et": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=100, max_features=0.6, n_jobs=-1, random_state=0),
      "mlp": make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, early_stopping=True, max_iter=60, random_state=0)),
    }
    if full:
        import xgboost as xgb
        ms["xgb"] = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, tree_method="hist", n_jobs=-1)
    return ms

def selective(y, p, thr=CONF):
    conf = np.maximum(p, 1-p); m = conf >= thr
    if m.sum() == 0: return dict(coverage=0.0, n=0, acc=None)
    pred = (p[m] > 0.5).astype(int)
    return dict(coverage=float(m.mean()), n=int(m.sum()), acc=float((pred == y[m]).mean()))

def run(full=False, max_train=250_000):
    base = build()
    results = {}
    for h in HORIZONS:
        d = label(base, h); tr, va, ca, te = split(d, h)
        X = lambda m: d.loc[m, FEATS].values.astype("float32"); Y = lambda m: d.loc[m, "y"].values
        Xtr, ytr = X(tr), Y(tr)
        if len(ytr) > max_train:
            ix = np.random.RandomState(0).choice(len(ytr), max_train, replace=False); Xtr, ytr = Xtr[ix], ytr[ix]
        Xva, yva, Xca, yca, Xte, yte = X(va), Y(va), X(ca), Y(ca), X(te), Y(te)
        models = base_models(full); P = {}
        for name, m in models.items():
            m.fit(Xtr, ytr)
            P[name] = tuple(m.predict_proba(Z)[:, 1] for Z in (Xva, Xca, Xte))
        names = list(P)
        stack = lambda i: np.column_stack([P[n][i] for n in names])
        meta = LogisticRegression(C=1.0).fit(stack(0), yva)             # meta-learner on validation
        pca, pte = meta.predict_proba(stack(1))[:, 1], meta.predict_proba(stack(2))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pca, yca)   # calibrate on held-out slice
        pte_cal = iso.predict(pte)
        base_rate = max(yte.mean(), 1-yte.mean())
        r = dict(n_test=int(len(yte)), majority_baseline=float(base_rate),
                 members={n: dict(auc=float(roc_auc_score(yte, P[n][2])), acc=float(accuracy_score(yte, P[n][2] > .5))) for n in names},
                 ensemble=dict(auc=float(roc_auc_score(yte, pte_cal)), acc=float(accuracy_score(yte, pte_cal > .5)),
                               brier_raw=float(brier_score_loss(yte, pte)), brier_cal=float(brier_score_loss(yte, pte_cal)),
                               ece_raw=float(ece(yte, pte)), ece_cal=float(ece(yte, pte_cal))),
                 selective={f"conf>={t:.2f}": selective(yte, pte_cal, t) for t in (0.55, 0.60, 0.65, 0.70)})
        results[f"{h}d"] = r
        print(f"\n=== horizon {h}d | test n={len(yte):,} | majority baseline={base_rate:.3f}")
        for n, v in r["members"].items(): print(f"  {n:6s} acc={v['acc']:.3f} auc={v['auc']:.3f}")
        e = r["ensemble"]; print(f"  ENSEMBLE acc={e['acc']:.3f} auc={e['auc']:.3f} brier {e['brier_raw']:.4f}->{e['brier_cal']:.4f} ece {e['ece_raw']:.4f}->{e['ece_cal']:.4f}")
        for k, v in r["selective"].items(): print(f"  {k}: coverage={v['coverage']:.4f} n={v['n']} acc={v['acc']}")
    json.dump(results, open("ensemble_results.json", "w"), indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); a = ap.parse_args()
    run(a.full)
