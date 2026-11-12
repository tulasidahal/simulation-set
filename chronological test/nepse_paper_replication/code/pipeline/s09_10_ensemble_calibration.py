"""
Stage 09  Ensembles over the OOF probabilities: simple average, weighted (log-loss-optimal simplex weights),
          stacking (logistic regression on logits), regime-aware stacking (market trend x volatility regimes).
Stage 10  Probability calibration: Platt scaling and Isotonic regression.

Nested walk-forward protocol (no look-ahead; EMBARGO sessions purged before every evaluation block):
  fold-k ensemble weights / stackers are fit on first-level OOF predictions from folds < k  -> outputs for folds 2..8
  fold-k calibrators are fit on second-level outputs from folds 2..k-1                      -> calibrated outputs for folds 3..8
Evaluation region for ALL reported comparisons = folds 3..8 (EVAL_FIRST_FOLD).
"""
import joblib
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from common import *
from metrics import classification_metrics
import s05_audit as A

EVAL_FIRST_FOLD = 3
APPROACHES = ["avg", "weighted", "stacking", "regime"]

def regime_ids(mret60, mvol, thr):
    return 2 * (mret60 > 0).astype(int) + (mvol > thr).astype(int)       # 0: down/calm 1: down/volatile 2: up/calm 3: up/volatile

class Ensembler:
    def __init__(self, models): self.models = models; self.M = len(models)
    def fit(self, P, y, mret60, mvol):
        self.thr = float(np.nanmedian(mvol)); L = logit(P)
        self.w = self._weights(P, y)
        self.stack = LogisticRegression(C=1.0, max_iter=500).fit(L, y)
        rid = regime_ids(mret60, mvol, self.thr); self.reg = {}
        for r in range(4):
            m = rid == r
            if m.sum() >= 3000 and 0 < y[m].mean() < 1: self.reg[r] = LogisticRegression(C=1.0, max_iter=500).fit(L[m], y[m])
        return self
    def _weights(self, P, y):
        M = P.shape[1]; yy = y.astype(float)
        def nll(w):
            p = np.clip(P @ w, 1e-6, 1 - 1e-6); return -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
        r = minimize(nll, np.full(M, 1 / M), method="SLSQP", bounds=[(0, 1)] * M, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        w = np.clip(r.x, 0, 1); return w / w.sum()
    def predict(self, P, mret60, mvol):
        L = logit(P); out = {"avg": P.mean(1), "weighted": np.clip(P @ self.w, 1e-6, 1 - 1e-6), "stacking": self.stack.predict_proba(L)[:, 1]}
        rid = regime_ids(mret60, mvol, self.thr); pr = out["stacking"].copy()
        for r, m in self.reg.items():
            s = rid == r
            if s.any(): pr[s] = m.predict_proba(L[s])[:, 1]
        out["regime"] = pr; return out

class Calibrator:
    def __init__(self, method): self.method = method
    def fit(self, p, y):
        if self.method == "platt": self.m = LogisticRegression(C=1e4, max_iter=500).fit(logit(p)[:, None], y)
        else: self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(p, y)
        return self
    def predict(self, p):
        return self.m.predict_proba(logit(p)[:, None])[:, 1] if self.method == "platt" else self.m.predict(p)

def run():
    panel = pd.read_pickle(WORK / "panel.pkl"); splits = load_json(WORK / "splits.json"); market = pd.read_csv(WORK / "market.csv")
    O = joblib.load(WORK / "oof.pkl"); oof, fold_id = O["oof"], O["fold_id"]; models = [m for m in oof if m != "ranker"]
    assert len(O["done_folds"]) == N_FOLDS, "OOF incomplete"
    sess = panel.sess.values; U = np.asarray(splits["U"]); pos = np.searchsorted(U, sess)
    mret60 = np.nan_to_num(market.mkt_ret60.values[sess]); mvol = np.nan_to_num(market.mkt_vol20.values[sess])
    val_start = {f["fold"]: f["val_pos"][0] for f in splits["folds"]}
    n = len(panel); out = {}; report = {"models": models, "eval_first_fold": EVAL_FIRST_FOLD, "backends": O["backends"], "weights": {}, "stack_coef": {}}
    for h in HORIZONS:
        y = panel[f"y_{h}"].values; P = np.column_stack([oof[m][h] for m in models]); avail = ~np.isnan(P).any(1) & (fold_id > 0); lab = avail & ~np.isnan(y)
        raw = {a: np.full(n, np.nan, "float32") for a in APPROACHES}; report["weights"][f"{h}d"] = {}; report["stack_coef"][f"{h}d"] = {}
        raw["avg"][avail] = P[avail].mean(1)
        for k in range(2, N_FOLDS + 1):
            tr = lab & (fold_id < k) & (pos < val_start[k] - EMBARGO); va = avail & (fold_id == k)
            ens = Ensembler(models).fit(P[tr], y[tr].astype(int), mret60[tr], mvol[tr]); pr = ens.predict(P[va], mret60[va], mvol[va])
            for a in APPROACHES[1:]: raw[a][va] = pr[a]
            report["weights"][f"{h}d"][k] = dict(zip(models, np.round(ens.w, 4).tolist()))
            report["stack_coef"][f"{h}d"][k] = dict(zip(models, np.round(ens.stack.coef_[0], 4).tolist()))
        res = {a: raw[a] for a in APPROACHES}
        for m in models: res[m] = oof[m][h].copy()
        # ---- stage 10: nested walk-forward calibration of ensembles and base models ----
        for name in list(res):
            first = 2 if name in APPROACHES else 1
            for meth in ("platt", "iso"):
                c = np.full(n, np.nan, "float32")
                for k in range(first + 1, N_FOLDS + 1):
                    tr = lab & (fold_id >= first) & (fold_id < k) & (pos < val_start[k] - EMBARGO) & ~np.isnan(res[name]); va = avail & (fold_id == k) & ~np.isnan(res[name])
                    if tr.sum() < 2000 or va.sum() == 0: continue
                    c[va] = Calibrator(meth).fit(res[name][tr].astype("float64"), y[tr].astype(int)).predict(res[name][va].astype("float64"))
                res[f"{name}__{meth}"] = c
        out[h] = res
        # quick console summary over the evaluation region
        reg = lab & (fold_id >= EVAL_FIRST_FOLD) & ~np.isnan(res["stacking"]) if False else lab & (fold_id >= EVAL_FIRST_FOLD)
        log(f"horizon {h}d | eval rows={reg.sum():,}")
        for name in ["avg", "weighted", "stacking", "regime", "stacking__platt", "stacking__iso", "regime__iso"] + models[:1]:
            ok = reg & ~np.isnan(res[name]); mm = classification_metrics(y[ok], res[name][ok])
            log(f"   {name:18s} acc={mm['accuracy']:.4f} (maj {mm['majority_baseline_acc']:.4f}) auc={mm['auc']:.4f} ll={mm['logloss']:.4f} ece={mm['ece']:.4f}")
    joblib.dump({"preds": out, "models": models, "fold_id": fold_id, "eval_first_fold": EVAL_FIRST_FOLD}, WORK / "ens_oof.pkl")
    save_json(report, REPORTS / "stage09_10_ensemble_report.json"); log("ensembles + calibration done -> ens_oof.pkl")

if __name__ == "__main__":
    run()
