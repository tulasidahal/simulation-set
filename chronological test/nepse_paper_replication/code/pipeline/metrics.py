"""Metrics, selective prediction, and session-block bootstrap utilities."""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, matthews_corrcoef, balanced_accuracy_score
from common import *

def ece_score(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def classification_metrics(y, p):
    y = np.asarray(y).astype(int); p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6); pred = (p > 0.5).astype(int)
    base = max(y.mean(), 1 - y.mean())
    out = {"n": int(len(y)), "up_rate": float(y.mean()), "majority_baseline_acc": float(base), "accuracy": float((pred == y).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
           "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
           "logloss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)), "ece": ece_score(y, p)}
    for t in CONF_THRESHOLDS: out[f"sel_{t:.2f}"] = selective(y, p, t)
    return out

def selective(y, p, thr):
    conf = np.maximum(p, 1 - p); m = conf >= thr
    if m.sum() == 0: return {"coverage": 0.0, "n": 0, "accuracy": None, "up_calls_share": None}
    pred = (p[m] > 0.5).astype(int)
    return {"coverage": float(m.mean()), "n": int(m.sum()), "accuracy": float((pred == y[m]).mean()), "up_calls_share": float(pred.mean())}

def _session_slices(sess):
    order = np.argsort(sess, kind="stable"); s = sess[order]
    u, start, cnt = np.unique(s, return_index=True, return_counts=True)
    return order, u, start, cnt

def block_bootstrap(y, p, sess, B=300, block=20, seed=SEED, with_auc=True, thr=0.70):
    """Circular moving-block bootstrap over SESSIONS (keeps cross-sectional and serial dependence). Returns dict of CI arrays."""
    rng = np.random.default_rng(seed); y = np.asarray(y).astype(int); p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    order, u, start, cnt = _session_slices(np.asarray(sess)); y, p = y[order], p[order]
    T = len(u); nb = int(np.ceil(T / block))
    correct = ((p > 0.5).astype(int) == y).astype(float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)); br = (p - y) ** 2
    conf_m = (np.maximum(p, 1 - p) >= thr)
    per = lambda a: np.add.reduceat(a, start)
    S_corr, S_ll, S_br, S_n = per(correct), per(ll), per(br), cnt.astype(float)
    S_sel, S_selc = per(conf_m.astype(float)), per((conf_m * correct))
    res = {k: [] for k in ["accuracy", "logloss", "brier", "auc", "sel_acc", "sel_cov"]}
    for _ in range(B):
        starts = rng.integers(0, T, nb); sess_idx = ((starts[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        n = S_n[sess_idx].sum()
        res["accuracy"].append(S_corr[sess_idx].sum() / n); res["logloss"].append(S_ll[sess_idx].sum() / n); res["brier"].append(S_br[sess_idx].sum() / n)
        sc = S_selc[sess_idx].sum(); sn = S_sel[sess_idx].sum()
        res["sel_acc"].append(sc / sn if sn > 0 else np.nan); res["sel_cov"].append(sn / n)
        if with_auc:
            rows = np.concatenate([np.arange(start[i], start[i] + cnt[i]) for i in sess_idx])
            yy = y[rows]; res["auc"].append(roc_auc_score(yy, p[rows]) if 0 < yy.sum() < len(yy) else np.nan)
    return {k: np.array(v) for k, v in res.items() if len(v)}

def ci(a, alpha=0.05):
    a = a[~np.isnan(a)]
    return [float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))] if len(a) else [None, None]
