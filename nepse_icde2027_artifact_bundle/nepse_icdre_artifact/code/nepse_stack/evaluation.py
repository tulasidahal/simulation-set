"""
Evaluation: predictive, probabilistic, selective, economic and statistical.

Every function takes y (int labels 0=DOWN, 1=FLAT, 2=UP) and P (n x 3 probabilities).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, log_loss, matthews_corrcoef)

from .common import N_CLASS, expected_calibration_error
from .config import ExperimentConfig


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    y = np.asarray(y, int)
    pred = P.argmax(1)
    pcf1 = f1_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
        "log_loss": log_loss(y, np.clip(P, 1e-12, 1), labels=[0, 1, 2]),
        "brier": float(np.mean(np.sum((P - np.eye(N_CLASS)[y]) ** 2, 1))),
        "ece": expected_calibration_error(P, y),
        "f1_down": pcf1[0], "f1_flat": pcf1[1], "f1_up": pcf1[2],
        "n": len(y),
    }


# ---------------------------------------------------------------------------
# Selective prediction: accuracy vs coverage, and the high-confidence flag
# ---------------------------------------------------------------------------
def selective_curve(y, P, coverages=np.linspace(0.05, 1.0, 20)) -> pd.DataFrame:
    y = np.asarray(y, int)
    order = np.argsort(-P.max(1), kind="stable")
    correct = (P.argmax(1) == y)[order]
    cum = np.cumsum(correct) / np.arange(1, len(y) + 1)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        rows.append({"coverage": c, "accuracy": cum[k - 1],
                     "min_confidence": P.max(1)[order][k - 1]})
    return pd.DataFrame(rows)


def accuracy_at_coverage(y, P, coverage: float) -> float:
    """Accuracy on the `coverage` share of predictions with the highest confidence."""
    k = max(1, int(round(coverage * len(y))))
    top = np.argsort(-P.max(1), kind="stable")[:k]
    return float((P[top].argmax(1) == np.asarray(y, int)[top]).mean())


def aurc(y, P) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    y = np.asarray(y, int)
    order = np.argsort(-P.max(1), kind="stable")
    err = (P.argmax(1) != y)[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(y) + 1)))


def high_confidence_stats(y, P, threshold: float) -> dict:
    y = np.asarray(y, int)
    m = P.max(1) >= threshold
    out = {"threshold": threshold, "coverage": float(m.mean()), "n_flagged": int(m.sum())}
    if m.sum():
        out["accuracy_flagged"] = float((P[m].argmax(1) == y[m]).mean())
        out["macro_f1_flagged"] = f1_score(y[m], P[m].argmax(1), average="macro",
                                           zero_division=0)
    return out


# ---------------------------------------------------------------------------
# Long-only trading evaluation, net of costs
# ---------------------------------------------------------------------------
def backtest(frame: pd.DataFrame, P: np.ndarray, cfg: ExperimentConfig,
             rule: str = "argmax", top_frac: float = 0.2) -> tuple[dict, pd.DataFrame]:
    """
    Every `horizon` trading days, hold an equal-weight long-only portfolio of
    the stocks predicted UP, for `horizon` days; cash when nothing qualifies.

    rule="argmax":   every stock whose most likely class is UP.
    rule="top_conf": only UP stocks whose confidence is in the top `top_frac`
                     of that day's predictions (same-day ranking, no look-ahead).

    Costs are charged on turnover only: half the round-trip cost per unit of
    weight bought or sold, so unchanged holdings cost nothing. CGT is applied
    to positive per-stock gains. Gross (pre-cost) returns are reported too.
    Benchmark: equal-weight universe, buy and hold (no costs).
    """
    d = frame[cfg.date_col].to_numpy()
    sym = frame[cfg.symbol_col].to_numpy()
    r = frame["fwd_ret_abs"].to_numpy()
    conf = P.max(1)
    up = P.argmax(1) == 2
    udates = np.sort(np.unique(d))[:: cfg.horizon]
    one_way = cfg.cost_bps_roundtrip / 2 / 1e4
    prev_w: dict = {}
    rec = []
    for t in udates:
        m = d == t
        sel = m & up
        if rule == "top_conf" and m.any():
            sel &= conf >= np.quantile(conf[m], 1 - top_frac)
        bench = float(r[m].mean())
        w = {s: 1.0 / sel.sum() for s in sym[sel]} if sel.any() else {}
        turnover = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in set(w) | set(prev_w))
        prev_w = w
        if sel.any():
            gross = float(r[sel].mean())
            after_tax = float((r[sel] - cfg.cgt_rate * np.maximum(r[sel], 0)).mean())
        else:
            gross = after_tax = 0.0
        rec.append((t, after_tax - one_way * turnover, gross, bench, int(sel.sum()), turnover))
    eq = pd.DataFrame(rec, columns=["date", "strategy", "gross", "benchmark", "n_holdings",
                                    "turnover"])
    ppy = cfg.trading_days_per_year / cfg.horizon
    out = {}
    for col in ("strategy", "gross", "benchmark"):
        x = eq[col].to_numpy()
        curve = np.cumprod(1 + x)
        dd = curve / np.maximum.accumulate(curve) - 1
        down = x[x < 0]
        out[col] = {
            "ann_return": float(curve[-1] ** (ppy / len(x)) - 1) if len(x) else np.nan,
            "ann_vol": float(x.std(ddof=1) * np.sqrt(ppy)),
            "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(ppy)) if x.std() > 0 else np.nan,
            "sortino": float(x.mean() / down.std(ddof=1) * np.sqrt(ppy)) if len(down) > 1 else np.nan,
            "max_drawdown": float(dd.min()),
            "hit_rate": float((x > 0).mean()),
        }
    out["strategy"]["exposure"] = float((eq["n_holdings"] > 0).mean())
    out["strategy"]["avg_turnover"] = float(eq["turnover"].mean())
    out["strategy"]["gross_ann_return"] = out["gross"]["ann_return"]
    out["strategy"]["gross_sharpe"] = out["gross"]["sharpe"]
    # Round-trip cost (bps) at which the strategy's average net return is zero.
    mt = eq["turnover"].mean()
    out["strategy"]["breakeven_cost_bps"] = float(2 * eq["gross"].mean() / mt * 1e4) if mt > 0 else np.nan
    out["strategy"]["avg_holdings"] = float(eq.loc[eq.n_holdings > 0, "n_holdings"].mean()) \
        if (eq.n_holdings > 0).any() else 0.0
    eq["strategy_curve"] = np.cumprod(1 + eq["strategy"])
    eq["benchmark_curve"] = np.cumprod(1 + eq["benchmark"])
    return out, eq


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired correctness."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only": b, "b_only": c, "p_value": float(p)}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, dates: np.ndarray,
                    horizon: int = 1) -> dict:
    """
    DM test on per-date mean loss differentials (a - b), Newey-West variance
    with lag = horizon, Harvey-Leybourne-Newbold small-sample correction.
    Negative statistic -> model a has lower loss.
    """
    diff = pd.Series(loss_a - loss_b).groupby(dates).mean().to_numpy()
    T = len(diff)
    dbar = diff.mean()
    x = diff - dbar
    lag = max(1, horizon)
    var = x @ x / T
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / T
    dm = dbar / np.sqrt(var / T)
    hln = np.sqrt((T + 1 - 2 * lag + lag * (lag - 1) / T) / T)
    stat = dm * hln
    return {"dm_stat": float(stat), "p_value": float(2 * stats.t.sf(abs(stat), T - 1)),
            "mean_loss_diff": float(dbar), "n_dates": T}


def per_row_log_loss(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1))


def confusion_by_date(y, pred, date_codes, n_dates) -> np.ndarray:
    cm = np.zeros((n_dates, N_CLASS, N_CLASS))
    np.add.at(cm, (date_codes, np.asarray(y, int), pred), 1)
    return cm


def macro_f1_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return float(f1.mean())


def mcc_cm(cm: np.ndarray) -> float:
    t, p, c, s = cm.sum(1), cm.sum(0), np.trace(cm), cm.sum()
    den = np.sqrt((s ** 2 - p @ p) * (s ** 2 - t @ t))
    return float((c * s - t @ p) / den) if den > 0 else 0.0


def block_bootstrap_diff(cm_a: np.ndarray, cm_b: np.ndarray, metric=mcc_cm,
                         n_boot: int = 1000, block: int = 20, seed: int = 0) -> dict:
    """
    Moving-block bootstrap over trading dates (keeps serial dependence).
    Returns metric(a) - metric(b), a 95% CI and a one-sided p-value.
    """
    rng = np.random.default_rng(seed)
    T = len(cm_a)
    nb = int(np.ceil(T / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, T - block + 1), nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        diffs[i] = metric(cm_a[idx].sum(0)) - metric(cm_b[idx].sum(0))
    point = metric(cm_a.sum(0)) - metric(cm_b.sum(0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "p_one_sided": float((diffs <= 0).mean())}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out
