"""
End-to-end walk-forward pipeline:

  prepare()          labels, audits, rank-normalised features, sequence source
  run_base_models()  out-of-fold calibrated log-probabilities per model/block
  run_stacker()      chronological meta-learner on earlier blocks' OOF outputs
  run_protocols()    leakage experiments: purge/embargo/random k-fold/shuffled labels
"""
from __future__ import annotations

import resource
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression

from . import data as D
from .common import N_CLASS, SeqSource, SeqView
from .config import ExperimentConfig
from .models import make_model
from .splits import Fold, random_kfold_masks, stacker_train_mask, walk_forward


@dataclass
class Prepared:
    df: pd.DataFrame
    feats: list[str]
    tau: float
    seq: SeqSource
    quality: dict
    leakage: dict

    @property
    def y(self):
        return self.df["label"].to_numpy(int)


def prepare(raw: pd.DataFrame, cfg: ExperimentConfig, feats: list[str] | None = None) -> Prepared:
    df = D.validate_schema(raw, cfg)
    df, tau = D.add_returns_and_labels(df, cfg)
    dropped = {"label_breaches_dropped": df.attrs.get("label_breaches_dropped", 0),
               "label_breach_examples": df.attrs.get("label_breach_examples", [])}
    feats = feats or D.feature_columns(df, cfg)
    quality = {**D.data_quality_report(df, cfg, feats), **dropped}
    leakage = D.leakage_ic_audit(df, feats, cfg)
    if cfg.cross_sectional_rank:
        df = D.rank_normalise(df, feats, cfg)
    seq = SeqSource(df, feats, cfg.symbol_col, cfg.date_col, cfg.seq_len)
    return Prepared(df, feats, tau, seq, quality, leakage)


def model_input(prep: Prepared, kind: str, cfg: ExperimentConfig, mask=None):
    rows = np.arange(len(prep.df)) if mask is None else np.flatnonzero(mask)
    if kind == "sequence":
        return SeqView(prep.seq, rows)
    if kind == "frame":
        return prep.df.iloc[rows]
    cols = prep.feats + [cfg.symbol_col, cfg.sector_col]
    return prep.df.iloc[rows][cols]


def block_of_rows(dates: pd.Series, folds: list[Fold]) -> np.ndarray:
    b = np.full(len(dates), -1)
    for f in folds:
        b[f.test_mask(dates)] = f.k
    return b


# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------
def run_base_models(prep: Prepared, cfg: ExperimentConfig, specs: dict, seed: int,
                    folds: list[Fold], label_override: np.ndarray | None = None,
                    log=print) -> tuple[dict, list]:
    """
    specs: {output_name: (model_name, overrides)}.
    Returns {output_name: (n, 3) log-proba, NaN outside test blocks} and timings.
    """
    y = prep.y if label_override is None else label_override
    dates = prep.df[cfg.date_col]
    oof = {k: np.full((len(y), N_CLASS), np.nan) for k in specs}
    timings = []
    for f in folds:
        tr, te = f.train_mask(dates), f.test_mask(dates)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        for out_name, (model_name, overrides) in specs.items():
            m = make_model(model_name, cfg, seed, **overrides)
            m.fit(model_input(prep, m.input_kind, cfg, tr), y[tr], dates[tr])
            oof[out_name][te] = m.predict_log_proba(model_input(prep, m.input_kind, cfg, te))
            timings.append({"seed": seed, "model": out_name, "block": f.k,
                            "fallback": m.is_fallback, "n_train": int(tr.sum()),
                            "n_test": int(te.sum()), "fit_s": m.fit_seconds,
                            "predict_s": m.predict_seconds, "n_iter": m.n_iter,
                            "temperature": m.T,
                            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
            log(f"  seed {seed} block {f.k} {out_name:<24} fit {m.fit_seconds:6.1f}s"
                f"{'  [FALLBACK]' if m.is_fallback else ''}")
    return oof, timings


# ---------------------------------------------------------------------------
# Stacker
# ---------------------------------------------------------------------------
STACK_VARIANTS = ("lr_logits", "lr_logits_balanced", "lr_probs", "mean_prob", "mean_logprob")


def run_stacker(oof: dict, base_names: list[str], prep: Prepared, cfg: ExperimentConfig,
                folds: list[Fold], variant: str = "lr_logits", purge: bool = True,
                return_coefs: bool = False):
    dates = prep.df[cfg.date_col]
    y = prep.y
    Z = np.concatenate([oof[b] for b in base_names], axis=1)       # (n, 3M) log-probs
    blocks = block_of_rows(dates, folds)
    out = np.full((len(y), N_CLASS), np.nan)
    coefs = []
    for f in folds:
        if f.k < cfg.min_stack_blocks:
            continue
        te = f.test_mask(dates)
        if variant == "mean_prob":
            P = np.mean([np.exp(oof[b][te]) for b in base_names], axis=0)
            out[te] = np.log(np.clip(P, 1e-12, 1))
            continue
        if variant == "mean_logprob":
            out[te] = log_softmax(np.mean([oof[b][te] for b in base_names], axis=0), axis=1)
            continue
        tr = stacker_train_mask(blocks, dates, f, cfg, purge) & ~np.isnan(Z).any(1)
        feats = np.exp(Z) if variant == "lr_probs" else Z
        cw = "balanced" if variant == "lr_logits_balanced" else None
        lr = LogisticRegression(C=cfg.stacker_C, max_iter=5000, class_weight=cw)
        lr.fit(feats[tr], y[tr])
        out[te] = np.log(np.clip(lr.predict_proba(feats[te]), 1e-12, 1))
        coefs.append({"block": f.k, "n_train": int(tr.sum()),
                      **{f"{b}_abs_weight": float(np.abs(lr.coef_[:, 3 * i:3 * i + 3]).sum())
                         for i, b in enumerate(base_names)}})
    return (out, coefs) if return_coefs else out


# ---------------------------------------------------------------------------
# Leakage / protocol experiments
# ---------------------------------------------------------------------------
def run_protocols(prep: Prepared, cfg: ExperimentConfig, seed: int, model: str = "xgboost",
                  embargo_sweep=(0, 5, 10, 25, 50), log=print) -> dict:
    """
    Same model, same evaluation rows, different protocols. Shows how much
    leaky protocols inflate scores - a central figure for the paper.
    """
    dates = prep.df[cfg.date_col]
    y = prep.y
    folds = walk_forward(dates, cfg)
    eval_rows = block_of_rows(dates, folds) >= cfg.min_stack_blocks
    spec = {model: (model, {})}
    res = {}

    for emb in embargo_sweep:
        fl = walk_forward(dates, cfg, purge=True, embargo=emb)
        o, _ = run_base_models(prep, cfg, {model: (model, {"embargo_days": emb})}, seed, fl, log=log)
        res[f"walk_forward_embargo_{emb}"] = o[model]

    fl = walk_forward(dates, cfg, purge=False)
    o, _ = run_base_models(prep, cfg, {model: (model, {"embargo_days": 0})}, seed, fl, log=log)
    res["walk_forward_no_purge"] = o[model]

    # Random k-fold over the evaluation period (the classic leak).
    rk = np.full((len(y), N_CLASS), np.nan)
    idx = np.flatnonzero(eval_rows | (dates <= dates[eval_rows].min()).to_numpy())
    for tr_m, te_m in random_kfold_masks(len(idx), 5, seed):
        tr, te = idx[tr_m], idx[te_m]
        m = make_model(model, cfg, seed, embargo_days=0)
        mtr = np.zeros(len(y), bool); mtr[tr] = True
        mte = np.zeros(len(y), bool); mte[te] = True
        # dates are shuffled within the fold, so inner early stopping sees future rows too
        m.fit(model_input(prep, m.input_kind, cfg, mtr), y[mtr], dates[mtr])
        rk[mte] = m.predict_log_proba(model_input(prep, m.input_kind, cfg, mte))
    res["random_kfold"] = rk

    # Permutation test: shuffled training labels must give chance performance.
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    o, _ = run_base_models(prep, cfg, spec, seed, folds, label_override=y_shuf, log=log)
    res["shuffled_labels"] = o[model]
    return {"eval_rows": eval_rows, "preds": res}
