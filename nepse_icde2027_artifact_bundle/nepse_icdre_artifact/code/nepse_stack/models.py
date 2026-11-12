"""
Base learners for the NEPSE stack, all behind one interface:

    m.fit(X, y, dates)          -> trains on one walk-forward training window
    m.predict_log_proba(X)      -> (n, 3) calibrated log-probabilities

Shared training recipe (each switch is an ablation):
  * embargoed tail of the training window for early stopping + calibration
  * class-balanced x time-decayed sample weights
  * temperature scaling of the output logits before stacking

The customised XGBoost adds a multiclass focal-loss objective, a choice of
early-stopping metric and native categorical splits for symbol/sector.

If a library is missing and cfg.allow_fallbacks is True, a scikit-learn
stand-in is used and flagged with is_fallback=True. Fallback results are
for smoke tests only and must never be reported in the paper.
"""
from __future__ import annotations

import importlib.util
import time

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score, log_loss, matthews_corrcoef

from .common import (N_CLASS, SeqView, apply_temperature, build_weights,
                     fit_temperature, inner_train_val_split)
from .config import ExperimentConfig


def has(lib: str) -> bool:
    return importlib.util.find_spec(lib) is not None


# ===========================================================================
# Base template
# ===========================================================================
class BaseLearner:
    name = "base"
    input_kind = "tabular"          # "tabular" | "sequence" | "frame"
    is_fallback = False

    def __init__(self, cfg: ExperimentConfig, seed: int = 0, class_balance: bool = True,
                 halflife_days: float | None = 500.0, calibrate: bool = True,
                 embargo_days: int | None = None, **kw):
        self.cfg, self.seed = cfg, seed
        self.class_balance, self.halflife_days = class_balance, halflife_days
        self.calibrate = calibrate
        self.inner_embargo = cfg.embargo_days if embargo_days is None else embargo_days
        self.kw = kw
        self.T = 1.0
        self.fit_seconds = self.predict_seconds = 0.0
        self.n_iter = None

    # subclasses implement these two
    def _fit_core(self, Xtr, ytr, wtr, Xva, yva): ...
    def _raw_log_proba(self, X) -> np.ndarray: ...

    def _prep(self, X, fit: bool):
        return X

    def fit(self, X, y, dates):
        t0 = time.perf_counter()
        y = np.asarray(y, int)
        dates = pd.Series(pd.to_datetime(np.asarray(dates)))
        Xp = self._prep(X, fit=True)
        tr, va = inner_train_val_split(dates, self.cfg.inner_val_frac, self.inner_embargo)
        w = build_weights(y[tr], dates[tr], self.class_balance, self.halflife_days)
        self._fit_core(Xp[tr], y[tr], w, Xp[va], y[va])
        if self.calibrate:
            self.T = fit_temperature(self._raw_log_proba(Xp[va]), y[va])
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict_log_proba(self, X) -> np.ndarray:
        t0 = time.perf_counter()
        lp = self._raw_log_proba(self._prep(X, fit=False))
        out = apply_temperature(lp, self.T) if self.calibrate else lp
        self.predict_seconds = time.perf_counter() - t0
        return out


# ===========================================================================
# Tabular helpers: categorical symbol / sector with frozen levels
# ===========================================================================
class TabularLearner(BaseLearner):
    cat_as_string = False

    def __init__(self, cfg, seed=0, use_categorical: bool = True, **kw):
        super().__init__(cfg, seed, **kw)
        self.use_categorical = use_categorical
        self.cat_cols = [cfg.symbol_col, cfg.sector_col]
        self._levels: dict = {}
        self.columns: list[str] = []

    def _prep(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.copy()
        cats = [c for c in self.cat_cols if c in X.columns]
        if not self.use_categorical:
            X = X.drop(columns=cats)
            cats = []
        for c in cats:
            if fit:
                self._levels[c] = pd.Index(sorted(X[c].astype(str).unique()))
            X[c] = pd.Categorical(X[c].astype(str), categories=self._levels[c])
            if self.cat_as_string:
                X[c] = X[c].astype(str).replace("nan", "__unseen__")
        if fit:
            self.columns = list(X.columns)
        return X[self.columns]

    @property
    def cat_features(self):
        return [c for c in self.cat_cols if c in self.columns]


# ===========================================================================
# Customised XGBoost
# ===========================================================================
def focal_grad_hess(margin, y, gamma, weight=None):
    """Exact gradient of FL = -(1-p_y)^g log p_y; positive diagonal Hessian approx."""
    margin = margin.reshape(-1, N_CLASS)
    p = softmax(margin, axis=1)
    n = len(y)
    py = np.clip(p[np.arange(n), y], 1e-12, 1.0)
    onehot = np.eye(N_CLASS)[y]
    if gamma == 0:
        dfl = -1.0 / py
    else:
        dfl = gamma * (1 - py) ** (gamma - 1) * np.log(py) - (1 - py) ** gamma / py
    grad = (dfl * py)[:, None] * (onehot - p)
    hess = np.maximum(np.maximum((1 - py) ** gamma, 1e-3)[:, None] * p * (1 - p), 1e-6)
    if weight is not None:
        grad, hess = grad * weight[:, None], hess * weight[:, None]
    return grad, hess


def _metric(name, is_margin):
    def f(predt, dmat):
        y = dmat.get_label().astype(int)
        p = predt.reshape(-1, N_CLASS)
        p = softmax(p, axis=1) if is_margin else p
        if name == "macro_f1":
            return "neg_macro_f1", -f1_score(y, p.argmax(1), average="macro")
        if name == "mcc":
            return "neg_mcc", -matthews_corrcoef(y, p.argmax(1))
        return "mlogloss", log_loss(y, p, labels=list(range(N_CLASS)))
    return f


class XGBLearner(TabularLearner):
    name = "xgboost"
    DEFAULT = {"tree_method": "hist", "max_depth": 4, "eta": 0.035, "min_child_weight": 20,
               "subsample": 0.8, "colsample_bytree": 0.7, "colsample_bynode": 0.8,
               "gamma": 0.5, "lambda": 5.0, "alpha": 0.5, "max_bin": 128,
               "max_cat_to_onehot": 1, "nthread": -1}

    def __init__(self, cfg, seed=0, objective="softprob", focal_gamma=2.0,
                 stop_metric="mlogloss", num_boost_round=2000, early_stopping_rounds=100,
                 params=None, **kw):
        super().__init__(cfg, seed, **kw)
        self.objective, self.focal_gamma, self.stop_metric = objective, focal_gamma, stop_metric
        self.num_boost_round, self.esr = num_boost_round, early_stopping_rounds
        self.params = {**self.DEFAULT, **(params or {}), "seed": seed}

    def _dm(self, X, y=None, w=None):
        import xgboost as xgb
        return xgb.DMatrix(X, label=y, weight=w, enable_categorical=self.use_categorical)

    def _fit_core(self, Xtr, ytr, wtr, Xva, yva):
        import xgboost as xgb
        p = {**self.params, "num_class": N_CLASS, "disable_default_eval_metric": 1}
        custom = self.objective == "focal"
        obj = None
        if custom:
            g = self.focal_gamma

            def obj(predt, dtrain):
                w = dtrain.get_weight()
                return focal_grad_hess(predt, dtrain.get_label().astype(int), g,
                                       w if w.size else None)
        else:
            p["objective"] = "multi:softprob"
        self.booster = xgb.train(p, self._dm(Xtr, ytr, wtr), self.num_boost_round,
                                 evals=[(self._dm(Xva, yva), "val")], obj=obj,
                                 custom_metric=_metric(self.stop_metric, custom),
                                 early_stopping_rounds=self.esr, verbose_eval=False)
        self.n_iter = self.booster.best_iteration + 1

    def _raw_log_proba(self, X):
        m = self.booster.predict(self._dm(X), output_margin=True,
                                 iteration_range=(0, self.n_iter))
        return log_softmax(m.reshape(-1, N_CLASS), axis=1)

    def shap_values(self, X):
        """(n, 3, n_features + 1) contributions; last column is the bias."""
        return self.booster.predict(self._dm(self._prep(X, False)), pred_contribs=True,
                                    iteration_range=(0, self.n_iter))


# ===========================================================================
# LightGBM
# ===========================================================================
class LGBMLearner(TabularLearner):
    name = "lightgbm"

    def __init__(self, cfg, seed=0, params=None, num_boost_round=2000,
                 early_stopping_rounds=100, **kw):
        super().__init__(cfg, seed, **kw)
        self.params = {"objective": "multiclass", "num_class": N_CLASS,
                       "learning_rate": 0.035, "num_leaves": 31, "min_data_in_leaf": 50,
                       "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
                       "lambda_l2": 5.0, "max_cat_to_onehot": 4, "cat_smooth": 20,
                       "verbosity": -1, "seed": seed, "deterministic": True,
                       **(params or {})}
        self.nbr, self.esr = num_boost_round, early_stopping_rounds

    def _fit_core(self, Xtr, ytr, wtr, Xva, yva):
        import lightgbm as lgb
        cats = self.cat_features or "auto"
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cats)
        dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cats)
        self.model = lgb.train(self.params, dtr, self.nbr, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(self.esr, verbose=False)])
        self.n_iter = self.model.best_iteration

    def _raw_log_proba(self, X):
        m = self.model.predict(X, raw_score=True, num_iteration=self.n_iter)
        return log_softmax(np.asarray(m).reshape(-1, N_CLASS), axis=1)


# ===========================================================================
# CatBoost
# ===========================================================================
class CatBoostLearner(TabularLearner):
    name = "catboost"
    cat_as_string = True

    def __init__(self, cfg, seed=0, params=None, **kw):
        super().__init__(cfg, seed, **kw)
        self.params = {"loss_function": "MultiClass", "iterations": 2000, "depth": 7,
                       "learning_rate": 0.04, "l2_leaf_reg": 5.0, "random_strength": 1.0,
                       "od_type": "Iter", "od_wait": 100, "use_best_model": True,
                       "random_seed": seed, "verbose": 0, "allow_writing_files": False,
                       **(params or {})}

    def _fit_core(self, Xtr, ytr, wtr, Xva, yva):
        from catboost import CatBoostClassifier, Pool
        cats = self.cat_features
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(Pool(Xtr, ytr, weight=wtr, cat_features=cats),
                       eval_set=Pool(Xva, yva, cat_features=cats))
        self.n_iter = self.model.get_best_iteration()

    def _raw_log_proba(self, X):
        m = self.model.predict(X, prediction_type="RawFormulaVal")
        return log_softmax(np.asarray(m).reshape(-1, N_CLASS), axis=1)


# ===========================================================================
# Tree fallback (scikit-learn HistGradientBoosting) - smoke tests only
# ===========================================================================
class HistGBFallback(TabularLearner):
    is_fallback = True

    def __init__(self, cfg, seed=0, stands_for="tree", **kw):
        super().__init__(cfg, seed, **kw)
        self.name = stands_for

    def _fit_core(self, Xtr, ytr, wtr, Xva, yva):
        from sklearn.ensemble import HistGradientBoostingClassifier
        # Slightly different settings per stand-in so the stack ablation is not degenerate.
        leaves, lr, off = {"catboost": (63, 0.04, 11), "lightgbm": (31, 0.035, 22),
                           "xgboost": (15, 0.035, 33)}.get(self.name, (31, 0.05, 0))
        self.model = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=600, max_leaf_nodes=leaves, min_samples_leaf=50,
            l2_regularization=5.0, max_features=0.7, early_stopping=True,
            n_iter_no_change=40, categorical_features="from_dtype",
            random_state=self.seed + off)
        self.model.fit(Xtr, ytr, sample_weight=wtr, X_val=Xva, y_val=yva)
        self.n_iter = self.model.n_iter_

    def _raw_log_proba(self, X):
        return np.log(np.clip(self.model.predict_proba(X), 1e-12, 1))


# ===========================================================================
# Transformer over 20 x d sequences
# ===========================================================================
class TransformerLearner(BaseLearner):
    name = "transformer"
    input_kind = "sequence"

    def __init__(self, cfg, seed=0, d_model=64, nhead=4, ff=96, layers=2, dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, batch_size=512, max_epochs=30, patience=4,
                 **kw):
        super().__init__(cfg, seed, **kw)
        self.hp = dict(d_model=d_model, nhead=nhead, ff=ff, layers=layers, dropout=dropout)
        self.lr, self.wd, self.bs = lr, weight_decay, batch_size
        self.max_epochs, self.patience = max_epochs, patience

    def _norm_stats(self, X: SeqView):
        F = X.src.F[X.rows]
        self.mu = np.nanmean(F, 0).astype(np.float32)
        self.sd = (np.nanstd(F, 0) + 1e-6).astype(np.float32)

    def _fit_core(self, Xtr: SeqView, ytr, wtr, Xva: SeqView, yva):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._norm_stats(Xtr)
        self.net = _build_tst(Xtr.src.d, Xtr.src.L, **self.hp).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        ce = nn.CrossEntropyLoss(reduction="none")
        yt = torch.as_tensor(ytr)
        wt = torch.as_tensor(wtr, dtype=torch.float32)
        best, best_state, bad = np.inf, None, 0
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.max_epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(len(Xtr)), max(1, len(Xtr) // self.bs)):
                xb = torch.as_tensor(Xtr[b].materialize(self.mu, self.sd), device=self.device)
                loss = (ce(self.net(xb), yt[b].to(self.device)) * wt[b].to(self.device)).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            val = log_loss(yva, np.exp(self._raw_log_proba(Xva)), labels=[0, 1, 2])
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.n_iter = epoch + 1

    def _raw_log_proba(self, X: SeqView):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for b in np.array_split(np.arange(len(X)), max(1, len(X) // 4096)):
                xb = torch.as_tensor(X[b].materialize(self.mu, self.sd), device=self.device)
                out.append(torch.log_softmax(self.net(xb), 1).cpu().numpy())
        return np.concatenate(out)


def _build_tst(d_in, L, d_model, nhead, ff, layers, dropout):
    import torch
    from torch import nn

    class TST(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d_in, d_model)
            self.pos = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d_model, N_CLASS))

        def forward(self, x):
            pad = x.abs().sum(-1) == 0                       # zero-padded history
            h = self.norm(self.enc(self.inp(x) + self.pos, src_key_padding_mask=pad))
            keep = (~pad).unsqueeze(-1).float()
            mean = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
            return self.head(torch.cat([h[:, -1], mean], 1))
    return TST()


class MLPSeqFallback(TransformerLearner):
    """Stands in for the Transformer when torch is missing. Smoke tests only."""
    is_fallback = True

    def _fit_core(self, Xtr, ytr, wtr, Xva, yva):
        from sklearn.neural_network import MLPClassifier
        self._norm_stats(Xtr)
        flat = Xtr.materialize(self.mu, self.sd).reshape(len(Xtr), -1)
        self.model = MLPClassifier((128, 64), alpha=1e-3, learning_rate_init=1e-3,
                                   max_iter=60, early_stopping=True, n_iter_no_change=5,
                                   random_state=self.seed)
        self.model.fit(flat, ytr, sample_weight=wtr)
        self.n_iter = self.model.n_iter_

    def _raw_log_proba(self, X):
        flat = X.materialize(self.mu, self.sd).reshape(len(X), -1)
        return np.log(np.clip(self.model.predict_proba(flat), 1e-12, 1))


# ===========================================================================
# Baselines
# ===========================================================================
class PriorBaseline(BaseLearner):
    """Always predicts the training class distribution (majority-class baseline)."""
    name = "majority"
    input_kind = "frame"

    def fit(self, X, y, dates):
        c = np.bincount(np.asarray(y, int), minlength=N_CLASS) + 1.0
        self.logp = np.log(c / c.sum())
        return self

    def predict_log_proba(self, X):
        return np.tile(self.logp, (len(X), 1))


class RuleBaseline(BaseLearner):
    """
    Turns a hard rule (persistence, momentum) into probabilities with the
    empirical P(y | rule) from the training window, so log loss is defined.
    """
    input_kind = "frame"

    def __init__(self, cfg, seed=0, rule_col="label_prev", name="persistence", **kw):
        super().__init__(cfg, seed, **kw)
        self.rule_col, self.name = rule_col, name

    def fit(self, X, y, dates):
        r = X[self.rule_col].to_numpy(int)
        tab = np.ones((N_CLASS, N_CLASS))
        np.add.at(tab, (r, np.asarray(y, int)), 1)
        self.logp = np.log(tab / tab.sum(1, keepdims=True))
        return self

    def predict_log_proba(self, X):
        return self.logp[X[self.rule_col].to_numpy(int)]


class LogRegBaseline(TabularLearner):
    name = "logreg"

    def __init__(self, cfg, seed=0, **kw):
        super().__init__(cfg, seed, use_categorical=False, **kw)

    def _fit_core(self, Xtr, ytr, wtr, Xva, yva):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        self.model.fit(Xtr.fillna(0), ytr, logisticregression__sample_weight=wtr)

    def _raw_log_proba(self, X):
        return np.log(np.clip(self.model.predict_proba(X.fillna(0)), 1e-12, 1))


# ===========================================================================
# Factory
# ===========================================================================
LIB = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
       "transformer": "torch"}
REAL = {"xgboost": XGBLearner, "lightgbm": LGBMLearner, "catboost": CatBoostLearner,
        "transformer": TransformerLearner}


def make_model(name: str, cfg: ExperimentConfig, seed: int = 0, **overrides) -> BaseLearner:
    if name == "majority":
        return PriorBaseline(cfg, seed)
    if name == "persistence":
        return RuleBaseline(cfg, seed, "label_prev", "persistence")
    if name == "momentum20":
        return RuleBaseline(cfg, seed, "label_mom", "momentum20")
    if name == "logreg":
        return LogRegBaseline(cfg, seed, **overrides)
    if name not in REAL:
        raise ValueError(f"unknown model {name}")
    if has(LIB[name]):
        return REAL[name](cfg, seed, **overrides)
    if not cfg.allow_fallbacks:
        raise ImportError(f"{LIB[name]} is not installed (needed for {name})")
    if name == "transformer":
        keep = {k: v for k, v in overrides.items()
                if k in ("class_balance", "halflife_days", "calibrate", "embargo_days")}
        return MLPSeqFallback(cfg, seed, **keep)
    keep = {k: v for k, v in overrides.items()
            if k in ("class_balance", "halflife_days", "calibrate", "embargo_days",
                     "use_categorical")}
    return HistGBFallback(cfg, seed, stands_for=name, **keep)


# One switch at a time relative to the full XGBoost configuration.
XGB_ABLATIONS = {
    "xgb_full": {},
    "xgb_no_class_balance": {"class_balance": False},
    "xgb_no_time_decay": {"halflife_days": None},
    "xgb_no_symbol_sector": {"use_categorical": False},
    "xgb_focal_g1": {"objective": "focal", "focal_gamma": 1.0},
    "xgb_focal_g2": {"objective": "focal", "focal_gamma": 2.0},
    "xgb_stop_macro_f1": {"stop_metric": "macro_f1"},
    "xgb_stop_mcc": {"stop_metric": "mcc"},
    "xgb_no_calibration": {"calibrate": False},
    # Both switches that helped in the first NEPSE run, combined. Choose it only if
    # it also wins on the development block (block 2), not just on the pooled test.
    "xgb_recommended": {"halflife_days": None, "use_categorical": False},
    "xgb_original_350x4": {"num_boost_round": 350, "early_stopping_rounds": 10_000,
                           "class_balance": False, "halflife_days": None,
                           "calibrate": False,
                           "params": {"min_child_weight": 1, "subsample": 1.0,
                                      "colsample_bytree": 1.0, "colsample_bynode": 1.0,
                                      "gamma": 0.0, "lambda": 1.0, "alpha": 0.0,
                                      "max_bin": 256}},
}
