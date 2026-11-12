"""
XGBoost base learner for the NEPSE 4-model stack (UP / DOWN / FLAT).

Customizations, each switchable so they can be reported as ablations:
  1. Native categorical handling for symbol and sector (or drop them).
  2. Sample weights: balanced classes x exponential time decay.
  3. Objective: built-in softmax, or a custom multiclass focal loss.
  4. Early stopping on macro-F1 / MCC / log loss using an embargoed
     validation slice at the END of the training window (never the test fold).
  5. Nested hyperparameter search inside each walk-forward fold
     (Optuna if installed, otherwise random search).
  6. Temperature scaling so the log-probabilities sent to the stacker
     are calibrated.
  7. SHAP values via XGBoost's built-in pred_contribs.

Requires: xgboost>=2.0, numpy, pandas, scikit-learn, scipy. Optuna optional.
Class encoding everywhere: 0 = DOWN, 1 = FLAT, 2 = UP.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score, log_loss, matthews_corrcoef

N_CLASS = 3
CLASS_NAMES = ["DOWN", "FLAT", "UP"]


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
@dataclass
class XGBConfig:
    # Your current setting is 350 rounds, depth 4, eta 0.035.
    # These are a starting point; tune them per fold.
    params: dict = field(default_factory=lambda: {
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.035,
        "min_child_weight": 20,     # larger than default: daily returns are noisy
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bynode": 0.8,
        "gamma": 0.5,
        "lambda": 5.0,
        "alpha": 0.5,
        "max_bin": 128,
        "max_cat_to_onehot": 1,     # use partition splits for symbol/sector
        "seed": 42,
        "nthread": -1,
    })
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    objective: str = "softprob"        # "softprob" | "focal"
    focal_gamma: float = 2.0
    stop_metric: str = "macro_f1"      # "macro_f1" | "mcc" | "mlogloss"
    class_balance: bool = True
    time_decay_halflife_days: float | None = 500.0   # None disables decay
    use_categorical: bool = True       # False drops symbol/sector entirely
    val_fraction: float = 0.15         # tail of training window for early stopping
    embargo_days: int = 25             # gap between inner-train and inner-val
    calibrate: bool = True


# ---------------------------------------------------------------------------
# 2. Sample weights
# ---------------------------------------------------------------------------
def class_balance_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights, normalised so the mean weight is 1."""
    counts = np.bincount(y, minlength=N_CLASS).astype(float)
    counts[counts == 0] = 1.0
    w = (len(y) / (N_CLASS * counts))[y]
    return w / w.mean()


def time_decay_weights(dates: pd.Series, halflife_days: float) -> np.ndarray:
    """Exponential decay so recent trading days count more."""
    d = pd.to_datetime(dates)
    age = (d.max() - d).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (age / halflife_days)
    return w / w.mean()


def build_weights(y: np.ndarray, dates: pd.Series, cfg: XGBConfig) -> np.ndarray:
    w = np.ones(len(y))
    if cfg.class_balance:
        w *= class_balance_weights(y)
    if cfg.time_decay_halflife_days:
        w *= time_decay_weights(dates, cfg.time_decay_halflife_days)
    return w / w.mean()


# ---------------------------------------------------------------------------
# 3. Custom objective: multiclass focal loss
# ---------------------------------------------------------------------------
def focal_grad_hess(margin: np.ndarray, y: np.ndarray, gamma: float,
                    weight: np.ndarray | None = None):
    """
    Focal loss  FL = -(1 - p_y)^gamma * log(p_y),  p = softmax(margin).

    Gradient is exact. The Hessian uses the standard positive diagonal
    approximation (softmax Hessian scaled by the focal modulating factor),
    because the exact focal Hessian can be negative and XGBoost needs h > 0.
    gamma = 0 reduces exactly to ordinary softmax cross-entropy.
    """
    margin = margin.reshape(-1, N_CLASS)
    p = softmax(margin, axis=1)
    n = len(y)
    py = np.clip(p[np.arange(n), y], 1e-12, 1.0)
    onehot = np.eye(N_CLASS)[y]

    # dFL/dp_y
    dfl_dpy = gamma * (1 - py) ** (gamma - 1) * np.log(py) - (1 - py) ** gamma / py
    if gamma == 0:
        dfl_dpy = -1.0 / py
    # dp_y/dz_j = p_y (1[j=y] - p_j)
    grad = (dfl_dpy * py)[:, None] * (onehot - p)

    modulator = np.maximum((1 - py) ** gamma, 1e-3)
    hess = np.maximum(modulator[:, None] * p * (1 - p), 1e-6)

    if weight is not None:
        grad *= weight[:, None]
        hess *= weight[:, None]
    return grad, hess


def focal_loss_value(margin: np.ndarray, y: np.ndarray, gamma: float) -> float:
    p = softmax(margin.reshape(-1, N_CLASS), axis=1)
    py = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    return float(np.mean(-(1 - py) ** gamma * np.log(py)))


def make_focal_objective(gamma: float):
    def obj(predt: np.ndarray, dtrain):
        y = dtrain.get_label().astype(int)
        w = dtrain.get_weight()
        w = w if w.size else None
        g, h = focal_grad_hess(predt, y, gamma, w)
        return g, h        # shape (n, 3) is accepted by xgboost>=2.0
    return obj


# ---------------------------------------------------------------------------
# 4. Custom evaluation metrics for early stopping
# ---------------------------------------------------------------------------
def to_proba(predt: np.ndarray, is_margin: bool) -> np.ndarray:
    predt = predt.reshape(-1, N_CLASS)
    return softmax(predt, axis=1) if is_margin else predt


def make_metric(name: str, is_margin: bool):
    """
    With a built-in objective XGBoost passes probabilities to custom_metric;
    with a custom objective it passes raw margins. `is_margin` handles both.
    Lower-is-better is reported for every metric so early stopping is uniform.
    """
    def metric(predt: np.ndarray, dmat):
        y = dmat.get_label().astype(int)
        p = to_proba(predt, is_margin)
        if name == "macro_f1":
            return "neg_macro_f1", -f1_score(y, p.argmax(1), average="macro")
        if name == "mcc":
            return "neg_mcc", -matthews_corrcoef(y, p.argmax(1))
        return "mlogloss", log_loss(y, p, labels=list(range(N_CLASS)))
    return metric


# ---------------------------------------------------------------------------
# 5. Embargoed inner split (for early stopping and tuning)
# ---------------------------------------------------------------------------
def inner_train_val_split(dates: pd.Series, val_fraction: float, embargo_days: int):
    """
    Last `val_fraction` of unique trading dates -> validation.
    The `embargo_days` trading dates before it are dropped from training,
    so forward-looking labels cannot leak from train into validation.
    """
    udates = np.sort(pd.to_datetime(dates).unique())
    n_val = max(1, int(len(udates) * val_fraction))
    val_start = udates[-n_val]
    cut_idx = max(0, len(udates) - n_val - embargo_days)
    train_end = udates[cut_idx - 1] if cut_idx > 0 else udates[0]
    d = pd.to_datetime(dates).to_numpy()
    train_mask = d <= train_end
    val_mask = d >= val_start
    return train_mask, val_mask


# ---------------------------------------------------------------------------
# 6. Temperature scaling
# ---------------------------------------------------------------------------
def fit_temperature(log_proba: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T minimising NLL of softmax(log_proba / T)."""
    def nll(t):
        return -np.mean(log_softmax(log_proba / t, axis=1)[np.arange(len(y)), y])
    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(log_proba: np.ndarray, t: float) -> np.ndarray:
    return log_softmax(log_proba / t, axis=1)


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(1)
    correct = (proba.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# 7. The base learner
# ---------------------------------------------------------------------------
class NepseXGB:
    """
    fit() on one walk-forward training window; predict_log_proba() on the
    test fold produces the calibrated log-probabilities the stacker consumes.
    """

    def __init__(self, cfg: XGBConfig | None = None,
                 cat_cols: tuple[str, ...] = ("symbol", "sector")):
        self.cfg = cfg or XGBConfig()
        self.cat_cols = cat_cols
        self.booster = None
        self.temperature = 1.0
        self.best_iteration = None
        self.feature_names: list[str] = []

    # -- data -------------------------------------------------------------
    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        present = [c for c in self.cat_cols if c in X.columns]
        if not self.cfg.use_categorical:
            return X.drop(columns=present)
        for c in present:
            X[c] = X[c].astype("category")
        return X

    def _dmatrix(self, X, y=None, w=None):
        import xgboost as xgb
        return xgb.DMatrix(X, label=y, weight=w, enable_categorical=self.cfg.use_categorical)

    # -- training ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, dates: pd.Series, params: dict | None = None):
        import xgboost as xgb
        cfg = self.cfg
        y = np.asarray(y, dtype=int)
        X = self._prep(X)
        if cfg.use_categorical:   # freeze category levels so test data maps identically
            self._categories = {c: X[c].cat.categories for c in X.select_dtypes("category")}
        self.feature_names = list(X.columns)

        tr, va = inner_train_val_split(dates, cfg.val_fraction, cfg.embargo_days)
        w = build_weights(y, dates, cfg)
        dtr = self._dmatrix(X[tr], y[tr], w[tr])
        dva = self._dmatrix(X[va], y[va])

        p = {**cfg.params, **(params or {}), "num_class": N_CLASS}
        custom = cfg.objective == "focal"
        if custom:
            p["disable_default_eval_metric"] = 1
            obj = make_focal_objective(cfg.focal_gamma)
        else:
            p["objective"] = "multi:softprob"
            p["disable_default_eval_metric"] = 1
            obj = None

        self.booster = xgb.train(
            p, dtr,
            num_boost_round=cfg.num_boost_round,
            evals=[(dva, "val")],
            obj=obj,
            custom_metric=make_metric(cfg.stop_metric, is_margin=custom),
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_iteration = self.booster.best_iteration

        if cfg.calibrate:
            self.temperature = fit_temperature(self._raw_log_proba(X[va]), y[va])
        return self

    # -- inference --------------------------------------------------------
    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._prep(X)
        if self.cfg.use_categorical:
            for c, cats in self._categories.items():
                X[c] = pd.Categorical(X[c], categories=cats)  # unseen symbol -> NaN
        return X[self.feature_names]

    def _raw_log_proba(self, X_prepped: pd.DataFrame) -> np.ndarray:
        d = self._dmatrix(X_prepped)
        rng = (0, self.best_iteration + 1)
        margin = self.booster.predict(d, output_margin=True, iteration_range=rng)
        return log_softmax(margin.reshape(-1, N_CLASS), axis=1)

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated log-probabilities: feed these to the logistic stacker."""
        lp = self._raw_log_proba(self._align(X))
        return apply_temperature(lp, self.temperature) if self.cfg.calibrate else lp

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 3, n_features + 1) SHAP contributions; last column is the bias."""
        d = self._dmatrix(self._align(X))
        return self.booster.predict(d, pred_contribs=True,
                                    iteration_range=(0, self.best_iteration + 1))


# ---------------------------------------------------------------------------
# 8. Nested hyperparameter search (inside ONE walk-forward training window)
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "max_depth":        ("int",   3, 6),
    "eta":              ("float", 0.01, 0.06, "log"),
    "min_child_weight": ("float", 5, 150, "log"),
    "subsample":        ("float", 0.55, 0.95),
    "colsample_bytree": ("float", 0.4, 0.9),
    "colsample_bynode": ("float", 0.5, 1.0),
    "gamma":            ("float", 0.0, 5.0),
    "lambda":           ("float", 1.0, 30.0, "log"),
    "alpha":            ("float", 0.0, 5.0),
    "max_bin":          ("cat",   [64, 128, 256]),
}


def _sample(rng: np.random.Generator) -> dict:
    out = {}
    for k, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "int":
            out[k] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "cat":
            out[k] = spec[1][rng.integers(len(spec[1]))]
        elif len(spec) == 4:
            out[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            out[k] = float(rng.uniform(spec[1], spec[2]))
    return out


def tune(X: pd.DataFrame, y: np.ndarray, dates: pd.Series,
         cfg: XGBConfig | None = None, n_trials: int = 40, seed: int = 0) -> dict:
    """
    Scores each candidate on the embargoed tail of the training window only.
    Never pass test-fold data here: the test fold must stay untouched.
    Returns the best params; refit NepseXGB with them on the full window.
    """
    cfg = cfg or XGBConfig()
    y = np.asarray(y, dtype=int)
    # Hold out the tail of the window for scoring, with its own embargo.
    tr, va = inner_train_val_split(dates, cfg.val_fraction, cfg.embargo_days)
    Xtr, ytr, dtr = X[tr], y[tr], dates[tr]
    Xva, yva = X[va], y[va]

    def score(params):
        m = NepseXGB(cfg).fit(Xtr, ytr, dtr, params)
        p = m.predict_proba(Xva)
        return log_loss(yva, p, labels=list(range(N_CLASS)))

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            for k, spec in SEARCH_SPACE.items():
                if spec[0] == "int":
                    params[k] = trial.suggest_int(k, spec[1], spec[2])
                elif spec[0] == "cat":
                    params[k] = trial.suggest_categorical(k, spec[1])
                else:
                    params[k] = trial.suggest_float(k, spec[1], spec[2], log=len(spec) == 4)
            return score(params)

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
    except ImportError:
        rng = np.random.default_rng(seed)
        best, best_s = None, np.inf
        for _ in range(n_trials):
            cand = _sample(rng)
            s = score(cand)
            if s < best_s:
                best, best_s = cand, s
        return best


# ---------------------------------------------------------------------------
# 9. Ablation grid for the paper
# ---------------------------------------------------------------------------
def ablation_configs() -> dict[str, XGBConfig]:
    """One switch changed at a time relative to the full configuration."""
    base = XGBConfig()
    variants = {
        "full": {},
        "no_class_balance": {"class_balance": False},
        "no_time_decay": {"time_decay_halflife_days": None},
        "no_symbol_sector": {"use_categorical": False},
        "focal_g1": {"objective": "focal", "focal_gamma": 1.0},
        "focal_g2": {"objective": "focal", "focal_gamma": 2.0},
        "stop_on_logloss": {"stop_metric": "mlogloss"},
        "stop_on_mcc": {"stop_metric": "mcc"},
        "no_calibration": {"calibrate": False},
        "no_embargo": {"embargo_days": 0},
        "original_350x4": {"num_boost_round": 350, "early_stopping_rounds": 10_000,
                           "class_balance": False, "time_decay_halflife_days": None,
                           "calibrate": False},
    }
    out = {}
    for name, overrides in variants.items():
        c = XGBConfig(**{**base.__dict__, "params": dict(base.params)})
        for k, v in overrides.items():
            setattr(c, k, v)
        out[name] = c
    return out
