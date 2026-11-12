"""Shared pieces for all base learners: weights, inner split, calibration, sequences."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

N_CLASS = 3


# ---------------------------------------------------------------------------
# Sample weights: balanced classes x exponential time decay
# ---------------------------------------------------------------------------
def class_balance_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=N_CLASS).astype(float)
    counts[counts == 0] = 1.0
    w = (len(y) / (N_CLASS * counts))[y]
    return w / w.mean()


def time_decay_weights(dates: pd.Series, halflife_days: float) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates))
    age = (d.max() - d).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (age / halflife_days)
    return w / w.mean()


def build_weights(y, dates, class_balance=True, halflife_days=500.0) -> np.ndarray:
    w = np.ones(len(y))
    if class_balance:
        w *= class_balance_weights(np.asarray(y, int))
    if halflife_days:
        w *= time_decay_weights(dates, halflife_days)
    return w / w.mean()


# ---------------------------------------------------------------------------
# Embargoed inner split: tail of the training window for early stopping
# ---------------------------------------------------------------------------
def inner_train_val_split(dates, val_fraction: float, embargo_days: int):
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    udates = np.sort(np.unique(d))
    n_val = max(1, int(len(udates) * val_fraction))
    val_start = udates[-n_val]
    cut = max(1, len(udates) - n_val - embargo_days)
    train_end = udates[cut - 1]
    return d <= train_end, d >= val_start


# ---------------------------------------------------------------------------
# Temperature scaling and calibration error
# ---------------------------------------------------------------------------
def fit_temperature(log_proba: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, int)

    def nll(t):
        return -np.mean(log_softmax(log_proba / t, axis=1)[np.arange(len(y)), y])
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(log_proba: np.ndarray, t: float) -> np.ndarray:
    return log_softmax(log_proba / t, axis=1)


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(1)
    correct = (proba.argmax(1) == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Lazy sequence windows for the Transformer (avoids an n x L x d tensor in RAM)
# ---------------------------------------------------------------------------
class SeqSource:
    """Feature matrix plus each row's position inside its symbol's history."""

    def __init__(self, frame: pd.DataFrame, feats: list[str], symbol_col: str,
                 date_col: str, seq_len: int):
        order = frame.sort_values([symbol_col, date_col]).index
        if not (order == frame.index).all():
            raise ValueError("frame must be sorted by (symbol, date) with a RangeIndex")
        self.F = frame[feats].to_numpy(np.float32)
        self.pos = frame.groupby(symbol_col, sort=False).cumcount().to_numpy()
        self.L = seq_len
        self.d = len(feats)

    def windows(self, rows: np.ndarray, mean=None, std=None) -> np.ndarray:
        offs = np.arange(self.L - 1, -1, -1)                    # L-1 ... 0
        idx = rows[:, None] - offs[None, :]
        valid = (self.pos[rows][:, None] - offs[None, :]) >= 0
        out = self.F[np.where(valid, idx, 0)]
        if mean is not None:
            out = (out - mean) / std
        out[~valid] = 0.0                                       # zero-pad short histories
        return np.nan_to_num(out, copy=False)


class SeqView:
    """Row subset of a SeqSource; supports X[mask] like a DataFrame."""

    def __init__(self, src: SeqSource, rows: np.ndarray):
        self.src, self.rows = src, np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, m):
        return SeqView(self.src, self.rows[m])

    def materialize(self, mean=None, std=None) -> np.ndarray:
        return self.src.windows(self.rows, mean, std)
