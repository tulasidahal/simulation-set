"""
Chronological walk-forward protocol with purging and embargo.

    |---- warm-up (train only) ----|-- block 0 --|-- block 1 --| ... |-- block K-1 --|

For test block k starting at date index s, base models train on dates with
index < s - horizon - embargo when purging is on: a training label at index i
looks forward to i + horizon, so it can never reach the test block, and the
extra embargo absorbs serial correlation in features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExperimentConfig


@dataclass
class Fold:
    k: int
    train_end_idx: int          # exclusive index into unique dates
    test_start_idx: int
    test_end_idx: int           # exclusive
    udates: np.ndarray

    @property
    def test_start(self):
        return self.udates[self.test_start_idx]

    @property
    def test_end(self):
        return self.udates[self.test_end_idx - 1]

    def train_mask(self, dates: pd.Series) -> np.ndarray:
        if self.train_end_idx <= 0:
            return np.zeros(len(dates), bool)
        return dates.to_numpy() <= self.udates[self.train_end_idx - 1]

    def test_mask(self, dates: pd.Series) -> np.ndarray:
        d = dates.to_numpy()
        return (d >= self.test_start) & (d <= self.test_end)

    def describe(self) -> dict:
        return {"block": self.k,
                "train_end": str(self.udates[self.train_end_idx - 1])[:10],
                "test_start": str(self.test_start)[:10],
                "test_end": str(self.test_end)[:10],
                "gap_trading_days": int(self.test_start_idx - self.train_end_idx)}


def walk_forward(dates: pd.Series, cfg: ExperimentConfig, purge: bool = True,
                 embargo: int | None = None) -> list[Fold]:
    udates = np.sort(pd.to_datetime(dates).unique())
    n = len(udates)
    warm = int(n * cfg.warmup_frac)
    edges = np.linspace(warm, n, cfg.n_blocks + 1).astype(int)
    emb = cfg.embargo_days if embargo is None else embargo
    gap = (cfg.horizon + emb) if purge else 0
    folds = []
    for k in range(cfg.n_blocks):
        s, e = edges[k], edges[k + 1]
        folds.append(Fold(k, max(0, s - gap), s, e, udates))
    return folds


def stacker_train_mask(block_of_row: np.ndarray, dates: pd.Series, fold: Fold,
                       cfg: ExperimentConfig, purge: bool = True) -> np.ndarray:
    """
    Meta-learner for block k trains on out-of-fold predictions from earlier
    blocks only, with the same purge + embargo before block k.
    """
    earlier = (block_of_row >= 0) & (block_of_row < fold.k)
    gap = (cfg.horizon + cfg.embargo_days) if purge else 0
    cut_idx = fold.test_start_idx - gap
    if cut_idx <= 0:
        return np.zeros(len(dates), bool)
    return earlier & (dates.to_numpy() <= fold.udates[cut_idx - 1])


def random_kfold_masks(n: int, k: int, seed: int):
    """The WRONG protocol, included only to measure how much it inflates scores."""
    rng = np.random.default_rng(seed)
    fold_id = rng.integers(0, k, n)
    for j in range(k):
        yield fold_id != j, fold_id == j
