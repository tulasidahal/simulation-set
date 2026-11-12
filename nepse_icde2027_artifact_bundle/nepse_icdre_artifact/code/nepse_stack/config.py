"""Experiment configuration. Every field is reported in the paper's setup section."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    # ---- data / labels -------------------------------------------------
    date_col: str = "date"
    symbol_col: str = "symbol"
    sector_col: str = "sector"
    price_col: str = "close"            # corporate-action ADJUSTED close
    raw_price_col: str | None = "close_raw"   # unadjusted close, for the audit only
    volume_col: str | None = "volume"
    horizon: int = 1                    # predict direction over the next h trading days
    label_mode: str = "absolute"        # "absolute" | "market_relative"
    flat_mode: str = "fixed"            # "fixed" | "warmup_quantile"
    flat_threshold: float = 0.005       # |r| < tau  -> FLAT  (fixed mode)
    flat_quantile: float = 1 / 3        # share of FLAT labels in warm-up (quantile mode)
    cross_sectional_rank: bool = True   # rank-normalise features within each date
    circuit_limit: float = 0.10         # NEPSE daily price band used by the audit
    drop_label_breaches: bool = True    # drop labels whose move exceeds the circuit band
    leak_ic_threshold: float = 0.15     # |IC| above this flags a feature as leaky

    # ---- walk-forward protocol ----------------------------------------
    warmup_frac: float = 0.40           # first 40% of dates: training only, never tested
    n_blocks: int = 6                   # chronological test blocks after warm-up
    embargo_days: int = 25              # trading days dropped between train and test
    min_stack_blocks: int = 2           # stacker needs >= this many OOF blocks
    inner_val_frac: float = 0.15        # tail of each training window for early stopping
    seq_len: int = 20                   # Transformer lookback (20 x n_features)

    # ---- models --------------------------------------------------------
    models: tuple[str, ...] = ("catboost", "lightgbm", "xgboost", "transformer")
    stacker_C: float = 0.1
    confidence_threshold: float = 0.80  # legacy fixed flag; coverage metrics are primary
    coverages: tuple[float, ...] = (0.1, 0.2, 0.3)   # accuracy on the most confident x%
    trade_top_frac: float = 0.2         # top-confidence trading rule: top 20% per day
    proposed: str = "stack_lr_logits"   # method all significance tests compare against
    allow_fallbacks: bool = True        # sklearn stand-ins when a library is missing

    # ---- trading evaluation (set from the CURRENT NEPSE fee schedule) ---
    long_only: bool = True              # no short selling on NEPSE
    cost_bps_roundtrip: float = 80.0    # commission + SEBON + DP, both legs (placeholder)
    cgt_rate: float = 0.0               # capital gains tax on positive trade P&L (placeholder)
    trading_days_per_year: int = 240

    # ---- reproducibility ------------------------------------------------
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_bootstrap: int = 1000
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
