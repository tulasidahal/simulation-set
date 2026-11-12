"""
Data layer: schema validation, point-in-time labels, cross-sectional features,
and the data-integrity audits that form the data-engineering core of the paper.

Expected input: a long panel, one row per (date, symbol), with
    date, symbol, sector, close (adjusted), [close_raw], [volume], <numeric features>
Feature values on row t must be computable at the close of day t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig

CLASS_NAMES = ["DOWN", "FLAT", "UP"]
DERIVED = {"fwd_ret", "fwd_ret_abs", "label", "ret_1d", "ret_20d", "label_prev", "label_mom"}


# ---------------------------------------------------------------------------
# Loading and schema
# ---------------------------------------------------------------------------
def load_panel(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def reserved_columns(cfg: ExperimentConfig) -> set[str]:
    cols = {cfg.date_col, cfg.symbol_col, cfg.sector_col, cfg.price_col}
    cols |= {c for c in (cfg.raw_price_col, cfg.volume_col) if c}
    return cols | DERIVED


def feature_columns(df: pd.DataFrame, cfg: ExperimentConfig) -> list[str]:
    reserved = reserved_columns(cfg)
    return [c for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]


def validate_schema(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    need = [cfg.date_col, cfg.symbol_col, cfg.price_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    dup = df.duplicated([cfg.date_col, cfg.symbol_col]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (date, symbol) rows")
    if (df[cfg.price_col] <= 0).any():
        raise ValueError("non-positive prices found")
    if cfg.sector_col not in df.columns:
        df[cfg.sector_col] = "UNKNOWN"
    return df.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Labels (point-in-time) and helper returns
# ---------------------------------------------------------------------------
def add_returns_and_labels(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, float]:
    """
    fwd_ret(t) = P(t+h)/P(t) - 1 per symbol, on the ADJUSTED price.
    The label uses future data by design; the walk-forward splitter purges
    h days plus the embargo so no training label overlaps a test date.
    Returns the frame (rows without a full forward window dropped) and tau.
    """
    g = df.groupby(cfg.symbol_col, sort=False)[cfg.price_col]
    h = cfg.horizon
    df["fwd_ret"] = g.shift(-h) / df[cfg.price_col] - 1
    df["fwd_ret_abs"] = df["fwd_ret"]                           # for the trading backtest
    df["ret_1d"] = df[cfg.price_col] / g.shift(1) - 1           # past, used by baselines
    df["ret_20d"] = df[cfg.price_col] / g.shift(20) - 1
    if cfg.drop_label_breaches:
        # A forward move beyond the compounded circuit band cannot be a real trade:
        # it is an unadjusted bonus/rights issue or a data error, so the label is wrong.
        bound = (1 + cfg.circuit_limit) ** h - 1 + 0.005
        bad = df["fwd_ret"].abs() > bound
        df.attrs["label_breaches_dropped"] = int(bad.sum())
        df.attrs["label_breach_examples"] = df.loc[bad, [cfg.date_col, cfg.symbol_col, "fwd_ret"]] \
            .head(20).astype(str).to_dict("records")
        df.loc[bad, "fwd_ret"] = np.nan
    if cfg.label_mode == "market_relative":
        df["fwd_ret"] -= df.groupby(cfg.date_col)["fwd_ret"].transform("mean")
    df = df.dropna(subset=["fwd_ret"]).reset_index(drop=True)

    if cfg.flat_mode == "warmup_quantile":
        dates = np.sort(df[cfg.date_col].unique())
        warm_end = dates[int(len(dates) * cfg.warmup_frac) - 1]
        warm = df.loc[df[cfg.date_col] <= warm_end, "fwd_ret"].abs()
        tau = float(warm.quantile(cfg.flat_quantile))          # warm-up only: no leakage
    else:
        tau = cfg.flat_threshold

    df["label"] = np.where(df["fwd_ret"] > tau, 2, np.where(df["fwd_ret"] < -tau, 0, 1))
    prev = df["ret_1d"]                     # today's realised move, known at t
    df["label_prev"] = np.where(prev > tau, 2, np.where(prev < -tau, 0, 1))
    mom = df["ret_20d"].fillna(0.0)
    df["label_mom"] = np.where(mom > tau, 2, np.where(mom < -tau, 0, 1))
    return df, tau


def rank_normalise(df: pd.DataFrame, cols: list[str], cfg: ExperimentConfig) -> pd.DataFrame:
    """Within-date percentile rank in [-0.5, 0.5]. Uses only same-day data."""
    out = df.copy()
    out[cols] = df.groupby(cfg.date_col)[cols].rank(pct=True) - 0.5
    out[cols] = out[cols].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Data-quality and integrity audits
# ---------------------------------------------------------------------------
def corporate_action_audit(df: pd.DataFrame, cfg: ExperimentConfig, tol: float = 0.005) -> dict:
    """
    Day-over-day moves outside the circuit band cannot be real trades, so
    they reveal unadjusted bonus/rights issues. Expect many on the raw price,
    ~zero on a correctly adjusted price.
    """
    res = {}
    for col in [c for c in (cfg.raw_price_col, cfg.price_col) if c and c in df.columns]:
        r = df.groupby(cfg.symbol_col, sort=False)[col].pct_change()
        breach = r.abs() > cfg.circuit_limit + tol
        ev = df.loc[breach, [cfg.date_col, cfg.symbol_col]].assign(move=r[breach])
        res[col] = {"n_breaches": int(breach.sum()),
                    "n_symbols": int(ev[cfg.symbol_col].nunique()),
                    "examples": ev.head(10).to_dict("records")}
    return res


def data_quality_report(df: pd.DataFrame, cfg: ExperimentConfig, feats: list[str]) -> dict:
    d = df[cfg.date_col]
    all_dates = np.sort(d.unique())
    per_sym = df.groupby(cfg.symbol_col)[cfg.date_col].agg(["min", "max", "count"])
    date_idx = pd.Series(np.arange(len(all_dates)), index=all_dates)
    expected = date_idx.loc[per_sym["max"]].to_numpy() - date_idx.loc[per_sym["min"]].to_numpy() + 1
    missing_days = int((expected - per_sym["count"].to_numpy()).sum())
    rep = {
        "rows": len(df), "symbols": int(df[cfg.symbol_col].nunique()),
        "dates": len(all_dates), "first_date": str(all_dates[0])[:10],
        "last_date": str(all_dates[-1])[:10], "n_features": len(feats),
        "symbol_days_missing_inside_listing": missing_days,
        "feature_missing_rate_top5": df[feats].isna().mean().sort_values(ascending=False)
                                       .head(5).round(4).to_dict(),
        "label_distribution": df["label"].value_counts(normalize=True).sort_index()
                                         .rename(index=dict(enumerate(CLASS_NAMES))).round(4).to_dict()
        if "label" in df else None,
    }
    if cfg.volume_col and cfg.volume_col in df:
        rep["zero_volume_share"] = float((df[cfg.volume_col] == 0).mean())
    rep["corporate_actions"] = corporate_action_audit(df, cfg)
    return rep


def information_coefficients(df: pd.DataFrame, feats: list[str], cfg: ExperimentConfig) -> pd.Series:
    """Mean per-date Spearman correlation between each feature and fwd_ret."""
    keys = df[cfg.date_col]
    y = df.groupby(keys)["fwd_ret"].rank(pct=True)
    y = y - y.groupby(keys).transform("mean")
    X = df.groupby(keys)[feats].rank(pct=True)
    X = X - X.groupby(keys).transform("mean")
    num = (X.mul(y, axis=0)).groupby(keys).sum()
    den = np.sqrt((X ** 2).groupby(keys).sum().mul((y ** 2).groupby(keys).sum(), axis=0))
    return (num / den.replace(0, np.nan)).mean().sort_values(key=np.abs, ascending=False)


def leakage_ic_audit(df: pd.DataFrame, feats: list[str], cfg: ExperimentConfig,
                     threshold: float | None = None) -> dict:
    """
    Real daily signals rarely exceed |IC| ~ 0.1. A feature far above the
    threshold almost certainly contains future information.
    """
    threshold = cfg.leak_ic_threshold if threshold is None else threshold
    ic = information_coefficients(df, feats, cfg)
    return {"threshold": threshold, "top_ic": ic.head(10).round(4).to_dict(),
            "flagged": ic[ic.abs() > threshold].round(4).to_dict()}


def canary_validation(df: pd.DataFrame, feats: list[str], cfg: ExperimentConfig,
                      seed: int = 0) -> dict:
    """
    Plant a feature = future return + noise and check the audit flags it.
    Noise uses a robust scale (MAD), because heavy-tailed returns make the
    standard deviation dominated by a few extreme days.
    """
    r = df["fwd_ret"]
    scale = 1.4826 * (r - r.median()).abs().median()
    canary = df.assign(canary_future=r + np.random.default_rng(seed).normal(0, scale, len(df)))
    audit = leakage_ic_audit(canary, feats + ["canary_future"], cfg)
    return {"canary_flagged": "canary_future" in audit["flagged"],
            "canary_ic": audit["top_ic"].get("canary_future"),
            "threshold": audit["threshold"], "noise_scale_mad": float(scale)}


def truncation_invariance_test(feature_fn, raw: pd.DataFrame, cfg: ExperimentConfig,
                               cut_date) -> dict:
    """
    Point-in-time test: features for dates <= cut_date must be identical
    whether or not data after cut_date exists. Any difference is look-ahead.
    """
    cut_date = pd.Timestamp(cut_date)
    full = feature_fn(raw)
    trunc = feature_fn(raw[raw[cfg.date_col] <= cut_date])
    key = [cfg.date_col, cfg.symbol_col]
    a = full[full[cfg.date_col] <= cut_date].set_index(key).sort_index()
    b = trunc.set_index(key).sort_index()
    cols = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c])]
    A, B = a.loc[b.index, cols], b[cols]
    diff = (A - B).abs().max().fillna(0.0)
    nan_mismatch = (A.isna() != B.isna()).sum()     # value exists only with future data
    bad = (diff > 1e-9) | (nan_mismatch > 0)
    return {"cut_date": str(cut_date)[:10], "max_abs_diff": diff.round(10).to_dict(),
            "nan_mismatches": nan_mismatch[nan_mismatch > 0].to_dict(),
            "violations": bad[bad].index.tolist()}


# ---------------------------------------------------------------------------
# Synthetic panel (smoke tests and the reproducibility demo only)
# ---------------------------------------------------------------------------
def demo_features(raw: pd.DataFrame, cfg: ExperimentConfig | None = None) -> pd.DataFrame:
    """Backward-looking features from adjusted close and volume (point-in-time)."""
    cfg = cfg or ExperimentConfig()
    df = raw.sort_values([cfg.symbol_col, cfg.date_col]).copy()
    g = df.groupby(cfg.symbol_col, sort=False)
    p = df[cfg.price_col]
    r = p / g[cfg.price_col].shift(1) - 1
    df["f_mom5"] = p / g[cfg.price_col].shift(5) - 1
    df["f_mom20"] = p / g[cfg.price_col].shift(20) - 1
    df["f_vol20"] = r.groupby(df[cfg.symbol_col]).transform(lambda s: s.rolling(20).std())
    df["f_rev1"] = -r
    df["f_ma_gap"] = p / g[cfg.price_col].transform(lambda s: s.rolling(20).mean()) - 1
    if cfg.volume_col in df:
        lv = np.log1p(df[cfg.volume_col])
        df["f_vol_z"] = (lv - lv.groupby(df[cfg.symbol_col]).transform(
            lambda s: s.rolling(20).mean())) / lv.groupby(df[cfg.symbol_col]).transform(
            lambda s: s.rolling(20).std()).replace(0, np.nan)
    return df


def make_synthetic_panel(n_symbols: int = 40, n_dates: int = 900, n_noise: int = 8,
                         signal: float = 0.35, seed: int = 0) -> pd.DataFrame:
    """
    Panel with a weak, persistent latent signal observed through noisy
    features, regime shifts, circuit-limited returns, and two unadjusted
    bonus-share events in close_raw (for the corporate-action audit).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates)
    sectors = np.array(["Banking", "Hydropower", "Microfinance", "Insurance", "Others"])
    rows = []
    regime = np.where(np.arange(n_dates) < n_dates // 2, 0.0006, -0.0003)
    mkt = rng.normal(regime, 0.010)
    for s in range(n_symbols):
        latent = np.zeros(n_dates)
        for t in range(1, n_dates):
            latent[t] = 0.95 * latent[t - 1] + rng.normal(0, 0.3)
        eps = rng.normal(0, 0.016, n_dates)
        ret = np.zeros(n_dates)
        ret[1:] = mkt[1:] + signal * 0.01 * latent[:-1] + eps[1:]
        ret = np.clip(ret, -0.10, 0.10)
        close = 100 * np.cumprod(1 + ret)
        raw = close.copy()
        if s < 2:                                  # unadjusted bonus share
            k = n_dates // 3 + 50 * s
            raw[k:] = raw[k:] / 1.25
        vol = rng.lognormal(10, 1, n_dates) * (rng.random(n_dates) > 0.03)
        df = pd.DataFrame({"date": dates, "symbol": f"SYM{s:03d}",
                           "sector": sectors[s % len(sectors)], "close": close,
                           "close_raw": raw, "volume": vol,
                           "f_latent_obs": latent + rng.normal(0, 1.0, n_dates)})
        for j in range(n_noise):
            df[f"f_noise{j}"] = rng.normal(size=n_dates)
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    panel = demo_features(panel)
    return panel.dropna().reset_index(drop=True)
