"""Shared configuration and utilities for the NEPSE multi-horizon forecasting pipeline."""
import os, json, math, time
from pathlib import Path
import numpy as np, pandas as pd

UPLOADS = Path(os.environ.get("NEPSE_UPLOADS", "/mnt/user-data/uploads"))
WORK = Path(os.environ.get("NEPSE_WORK", "./work")); WORK.mkdir(parents=True, exist_ok=True)
REPORTS = WORK / "reports"; REPORTS.mkdir(exist_ok=True)
MODELS = WORK / "models"; MODELS.mkdir(exist_ok=True)
ZIPS = [UPLOADS / f"floorsheet_data_part{i}.zip" for i in (1, 2, 3)]

HORIZONS = [1, 3, 5, 10]
MIN_TRAIN = 504        # sessions in the first training window
EMBARGO = 25           # sessions purged between any train end and the next evaluation block (> max horizon)
N_FOLDS = 8            # expanding-window walk-forward validation blocks
FINAL_TEST = 250       # untouched final test sessions
CONF_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
SEED = 42
SEQ_LEN = 30           # sequence length for Transformer / TFT
LARGE_TRADE = 1000     # shares; "large trade" flag for floorsheet features
CORP_ACTION_JUMP = 0.11  # NEPSE circuit limit is 10%; larger close-to-close jumps are treated as price adjustments

# ---- thin-floorsheet exclusion -------------------------------------------------------------------
# In Feb 2022, Apr 2022 - Mar 2023 and Mar 2024 the floorsheet holds only 20-60 of ~200 stocks a session.
# Prices there are complete (adjusted OHLCV) but every broker/trade-size feature is missing, so those
# sessions carry a different feature distribution from the rest of the sample. With NEPSE_EXCLUDE_GAP=1
# a session is dropped from the modelling panel when it belongs to a run of >= GAP_MIN_BLOCK consecutive
# sessions whose floorsheet coverage is below GAP_MIN_COVERAGE, and so are the GAP_BUFFER sessions that
# follow each such run, so that no rolling window and no sequence window reads a thin session.
EXCLUDE_GAP = os.environ.get("NEPSE_EXCLUDE_GAP", "0") == "1"
GAP_MIN_COVERAGE = 0.50   # share of that session's eligible stocks present in the floorsheet
GAP_MIN_BLOCK = 5         # consecutive thin sessions needed to call it a structural gap (isolated thin days are kept)
GAP_BUFFER = 19           # = SEQ_WIN - 1 and >= the longest floorsheet rolling window, so windows stay gap-free

FEATURES = [
    # price/return (7)
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ret60",
    # volatility (4)
    "vol5", "vol10", "vol20", "vol60",
    # candle (5)
    "hl_range", "oc_ret", "close_vwap", "gap", "close_pos",
    # moving-average distance (4)
    "ma5_dist", "ma10_dist", "ma20_dist", "ma50_dist",
    # oscillators (5)
    "rsi14", "macd_hist", "bb_pos", "atr14_pct", "stoch_k14",
    # range position (3)
    "drawdown60", "dist_high252", "dist_low252",
    # liquidity / volume (9)
    "volratio_1_20", "volratio_5_20", "log_amount", "log_trades", "amihud20", "stale20",
    "turnover_trend", "avg_trade_size", "trades_ratio20",
    # floorsheet / broker microstructure (13)
    "buyer_hhi", "seller_hhi", "buyer_top1", "seller_top1", "buyer_top5", "seller_top5",
    "log_buyers", "log_sellers", "self_share", "large_share", "size_cv", "broker_imbalance", "buyer_hhi_chg5",
    # market context (3)
    "mkt_ret1", "mkt_vol20", "rel_ret5",
]
assert len(FEATURES) == 53, len(FEATURES)

# point-in-time fundamentals (extension block; quarterly reports usable only from quarter end + publication lag)
FUND_FEATURES = ["fund_ep", "fund_bp", "fund_roe", "fund_de", "fund_rev_growth", "fund_eps_yoy", "fund_age"]
ALL_FEATURES = FEATURES + FUND_FEATURES
FEATURE_SET = os.environ.get("NEPSE_FEATURESET", "all")          # "core" = 53 predictors, "all" = 53 + 7 fundamentals
MODEL_FEATURES = FEATURES if FEATURE_SET == "core" else ALL_FEATURES

STUDY_START = pd.Timestamp("2018-10-01")     # first floorsheet session; earlier OHLCV history only warms up rolling windows
WARMUP_START = pd.Timestamp("2017-06-01")
INDEX_SYMBOLS = {"NEPSE", "SENSITIVE", "FLOAT", "SENFLOAT", "BANKING", "DEVBANK", "FINANCE", "HOTELS", "HYDROPOWER", "INVESTMENT",
                 "LIFEINSU", "MANUFACTURE", "MICROFINANCE", "MUTUAL", "NONLIFEINSU", "OTHERS", "TRADING"}
FUND_PRICE_MAX = 30    # median traded price below NPR 30 => closed-end mutual fund (NAV around NPR 10), not an ordinary share

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, (pd.Timestamp,)): return str(o)
        if isinstance(o, Path): return str(o)
        raise TypeError(type(o))
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=conv)

def load_json(path):
    with open(path) as f: return json.load(f)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
