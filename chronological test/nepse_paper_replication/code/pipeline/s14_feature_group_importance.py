"""Stage 14 (optional): grouped permutation importance of the 60 predictors on development folds 7 and 8 (the final test is never used).
Writes reports/drivers.json: rise in validation log-loss when a feature group is shuffled, by horizon."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import s05_audit as A
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
panel = pd.read_pickle(WORK / "panel.pkl")
splits = json.load(open(WORK / "splits.json"))
G = {
 "Returns & momentum": ["ret1","ret2","ret3","ret5","ret10","ret20","ret60","rel_ret5","drawdown60","dist_high252","dist_low252"],
 "Volatility & range": ["vol5","vol10","vol20","vol60","hl_range","atr14_pct","bb_pos"],
 "Candle shape & gaps": ["oc_ret","close_vwap","gap","close_pos"],
 "Trend & oscillators": ["ma5_dist","ma10_dist","ma20_dist","ma50_dist","rsi14","macd_hist","stoch_k14"],
 "Liquidity & volume": ["volratio_1_20","volratio_5_20","log_amount","log_trades","amihud20","stale20","turnover_trend","avg_trade_size","trades_ratio20"],
 "Broker concentration": ["buyer_hhi","seller_hhi","buyer_top1","seller_top1","buyer_top5","seller_top5","log_buyers","log_sellers","self_share","large_share","size_cv","broker_imbalance","buyer_hhi_chg5"],
 "NEPSE market": ["mkt_ret1","mkt_vol20"],
 "Fundamentals": ["fund_ep","fund_bp","fund_roe","fund_de","fund_rev_growth","fund_eps_yoy","fund_age"],
}
assert sorted(sum(G.values(), [])) == sorted(MODEL_FEATURES), set(MODEL_FEATURES) ^ set(sum(G.values(), []))
rng = np.random.default_rng(42); out = {}
for h in HORIZONS:
    y = panel[f"y_{h}"].values; res = {g: [] for g in G}; base = []
    for k in (7, 8):
        tr, va = A.fold_masks(panel, splits, k)
        tr &= ~np.isnan(y); va &= ~np.isnan(y)
        Xtr = panel.loc[tr, MODEL_FEATURES].values.astype("float32"); ytr = y[tr]
        Xva = panel.loc[va, MODEL_FEATURES].values.astype("float32"); yva = y[va]
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
        b = log_loss(yva, m.predict_proba(Xva)[:, 1]); base.append(b)
        for g, cols in G.items():
            idx = [MODEL_FEATURES.index(c) for c in cols]; d = []
            for _ in range(2):
                Xp = Xva.copy(); perm = rng.permutation(len(Xp)); Xp[:, idx] = Xva[perm][:, idx]
                d.append(log_loss(yva, m.predict_proba(Xp)[:, 1]) - b)
            res[g].append(float(np.mean(d)))
        log(f"h={h} fold={k} base_ll={b:.4f}")
    out[f"{h}d"] = {"base_logloss": float(np.mean(base)), "delta_logloss": {g: float(np.mean(v)) for g, v in res.items()}}
json.dump(out, open(REPORTS / "drivers.json", "w"), indent=1)
print("DRIVERS_OK")
