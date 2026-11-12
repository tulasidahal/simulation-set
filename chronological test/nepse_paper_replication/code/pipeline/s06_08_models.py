"""
Stage 06  Five classifiers: CatBoost, LightGBM, XGBoost, Logistic Regression, Transformer.
Stage 07  LightGBM LambdaRank ranker and a Temporal-Fusion-Transformer-style sequence model ("TFT-lite").
Stage 08  Out-of-fold (OOF) predictions for every model and horizon over the 8 walk-forward folds.

Two model families:
  --family native   CatBoost, LightGBM, XGBoost, logistic, Transformer, TFT-lite, LightGBM LambdaRank (needs those libraries)
  --family sklearn  HistGradientBoosting, ExtraTrees, logistic, MLP, windowed (20-session) MLP, HGB pointwise ranker -
                    every model is the algorithm its name says; no stand-ins.
`--allow-fallback` (native family only) substitutes scikit-learn models for missing libraries - SMOKE TESTING ONLY; every
output records the backend actually used ("native" or "fallback:<what>").
"""
import argparse, copy, importlib, time
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from common import *
from metrics import classification_metrics
import s05_audit as A

CLASSIFIERS = ["catboost", "lightgbm", "xgboost", "logistic", "transformer"]
SEQUENCE_EXTRA = ["tft"]
ALL_PROB_MODELS = CLASSIFIERS + SEQUENCE_EXTRA
FAMILIES = {"native": ALL_PROB_MODELS + ["ranker"], "sklearn": ["hgb", "extratrees", "logistic", "mlp", "seq_mlp", "ranker"]}
ALGORITHMS = {
    "catboost": "CatBoostClassifier (depth 6, lr 0.05, early stopping)", "lightgbm": "LGBMClassifier (31 leaves, lr 0.03, early stopping)",
    "xgboost": "XGBClassifier (depth 6, lr 0.03, early stopping)", "transformer": "Transformer encoder (3 layers, d=64, 4 heads) on 30-session windows",
    "tft": "TFT-lite: variable selection + LSTM + gated attention on 30-session windows", "ranker_native": "LightGBM LambdaRank on within-session return quintiles",
    "hgb": "HistGradientBoostingClassifier (31 leaves, lr 0.05, L2 5, early stopping on a time-ordered hold-out)",
    "extratrees": "ExtraTreesClassifier (200 trees, min leaf 100, 30% features, 50% bootstrap)",
    "logistic": "Logistic regression (L2, C=0.1) on quantile-normalised inputs",
    "mlp": "Multilayer perceptron (128-64, ReLU, Adam, early stopping), one network for all four horizons",
    "seq_mlp": "Windowed MLP: 12 key features over the last 20 sessions + the current 60-feature vector, one network for all horizons",
    "ranker_sklearn": "HistGradientBoostingRegressor on within-session return quintiles (pointwise ranker)"}
SEQ_WIN = 20
SEQ_COLS = ["ret1", "oc_ret", "gap", "hl_range", "close_pos", "close_vwap", "volratio_1_20", "log_trades", "buyer_top5", "seller_top5", "broker_imbalance", "mkt_ret1"]
MISS_COLS = ["close_vwap", "fund_ep"]                 # missingness flags for the networks (floorsheet gap, no fundamentals)
DEFAULT_CAPS = {"extratrees": 150_000, "mlp": 150_000, "seq_mlp": 120_000}

def cap_for(kind, cfg):
    caps = [c for c in (cfg.get("caps", {}).get(kind, 0), cfg["max_train_rows"]) if c]
    return min(caps) if caps else 0
LIB = {"catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost", "transformer": "torch", "tft": "torch", "ranker": "lightgbm"}

def have(mod):
    try: importlib.import_module(mod); return True
    except Exception: return False

def backend_of(kind, allow_fallback):
    lib = LIB.get(kind)
    if lib is None or have(lib): return "native"
    if allow_fallback: return f"fallback:sklearn(no {lib})"
    raise RuntimeError(f"Model '{kind}' needs the '{lib}' package. Install it (pip install {lib}) or pass --allow-fallback for a smoke test only.")

def inner_split(sess, frac=0.12, purge=15):
    u = np.unique(sess); cut = u[int(len(u) * (1 - frac))]
    return sess < cut - purge, sess >= cut

def robust_scale(Xtr):
    mu = np.nanmedian(Xtr, 0); q75, q25 = np.nanpercentile(Xtr, 75, 0), np.nanpercentile(Xtr, 25, 0)
    sd = (q75 - q25) / 1.349; sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0; mu[~np.isfinite(mu)] = 0.0
    return mu.astype("float32"), sd.astype("float32")

def apply_scale(X, mu, sd): return np.nan_to_num(np.clip((X - mu) / sd, -5, 5), nan=0.0).astype("float32")

# ------------------------------------------------------------------ tabular classifiers
def make_tabular(kind, cfg, n_threads):
    be = backend_of(kind, cfg["allow_fallback"]); fb = be != "native"
    if kind == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=5.0,
                                              max_features=0.8, early_stopping=True, n_iter_no_change=50, scoring="loss", random_state=SEED), "native"
    if kind == "extratrees":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=200, min_samples_leaf=100, max_features=0.3, bootstrap=True,
                             max_samples=0.5, n_jobs=n_threads, random_state=SEED)), "native"
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(output_distribution="normal", subsample=100000, random_state=SEED),
                             LogisticRegression(C=0.1, max_iter=500)), "native"
    if kind == "lightgbm" and not fb:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=cfg["n_estimators"], learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=5.0, n_jobs=n_threads, random_state=SEED, verbose=-1), be
    if kind == "xgboost" and not fb:
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=cfg["n_estimators"], learning_rate=0.03, max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5.0, tree_method="hist", early_stopping_rounds=100, eval_metric="logloss", n_jobs=n_threads, random_state=SEED), be
    if kind == "catboost" and not fb:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=cfg["n_estimators"], learning_rate=0.05, depth=6, l2_leaf_reg=5, loss_function="Logloss", eval_metric="Logloss",
                                  random_seed=SEED, verbose=0, thread_count=n_threads, early_stopping_rounds=100), be
    # ---- scikit-learn stand-ins (smoke test only) ----
    if kind == "lightgbm": return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=5.0, early_stopping=True, random_state=SEED), be
    if kind == "xgboost": return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=100, l2_regularization=5.0, early_stopping=True, random_state=SEED + 1), be
    if kind == "catboost": return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=150, min_samples_leaf=150, max_features=0.5, n_jobs=n_threads, random_state=SEED)), be
    raise ValueError(kind)

def fit_tabular(kind, cfg, n_threads, Xtr, ytr, str_):
    model, be = make_tabular(kind, cfg, n_threads)
    native_gbm = be == "native" and kind in ("lightgbm", "xgboost", "catboost")
    if kind == "hgb":                                   # early stopping on the last 12% of training sessions (purged), never on random rows
        a, e = inner_split(str_); model.fit(Xtr[a], ytr[a], X_val=Xtr[e], y_val=ytr[e])
    elif native_gbm:
        a, e = inner_split(str_); Xa, ya, Xe, ye = Xtr[a], ytr[a], Xtr[e], ytr[e]
        if kind == "lightgbm":
            import lightgbm as lgb
            model.fit(Xa, ya, eval_set=[(Xe, ye)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        elif kind == "xgboost": model.fit(Xa, ya, eval_set=[(Xe, ye)], verbose=False)
        else: model.fit(Xa, ya, eval_set=(Xe, ye), use_best_model=True)
    else:
        model.fit(Xtr, ytr)
    return model, be

def save_tabular(model, kind, be, path):
    if be == "native" and kind == "lightgbm": model.booster_.save_model(str(path) + ".txt")
    elif be == "native" and kind == "xgboost": model.save_model(str(path) + ".json")
    elif be == "native" and kind == "catboost": model.save_model(str(path) + ".cbm")
    else: joblib.dump(model, str(path) + ".joblib")

# ------------------------------------------------------------------ sequence models (PyTorch)
def _torch():
    import torch, torch.nn as nn
    return torch, nn

def build_net(kind, nf, nout):
    torch, nn = _torch()
    if kind == "transformer":
        class Net(nn.Module):
            def __init__(s, d=64, heads=4, layers=3, ff=128, drop=0.2):
                super().__init__()
                s.inp = nn.Linear(nf, d); s.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d)); nn.init.normal_(s.pos, std=0.02)
                s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
                s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, nout))
            def forward(s, x): return s.head(s.enc(s.inp(x) + s.pos)[:, -1])
        return Net()
    class GRN(nn.Module):                                   # gated residual network
        def __init__(s, d_in, d_h, d_out, drop=0.1):
            super().__init__()
            s.fc1, s.fc2, s.gate = nn.Linear(d_in, d_h), nn.Linear(d_h, d_out), nn.Linear(d_h, d_out * 2)
            s.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity(); s.norm = nn.LayerNorm(d_out); s.drop = nn.Dropout(drop)
        def forward(s, x):
            h = s.drop(torch.nn.functional.elu(s.fc1(x))); a, b = s.gate(h).chunk(2, -1)
            return s.norm(s.skip(x) + a * torch.sigmoid(b))
    class TFTLite(nn.Module):
        """Variable-selection + LSTM encoder + gated skips + attention from the last step + multi-horizon heads."""
        def __init__(s, d=32, heads=4, drop=0.1):
            super().__init__()
            s.var_w = nn.Parameter(torch.randn(nf, d) * 0.1); s.var_b = nn.Parameter(torch.zeros(nf, d))
            s.sel = GRN(nf * d, d, nf, drop); s.lstm = nn.LSTM(d, d, batch_first=True)
            s.gate1 = nn.Linear(d, d * 2); s.n1 = nn.LayerNorm(d); s.enrich = GRN(d, d, d, drop)
            s.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            s.gate2 = nn.Linear(d, d * 2); s.n2 = nn.LayerNorm(d); s.post = GRN(d, d, d, drop); s.head = nn.Linear(d, nout)
        def forward(s, x):
            emb = x.unsqueeze(-1) * s.var_w + s.var_b                       # (B,L,F,d)
            w = torch.softmax(s.sel(emb.flatten(2)), -1)                    # (B,L,F) variable-selection weights
            h = (emb * w.unsqueeze(-1)).sum(2)                              # (B,L,d)
            o, _ = s.lstm(h); a, b = s.gate1(o).chunk(2, -1); o = s.n1(h + a * torch.sigmoid(b))
            e = s.enrich(o); q = e[:, -1:, :]
            att, _ = s.attn(q, e, e, need_weights=False)                    # last step attends over the window (past only)
            a, b = s.gate2(att).chunk(2, -1); z = s.n2(q + a * torch.sigmoid(b))
            return s.head(s.post(z)).squeeze(1)
    return TFTLite()

def fit_predict_seq(kind, cfg, Xall, Ymat, sess, tr_idx, pred_idx, save_path=None):
    """Multi-horizon sequence model: one network, four heads. Xall/Ymat are symbol-major panel arrays."""
    torch, nn = _torch(); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu"
    mu, sd = robust_scale(Xall[tr_idx]); Z = apply_scale(Xall, mu, sd)
    offs = np.arange(-SEQ_LEN + 1, 1)
    if cfg["max_train_rows"] and len(tr_idx) > cfg["max_train_rows"]: tr_idx = np.sort(rng.choice(tr_idx, cfg["max_train_rows"], replace=False))
    a, e = inner_split(sess[tr_idx]); fit_idx, es_idx = tr_idx[a], tr_idx[e]
    net = build_net(kind, Z.shape[1], len(HORIZONS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); bs = cfg["batch_size"]
    def batch(idx):
        x = torch.from_numpy(Z[idx[:, None] + offs[None, :]]).to(dev); y = torch.from_numpy(np.nan_to_num(Ymat[idx], nan=0.0)).to(dev)
        m = torch.from_numpy(~np.isnan(Ymat[idx])).to(dev); return x, y, m
    def loss_fn(logit, y, m):
        l = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none"); return (l * m).sum() / m.sum().clamp(min=1)
    best, best_state, bad = 1e9, None, 0
    for ep in range(cfg["epochs"]):
        net.train(); perm = rng.permutation(fit_idx)
        for i in range(0, len(perm), bs):
            x, y, m = batch(perm[i:i + bs]); opt.zero_grad(); loss = loss_fn(net(x), y, m); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval(); tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(es_idx), 2048):
                x, y, m = batch(es_idx[i:i + 2048]); l = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y, reduction="none")
                tot += float((l * m).sum()); cnt += float(m.sum())
        v = tot / max(cnt, 1); log(f"    [{kind}] epoch {ep+1} val_logloss={v:.4f}")
        if v < best - 1e-4: best, best_state, bad = v, copy.deepcopy({k: t.cpu() for k, t in net.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    net.load_state_dict(best_state); net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(pred_idx), 2048):
            x = torch.from_numpy(Z[pred_idx[i:i + 2048][:, None] + offs[None, :]]).to(dev); out.append(torch.sigmoid(net(x)).cpu().numpy())
    P = np.concatenate(out) if out else np.zeros((0, len(HORIZONS)))
    if save_path is not None: torch.save({"state": best_state, "mu": mu, "sd": sd, "kind": kind, "features": MODEL_FEATURES, "seq_len": SEQ_LEN}, str(save_path) + ".pt")
    return {h: P[:, j].astype("float32") for j, h in enumerate(HORIZONS)}

def fit_predict_mlp_fallback(kind, cfg, Xall, Ymat, tr_idx, pred_idx, rng):
    out = {}
    for j, h in enumerate(HORIZONS):
        idx = tr_idx[~np.isnan(Ymat[tr_idx, j])]
        if cfg["max_train_rows"] and len(idx) > cfg["max_train_rows"]: idx = rng.choice(idx, cfg["max_train_rows"], replace=False)
        mu, sd = robust_scale(Xall[idx]); m = MLPClassifier((64, 32) if kind == "transformer" else (128, 64), alpha=1e-3, early_stopping=True, max_iter=40, random_state=SEED + (kind == "tft"))
        m.fit(apply_scale(Xall[idx], mu, sd), Ymat[idx, j].astype(int)); out[h] = m.predict_proba(apply_scale(Xall[pred_idx], mu, sd))[:, 1].astype("float32")
    return out

def mlp_inputs(kind, X, idx, mu, sd):
    cols = {c: i for i, c in enumerate(MODEL_FEATURES)}
    parts = [apply_scale(X[idx], mu, sd)]
    mc = [cols[c] for c in MISS_COLS if c in cols]
    if mc: parts.append(np.isnan(X[np.ix_(idx, mc)]).astype("float32"))
    if kind == "seq_mlp":                               # previous SEQ_WIN-1 rows of the same symbol (eligible rows have >= SEQ_LEN-1)
        sc = [cols[c] for c in SEQ_COLS if c in cols]; offs = np.arange(-SEQ_WIN + 1, 0)
        W = X[(idx[:, None] + offs[None, :])][:, :, sc]
        parts.append(apply_scale(W, mu[sc], sd[sc]).reshape(len(idx), -1))
    return np.hstack(parts).astype("float32")

def fit_predict_mlp(kind, cfg, X, Y, tr_idx, pred_idx, rng):
    """One network, four outputs (multi-label): trained on rows where all four labels exist."""
    idx = tr_idx[~np.isnan(Y[tr_idx]).any(1)]
    cap = cap_for(kind, cfg)
    if cap and len(idx) > cap: idx = np.sort(rng.choice(idx, cap, replace=False))
    mu, sd = robust_scale(X[idx])
    m = MLPClassifier((128, 64), alpha=1e-3, batch_size=512, learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=4, max_iter=40, random_state=SEED + (kind == "seq_mlp"))
    m.fit(mlp_inputs(kind, X, idx, mu, sd), Y[idx].astype(int))
    P = np.vstack([m.predict_proba(mlp_inputs(kind, X, pred_idx[i:i + 20000], mu, sd)) for i in range(0, len(pred_idx), 20000)]) if len(pred_idx) else np.zeros((0, len(HORIZONS)))
    return {h: P[:, j].astype("float32") for j, h in enumerate(HORIZONS)}, {"model": m, "mu": mu, "sd": sd, "kind": kind, "features": MODEL_FEATURES,
                                                                              "seq_cols": SEQ_COLS, "seq_win": SEQ_WIN, "miss_cols": MISS_COLS}

# ------------------------------------------------------------------ ranker
def quintile_relevance(sess, fwd, min_names=10):
    df = pd.DataFrame({"s": sess, "f": fwd}); r = df.groupby("s").f.rank(pct=True); cnt = df.groupby("s").f.transform("size")
    lab = np.minimum((r.values * 5).astype(int), 4); ok = (cnt.values >= min_names) & ~np.isnan(fwd)
    return lab, ok

def fit_predict_ranker(cfg, n_threads, X, fwd_h, sess, tr_idx, pred_idx, rng):
    be = "native" if cfg.get("family") == "sklearn" else backend_of("ranker", cfg["allow_fallback"])
    sk = cfg.get("family") == "sklearn" or be != "native"
    if cfg["max_train_rows"] and len(tr_idx) > cfg["max_train_rows"]:
        # a ranker learns within-session orderings: subsample whole sessions, never rows inside a session
        s_tr = sess[tr_idx]; us = np.unique(s_tr); rng.shuffle(us)
        cum = np.cumsum(pd.Series(s_tr).value_counts().reindex(us).values)
        tr_idx = tr_idx[np.isin(s_tr, us[:max(1, int(np.searchsorted(cum, cfg["max_train_rows"])) + 1)])]
    lab, ok = quintile_relevance(sess[tr_idx], fwd_h[tr_idx]); tr = tr_idx[ok]; lab = lab[ok]
    order = np.argsort(sess[tr], kind="stable"); tr, lab = tr[order], lab[order]
    _, grp = np.unique(sess[tr], return_counts=True)
    if not sk:
        import lightgbm as lgb
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=min(cfg["n_estimators"], 600), learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=n_threads, random_state=SEED, verbose=-1)
        m.fit(X[tr], lab, group=grp)
    else:
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=5.0, max_features=0.8,
                                          early_stopping=True, n_iter_no_change=30, random_state=SEED); m.fit(X[tr], lab / 4.0)
    return m.predict(X[pred_idx]).astype("float32"), m, be

# ------------------------------------------------------------------ orchestration
def run_models(panel, X, Y, sess, train_mask, pred_mask, models, cfg, n_threads, save_dir=None, tag=""):
    """Fit `models` on train_mask rows, predict pred_mask rows. Returns {model: {h: array aligned to np.where(pred_mask)}}, backends, ranker scores."""
    rng = np.random.default_rng(SEED); tr_idx = np.where(train_mask)[0]; pr_idx = np.where(pred_mask)[0]
    res, backends, t_fit = {}, {}, {}
    for kind in models:
        t0 = time.time(); sp = None if save_dir is None else Path(save_dir) / f"{kind}{tag}"
        if kind in ("transformer", "tft"):
            be = backend_of(kind, cfg["allow_fallback"]); backends[kind] = be
            if be == "native": res[kind] = fit_predict_seq(kind, cfg, X, Y, sess, tr_idx, pr_idx, save_path=None if sp is None else Path(str(sp)))
            else: res[kind] = fit_predict_mlp_fallback(kind, cfg, X, Y, tr_idx, pr_idx, rng)
        elif kind in ("mlp", "seq_mlp"):
            res[kind], obj = fit_predict_mlp(kind, cfg, X, Y, tr_idx, pr_idx, rng); backends[kind] = "native"
            if sp is not None: joblib.dump(obj, f"{sp}.joblib")
        elif kind == "ranker":
            res[kind] = {}
            for j, h in enumerate(HORIZONS):
                s, m, be = fit_predict_ranker(cfg, n_threads, X, panel[f"fwd_ret_{h}"].values, sess, tr_idx, pr_idx, rng); res[kind][h] = s; backends[kind] = be
                if sp is not None:
                    if hasattr(m, "booster_"): m.booster_.save_model(f"{sp}_h{h}.txt")
                    else: joblib.dump(m, f"{sp}_h{h}.joblib")
        else:
            res[kind] = {}
            for j, h in enumerate(HORIZONS):
                idx = tr_idx[~np.isnan(Y[tr_idx, j])]; cap = cap_for(kind, cfg)
                if cap and len(idx) > cap: idx = np.sort(rng.choice(idx, cap, replace=False))
                model, be = fit_tabular(kind, cfg, n_threads, X[idx], Y[idx, j].astype(int), sess[idx]); backends[kind] = be
                res[kind][h] = model.predict_proba(X[pr_idx])[:, 1].astype("float32")
                if sp is not None: save_tabular(model, kind, be, Path(f"{sp}_h{h}"))
        t_fit[kind] = time.time() - t0; log(f"  {tag or 'fit'} {kind:11s} [{backends[kind]}] {t_fit[kind]:.0f}s")
    return res, backends, t_fit

def prepare(panel):
    X = panel[MODEL_FEATURES].values.astype("float32"); Y = np.column_stack([panel[f"y_{h}"].values for h in HORIZONS]).astype("float32")
    return X, Y, panel.sess.values

def run_oof(panel, splits, cfg, models, folds, resume=True):
    X, Y, sess = prepare(panel); n = len(panel); n_threads = cfg["threads"]
    oof = {m: {h: np.full(n, np.nan, "float32") for h in HORIZONS} for m in models}; fold_id = np.zeros(n, "int8")
    rep = {"folds": {}, "backends": {}, "family": cfg.get("family", "native"), "feature_set": FEATURE_SET, "n_features": len(MODEL_FEATURES),
           "algorithms": {m: ALGORITHMS.get(m if m != "ranker" else ("ranker_sklearn" if cfg.get("family") == "sklearn" else "ranker_native"), m) for m in models},
           "row_caps": {m: cap_for(m, cfg) for m in models}}
    done = []
    if resume and (WORK / "oof.pkl").exists():
        O = joblib.load(WORK / "oof.pkl")
        if sorted(O["oof"]) == sorted(models) and O["cfg"].get("family") == cfg.get("family") and len(O["fold_id"]) == n:
            oof, fold_id, done = O["oof"], O["fold_id"], list(O["done_folds"])
            if (REPORTS / "stage06_08_models_report.json").exists(): rep = load_json(REPORTS / "stage06_08_models_report.json")
            log(f"resuming: folds already done {done}")
    for k in folds:
        if k in done: continue
        tr, va = A.fold_masks(panel, splits, k); log(f"=== fold {k}: train rows={tr.sum():,} val rows={va.sum():,}")
        res, be, tf = run_models(panel, X, Y, sess, tr, va, models, cfg, n_threads, tag=f"_f{k}"); vi = np.where(va)[0]; fold_id[vi] = k
        rep["backends"].update(be); rep["folds"][str(k)] = {"fit_seconds": tf, "metrics": {}}
        for m in models:
            for h in HORIZONS:
                oof[m][h][vi] = res[m][h] if m != "ranker" else res[m][h]
                if m == "ranker": continue
                j = HORIZONS.index(h); y = Y[vi, j]; ok = ~np.isnan(y)
                if ok.sum() > 100 and len(np.unique(y[ok])) > 1:
                    mm = classification_metrics(y[ok], res[m][h][ok]); rep["folds"][str(k)]["metrics"][f"{m}_{h}d"] = {x: mm[x] for x in ("accuracy", "auc", "logloss", "brier", "ece", "majority_baseline_acc")}
        save_json(rep, REPORTS / "stage06_08_models_report.json")
        done = sorted(set(done) | {k})
        joblib.dump({"oof": oof, "fold_id": fold_id, "backends": rep["backends"], "cfg": cfg, "done_folds": done}, WORK / "oof.pkl.tmp")
        os.replace(WORK / "oof.pkl.tmp", WORK / "oof.pkl")
    return oof, fold_id, rep

def default_cfg(args):
    return {"family": args.family, "caps": DEFAULT_CAPS if args.family == "sklearn" else {}, "allow_fallback": args.allow_fallback, "max_train_rows": args.max_train_rows, "n_estimators": args.n_estimators, "epochs": args.epochs,
            "patience": 3, "batch_size": args.batch_size, "device": args.device, "threads": args.threads}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), default="native"); ap.add_argument("--models", default="")
    ap.add_argument("--folds", default="1-8"); ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--allow-fallback", action="store_true"); ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=2000); ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto"); ap.add_argument("--threads", type=int, default=-1)
    a = ap.parse_args(); cfg = default_cfg(a)
    lo, hi = (int(x) for x in a.folds.split("-")) if "-" in a.folds else (int(a.folds), int(a.folds)); folds = list(range(lo, hi + 1))
    panel = pd.read_pickle(WORK / "panel.pkl"); splits = load_json(WORK / "splits.json")
    assert load_json(REPORTS / "stage05_audit_report.json")["all_checks_passed"], "run stage 05 audit first"
    models = a.models.split(",") if a.models else FAMILIES[a.family]
    run_oof(panel, splits, cfg, models, folds, resume=not a.no_resume); log("OOF done -> oof.pkl")

if __name__ == "__main__":
    main()
