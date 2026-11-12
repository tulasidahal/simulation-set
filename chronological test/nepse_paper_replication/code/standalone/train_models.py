"""
Train XGBoost and a Transformer on NEPSE daily floorsheet data.
Task (assumed): predict whether a stock's NEXT-day close is higher than today's close.

Usage:
  pip install pandas numpy scikit-learn xgboost torch
  python train_models.py --prep-only     # features + split only (no xgboost/torch needed)
  python train_models.py --model xgb
  python train_models.py --model transformer
  python train_models.py --model both
"""
import argparse, numpy as np, pandas as pd

CSV = "floorsheet_daily.csv"
TRAIN_END, VAL_END = "2025-06-30", "2025-12-31"   # test = after VAL_END
SEQ = 30
FEATS = ["ret1","ret5","ret10","ret20","vol10","vol20","hl_range","close_vwap",
         "volume_ratio","log_amount","log_trades","oc_ret"]

def build(csv=CSV):
    d = pd.read_csv(csv, parse_dates=["trade_date"]).sort_values(["symbol","trade_date"])
    d = d[(d.close > 0) & (d.low > 0) & (d.volume > 0)]
    d = d.groupby("symbol").filter(lambda g: len(g) >= 250)
    g = d.groupby("symbol", sort=False)
    d["ret1"]  = g.close.pct_change()
    # NEPSE has a +/-10% daily circuit limit: larger moves are bonus/right/split adjustments, not trading
    bad = d.ret1.abs() > 0.11
    d["ret1"] = d.ret1.where(~bad)
    for n in (5, 10, 20):
        d[f"ret{n}"] = g.close.pct_change(n)
    d["vol10"] = d.groupby("symbol").ret1.transform(lambda s: s.rolling(10).std())
    d["vol20"] = d.groupby("symbol").ret1.transform(lambda s: s.rolling(20).std())
    d["hl_range"] = (d.high - d.low) / d.close
    d["close_vwap"] = d.close / d.vwap - 1
    d["volume_ratio"] = d.volume / d.groupby("symbol").volume.transform(lambda s: s.rolling(20).mean())
    d["log_amount"] = np.log1p(d.amount)
    d["log_trades"] = np.log1p(d.trades)
    d["oc_ret"] = d.close / d.open - 1
    nxt = g.close.shift(-1)
    d["target_ret"] = nxt / d.close - 1
    d["y"] = (d.target_ret > 0).astype(int)
    d["valid_target"] = d.target_ret.notna() & (d.target_ret.abs() <= 0.11)
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.reset_index(drop=True)

def split_masks(d):
    t = d.trade_date
    return t <= TRAIN_END, (t > TRAIN_END) & (t <= VAL_END), t > VAL_END

def tabular(d):
    ok = d.valid_target & d[FEATS].notna().all(axis=1)
    tr, va, te = split_masks(d)
    return [(d.loc[ok & m, FEATS].values.astype("float32"), d.loc[ok & m, "y"].values) for m in (tr, va, te)]

def run_xgb(d):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score
    (Xtr,ytr),(Xva,yva),(Xte,yte) = tabular(d)
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="auc", early_stopping_rounds=100,
                          tree_method="hist", n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=100)
    p = m.predict_proba(Xte)[:,1]
    print(f"[XGB] test acc={accuracy_score(yte,p>0.5):.4f} auc={roc_auc_score(yte,p):.4f} "
          f"(base rate up={yte.mean():.3f})")
    m.save_model("xgb_model.json")
    imp = sorted(zip(FEATS, m.feature_importances_), key=lambda x: -x[1])
    print("[XGB] importances:", [(f, round(float(v),3)) for f,v in imp])

def make_sequences(d, mask_dates):
    X, y, sym = [], [], []
    codes = {s:i for i,s in enumerate(sorted(d.symbol.unique()))}
    for s, g in d.groupby("symbol", sort=False):
        v = g[FEATS].values.astype("float32"); ok = g.valid_target.values
        yy = g.y.values; dt = g.trade_date.values
        for i in range(SEQ-1, len(g)):
            w = v[i-SEQ+1:i+1]
            if ok[i] and dt[i] in mask_dates and not np.isnan(w).any():
                X.append(w); y.append(yy[i]); sym.append(codes[s])
    return np.array(X), np.array(y), np.array(sym), len(codes)

def run_transformer(d, epochs=15, bs=512):
    import torch, torch.nn as nn
    from sklearn.metrics import accuracy_score, roc_auc_score
    tr, va, te = split_masks(d)
    sets = [set(d.loc[m, "trade_date"].values) for m in (tr, va, te)]
    (Xtr,ytr,Str,ns),(Xva,yva,Sva,_),(Xte,yte,Ste,_) = [make_sequences(d, s) for s in sets]
    mu, sd = Xtr.mean((0,1)), Xtr.std((0,1)) + 1e-6
    norm = lambda X: np.clip((X-mu)/sd, -5, 5).astype("float32")
    Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(s, nf, ns, dm=64):
            super().__init__()
            s.inp = nn.Linear(nf, dm); s.pos = nn.Parameter(torch.randn(1, SEQ, dm)*0.02)
            s.emb = nn.Embedding(ns, dm)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, 4, 128, 0.2, batch_first=True), 3)
            s.head = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 1))
        def forward(s, x, sym):
            h = s.inp(x) + s.pos + s.emb(sym)[:,None,:]
            return s.head(s.enc(h)[:,-1]).squeeze(-1)

    net = Net(len(FEATS), ns).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    def predict(X, S):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                out.append(torch.sigmoid(net(T(X[i:i+2048]), T(S[i:i+2048], torch.long))).cpu().numpy())
        return np.concatenate(out)
    best, best_state = 0, None
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(len(Xtr)); tot = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(net(T(Xtr[b]), T(Str[b], torch.long)), T(ytr[b]))
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += loss.item()*len(b)
        auc = roc_auc_score(yva, predict(Xva, Sva))
        print(f"[TF] epoch {ep+1} loss={tot/len(perm):.4f} val_auc={auc:.4f}")
        if auc > best: best, best_state = auc, {k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state)
    p = predict(Xte, Ste)
    print(f"[TF] test acc={accuracy_score(yte,p>0.5):.4f} auc={roc_auc_score(yte,p):.4f} (base rate up={yte.mean():.3f})")
    torch.save(best_state, "transformer_model.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb","transformer","both"], default="both")
    ap.add_argument("--prep-only", action="store_true")
    a = ap.parse_args()
    d = build()
    tr, va, te = split_masks(d)
    print(f"rows={len(d):,} symbols={d.symbol.nunique()} train={tr.sum():,} val={va.sum():,} test={te.sum():,}")
    (Xtr,ytr),(Xva,yva),(Xte,yte) = tabular(d)
    print(f"usable samples train={len(ytr):,} val={len(yva):,} test={len(yte):,} | up-rate train={ytr.mean():.3f} test={yte.mean():.3f}")
    if not a.prep_only:
        if a.model in ("xgb","both"): run_xgb(d)
        if a.model in ("transformer","both"): run_transformer(d)
