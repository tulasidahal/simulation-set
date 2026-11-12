"""
Stage 03  Build the modeling panel: daily bars + 53 core predictors + 7 point-in-time fundamental predictors.
Stage 04  Construct four direction targets (1D, 3D, 5D, 10D) and tradable returns.

Sources (all cleaned and de-duplicated in stages 01-02 and 01b):
  * prices/volume  adjusted OHLCV table for listed symbols (complete coverage, official corporate-action adjustment);
                   floorsheet bars for symbols that were delisted/merged (the OHLCV table holds survivors only) and for
                   dates before a symbol's first OHLCV bar, rescaled onto the adjusted basis at the first overlapping date.
  * microstructure floorsheet aggregates (broker concentration, trade counts, turnover, VWAP); missing where the floorsheet
                   itself is missing (Feb 2022 - Mar 2023 and Mar 2024 are mostly absent), never imputed from the future.
  * market         official NEPSE index (equal-weighted universe return on the few sessions the index is missing).
  * fundamentals   quarterly reports, usable from quarter end + publication lag (stage 01b); valuation ratios are rebuilt
                   from the quarter-end price and the adjusted return since, so no price after date t is used.
Universe: ordinary shares only (promoter shares, debentures and closed-end mutual funds excluded); no filter that uses
a symbol's future history length. Time unit = market session; horizons are counted in sessions.
"""
import re
import numpy as np, pandas as pd
from common import *

# ----------------------------------------------------------------------------- sources and universe
def load_sources(cutoff=None):
    daily = pd.read_csv(WORK / "daily_floorsheet.csv.gz", parse_dates=["trade_date"])
    ohl = pd.read_pickle(WORK / "aux" / "ohlcv_adjusted.pkl"); fund = pd.read_pickle(WORK / "aux" / "fundamentals_pit.pkl")
    daily["symbol"] = daily.symbol.str.upper(); ohl["symbol"] = ohl.symbol.str.upper(); fund["symbol"] = fund.symbol.str.upper()
    daily["trade_date"] = daily.trade_date.astype("datetime64[ns]"); ohl["date"] = ohl.date.astype("datetime64[ns]")
    for c in ("quarter_end", "available_date", "date"): fund[c] = pd.to_datetime(fund[c]).astype("datetime64[ns]")
    daily = daily.sort_values("n").drop_duplicates(["symbol", "trade_date"], keep="last")
    ohl = ohl.drop_duplicates(["symbol", "date"], keep="last")
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        daily, ohl, fund = daily[daily.trade_date <= cutoff], ohl[ohl.date <= cutoff], fund[fund.available_date <= cutoff]
    return daily, ohl, fund

def instrument_meta(daily, ohl):
    """Static instrument classification (type of security and which price source is used)."""
    med = daily.groupby("symbol").close.median()
    syms = sorted(set(daily.symbol) | (set(ohl.symbol) - INDEX_SYMBOLS)); allset = set(syms); o_syms = set(ohl.symbol)
    rows = []
    for s in syms:
        if re.search(r"[\d/]", s): k = "debenture_or_fund_series"
        elif s.endswith("PO") or (s.endswith("P") and s[:-1] in allset): k = "promoter_share"
        elif med.get(s, np.inf) < FUND_PRICE_MAX: k = "mutual_fund"
        else: k = "equity"
        rows.append({"symbol": s, "kind": k, "price_src": "ohlcv_adjusted" if s in o_syms else "floorsheet"})
    return pd.DataFrame(rows).set_index("symbol")

def build_bars(daily, ohl, eq_ohlcv, eq_fs):
    """Daily bars on an adjusted basis. Returns long DataFrame (symbol, trade_date, open, high, low, close, volume, src) and stats."""
    st = {}
    o = ohl[ohl.symbol.isin(eq_ohlcv) & (ohl.volume > 0) & (ohl.close > 0)].rename(columns={"date": "trade_date"})
    px = o[["open", "high", "low", "close"]]
    st["ohlcv_ohlc_repaired"] = int(((o.high < px.max(axis=1)) | (o.low > px.min(axis=1))).sum())
    o = o.assign(high=px.max(axis=1), low=px.min(axis=1))[["symbol", "trade_date", "open", "high", "low", "close", "volume"]].assign(src=1)
    f = daily.rename(columns={"vol": "volume"})[["symbol", "trade_date", "open", "high", "low", "close", "volume"]]
    # floorsheet bars of OHLCV symbols on dates the OHLCV table lacks -> rescale with the ratio at the nearest overlap
    fo = f[f.symbol.isin(eq_ohlcv)].merge(o[["symbol", "trade_date"]].assign(_in=1), on=["symbol", "trade_date"], how="left")
    fill = fo[fo._in.isna()].drop(columns="_in").sort_values("trade_date")
    ov = f.merge(o[["symbol", "trade_date", "close"]], on=["symbol", "trade_date"], suffixes=("", "_o"))
    ov = ov.assign(k=ov.close_o / ov.close)[["symbol", "trade_date", "k"]].sort_values("trade_date")
    kb = pd.merge_asof(fill, ov, on="trade_date", by="symbol", direction="backward")
    kf = pd.merge_asof(fill, ov, on="trade_date", by="symbol", direction="forward")
    # the scale only sets the price level (returns and every feature are scale-free); 1.0 when no overlap exists yet
    fill = fill.assign(k=kb.k.fillna(kf.k).fillna(1.0).values)
    for c in ("open", "high", "low", "close"): fill[c] = fill[c] * fill.k
    st["floorsheet_bars_chained_into_ohlcv_symbols"] = int(len(fill))
    b = pd.concat([o, fill.drop(columns="k").assign(src=2), f[f.symbol.isin(eq_fs)].assign(src=3)], ignore_index=True)
    b = b[b.trade_date >= WARMUP_START]
    st["bars_by_source"] = {"ohlcv_adjusted": int((b.src == 1).sum()), "floorsheet_chained": int((b.src == 2).sum()), "floorsheet_delisted": int((b.src == 3).sum())}
    return b, st

# ----------------------------------------------------------------------------- fundamentals (point in time)
def fundamental_features(fund, sessions, cols, adj, fs_close):
    T, S = len(sessions), len(cols); col_ix = {c: i for i, c in enumerate(cols)}
    f = fund[fund.symbol.isin(col_ix)].dropna(subset=["available_date", "quarter_end"]).copy()
    f = f.sort_values(["symbol", "quarter_end"])
    # EPS growth vs the same quarter of the previous fiscal year (that report is older, hence already public)
    f["fy0"] = f.fiscal_year.str[:4].astype(float)
    prev = f[["symbol", "fy0", "quarter", "eps"]].assign(fy0=lambda x: x.fy0 + 1).rename(columns={"eps": "eps_ly"})
    f = f.merge(prev, on=["symbol", "fy0", "quarter"], how="left")
    f["eps_yoy"] = ((f.eps - f.eps_ly) / f.eps_ly.abs().replace(0, np.nan)).clip(-5, 5)
    # prices at quarter end (the P/E and P/B in the table were computed with this price)
    qe_i = np.searchsorted(sessions, f.quarter_end.values.astype("datetime64[ns]"), side="right") - 1
    av_i = np.searchsorted(sessions, f.available_date.values.astype("datetime64[ns]"), side="left")
    ok = (qe_i >= 0) & (av_i < T); f, qe_i, av_i = f[ok], qe_i[ok], av_i[ok]
    ci = f.symbol.map(col_ix).values
    raw_ff = fs_close.ffill(limit=10).values; adj_ff = adj.ffill().values
    price_qe = raw_ff[qe_i, ci]
    price_qe = np.where(np.isnan(price_qe), (f.pe_ratio * f.eps).where(f.pe_ratio > 0).values, price_qe)
    f["ep_qe"] = f.eps.values / price_qe
    f["bp_qe"] = 1.0 / f.pb_ratio.where(f.pb_ratio > 0).values
    f["adj_qe"] = adj_ff[qe_i, ci]; f["av_i"] = av_i; f["ci"] = ci
    f = f.sort_values(["av_i", "quarter_end"]).drop_duplicates(["ci", "av_i"], keep="last")
    W = {}
    for k in ("ep_qe", "bp_qe", "adj_qe", "roe", "debt_to_equity", "revenue_growth", "eps_yoy", "av_i"):
        m = np.full((T, S), np.nan); m[f.av_i.values, f.ci.values] = f[k].values.astype(float)
        W[k] = pd.DataFrame(m, index=adj.index, columns=adj.columns).ffill(limit=260)     # a report goes stale after ~1 year
    tidx = pd.DataFrame(np.tile(np.arange(T, dtype=float)[:, None], (1, S)), index=adj.index, columns=adj.columns)
    ret_since = adj / W["adj_qe"]                                                          # adjusted return since quarter end
    F = {"fund_ep": (W["ep_qe"] / ret_since).clip(-1, 1), "fund_bp": (W["bp_qe"] / ret_since).clip(0, 10),
         "fund_roe": W["roe"].clip(-100, 100), "fund_de": W["debt_to_equity"].clip(0, 50), "fund_rev_growth": W["revenue_growth"].clip(-100, 500),
         "fund_eps_yoy": W["eps_yoy"], "fund_age": np.log1p(tidx - W["av_i"])}
    return F, {"reports_used": int(len(f)), "symbols_with_reports": int(f.ci.nunique())}

# ----------------------------------------------------------------------------- panel
def build_panel(daily, ohl, fund, meta):
    eq = meta[meta.kind == "equity"]
    eq_ohlcv = set(eq[eq.price_src == "ohlcv_adjusted"].index); eq_fs = set(eq[eq.price_src == "floorsheet"].index)
    bars, st = build_bars(daily, ohl, eq_ohlcv, eq_fs)
    cols = sorted(bars.symbol.unique()); sessions = np.sort(bars.trade_date.unique()); T, S = len(sessions), len(cols)
    Wb = lambda c: bars.pivot(index="trade_date", columns="symbol", values=c).reindex(index=sessions, columns=cols).astype("float64")
    fsd = daily[daily.symbol.isin(cols)]
    Wf = lambda c: fsd.pivot(index="trade_date", columns="symbol", values=c).reindex(index=sessions, columns=cols).astype("float64")
    open_, high, low, close, vol = (Wb(c) for c in ("open", "high", "low", "close", "volume"))
    fs_close, fs_vwap, fs_amt, fs_n, fs_vol = (Wf(c) for c in ("close", "vwap", "amt", "n", "vol"))
    traded = close.notna(); fs_avail = fs_n.notna()
    close_ff = close.ffill(); ret_raw = close_ff / close_ff.shift(1) - 1
    event = traded & (ret_raw.abs() > CORP_ACTION_JUMP)            # residual jumps (unadjusted corporate actions or bad prints)
    oc = close / open_ - 1
    clean = ret_raw.where(~event, oc.clip(-CORP_ACTION_JUMP, CORP_ACTION_JUMP))
    listed = close_ff.notna(); clean = clean.fillna(0.0).where(listed)
    adj = np.exp(np.log1p(clean.fillna(0.0)).cumsum()).where(listed)
    adj_open, adj_high, adj_low = adj * (open_ / close), adj * (high / close), adj * (low / close)
    r_tr = clean.where(traded)
    F = {}
    for k in (1, 2, 3, 5, 10, 20, 60): F[f"ret{k}"] = adj / adj.shift(k) - 1
    for k, mp in ((5, 3), (10, 6), (20, 12), (60, 30)): F[f"vol{k}"] = r_tr.rolling(k, min_periods=mp).std()
    F["hl_range"] = (high - low) / close
    F["oc_ret"] = oc.clip(-0.2, 0.2)
    F["close_vwap"] = fs_close / fs_vwap - 1                        # same-day floorsheet ratio
    F["gap"] = adj_open / adj.shift(1) - 1
    F["close_pos"] = ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5).where(traded)
    for k in (5, 10, 20, 50): F[f"ma{k}_dist"] = adj / adj.rolling(k, min_periods=int(0.8 * k)).mean() - 1
    d = adj.diff(); gain = d.clip(lower=0).rolling(14, min_periods=10).mean(); loss = (-d.clip(upper=0)).rolling(14, min_periods=10).mean(); s = gain + loss
    F["rsi14"] = (100 * gain / s.replace(0, np.nan)).fillna(50.0).where(s.notna())
    macd = adj.ewm(span=12, adjust=False).mean() - adj.ewm(span=26, adjust=False).mean(); F["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / adj
    m20, s20 = adj.rolling(20, min_periods=15).mean(), adj.rolling(20, min_periods=15).std()
    F["bb_pos"] = ((adj - m20) / (2 * s20.replace(0, np.nan))).fillna(0.0).where(s20.notna())
    prev = adj.shift(1)
    tr = pd.DataFrame(np.fmax(np.fmax((adj_high - adj_low).values, (adj_high - prev).abs().values), (adj_low - prev).abs().values), index=adj.index, columns=adj.columns)
    F["atr14_pct"] = tr.rolling(14, min_periods=8).mean() / adj
    ll, hh = adj_low.rolling(14, min_periods=8).min(), adj_high.rolling(14, min_periods=8).max()
    F["stoch_k14"] = ((adj - ll) / (hh - ll).replace(0, np.nan)).fillna(0.5).where(ll.notna())
    F["drawdown60"] = adj / adj.rolling(60, min_periods=30).max() - 1
    F["dist_high252"] = adj / adj.rolling(252, min_periods=120).max() - 1
    F["dist_low252"] = adj / adj.rolling(252, min_periods=120).min() - 1
    m20v = vol.rolling(20, min_periods=5).mean()
    F["volratio_1_20"] = vol / m20v; F["volratio_5_20"] = vol.rolling(5, min_periods=2).mean() / m20v
    F["log_amount"] = np.log1p(fs_amt); F["log_trades"] = np.log1p(fs_n)
    F["amihud20"] = np.log1p((r_tr.abs() / fs_amt.replace(0, np.nan)).rolling(20, min_periods=5).mean() * 1e9)
    F["stale20"] = 1 - (traded & (clean != 0)).astype("float64").rolling(20, min_periods=10).mean()
    F["turnover_trend"] = fs_amt.rolling(5, min_periods=2).mean() / fs_amt.rolling(20, min_periods=5).mean()
    F["avg_trade_size"] = np.log(fs_vol / fs_n); F["trades_ratio20"] = fs_n / fs_n.rolling(20, min_periods=5).mean()
    for c in ("buyer_hhi", "seller_hhi", "buyer_top1", "seller_top1", "buyer_top5", "seller_top5"): F[c] = Wf(c)
    F["log_buyers"] = np.log1p(Wf("n_buyers")); F["log_sellers"] = np.log1p(Wf("n_sellers"))
    F["self_share"] = Wf("self_q") / fs_vol; F["large_share"] = Wf("large_q") / fs_vol
    mean_sz = fs_vol / fs_n; F["size_cv"] = np.sqrt((Wf("q2") / fs_n - mean_sz ** 2).clip(lower=0)) / mean_sz
    F["broker_imbalance"] = F["buyer_top5"] - F["seller_top5"]
    F["buyer_hhi_chg5"] = F["buyer_hhi"] - F["buyer_hhi"].rolling(5, min_periods=2).mean()
    # ---- market: official NEPSE index, equal-weighted fallback on sessions the index is missing
    nep = ohl[ohl.symbol == "NEPSE"].set_index("date").close.reindex(sessions)
    ew = r_tr.clip(-CORP_ACTION_JUMP, CORP_ACTION_JUMP).mean(axis=1)
    mret1 = (nep / nep.ffill().shift(1) - 1).where(nep.notna()).fillna(ew)
    midx = np.exp(np.log1p(mret1.fillna(0.0)).cumsum())
    market = pd.DataFrame({"trade_date": sessions, "mkt_ret1": mret1.values, "mkt_vol20": mret1.rolling(20, min_periods=10).std().values,
                           "mkt_ret5": (midx / midx.shift(5) - 1).values, "mkt_ret60": (midx / midx.shift(60) - 1).values,
                           "nepse_index_available": nep.notna().values})
    bc = lambda v: pd.DataFrame(np.tile(np.asarray(v, dtype="float64")[:, None], (1, S)), index=adj.index, columns=adj.columns)
    F["mkt_ret1"] = bc(market.mkt_ret1); F["mkt_vol20"] = bc(market.mkt_vol20); F["rel_ret5"] = F["ret5"] - bc(market.mkt_ret5)
    # ---- fundamentals
    FF, fst = fundamental_features(fund, sessions, cols, adj, fs_close); F.update(FF); st["fundamentals"] = fst
    # ---- stage 04: targets (future information; never used as features)
    TG = {}
    for h in HORIZONS:
        fwd = adj.shift(-h) / adj - 1
        TG[f"fwd_ret_{h}"] = fwd
        TG[f"y_{h}"] = (fwd > 0).astype("float64").where(fwd.notna() & (fwd != 0))   # exactly-flat outcomes are excluded
        TG[f"trade_ret_{h}"] = adj.shift(-h) / adj_open.shift(-1) - 1                 # enter next-session open, exit close of t+h
    traded20 = traded.rolling(20, min_periods=1).sum()
    in_study = pd.DataFrame(np.tile((sessions >= np.datetime64(STUDY_START))[:, None], (1, S)), index=adj.index, columns=adj.columns)
    eligible = traded & (traded20 >= 10) & F["ret60"].notna() & in_study
    st["thin_floorsheet"] = thin_stats = {"excluded": bool(EXCLUDE_GAP)}
    if EXCLUDE_GAP:
        n_el = eligible.sum(axis=1)
        cov = ((fs_avail & eligible).sum(axis=1) / n_el.where(n_el > 0)).values
        thin = np.nan_to_num(cov, nan=1.0) < GAP_MIN_COVERAGE
        drop = np.zeros(len(thin), bool); blocks = []
        i = 0
        while i < len(thin):
            if thin[i]:
                j = i
                while j + 1 < len(thin) and thin[j + 1]: j += 1
                if j - i + 1 >= GAP_MIN_BLOCK:
                    blocks.append((i, j)); drop[i:min(j + 1 + GAP_BUFFER, len(thin))] = True
                i = j + 1
            else: i += 1
        eligible = eligible & ~drop[:, None]
        thin_stats.update({"min_coverage": GAP_MIN_COVERAGE, "min_block": GAP_MIN_BLOCK, "buffer_sessions": GAP_BUFFER,
                           "thin_sessions": int(thin.sum()), "excluded_sessions": int(drop.sum()),
                           "blocks": [{"from": str(pd.Timestamp(sessions[a]).date()), "to": str(pd.Timestamp(sessions[b]).date()),
                                       "sessions": int(b - a + 1),
                                       "buffer_to": str(pd.Timestamp(sessions[min(b + GAP_BUFFER, len(thin) - 1)]).date())} for a, b in blocks]})
        log(f"thin-floorsheet exclusion: {thin_stats['excluded_sessions']} sessions dropped "
            f"({len(blocks)} blocks + {GAP_BUFFER}-session buffers)")
    # ---- long panel, symbol-major
    mask = traded.values.T; flat = lambda df: df.values.T[mask]
    P = {"sym_id": np.repeat(np.arange(S)[:, None], T, axis=1)[mask].astype("int32"), "sess": np.tile(np.arange(T), (S, 1))[mask].astype("int32")}
    P["symbol"] = np.array(cols, dtype=object)[P["sym_id"]]; P["date"] = sessions[P["sess"]]
    for k in ALL_FEATURES: P[k] = flat(F[k]).astype("float32")
    sym = P["sym_id"]; valid_seq = np.zeros(len(sym), bool)
    valid_seq[SEQ_LEN - 1:] = sym[SEQ_LEN - 1:] == sym[:len(sym) - SEQ_LEN + 1]
    P["amount"] = flat(fs_amt).astype("float32"); P["adj_close"] = flat(adj); P["eligible"] = flat(eligible) & valid_seq
    P["event_day"] = flat(event); P["fs_available"] = flat(fs_avail)
    P["price_src"] = flat(Wb("src")).astype("int8")
    for k, v in TG.items(): P[k] = flat(v).astype("float32")
    return pd.DataFrame(P), market, sessions, st

def summarize(panel, meta, st):
    el = panel[panel.eligible]
    out = {"universe": meta.kind.value_counts().to_dict(), "equity_price_source": meta[meta.kind == "equity"].price_src.value_counts().to_dict(),
           "bars": st, "rows_all_traded": int(len(panel)), "rows_eligible": int(len(el)), "symbols_eligible": int(el.symbol.nunique()),
           "sessions_eligible": int(el.sess.nunique()), "first_eligible_date": str(el.date.min().date()), "last_date": str(panel.date.max().date()),
           "n_core_features": len(FEATURES), "n_fundamental_features": len(FUND_FEATURES), "corporate_action_or_bad_print_days": int(el.event_day.sum()),
           "floorsheet_coverage_eligible": float(el.fs_available.mean()),
           "floorsheet_coverage_by_year": el.groupby(el.date.dt.year).fs_available.mean().round(3).to_dict(),
           "fundamental_coverage_eligible": float(el.fund_ep.notna().mean()),
           "feature_missing_rate_eligible": {k: round(float(el[k].isna().mean()), 4) for k in ALL_FEATURES if el[k].isna().mean() > 0}, "targets": {}}
    for h in HORIZONS:
        y = el[f"y_{h}"]; fr = el[f"fwd_ret_{h}"]
        out["targets"][f"{h}d"] = {"labeled": int(y.notna().sum()), "up_rate": float(y.mean()), "flat_excluded_share": float((fr == 0).mean()),
                                   "fwd_ret_mean": float(fr.mean()), "fwd_ret_std": float(fr.std()),
                                   "tradable_ret_mean": float(el[f"trade_ret_{h}"].mean()), "tradable_missing": float(el[f"trade_ret_{h}"].isna().mean())}
    return out

def run():
    daily, ohl, fund = load_sources(); meta = instrument_meta(daily, ohl); meta.to_csv(WORK / "instrument_meta.csv")
    log(f"universe: {meta.kind.value_counts().to_dict()}")
    panel, market, sessions, st = build_panel(daily, ohl, fund, meta)
    panel.to_pickle(WORK / "panel.pkl"); market.to_csv(WORK / "market.csv", index=False)
    summ = summarize(panel, meta, st); save_json(summ, REPORTS / "stage03_04_panel_report.json")
    log(f"panel rows={len(panel):,} eligible={summ['rows_eligible']:,} symbols={summ['symbols_eligible']} sessions={summ['sessions_eligible']} "
        f"floorsheet coverage={summ['floorsheet_coverage_eligible']:.3f} fundamentals coverage={summ['fundamental_coverage_eligible']:.3f}")
    for k, v in summ["targets"].items(): log(f"  target {k}: labeled={v['labeled']:,} up_rate={v['up_rate']:.3f}")

if __name__ == "__main__":
    run()
