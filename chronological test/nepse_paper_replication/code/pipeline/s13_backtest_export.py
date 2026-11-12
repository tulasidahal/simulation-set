"""
Stage 13  Backtest, export and research report.
Strategy (long-only; NEPSE retail investors cannot short): at the close of session t, buy equal-weight the (up to MAX_POS) liquid names
whose calibrated P(up over h sessions) >= threshold; enter at the NEXT session open (adjusted), exit at the close of t+h.
A tranche of 1/h of capital is deployed each signal session. Costs: broker commission + SEBON fee + slippage per side, 7.5% capital-gains
tax on positive trades (short-term, individual). Signals whose name does not trade at t+1 are counted as unfilled.
Benchmarks: all-eligible equal-weight under identical rules, and a random-selection placebo with the same pick counts.
"""
import hashlib, shutil, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
import s05_audit as A

COSTS = {"commission": 0.0036, "sebon": 0.00015, "slippage": 0.0010}   # per side (NPR retail broker slab ~0.36% at small tickets)
CGT, MIN_AMOUNT, MAX_POS, SESS_YEAR = 0.075, 1_000_000, 10, 240
FIGS = REPORTS / "figs"; FIGS.mkdir(exist_ok=True)

def net_return(g, cm=1.0):
    c = cm * sum(COSTS.values()); r = (1 + g) * (1 - c) / (1 + c) - 1
    return np.where(r > 0, r * (1 - CGT), r)

def pick(df, score_col, thr, max_pos, top_k=None):
    d = df[df.amount >= MIN_AMOUNT]
    d = d[d[score_col] >= thr] if thr is not None else d
    d = d.sort_values(["sess", score_col], ascending=[True, False]); return d.groupby("sess").head(top_k or max_pos)

def tranche_metrics(R_by_sess, h, all_sessions):
    s0, s1 = int(all_sessions.min()), int(all_sessions.max()); idx = np.arange(s0, s1 + h + 1)
    R = pd.Series(R_by_sess).reindex(idx - h).fillna(0.0).values; rho = R / h
    eq = np.cumprod(1 + rho); dd = (eq / np.maximum.accumulate(eq) - 1).min(); T = len(rho)
    return {"cum_return": float(eq[-1] - 1), "ann_return": float(eq[-1] ** (SESS_YEAR / T) - 1), "ann_vol": float(rho.std() * np.sqrt(SESS_YEAR)),
            "sharpe": float(rho.mean() / rho.std() * np.sqrt(SESS_YEAR)) if rho.std() > 0 else None, "max_drawdown": float(dd),
            "exposure_share_sessions": float((pd.Series(R_by_sess).reindex(all_sessions).fillna(0) != 0).mean())}, eq

def block_boot_mean(sess, x, B=1000, block=10, seed=SEED):
    rng = np.random.default_rng(seed); df = pd.DataFrame({"s": sess, "x": x}).groupby("s").x.agg(["sum", "count"]); T = len(df); nb = int(np.ceil(T / block)); sm, ct = df["sum"].values, df["count"].values; out = []
    for _ in range(B):
        st = rng.integers(0, T, nb); ix = ((st[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]; out.append(sm[ix].sum() / ct[ix].sum())
    return np.array(out)

def run_strategy(df, score_col, h, thr, cm=1.0, top_k=None, placebo_draws=300, B=1000, all_sessions=None):
    d = df.copy(); d["g"] = d[f"trade_ret_{h}"]
    sig = pick(d, score_col, thr, MAX_POS, top_k); n_sig = len(sig); filled = sig[sig.g.notna()].copy()
    out = {"signals": int(n_sig), "filled": int(len(filled)), "fill_rate": float(len(filled) / n_sig) if n_sig else None}
    if len(filled) < 30: out["note"] = "too few trades"; return out, None
    filled["net"] = net_return(filled.g.values, cm); R = filled.groupby("sess").net.mean()
    pm, eq = tranche_metrics(R, h, all_sessions if all_sessions is not None else d.sess.unique())
    bs = block_boot_mean(filled.sess.values, filled.net.values, B=B)
    out.update({"trades": int(len(filled)), "signal_sessions": int(len(R)), "mean_net_per_trade": float(filled.net.mean()), "median_net_per_trade": float(filled.net.median()),
                "hit_rate_net": float((filled.net > 0).mean()), "mean_gross_per_trade": float(filled.g.mean()), "ci_mean_net": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                "p_mean_net_le_0": float(np.mean(bs <= 0)), **pm})
    # benchmark: all eligible liquid names, same rules
    pool = d[(d.amount >= MIN_AMOUNT) & d.g.notna()]; pool = pool.assign(net=net_return(pool.g.values, cm)); out["benchmark_all_eligible_mean_net"] = float(pool.net.mean())
    # placebo: same number of picks per signal session, drawn at random from that session's eligible pool
    rng = np.random.default_rng(SEED); cnt = filled.groupby("sess").size(); grp = {s: v.net.values for s, v in pool.groupby("sess")}; sims = []
    for _ in range(placebo_draws):
        tot, num = 0.0, 0
        for s_, k in cnt.items():
            a = grp.get(s_); 
            if a is None or len(a) == 0: continue
            tot += rng.choice(a, size=min(k, len(a)), replace=False).sum(); num += min(k, len(a))
        sims.append(tot / num if num else np.nan)
    sims = np.array(sims); out["placebo_mean_net"] = float(np.nanmean(sims)); out["p_vs_placebo"] = float(np.mean(sims >= filled.net.mean()))
    return out, eq

def backtests():
    T = joblib.load(WORK / "test_predictions.pkl"); meta, preds, sel = T["meta"], T["preds"], T["selection"]
    E = joblib.load(WORK / "ens_oof.pkl"); panel = pd.read_pickle(WORK / "panel.pkl"); fold_id = E["fold_id"]; first = E["eval_first_fold"]
    res = {"params": {"costs_per_side": COSTS, "cgt_short_term": CGT, "min_amount_npr": MIN_AMOUNT, "max_positions_per_signal_session": MAX_POS, "sessions_per_year": SESS_YEAR}, "test": {}, "oof": {}, "sweep": {}}
    curves = {}
    for h in HORIZONS:
        prim = sel[f"{h}d"]["primary"]
        d = meta[["sess", "amount", f"trade_ret_{h}"]].copy(); d["p_primary"] = preds[h][prim]; d["rank_score"] = preds[h]["ranker"]; d["p_avg"] = preds[h]["avg"]
        allsess = d.sess.unique(); res["test"][f"{h}d"] = {"primary": prim}
        for thr in (0.55, 0.60, 0.65, 0.70):
            r, eq = run_strategy(d.assign(sc=np.maximum(d.p_primary, 0)), "sc", h, thr, all_sessions=allsess); res["test"][f"{h}d"][f"thr_{thr:.2f}"] = r
            if thr == 0.70 and eq is not None: curves[h] = eq
        for cm in (0.0, 2.0):
            r, _ = run_strategy(d.assign(sc=d.p_primary), "sc", h, 0.70, cm=cm, all_sessions=allsess, placebo_draws=50, B=300); res["sweep"][f"{h}d_cost_x{cm}"] = r
        rk, _ = run_strategy(d.assign(sc=d.rank_score), "sc", h, None, top_k=MAX_POS, all_sessions=allsess); res["test"][f"{h}d"]["ranker_top10"] = rk
        # robustness on the OOF region (not the final test)
        m = (fold_id >= first) & ~np.isnan(E["preds"][h][prim]); o = panel.loc[m, ["sess", "amount", f"trade_ret_{h}"]].copy(); o["sc"] = E["preds"][h][prim][m]
        r, _ = run_strategy(o, "sc", h, 0.70, all_sessions=o.sess.unique(), placebo_draws=100, B=500); res["oof"][f"{h}d"] = r
    save_json(res, REPORTS / "stage13_backtest.json")
    plt.figure(figsize=(7, 4))
    for h, eq in curves.items(): plt.plot(eq, label=f"{h}D (conf>=0.70)")
    plt.axhline(1, color="gray", lw=.5); plt.title("Final-test equity curves (net of costs, long-only)"); plt.xlabel("sessions"); plt.ylabel("equity"); plt.legend(); plt.tight_layout(); plt.savefig(FIGS / "equity_curves.png", dpi=130); plt.close()
    return res

def figures():
    st = load_json(REPORTS / "stage11_stats.json"); fin = load_json(REPORTS / "stage12_final_test.json"); sel = fin["selection"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for a, h in zip(ax, HORIZONS):
        for lab, src in (("OOF", st["horizons"][f"{h}d"]["candidates"][sel[f"{h}d"]["primary"]].get("reliability")), ("Test", fin["horizons"][f"{h}d"]["reliability_primary"])):
            if src: a.plot([r["mean_p"] for r in src], [r["freq"] for r in src], marker="o", label=lab)
        a.plot([0, 1], [0, 1], "k--", lw=.7); a.set_title(f"{h}D reliability ({sel[f'{h}d']['primary']})", fontsize=8); a.set_xlabel("predicted P(up)")
    ax[0].set_ylabel("observed freq"); ax[0].legend(); plt.tight_layout(); plt.savefig(FIGS / "reliability.png", dpi=130); plt.close()
    fig, ax = plt.subplots(figsize=(7, 3.8)); xs = np.arange(len(HORIZONS)); w = 0.25
    for i, (lab, getter) in enumerate((("OOF acc", lambda h: st["horizons"][f"{h}d"]["candidates"][sel[f"{h}d"]["primary"]]), ("Test acc", lambda h: fin["horizons"][f"{h}d"]["candidates"][sel[f"{h}d"]["primary"]]))):
        v = [getter(h)["accuracy"] for h in HORIZONS]; ax.bar(xs + i * w, v, w, label=lab)
    ax.bar(xs + 2 * w, [fin["horizons"][f"{h}d"]["candidates"][sel[f"{h}d"]["primary"]]["majority_baseline_acc"] for h in HORIZONS], w, label="Test majority baseline", color="lightgray")
    ax.set_xticks(xs + w); ax.set_xticklabels([f"{h}D" for h in HORIZONS]); ax.set_ylim(0.4, 0.8); ax.legend(); ax.set_title("Accuracy of selected primary model"); plt.tight_layout(); plt.savefig(FIGS / "accuracy.png", dpi=130); plt.close()

def f(x, d=4): return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"
def pct(x, d=1): return "n/a" if x is None else f"{100*x:.{d}f}%"

def data_sections():
    """Sections 1-3 (data cleaning, auxiliary sources, panel, leakage audit) - need only stages 01-05."""
    ing = load_json(REPORTS / "stage01_02_ingest_report.json"); pan = load_json(REPORTS / "stage03_04_panel_report.json"); aud = load_json(REPORTS / "stage05_audit_report.json")
    C = ing["cleaning"]; L = []
    D = C["dedup"]; aux = load_json(REPORTS / "stage01b_aux_report.json") if (REPORTS / "stage01b_aux_report.json").exists() else None
    L += ["## 1. Data validation and cleaning (stages 01–02, 01b)", "", "### 1.1 Floorsheet trades", "", "| File | Size (MB) | CRC ok | SHA-256 | Rows | Dates |", "|---|---|---|---|---|---|"]
    L += [f"| {i['file']} | {i['zip_bytes']/1e6:.0f} | {i.get('crc_ok')} | `{i['sha256'][:16]}…` | {C['per_part'][i['file']]['rows_in']:,} | {C['per_part'][i['file']]['min_date']} → {C['per_part'][i['file']]['max_date']} |" for i in ing["inventory"]]
    L += ["", f"Trades parsed: **{C['rows_in']:,}**; kept after cleaning and de-duplication: **{C['rows_kept']:,}** ({pct(C['rows_kept']/C['rows_in'],3)}). "
          f"Duplicate transaction numbers removed: **{D['duplicates_removed']:,}** (exact copies {D['exact']:,}, conflicting {D['conflicting']:,}; by source {D['by_parts']}; months {D['months_with_duplicates']}). "
          f"De-duplication is global: trades are bucketed by the month encoded in the transaction number, so every copy of a transaction lands in the same bucket. "
          f"Dropped: missing required field {C['dropped_required_null']:,}, non-positive quantity/rate/amount {C['dropped_nonpositive']:,}, unusable broker id {C['dropped_bad_broker']:,}, bad date {C['dropped_bad_date']:,}. "
          f"Kept and flagged: {C.get('alpha_broker_fields', 0):,} broker fields with alphanumeric codes ({', '.join(ing.get('broker_alpha_codes', {}).keys()) or 'none'}; real dealer trades), "
          f"{C['malformed_sn_rows']:,} trades from {C['min_date']} to Dec 2018 whose transaction numbers use an older format, {C['amount_mismatch']:,} rows where amount ≠ quantity × rate by more than NPR 1. "
          f"`trade_time` is empty in {pct(C['nulls']['trade_time']/C['rows_in'])} of rows, so intraday order is taken from the transaction number. "
          f"OHLC violations after aggregation: {C['ohlc_violations']}; trade counts reconcile with daily aggregates: {C['trades_reconcile']}. "
          f"Result: {C['daily_rows']:,} symbol-sessions, {C['symbols']} symbols, {C['sessions']} sessions ({C['min_date']} → {C['max_date']}).", ""]
    if aux:
        O, Fd, FS = aux["ohlcv_clean"], aux["fundamentals_clean"], aux["featurestore_audit"]; sn = aux["snapshot_tables"]
        L += ["### 1.2 Auxiliary sources", "",
              f"- **Adjusted OHLCV** (`livedata_adjusted`): {O['rows']:,} rows, {O['symbols']} symbols incl. the NEPSE index and 16 sub-indices, {O['min_date']} → {O['max_date']}; no duplicate (symbol, date) keys; `sector_id` is empty; {O['ohlc_violations']:,} bars with open/close outside high–low ({pct(O.get('ohlc_violations_share_before_2019'))} dated before 2019; {pct(O.get('ohlc_violations_share_index_rows'))} on index rows) are repaired by widening high/low. "
              "Back-adjusted for corporate actions (price ratio to raw floorsheet close is 1 after the last action and steps at bonus/rights dates); volume is in raw shares. It lists **currently listed symbols only**, so it is used for listed stocks and the floorsheet supplies the delisted/merged ones.",
              f"- **Quarterly fundamentals** (`new_fundamentals`): {Fd['rows']:,} reports, {Fd['symbols']} companies, FY {Fd['fiscal_years'][0]} → {Fd['fiscal_years'][-1]}. The table's `date` column lies {Fd['date_column_days_before_quarter_end']['min']:.0f}–{Fd['date_column_days_before_quarter_end']['max']:.0f} days *before* the quarter ends in {pct(Fd['date_column_days_before_quarter_end']['share_before_quarter_end'])} of rows, so it is not a publication date. "
              f"Reports are used only from quarter end + {Fd['publication_lag_days_used']} days. "
              + (f"The table's P/E × EPS equals the raw closing price at quarter end within 2% for {pct(Fd['pe_price_reference_check']['quarter_end']['share_within_2pct'])} of reports (vs {pct(Fd['pe_price_reference_check']['table_date']['share_within_2pct'])} at the `date` column), so E/P and B/P are rebuilt from the quarter-end price and the adjusted return since." if 'pe_price_reference_check' in Fd else ""),
              f"- **Nine `fundamental_*` tables** ({', '.join(sn)}): each holds at most one fiscal period per company (latest quarter, created {list(sn.values())[0]['created_at_range'][0][:10]}). Applying them to earlier dates would leak future information, so they are **excluded** from modelling.",
              f"- **Feature store** (`nepse_prediction_featurestore`): {FS['rows']:,} rows, {FS['symbols']} symbols, {FS['min_date']} → {FS['max_date']}. Its target is the next-session return. Its fundamental columns switch to a new report on the table's `date` column — {pct(FS['fundamental_eps_switch_timing']['share_before_quarter_end'])} of switches happen before the quarter even ends — so it carries look-ahead bias and is **not** used as model input.", ""]
    L += ["## 2. Modeling panel and targets (stages 03–04)", "",
          f"Universe: {pan['universe']} → ordinary shares only; price source for those: {pan['equity_price_source']}. Bars: {pan['bars']['bars_by_source']}; {pan['bars']['floorsheet_bars_chained_into_ohlcv_symbols']:,} floorsheet bars chained onto adjusted series before a symbol's first OHLCV bar. "
          f"Eligible samples: {pan['rows_eligible']:,} ({pan['symbols_eligible']} symbols, {pan['sessions_eligible']} sessions, {pan['first_eligible_date']} → {pan['last_date']}). "
          f"Predictors: {pan['n_core_features']} core (price/return, volatility, candle, trend, oscillators, range, liquidity, floorsheet/broker, NEPSE market) + {pan['n_fundamental_features']} point-in-time fundamentals; this run used feature set `{FEATURE_SET}` ({len(MODEL_FEATURES)} predictors). "
          f"Floorsheet features are present for {pct(pan['floorsheet_coverage_eligible'])} of samples (by year {pan['floorsheet_coverage_by_year']}) and are left missing, not imputed, where the floorsheet has gaps; fundamentals cover {pct(pan['fundamental_coverage_eligible'])}. "
          f"Residual price jumps > 11% (bad prints or unadjusted actions) neutralised on {pan['corporate_action_or_bad_print_days']:,} symbol-sessions.",
          *([ "", "**Thin-floorsheet sessions excluded.** A session is thin when fewer than "
             f"{pct(pan['bars']['thin_floorsheet']['min_coverage'],0)} of its eligible stocks appear in the floorsheet. Runs of at least "
             f"{pan['bars']['thin_floorsheet']['min_block']} consecutive thin sessions are dropped from the modelling panel, together with the "
             f"{pan['bars']['thin_floorsheet']['buffer_sessions']} sessions that follow each run, so that no rolling or sequence window reads a thin session. "
             f"{pan['bars']['thin_floorsheet']['thin_sessions']} sessions are thin; {pan['bars']['thin_floorsheet']['excluded_sessions']} are excluded in total. Blocks: "
             + "; ".join(f"{b['from']} → {b['to']} ({b['sessions']} sessions, buffer to {b['buffer_to']})"
                         for b in pan['bars']['thin_floorsheet']['blocks']) + "."]
            if pan.get("bars", {}).get("thin_floorsheet", {}).get("excluded") else []), "",
          "| Horizon | Labeled | Up-rate | Flat outcomes excluded | Mean fwd ret |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['labeled']:,} | {pct(v['up_rate'])} | {pct(v['flat_excluded_share'])} | {pct(v['fwd_ret_mean'],2)} |" for k, v in pan["targets"].items()]
    L += ["", "## 3. Leakage and split audit (stage 05)", "", f"All checks passed: **{aud['all_checks_passed']}**.", "", "| Check | Passed | Detail |", "|---|---|---|"] + [f"| {k} | {v['passed']} | {v['detail']} |" for k, v in aud["checks"].items()]
    L += ["", "| Fold | Train sessions | Val sessions | Train dates | Val dates | Train rows | Val rows |", "|---|---|---|---|---|---|---|"] + [f"| {r['fold']} | {r['train_sessions']} | {r['val_sessions']} | {r['train_dates'][0]}→{r['train_dates'][1]} | {r['val_dates'][0]}→{r['val_dates'][1]} | {r['train_rows']:,} | {r['val_rows']:,} |" for r in aud["folds"]]
    L += ["", f"Final test: {aud['final_test']['sessions']} sessions, {aud['final_test']['dates'][0]} → {aud['final_test']['dates'][1]} ({aud['final_test']['rows']:,} rows). Embargo: {EMBARGO} sessions.", ""]
    return L

def data_report():
    L = ["# NEPSE data cleaning, panel construction and leakage audit", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the stage 01-05 reports.", ""] + data_sections()
    (REPORTS / "data_report.md").write_text("\n".join(L), encoding="utf-8"); log(f"data report written -> {REPORTS / 'data_report.md'}")

def report(bt):
    ing = load_json(REPORTS / "stage01_02_ingest_report.json"); pan = load_json(REPORTS / "stage03_04_panel_report.json"); aud = load_json(REPORTS / "stage05_audit_report.json")
    mod = load_json(REPORTS / "stage06_08_models_report.json"); ens = load_json(REPORTS / "stage09_10_ensemble_report.json"); st = load_json(REPORTS / "stage11_stats.json")
    fin = load_json(REPORTS / "stage12_final_test.json"); sel = fin["selection"]; C = ing["cleaning"]; L = []
    fb = {k: v for k, v in mod["backends"].items() if v != "native"}
    L += ["# Multi-horizon calibrated ensemble forecasting on NEPSE floorsheet data", ""]
    fam = mod.get("family", "native")
    if fam == "sklearn": L += ["> **Model family: scikit-learn stand-ins.** CatBoost, LightGBM, XGBoost, the Transformer and the TFT were not run; every number below comes from the scikit-learn models listed in section 4. Do not report these results as the named algorithms. Run `FAMILY=native START=6 bash run_pipeline.sh` with the libraries installed to produce them.", ""]
    if fb: L += ["> **WARNING – SMOKE-TEST RUN.** The following models used scikit-learn stand-ins and are NOT the named algorithms: " + ", ".join(f"`{k}` ({v})" for k, v in fb.items()) + ". Do not report these results as CatBoost/LightGBM/XGBoost/Transformer/TFT.", ""]
    L += data_sections()
    L += ["## 4. Models (stages 06–08)", ""]
    if fam == "sklearn": L += [f"Model family `sklearn`; feature set `{mod.get('feature_set')}` ({mod.get('n_features')} predictors).", ""] + [f"- `{k}`: {v}" for k, v in mod["algorithms"].items()] + [""]
    else: L += ["Backends actually used: " + ", ".join(f"`{k}`: {v}" for k, v in mod["backends"].items()), ""]
    L += [f"## 5. Out-of-fold results (nested region {st['region']}; stages 09–11)", ""]
    for h in HORIZONS:
        H = st["horizons"][f"{h}d"]; L += [f"### {h}-session horizon (n={H['n']:,}, sessions={H['sessions']})", "", "| Model | Acc | Majority | AUC | LogLoss | Brier | ECE | Acc@conf≥0.70 [95% CI] | Coverage |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in H["candidates"].items():
            if "__" in k and not k.startswith(("stacking", "regime", "avg", "weighted")): continue
            s7 = v["sel70"]; L.append(f"| {k} | {pct(v['accuracy'])} | {pct(v['majority_baseline_acc'])} | {f(v.get('auc'),3)} | {f(v['logloss'])} | {f(v['brier'])} | {f(v['ece'])} | {pct(s7['accuracy'])} {s7['ci_acc'] if s7['ci_acc'][0] is None else '['+pct(s7['ci_acc'][0])+', '+pct(s7['ci_acc'][1])+']'} | {pct(s7['coverage'],2)} |")
        L += [""]
    L += ["## 6. Significance tests (block bootstrap over sessions; DM/HAC; Holm)", "", "Primary candidate per horizon was **chosen on OOF log-loss before the test was touched**:", ""]
    for h in HORIZONS:
        H = st["horizons"][f"{h}d"]; p = sel[f"{h}d"]["primary"]; c = H["candidates"][p]; s7 = c["sel70"]
        L.append(f"- **{h}D** `{p}`: accuracy {pct(c['accuracy'])} vs majority {pct(c['majority_baseline_acc'])} (bootstrap p={f(c['p_acc_gt_majority'],3)}); AUC {f(c.get('auc'),3)}; ECE {f(c['ece'])}. At confidence ≥0.70: accuracy {pct(s7['accuracy'])}, coverage {pct(s7['coverage'],2)}, n={s7['n']:,}; p(acc > majority)={f(s7['p_acc_gt_majority'],3)}; **p(acc ≥ 0.70)={f(s7['p_acc_ge_0.70'],3)}**.")
        for ref in ("climatology", "best_single", "avg"):
            k = f"{p}_vs_{ref}"
            if k in H["dm"]: d = H["dm"][k]; L.append(f"  - DM log-loss vs {ref}: mean Δ={d['mean_loss_diff']:+.5f}, stat={d['dm_stat']:.2f}, one-sided p={d['p_one_sided_B_better']:.4f}, Holm p={d.get('p_holm', float('nan')):.4f}")
        rg = H["regime"].get(p) or H["regime"].get("stacking__iso")
        if rg: L.append(f"  - Regime split of confident calls: market-up n={rg['market_up']['n_conf']} acc={pct(rg['market_up']['accuracy'])}; market-down n={rg['market_down']['n_conf']} acc={pct(rg['market_down']['accuracy'])}, share of down-calls {pct(rg['market_down']['share_down_calls'])}")
        pf = H["per_fold"].get(p)
        if pf: L.append(f"  - Folds beating majority baseline: {pf['folds_beating_majority']}/{pf['n_folds']}")
    L += ["", f"### Ranker ({'pointwise HistGradientBoosting, scikit-learn stand-in' if fam == 'sklearn' else 'LightGBM LambdaRank'}) – cross-sectional experiment", "", "| Horizon | Mean rank-IC | IC t (HAC) | Top-bottom quintile spread | Top-decile up-rate |", "|---|---|---|---|---|"]
    for h in HORIZONS:
        r = st["horizons"][f"{h}d"].get("ranker")
        if r: L.append(f"| {h}D | {f(r['mean_ic'])} | {f(r['ic_t_hac'],2)} | {pct(r['spread'],2)} | {pct(r['top_decile_up_rate'])} |")
    L += ["", "![reliability](figs/reliability.png)", "", "## 7. Untouched final test (stage 12)", "", f"Dates {fin['test_dates'][0]} → {fin['test_dates'][1]}. Forced re-runs recorded in lock: {fin['forced_reruns']}.", "",
          "| Horizon | Primary | Acc [95% CI] | Majority | AUC | LogLoss | ECE | Acc@≥0.70 | Coverage | n@≥0.70 | p(acc≥0.70) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        c = fin["horizons"][f"{h}d"]["candidates"][sel[f"{h}d"]["primary"]]; s7 = c["sel70"]; a = c["ci"]["accuracy"]
        L.append(f"| {h}D | {sel[f'{h}d']['primary']} | {pct(c['accuracy'])} [{pct(a[0])}, {pct(a[1])}] | {pct(c['majority_baseline_acc'])} | {f(c.get('auc'),3)} | {f(c['logloss'])} | {f(c['ece'])} | {pct(s7['accuracy'])} | {pct(s7['coverage'],2)} | {s7['n']:,} | {f(s7['p_acc_ge_0.70'],3)} |")
    L += ["", "![accuracy](figs/accuracy.png)", "", "## 8. Backtest (stage 13, final test, net of costs)", "", f"Costs/side: {COSTS}; CGT {pct(CGT)}; long-only; entry next-session open.", "", "| Horizon | Thr | Signals | Filled | Trades | Mean net/trade [95% CI] | Hit rate | p(≤0) | Placebo mean | p vs placebo | Sharpe | MaxDD | Cum ret |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        for thr in ("0.60", "0.70"):
            r = bt["test"][f"{h}d"].get(f"thr_{thr}", {})
            if "trades" in r: L.append(f"| {h}D | {thr} | {r['signals']} | {r['filled']} | {r['trades']} | {pct(r['mean_net_per_trade'],2)} [{pct(r['ci_mean_net'][0],2)}, {pct(r['ci_mean_net'][1],2)}] | {pct(r['hit_rate_net'])} | {f(r['p_mean_net_le_0'],3)} | {pct(r['placebo_mean_net'],2)} | {f(r['p_vs_placebo'],3)} | {f(r['sharpe'],2)} | {pct(r['max_drawdown'])} | {pct(r['cum_return'])} |")
            else: L.append(f"| {h}D | {thr} | {r.get('signals','n/a')} | {r.get('filled','n/a')} | – | too few trades | | | | | | | |")
    L += ["", "![equity](figs/equity_curves.png)", ""]
    # ---- data-driven conclusions ----
    L += ["## 9. What the evidence supports", ""]
    for h in HORIZONS:
        c = fin["horizons"][f"{h}d"]["candidates"][sel[f"{h}d"]["primary"]]; s7 = c["sel70"]; a = c["ci"]["accuracy"]
        beats = a[0] is not None and a[0] > c["majority_baseline_acc"]
        L.append(f"- **{h}D:** overall test accuracy {pct(c['accuracy'])} – {'statistically above' if beats else 'NOT statistically distinguishable from'} the majority baseline ({pct(c['majority_baseline_acc'])}). "
                 + (f"At ≥0.70 confidence accuracy was {pct(s7['accuracy'])} on {pct(s7['coverage'],2)} of cases (n={s7['n']:,}); the 95% CI is [{pct(s7['ci_acc'][0])}, {pct(s7['ci_acc'][1])}], so a true accuracy of 70% is " + ("supported." if s7['ci_acc'][0] is not None and s7['ci_acc'][0] >= 0.70 and s7['n'] >= 100 else ("not assessable: fewer than 100 calls." if s7['n'] < 100 else "NOT statistically supported (lower bound below 70%).")) if s7['accuracy'] is not None else "No predictions reached 0.70 confidence."))
    L += ["", "70% is therefore a *selective-prediction operating point* (accuracy conditional on confidence ≥ 0.70 with the stated coverage), not a model-wide accuracy. Calibration (ECE) shows whether stated confidences are honest.", "",
          "## 10. Limitations", "", "- Daily aggregation hides intraday dynamics; no order-book data.", "- The floorsheet is largely missing for Feb 2022 – Mar 2023 and Mar 2024; broker features are missing there (prices come from the adjusted OHLCV table).", "- Fundamental publication dates are approximated as quarter end + 45 days; a sensitivity check with a longer lag is advisable.", "- Overlapping labels (h>1) are handled by block bootstrap and HAC tests but reduce the effective sample size.", "- Corporate-action adjustment is heuristic (jumps > 11%); dividends are not modelled.",
          "- Long-only execution at the next open assumes fills; thin names may fill worse than the 10 bp slippage assumed.", "- A single 250-session test window covers one market regime; repeated forward evaluation is recommended before any deployment claim.", "- Results from a smoke-test run (stand-in models) must not be cited as results of the named algorithms."]
    (REPORTS / "research_report.md").write_text("\n".join(L), encoding="utf-8"); log("report written")

def export():
    arts = sorted(p for p in list(MODELS.glob("*")) + list(REPORTS.glob("*.json")) + [WORK / "splits.json", WORK / "selection.json", WORK / "final_test.lock"] if p.is_file())
    man = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in arts}
    save_json({"features": MODEL_FEATURES, "feature_set": FEATURE_SET, "horizons": HORIZONS, "artifacts": man}, REPORTS / "export_manifest.json"); log(f"exported manifest for {len(man)} artifacts")

def run():
    bt = backtests(); figures(); report(bt); export()

if __name__ == "__main__":
    import sys
    data_report() if "--data-only" in sys.argv else run()
