"""
Stage 05  Leakage and split audit.
Split design (all units are market sessions):
  * final test  = last FINAL_TEST (250) sessions, untouched until stage 12
  * EMBARGO (25) sessions between the end of development data and the final test
  * development = expanding-window walk-forward with N_FOLDS (8) validation blocks; the first training window has
    >= MIN_TRAIN (504) sessions; each validation block is preceded by an EMBARGO (25) purge of the training data.
Audits: (1) embargo vs label horizon, (2) block ordering/overlap, (3) final-test isolation, (4) feature look-ahead
via truncation-recompute equality test, (5) target wrap-around, (6) feature-target correlation sanity, (7) fold sizes.
"""
import hashlib
import numpy as np, pandas as pd
from common import *
import s03_04_panel as P

def make_splits(panel):
    el = panel[panel.eligible]
    U = np.sort(el.sess.unique()); N = len(U)
    test_start = N - FINAL_TEST; dev_end = test_start - EMBARGO
    first_val = MIN_TRAIN + EMBARGO
    R = dev_end - first_val; B = R // N_FOLDS
    assert B >= 40, f"validation blocks too small ({B}); not enough sessions"
    folds = []
    for k in range(1, N_FOLDS + 1):
        vs = first_val + (k - 1) * B; ve = vs + B if k < N_FOLDS else dev_end
        folds.append({"fold": k, "train_pos": [0, vs - EMBARGO], "val_pos": [vs, ve]})
    for f in folds:
        f["train_sess"] = [int(U[f["train_pos"][0]]), int(U[f["train_pos"][1] - 1])]
        f["val_sess"] = [int(U[f["val_pos"][0]]), int(U[f["val_pos"][1] - 1])]
    return {"U": U.tolist(), "N": int(N), "folds": folds, "dev_end_pos": int(dev_end), "test_start_pos": int(test_start),
            "dev_sess": [int(U[0]), int(U[dev_end - 1])], "test_sess": [int(U[test_start]), int(U[-1])],
            "params": {"MIN_TRAIN": MIN_TRAIN, "EMBARGO": EMBARGO, "N_FOLDS": N_FOLDS, "FINAL_TEST": FINAL_TEST}}

def pos_of(sess_arr, splits):
    """Map calendar session index -> position in the eligible-session list U."""
    U = np.asarray(splits["U"]); return np.searchsorted(U, sess_arr)

def fold_masks(panel, splits, k):
    """Boolean masks (train, val) over ALL panel rows for fold k (1-based); rows must also be eligible to be used."""
    f = splits["folds"][k - 1]; s = panel.sess.values
    tr = (s >= f["train_sess"][0]) & (s <= f["train_sess"][1]) & panel.eligible.values
    va = (s >= f["val_sess"][0]) & (s <= f["val_sess"][1]) & panel.eligible.values
    return tr, va

def dev_mask(panel, splits):
    return (panel.sess.values <= splits["dev_sess"][1]) & panel.eligible.values

def test_mask(panel, splits):
    return (panel.sess.values >= splits["test_sess"][0]) & panel.eligible.values

def run():
    panel = pd.read_pickle(WORK / "panel.pkl"); meta = pd.read_csv(WORK / "instrument_meta.csv", index_col="symbol")
    sp = make_splits(panel); rep = {"checks": {}}; ok_all = True
    def check(name, passed, detail):
        nonlocal ok_all; ok_all &= bool(passed); rep["checks"][name] = {"passed": bool(passed), "detail": detail}
        log(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    hmax = max(HORIZONS)
    # 1. embargo vs label horizon (calendar sessions): every train label must resolve before the val block starts
    worst = min(f["val_sess"][0] - (f["train_sess"][1] + hmax) for f in sp["folds"])
    check("embargo_covers_label_horizon", worst > 0, f"min (val_start - (train_end + {hmax})) = {worst} sessions; embargo={EMBARGO}")
    for f in sp["folds"]:
        assert f["val_pos"][0] - f["train_pos"][1] == EMBARGO
    # 2. fold ordering and overlap
    ordered = all(sp["folds"][i]["val_pos"][1] == sp["folds"][i + 1]["val_pos"][0] for i in range(N_FOLDS - 1))
    check("validation_blocks_contiguous_non_overlapping", ordered, "blocks tile the development period")
    check("first_train_window_min_sessions", sp["folds"][0]["train_pos"][1] >= MIN_TRAIN, f"{sp['folds'][0]['train_pos'][1]} sessions")
    # 3. final-test isolation
    gap_pos = sp["test_start_pos"] - sp["dev_end_pos"]; last_dev = sp["dev_sess"][1]; first_test = sp["test_sess"][0]
    check("final_test_isolated", gap_pos == EMBARGO and first_test - last_dev > hmax, f"gap={gap_pos} eligible sessions, {first_test-last_dev} calendar sessions; test={FINAL_TEST} sessions")
    dm = dev_mask(panel, sp); tm = test_mask(panel, sp)
    check("dev_test_row_disjoint", not (dm & tm).any(), f"dev rows={dm.sum():,}, test rows={tm.sum():,}")
    # 4. duplicates
    check("no_duplicate_symbol_session", not panel.duplicated(["symbol", "sess"]).any(), "unique (symbol, session)")
    # 5. target wrap-around: last h sessions must have no label
    T = panel.sess.max()
    bad = {h: int(panel.loc[panel.sess > T - h, f"y_{h}"].notna().sum()) for h in HORIZONS}
    check("targets_unavailable_in_last_h_sessions", all(v == 0 for v in bad.values()), f"labelled rows in last h sessions: {bad}")
    # 6. feature look-ahead: truncate data at cutoff, recompute, compare
    cut = int(np.quantile(sp["U"], 0.70)); cut_date = pd.Timestamp(panel.loc[panel.sess == cut, "date"].iloc[0])
    log(f"truncation test: recomputing features on data <= {cut_date.date()}")
    ptr, _, _, _ = P.build_panel(*P.load_sources(cutoff=cut_date), meta)        # every source truncated (fundamentals by availability date)
    full = panel[panel.date <= cut_date].set_index(["symbol", "date"]); tr = ptr.set_index(["symbol", "date"])
    common_idx = full.index.intersection(tr.index)
    check("truncation_same_rows", len(common_idx) == len(full), f"{len(common_idx):,} / {len(full):,} rows present in both builds")
    worst_feat, mx = None, 0.0; diffs = {}
    for k in ALL_FEATURES:
        a = full.loc[common_idx, k].values.astype("float64"); b = tr.loc[common_idx, k].values.astype("float64")
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum()); m = ~np.isnan(a) & ~np.isnan(b)
        d = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0; diffs[k] = (d, nan_mismatch)
        if d > mx: mx, worst_feat = d, k
    nm = sum(v[1] for v in diffs.values())
    check("features_lookahead_free_truncation_test", mx < 1e-4 and nm == 0, f"max |diff|={mx:.2e} ({worst_feat}); NaN-pattern mismatches={nm}; {len(ALL_FEATURES)} features x {len(common_idx):,} rows")
    rep["truncation_feature_diffs"] = {k: {"max_abs_diff": v[0], "nan_mismatch": v[1]} for k, v in diffs.items()}
    # 7. feature-target correlation sanity (only ordinary Pearson, eligible dev rows)
    dv = panel[dm & panel.y_1.notna()]
    cors = {k: float(np.corrcoef(np.nan_to_num(dv[k].values.astype("float64")), dv.y_1.values)[0, 1]) for k in ALL_FEATURES}
    top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:5]
    check("no_suspicious_feature_target_correlation", abs(top[0][1]) < 0.15, f"top |corr| with y_1: {[(k, round(v,3)) for k,v in top]}")
    # 8. fold sizes
    rep["folds"] = []
    for f in sp["folds"]:
        tr_m, va_m = fold_masks(panel, sp, f["fold"])
        row = {"fold": f["fold"], "train_sessions": f["train_pos"][1] - f["train_pos"][0], "val_sessions": f["val_pos"][1] - f["val_pos"][0],
               "train_dates": [str(panel.loc[panel.sess == f["train_sess"][0], "date"].iloc[0].date()), str(panel.loc[panel.sess == f["train_sess"][1], "date"].iloc[0].date())],
               "val_dates": [str(panel.loc[panel.sess == f["val_sess"][0], "date"].iloc[0].date()), str(panel.loc[panel.sess == f["val_sess"][1], "date"].iloc[0].date())],
               "train_rows": int(tr_m.sum()), "val_rows": int(va_m.sum())}
        for h in HORIZONS: row[f"val_up_rate_{h}d"] = float(panel.loc[va_m, f"y_{h}"].mean())
        rep["folds"].append(row)
    rep["final_test"] = {"sessions": FINAL_TEST, "dates": [str(panel.loc[panel.sess == sp["test_sess"][0], "date"].iloc[0].date()), str(panel.loc[panel.sess == sp["test_sess"][1], "date"].iloc[0].date())],
                         "rows": int(tm.sum())}
    rep["notes"] = ["Labels overlap for h>1; block bootstrap and HAC tests (stage 11) account for serial dependence.",
                    "Exactly-flat forward returns are excluded from classification labels.",
                    "Universe is not filtered by future history length (no survivorship filter); eligibility uses only past data."]
    rep["all_checks_passed"] = ok_all
    sp["manifest_sha256"] = hashlib.sha256(json.dumps(sp["folds"], sort_keys=True).encode()).hexdigest()
    save_json(sp, WORK / "splits.json"); save_json(rep, REPORTS / "stage05_audit_report.json")
    log(f"AUDIT {'PASSED' if ok_all else 'FAILED'}; folds: " + "; ".join(f"{r['fold']}: train {r['train_sessions']}s val {r['val_sessions']}s" for r in rep["folds"]))
    if not ok_all: raise SystemExit("leakage/split audit failed")

if __name__ == "__main__":
    run()
