"""Unit tests: run with `python -m pytest tests -q` (or `python tests/test_pipeline.py`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score, matthews_corrcoef

from nepse_stack import common as C
from nepse_stack import data as D
from nepse_stack import evaluation as E
from nepse_stack.config import ExperimentConfig
from nepse_stack.models import focal_grad_hess
from nepse_stack.pipeline import block_of_rows, prepare
from nepse_stack.splits import stacker_train_mask, walk_forward

CFG = ExperimentConfig()
RAW = D.make_synthetic_panel(12, 400, seed=3)
PREP = prepare(RAW, CFG)


def test_focal_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    z, y = rng.normal(size=(300, 3)), rng.integers(0, 3, 300)
    for g in (0.0, 1.0, 2.0):
        grad, hess = focal_grad_hess(z, y, g)

        def per(zz):
            p = np.clip(softmax(zz, 1)[np.arange(len(y)), y], 1e-12, 1)
            return -(1 - p) ** g * np.log(p)
        num = np.stack([(per(z + e) - per(z - e)) / 2e-6 for e in np.eye(3) * 1e-6], 1)
        assert np.abs(grad - num).max() < 1e-5 and (hess > 0).all()
    assert np.allclose(focal_grad_hess(z, y, 0.0)[0], softmax(z, 1) - np.eye(3)[y])


def test_labels_are_forward_returns():
    df = PREP.df
    s = df[df.symbol == df.symbol.iloc[0]]
    raw = RAW[RAW.symbol == s.symbol.iloc[0]].set_index("date")["close"]
    t = s.date.iloc[10]
    nxt = raw.index[raw.index.get_loc(t) + 1]
    assert np.isclose(s.set_index("date").loc[t, "fwd_ret"], raw[nxt] / raw[t] - 1)
    assert df["label"].isin([0, 1, 2]).all() and not df["fwd_ret"].isna().any()


def test_walk_forward_purge_and_embargo():
    dates = PREP.df["date"]
    ud = np.sort(dates.unique())
    for f in walk_forward(dates, CFG):
        train_last = np.flatnonzero(ud == dates[f.train_mask(dates)].max())[0]
        test_first = np.flatnonzero(ud == dates[f.test_mask(dates)].min())[0]
        # a training label at index i looks ahead to i + horizon
        assert train_last + CFG.horizon < test_first
        assert test_first - train_last - 1 == CFG.horizon + CFG.embargo_days
        assert not (f.train_mask(dates) & f.test_mask(dates)).any()


def test_stacker_uses_only_earlier_blocks():
    dates = PREP.df["date"]
    folds = walk_forward(dates, CFG)
    blocks = block_of_rows(dates, folds)
    for f in folds[CFG.min_stack_blocks:]:
        m = stacker_train_mask(blocks, dates, f, CFG)
        assert m.any()
        assert blocks[m].max() < f.k
        assert dates[m].max() < f.test_start


def test_inner_split_embargo():
    d = pd.Series(np.repeat(pd.bdate_range("2020-01-01", periods=300), 4))
    tr, va = C.inner_train_val_split(d, 0.2, 10)
    ud = np.sort(d.unique())
    gap = np.searchsorted(ud, d[va].min()) - np.searchsorted(ud, d[tr].max()) - 1
    assert gap == 10 and not (tr & va).any()


def test_weights():
    y = np.array([0] * 10 + [1] * 30 + [2] * 60)
    w = C.class_balance_weights(y)
    assert np.allclose([w[y == k].sum() for k in range(3)], w[y == 0].sum())
    d = pd.Series(pd.date_range("2020-01-01", periods=1001))
    w = C.time_decay_weights(d, 500)
    assert np.isclose(w[500] / w[-1], 0.5)


def test_temperature_scaling_recovers_known_factor():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(20000, 3)) * 1.5
    y = (rng.random(20000)[:, None] > np.cumsum(softmax(z, 1), 1)).sum(1)
    T = C.fit_temperature(log_softmax(z * 2.0, 1), y)
    assert abs(T - 2.0) < 0.15


def test_corporate_action_audit_finds_unadjusted_bonus():
    ca = D.corporate_action_audit(PREP.df, CFG)
    assert ca["close_raw"]["n_breaches"] >= 1 and ca["close"]["n_breaches"] == 0


def test_truncation_invariance_detects_lookahead():
    cut = RAW.date.sort_values().iloc[len(RAW) // 2]
    base = RAW[["date", "symbol", "sector", "close", "close_raw", "volume"]]
    ok = D.truncation_invariance_test(lambda r: D.demo_features(r), base, CFG, cut)
    assert ok["violations"] == []

    def leaky(r):
        r = r.sort_values(["symbol", "date"]).copy()
        r["centered_ma"] = r.groupby("symbol")["close"].transform(
            lambda s: s.rolling(11, center=True).mean())
        return r
    bad = D.truncation_invariance_test(leaky, base, CFG, cut)
    assert "centered_ma" in bad["violations"]


def test_ic_audit_flags_future_canary():
    df = PREP.df.assign(canary=PREP.df["fwd_ret"] + np.random.default_rng(0).normal(
        0, PREP.df["fwd_ret"].std(), len(PREP.df)))
    audit = D.leakage_ic_audit(df, PREP.feats + ["canary"], CFG)
    assert "canary" in audit["flagged"]
    assert all(f not in audit["flagged"] for f in PREP.feats)


def test_sequence_windows_are_point_in_time_and_per_symbol():
    df = PREP.df
    src = C.SeqSource(df, PREP.feats, "symbol", "date", 5)
    sym = df.symbol.to_numpy()
    first_of_second_symbol = np.flatnonzero(sym != sym[0])[0]
    w = src.windows(np.array([first_of_second_symbol, first_of_second_symbol + 4]))
    assert np.allclose(w[0, :4], 0)                       # no bleed from previous symbol
    assert np.allclose(w[0, -1], src.F[first_of_second_symbol])
    assert np.allclose(w[1], src.F[first_of_second_symbol:first_of_second_symbol + 5])


def test_confusion_metrics_match_sklearn():
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 3, 2000), rng.integers(0, 3, 2000)
    cm = E.confusion_by_date(y, p, np.zeros(2000, int), 1)[0]
    assert np.isclose(E.mcc_cm(cm), matthews_corrcoef(y, p))
    assert np.isclose(E.macro_f1_cm(cm), f1_score(y, p, average="macro"))


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(3)
    d = np.repeat(np.arange(300), 10)
    la = rng.normal(1.0, 0.2, 3000)
    res = E.diebold_mariano(la - 0.05, la + rng.normal(0, 0.05, 3000), d)
    assert res["dm_stat"] < 0 and res["p_value"] < 0.01


def test_backtest_perfect_foresight_beats_benchmark():
    df = PREP.df.copy()
    P = np.full((len(df), 3), 0.05)
    P[np.arange(len(df)), np.where(df.fwd_ret_abs > 0.01, 2, 0)] = 0.9
    out, _ = E.backtest(df, P, CFG, "top_conf", 0.5)
    assert out["strategy"]["sharpe"] > out["benchmark"]["sharpe"]


def test_backtest_charges_costs_on_turnover_only():
    df = PREP.df.copy()
    first = df.symbol.iloc[0]
    P = np.tile([0.3, 0.4, 0.3], (len(df), 1))
    P[(df.symbol == first).to_numpy()] = [0.1, 0.1, 0.8]      # always hold one stock
    out, eq = E.backtest(df, P, CFG, "argmax")
    assert eq.turnover.iloc[0] == 1.0 and (eq.turnover.iloc[1:] == 0).all()
    gross_minus_net = (eq.gross - eq.strategy).to_numpy()
    assert np.isclose(gross_minus_net[0], CFG.cost_bps_roundtrip / 2 / 1e4)
    assert np.allclose(gross_minus_net[1:], 0)


def test_label_breaches_are_dropped():
    raw = RAW.copy()
    sym = raw.symbol.iloc[0]
    idx = raw.index[raw.symbol == sym][100]
    later = raw.index[(raw.symbol == sym) & (raw.index > idx)]
    raw.loc[later, "close"] *= 1.3                               # unadjusted bonus-like jump
    prep = prepare(raw, CFG)
    assert prep.quality["label_breaches_dropped"] >= 1
    assert (prep.df["fwd_ret"].abs() <= (1 + CFG.circuit_limit) - 1 + 0.006).all()


def test_robust_canary_is_detected_with_heavy_tails():
    df = PREP.df.copy()
    spikes = np.random.default_rng(5).random(len(df)) < 0.01   # 1% extreme days
    df.loc[spikes, "fwd_ret"] *= 25
    res = D.canary_validation(df, PREP.feats, CFG)
    assert res["canary_flagged"], res


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"\n{len(fns)} tests passed")


def test_incremental_update_matches_full_rebuild():
    import run_efficiency as RE
    cfg = ExperimentConfig()
    raw = D.validate_schema(D.make_synthetic_panel(n_symbols=15, n_dates=200), cfg)
    feats = D.feature_columns(raw, cfg)
    res = RE.exp_incremental(raw, cfg, feats, "lightgbm", reps=1)
    assert res["features_identical"] and res["labels_identical"]
