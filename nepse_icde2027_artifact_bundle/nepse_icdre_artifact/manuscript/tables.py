"""Build every results table for the manuscript directly from the run-1 files."""
import pandas as pd, numpy as np, json

R = "/tmp/claude-0/-home-claude/0f7c4175-42ce-5141-b235-87761a7e5912/scratchpad/bundle/r/"
M = pd.read_csv(R + "tables/metrics_by_seed.csv")
MINUS = "−"


def cmp(mean, std=None, d=3):
    """0.383(3): mean with std in units of the last digit."""
    s = f"{abs(mean):.{d}f}"
    if mean < 0 and float(s) != 0:
        s = MINUS + s
    if std is not None and std * 10 ** d >= 0.5:
        s += f"({int(round(std * 10 ** d))})"
    return s


def agg(method, col):
    x = M[M.method == method][col]
    return x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0


def pipe(header, rows, align=None):
    align = align or ["l"] + ["c"] * (len(header) - 1)
    sep = {"l": ":--", "c": ":-:", "r": "--:"}
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep[a] for a in align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


NAMES = {
    "majority": "Majority class", "persistence": "Persistence", "momentum20": "Momentum-20",
    "logreg": "Logistic regression", "catboost": "CatBoost", "lightgbm": "LightGBM",
    "xgboost": "XGBoost (custom.)", "transformer": "Transformer",
    "stack_mean_prob": "**Mean of probabilities**", "stack_mean_logprob": "Mean of log-probs",
    "stack_lr_probs": "Logistic stacker (probs)", "stack_lr_logits": "Logistic stacker (logits)",
    "stack_lr_logits_balanced": "Logistic stacker (balanced)",
}
ORDER = list(NAMES)


def t_main():
    rows = []
    for m in ORDER:
        r = [NAMES[m]]
        for col, d in [("balanced_accuracy", 3), ("macro_f1", 3), ("mcc", 3), ("log_loss", 3), ("ece", 3)]:
            mu, sd = agg(m, col)
            r.append(cmp(mu, sd, d))
        rows.append(r)
    return pipe(["Method", "Bal. acc.", "Macro-F1", "MCC", "Log loss", "ECE"], rows)


def t_perclass():
    keep = ["majority", "persistence", "logreg", "catboost", "lightgbm", "xgboost", "transformer",
            "stack_mean_prob", "stack_lr_logits"]
    rows = []
    for m in keep:
        g = M[M.method == m]
        rows.append([NAMES[m]] + [f"{g[c].mean():.3f}" for c in ["accuracy", "f1_down", "f1_flat", "f1_up", "aurc"]])
    return pipe(["Method", "Acc.", "F1 DOWN", "F1 FLAT", "F1 UP", "AURC"], rows)


def t_blocks():
    f = pd.read_csv(R + "tables/folds.csv")
    rows = [[str(int(r.block)), r.train_end, r.test_start, r.test_end, "yes" if r.evaluated else "no"]
            for r in f.itertuples()]
    return pipe(["Block", "Train ends", "Test start", "Test end", "Scored"], rows, ["c"] * 5)


def t_protocol():
    p = pd.read_csv(R + "tables/protocols.csv", index_col=0)
    lab = {"walk_forward_embargo_25": "Walk-forward, embargo 25 (reference)",
           "walk_forward_embargo_0": "Walk-forward, embargo 0",
           "walk_forward_embargo_5": "Walk-forward, embargo 5",
           "walk_forward_embargo_10": "Walk-forward, embargo 10",
           "walk_forward_embargo_50": "Walk-forward, embargo 50",
           "walk_forward_no_purge": "Walk-forward, no purge",
           "random_kfold": "**Random 5-fold**", "shuffled_labels": "Shuffled training labels"}
    rows = []
    for k, v in lab.items():
        r = p.loc[k]
        rows.append([v, cmp(r.balanced_accuracy), cmp(r.macro_f1), cmp(r.mcc), cmp(r.log_loss)])
    return pipe(["Protocol (seed 0)", "Bal. acc.", "Macro-F1", "MCC", "Log loss"], rows)


def _pm(s):
    a, b = s.replace("±", "|").split("|")
    a = float(a); b = float(b)
    return a, b


def t_xgb():
    x = pd.read_csv(R + "tables/xgb_ablation.csv", index_col=0)
    lab = {"xgb_full": "**Full customised model**", "xgb_no_class_balance": "− class-balanced weights",
           "xgb_no_time_decay": "− time-decay weights", "xgb_no_symbol_sector": "− symbol, sector features",
           "xgb_focal_g1": "Focal loss, γ = 1", "xgb_focal_g2": "Focal loss, γ = 2",
           "xgb_stop_macro_f1": "Early stop: macro-F1", "xgb_stop_mcc": "Early stop: MCC",
           "xgb_no_calibration": "− temperature scaling",
           "xgb_original_350x4": "Original 350 rounds, depth 4"}
    rows = []
    for k, v in lab.items():
        r = [v]
        for col in ["balanced_accuracy", "macro_f1", "mcc", "log_loss", "ece"]:
            a, b = _pm(x.loc[k, col]); r.append(cmp(a, b))
        rows.append(r)
    return pipe(["Configuration", "Bal. acc.", "Macro-F1", "MCC", "Log loss", "ECE"], rows)


def t_stackabl():
    x = pd.read_csv(R + "tables/stack_ablation.csv", index_col=0)
    lab = {"stack_lr_logits": "**All four models**", "stack_without_catboost": "Without CatBoost",
           "stack_without_lightgbm": "Without LightGBM", "stack_without_xgboost": "Without XGBoost",
           "stack_without_transformer": "Without Transformer"}
    rows = []
    for k, v in lab.items():
        r = [v]
        for col in ["balanced_accuracy", "mcc", "log_loss"]:
            a, b = _pm(x.loc[k, col]); r.append(cmp(a, b))
        rows.append(r)
    return pipe(["Stacker inputs", "Bal. acc.", "MCC", "Log loss"], rows)


def t_regime():
    p = pd.read_csv(R + "tables/per_regime_block.csv")
    piv = p.pivot(index="method", columns="block", values="mcc")
    lab = {"persistence": "Persistence", "logreg": "Logistic regression", "catboost": "CatBoost",
           "lightgbm": "LightGBM", "xgboost": "XGBoost (custom.)", "transformer": "Transformer",
           "stack_lr_logits": "Logistic stacker"}
    rows = [[v] + [cmp(piv.loc[k, b]) for b in [2, 3, 4, 5]] for k, v in lab.items()]
    return pipe(["Method", "Block 2 (dev.)", "Block 3", "Block 4", "Block 5"], rows)


def t_eff():
    e = pd.read_csv(R + "tables/efficiency.csv", index_col=0)
    rows = []
    for k, v, it in [("catboost", "CatBoost", "iterations"), ("lightgbm", "LightGBM", "iterations"),
                     ("xgboost", "XGBoost", "iterations"), ("transformer", "Transformer", "epochs"),
                     ("logreg", "Logistic regression", None)]:
        r = e.loc[k]
        rows.append([v, f"{r.fit_s_mean:.1f}", "–" if it is None else f"{r.iterations_mean:.0f}",
                     f"{r.predict_rows_per_s / 1000:,.0f}"])
    return pipe(["Model", "Fit (s)", "Iter./epochs", "Inference (k rows/s)"], rows)


def values():
    """Numbers quoted in the text, read from the files so they cannot drift."""
    v = {}
    pr = pd.read_csv(R + "tables/protocols.csv", index_col=0)
    v["infl"] = (pr.loc["random_kfold", "mcc"] - pr.loc["walk_forward_embargo_25", "mcc"]) / pr.loc["walk_forward_embargo_25", "mcc"]
    return v


if __name__ == "__main__":
    print(t_main()); print(); print(t_perclass()); print(); print(t_protocol()); print(values())
