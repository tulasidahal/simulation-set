#!/usr/bin/env python3
"""Export a run's panel, OOF predictions and final-test predictions to gzipped CSV."""
import sys, joblib, numpy as np, pandas as pd
from pathlib import Path

HOR = [1, 3, 5, 10]

def export(work, out, tag):
    work, out = Path(work), Path(out); out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_pickle(work / "panel.pkl")
    panel.to_csv(out / f"panel_{tag}.csv.gz", index=False)
    print(f"  panel_{tag}.csv.gz           {panel.shape[0]:,} x {panel.shape[1]}")

    O = joblib.load(work / "oof.pkl")
    key = ["symbol", "date", "sess", "eligible"] + [f"y_{h}" for h in HOR]
    df = panel[[c for c in key if c in panel.columns]].copy()
    df["fold_id"] = O["fold_id"]
    for mdl in O["oof"]:
        for h in HOR:
            v = O["oof"][mdl].get(h)
            if v is not None: df[f"{mdl}_h{h}"] = np.asarray(v, dtype="float32")
    df = df[df.fold_id >= 0] if (df.fold_id >= 0).any() else df
    df.to_csv(out / f"oof_predictions_{tag}.csv.gz", index=False, float_format="%.6g")
    print(f"  oof_predictions_{tag}.csv.gz  {df.shape[0]:,} x {df.shape[1]}  "
          f"({len(O['oof'])} models x {len(HOR)} horizons)")
    pd.Series(O["backends"]).rename("backend").to_csv(out / f"model_backends_{tag}.csv")

    T = joblib.load(work / "test_predictions.pkl")
    t = T["meta"].copy()
    for h in HOR:
        for cand, v in T["preds"][h].items():
            t[f"{cand}_h{h}"] = np.asarray(v, dtype="float32")
    t.to_csv(out / f"test_predictions_{tag}.csv.gz", index=False, float_format="%.6g")
    print(f"  test_predictions_{tag}.csv.gz {t.shape[0]:,} x {t.shape[1]}  "
          f"({len(T['preds'][1])} candidates x {len(HOR)} horizons)")
    pd.DataFrame(T["selection"]).T.rename_axis("horizon").to_csv(out / f"selection_{tag}.csv")

if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2], sys.argv[3])
