#!/usr/bin/env python3
"""Turn a full panel export into the model-ready training set.

    python make_training_set.py data/panel/panel_run_C_recovered.csv.gz out.csv.gz

Keeps only rows the pipeline marked eligible and drops the bookkeeping columns, giving the
same 71-column layout as training_set_run_A_full_362_stocks.csv.gz: symbol, date, sess,
the 60 predictors and the fwd_ret_*/y_* targets.
"""
import sys, pandas as pd

DROP = ["sym_id", "amount", "adj_close", "eligible", "event_day", "fs_available",
        "price_src", "trade_ret_1", "trade_ret_3", "trade_ret_5", "trade_ret_10"]

src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
if "eligible" in d.columns:
    d = d[d.eligible.astype(bool)]
d = d.drop(columns=[c for c in DROP if c in d.columns])
d.to_csv(dst, index=False)
print(f"{dst}: {d.shape[0]:,} rows x {d.shape[1]} columns")
