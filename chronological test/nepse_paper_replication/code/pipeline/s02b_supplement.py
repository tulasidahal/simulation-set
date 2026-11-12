"""
Stage 02b  Supplement the floorsheet with recovered day files.

The original three zips are missing or near-empty on a number of sessions. The user supplied the real
archive rows for some of them as per-day SQL dumps (days/*.sql) plus one multi-day dump. This stage
parses those dumps with the *same* validation, de-duplication and daily aggregation as stage 02, then
merges the resulting symbol-sessions into work/daily_floorsheet.csv.gz.

Merge rule: a (symbol, trade_date) bar from the recovered files replaces the existing bar for that pair,
because the recovered day is the complete archive for that session; existing bars for pairs the recovered
files do not contain are kept. Every replacement and every conflict is counted in the report.

  python s02b_supplement.py --src /home/claude/newfs/raw
"""
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from common import *

ALPHA_BASE = 10000
LARGE = LARGE_TRADE
ROW = re.compile(r"^\((\d+),'([^']*)','([^']*)','([^']*)',"
                 r"([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+),'(\d{4}-\d{2}-\d{2})'")

def parse_sql(path, stats):
    sn, sym, buy, sel, qty, rate, amt, dt = [], [], [], [], [], [], [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("("): continue
            m = ROW.match(line)
            if not m:
                stats["unparsed_lines"] += 1; continue
            sn.append(m.group(1)); sym.append(m.group(2)); buy.append(m.group(3)); sel.append(m.group(4))
            qty.append(m.group(5)); rate.append(m.group(6)); amt.append(m.group(7)); dt.append(m.group(8))
    return pd.DataFrame({"sn": pd.to_numeric(sn, errors="coerce"), "symbol": sym,
                         "buyer": pd.Series(buy, dtype="string"), "seller": pd.Series(sel, dtype="string"),
                         "quantity": pd.to_numeric(qty, errors="coerce"), "rate": pd.to_numeric(rate, errors="coerce"),
                         "amount": pd.to_numeric(amt, errors="coerce"), "trade_date": dt})

def broker_codes(col, amap, stats):
    num = pd.to_numeric(col, errors="coerce")
    alpha = num.isna() & col.notna() & (col.str.strip() != "")
    if alpha.any():
        for code in pd.unique(col[alpha].str.strip().dropna()):
            if code not in amap: amap[code] = ALPHA_BASE + len(amap)
        num = num.where(~alpha, col.str.strip().map(amap))
        stats["alpha_broker_fields"] += int(alpha.sum())
    return num.where(num.between(0, 32000))

def clean(df, amap, stats):
    stats["rows_in"] += len(df)
    n0 = len(df); df = df.dropna(subset=["sn", "symbol", "buyer", "seller", "quantity", "rate", "amount", "trade_date"])
    stats["dropped_required_null"] += n0 - len(df)
    n0 = len(df); df = df[(df.quantity > 0) & (df.rate > 0) & (df.amount > 0)]
    stats["dropped_nonpositive"] += n0 - len(df)
    b, s = broker_codes(df.buyer, amap, stats), broker_codes(df.seller, amap, stats)
    ok = b.notna() & s.notna(); stats["dropped_bad_broker"] += int((~ok).sum())
    df, b, s = df[ok], b[ok], s[ok]
    stats["amount_mismatch"] += int((np.abs(df.amount.values - df.quantity.values * df.rate.values) > 1.0).sum())
    d = pd.to_datetime(df.trade_date, format="%Y-%m-%d")
    dint = (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).values.astype("int32")
    stats["sn_date_mismatch"] += int((df.sn.values // 10 ** 8 != dint).sum())
    out = pd.DataFrame({"sn": df.sn.values.astype("int64"), "date": dint, "symbol": df.symbol.values,
                        "buyer": b.values.astype("int32"), "seller": s.values.astype("int32"),
                        "qty": df.quantity.values.astype("float32"), "rate": df.rate.values.astype("float64")})
    out = out.sort_values("sn", kind="stable")
    dup = out.sn.duplicated()
    stats["dup_rows"] += int(dup.sum())
    out = out[~dup]
    stats["rows_clean"] += len(out)
    return out

def broker_features(df, col, side):
    B = df.groupby(["symbol", "date", col], sort=False).qty.sum().reset_index()
    B["share"] = B.qty / B.groupby(["symbol", "date"], sort=False).qty.transform("sum")
    B = B.sort_values(["symbol", "date", "qty"], ascending=[True, True, False])
    rk = B.groupby(["symbol", "date"], sort=False).cumcount().values
    B["sh2"] = B.share ** 2; B["top1"] = np.where(rk == 0, B.share, 0.0); B["top5"] = np.where(rk < 5, B.share, 0.0)
    return B.groupby(["symbol", "date"], sort=False).agg(**{f"{side}_hhi": ("sh2", "sum"), f"{side}_top1": ("top1", "sum"),
                                                            f"{side}_top5": ("top5", "sum"), f"n_{side}s": (col, "size")})

def aggregate(df):
    df = df.copy()
    df["amt"] = df.qty.astype("float64") * df.rate
    df["q2"] = df.qty.astype("float64") ** 2
    df["self_q"] = np.where(df.buyer == df.seller, df.qty, 0.0)
    df["large_q"] = np.where(df.qty >= LARGE, df.qty, 0.0)
    g = df.groupby(["symbol", "date"], sort=False)      # rows are in sn order -> first/last = open/close
    day = g.agg(n=("sn", "size"), vol=("qty", "sum"), amt=("amt", "sum"), open=("rate", "first"),
                high=("rate", "max"), low=("rate", "min"), close=("rate", "last"), q2=("q2", "sum"),
                self_q=("self_q", "sum"), large_q=("large_q", "sum"))
    return day.join(broker_features(df, "buyer", "buyer")).join(broker_features(df, "seller", "seller")).reset_index()

def main(src):
    src = Path(src)
    files = sorted(src.glob("days/*.sql")) + sorted(src.glob("*.sql"))
    if not files: sys.exit(f"no .sql files under {src}")
    log(f"supplement: {len(files)} source files")
    stats = dict.fromkeys(["rows_in", "rows_clean", "dropped_required_null", "dropped_nonpositive",
                           "dropped_bad_broker", "dup_rows", "amount_mismatch", "sn_date_mismatch",
                           "alpha_broker_fields", "unparsed_lines"], 0)
    amap = {"D01": ALPHA_BASE}
    parts = []
    for k, f in enumerate(files):
        raw = parse_sql(f, stats)
        if raw.empty: continue
        parts.append(aggregate(clean(raw, amap, stats)))
        if k % 20 == 0 or k == len(files) - 1:
            log(f"  {k+1}/{len(files)} {f.name} rows_in={stats['rows_in']:,}")
    new = pd.concat(parts, ignore_index=True)
    new = new.groupby(["symbol", "date"], as_index=False).first() if new.duplicated(["symbol", "date"]).any() else new
    new["trade_date"] = pd.to_datetime(new.date.astype(str), format="%Y%m%d")
    new["vwap"] = new.amt / new.vol
    new = new.drop(columns=["date"])

    old = pd.read_csv(WORK / "daily_floorsheet.csv.gz", parse_dates=["trade_date"])
    new = new[old.columns]                                   # identical column order
    key = ["symbol", "trade_date"]
    oi, ni = old.set_index(key), new.set_index(key)
    both = oi.index.intersection(ni.index)
    rep = {"recovered_sessions": int(new.trade_date.nunique()),
           "recovered_symbol_sessions": int(len(new)),
           "recovered_trades": int(new.n.sum()),
           "bars_replaced": int(len(both)),
           "bars_added": int(len(ni.index.difference(oi.index))),
           "bars_kept_from_original_on_recovered_dates":
               int(oi.loc[oi.index.get_level_values(1).isin(new.trade_date.unique())].index.difference(ni.index).size),
           "trades_before_on_recovered_dates": int(old.loc[old.trade_date.isin(new.trade_date.unique()), "n"].sum()),
           "replaced_bars_with_fewer_trades_than_before":
               int((ni.loc[both, "n"].values < oi.loc[both, "n"].values).sum()) if len(both) else 0,
           "parse": stats, "broker_alpha_codes": amap}
    merged = pd.concat([old.set_index(key).drop(index=both, errors="ignore").reset_index(), new], ignore_index=True)
    merged = merged.sort_values(key).reset_index(drop=True)
    rep["daily_rows_before"], rep["daily_rows_after"] = int(len(old)), int(len(merged))
    rep["sessions_before"], rep["sessions_after"] = int(old.trade_date.nunique()), int(merged.trade_date.nunique())
    rep["trades_before"], rep["trades_after"] = int(old.n.sum()), int(merged.n.sum())
    rep["ohlc_violations"] = int(((merged.high < merged.low) | (merged.open > merged.high) | (merged.open < merged.low)
                                  | (merged.close > merged.high) | (merged.close < merged.low)).sum())
    assert not merged.duplicated(key).any(), "duplicate symbol-session after merge"
    merged.to_csv(WORK / "daily_floorsheet.csv.gz", index=False)
    save_json(rep, REPORTS / "stage02b_supplement_report.json")
    log(f"merged: daily rows {rep['daily_rows_before']:,} -> {rep['daily_rows_after']:,} | "
        f"sessions {rep['sessions_before']} -> {rep['sessions_after']} | "
        f"trades {rep['trades_before']:,} -> {rep['trades_after']:,} | "
        f"replaced {rep['bars_replaced']:,} added {rep['bars_added']:,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True)
    main(ap.parse_args().src)
