"""
Stage 01b  Validate, clean and de-duplicate the auxiliary sources, and audit them for look-ahead risk.

  ohlcv_data.sql                       livedata_adjusted   daily adjusted OHLCV (+ indices, sector ids)
  fundamental_data.sql                 new_fundamentals    quarterly history: EPS, P/E, P/B, ROE, D/E, revenue growth
  fundamental_*.sql (9 tables)         latest-quarter snapshots (checked below)
  nepse_prediction_featurestore.sql    precomputed feature store (audited, not used as model input)

Outputs (WORK/aux/*.pkl) and REPORTS/stage01b_aux_report.json.
Nothing here is used by a model unless it passes the point-in-time rules written in this file.
"""
import csv, io, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from common import *

AUX_DIR = Path(os.environ.get("NEPSE_AUX", "/mnt/user-data/attachments/cbb17335-a275-5aec-b41c-f2bdc2004a10"))
OUT = WORK / "aux"; OUT.mkdir(exist_ok=True)
PUBLICATION_LAG_DAYS = 45      # quarterly report assumed public 45 days after quarter end (regulatory deadline 30 days + buffer)

def find(pattern):
    hits = sorted(list(AUX_DIR.glob(pattern)) + list(UPLOADS.glob(pattern)))
    if not hits: raise FileNotFoundError(pattern)
    return hits[0]

# ----------------------------------------------------------------------------- generic MySQL dump parser
TYPE_NUM = re.compile(r"^(tiny|small|medium|big)?int|^decimal|^float|^double|^numeric", re.I)
TYPE_DATE = re.compile(r"^(date|datetime|timestamp)", re.I)

def parse_dump(path):
    """Return {table: DataFrame} for a phpMyAdmin/mysqldump file (one row per line, multi-row INSERTs)."""
    tables, cur, schema, buf = {}, None, {}, {}
    with open(path, "r", encoding="utf8", errors="replace") as f:
        in_create = None
        for line in f:
            s = line.strip()
            if s.startswith("CREATE TABLE"):
                in_create = re.search(r"`([^`]+)`", s).group(1); schema[in_create] = []; continue
            if in_create:
                m = re.match(r"`([^`]+)`\s+(\w+)", s)
                if m: schema[in_create].append((m.group(1), m.group(2)))
                if s.startswith(")"): in_create = None
                continue
            if s.startswith("INSERT INTO"):
                cur = re.search(r"INSERT INTO `?([^`\s]+)`?", s).group(1); buf.setdefault(cur, [])
                rest = s[s.find("VALUES") + 6:].strip()
                if rest.startswith("("): buf[cur].append(rest)
                continue
            if cur and s.startswith("("): buf[cur].append(s)
    for t, rows in buf.items():
        cols = [c for c, _ in schema[t]]; types = dict(schema[t])
        body = "\n".join(r[1:-2] if r.endswith(("),", ");")) else r[1:-1] for r in rows)
        df = pd.read_csv(io.StringIO(body), names=cols, quotechar="'", escapechar="\\", skipinitialspace=True,
                         na_values=["NULL"], keep_default_na=False, dtype=str, engine="c")
        for c in cols:
            if TYPE_NUM.match(types[c]): df[c] = pd.to_numeric(df[c], errors="coerce")
            elif TYPE_DATE.match(types[c]): df[c] = pd.to_datetime(df[c], errors="coerce")
        tables[t] = df
    return tables, schema

def profile(df, key, name):
    p = {"rows": int(len(df)), "columns": int(df.shape[1]), "id_duplicates": int(df["id"].duplicated().sum()) if "id" in df else None,
         "natural_key": key, "key_duplicates": int(df.duplicated(key).sum()),
         "key_duplicates_conflicting": int((df.duplicated(key, keep=False) & ~df.drop(columns=[c for c in ("id", "created_at", "updated_at") if c in df]).duplicated(keep=False)).sum()),
         "fully_null_columns": [c for c in df if df[c].isna().all()],
         "null_rate": {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0},
         "symbols": int(df["symbol"].nunique()) if "symbol" in df else None}
    log(f"{name}: rows={p['rows']:,} key_dups={p['key_duplicates']:,} (conflicting {p['key_duplicates_conflicting']:,}) symbols={p['symbols']}")
    return p

def dedup(df, key):
    """Keep one row per natural key: the most recently updated (then highest id)."""
    order = [c for c in ("updated_at", "id") if c in df]
    return df.sort_values(order).drop_duplicates(key, keep="last").reset_index(drop=True) if order else df.drop_duplicates(key, keep="last")

# ----------------------------------------------------------------------------- Nepali fiscal calendar
def quarter_end(fy, q):
    """Approximate AD quarter-end date for a Nepali fiscal year 'YYYY/YYYY' (BS) and quarter 1-4.
    The fiscal year starts on 1 Shrawan (~16 July); quarters end ~mid-Oct, ~mid-Jan, ~mid-Apr, ~mid-Jul."""
    try: y = int(str(fy).split("/")[0]) - 57; q = int(q)
    except Exception: return pd.NaT
    return {1: pd.Timestamp(y, 10, 17), 2: pd.Timestamp(y + 1, 1, 14), 3: pd.Timestamp(y + 1, 4, 13), 4: pd.Timestamp(y + 1, 7, 16)}.get(q, pd.NaT)

# ----------------------------------------------------------------------------- sources
def ohlcv(rep):
    t, _ = parse_dump(find("*ohlcv_data*.sql")); df = t["livedata_adjusted"]
    df["date"] = pd.to_datetime(df["date"], unit="s").dt.normalize()
    rep["ohlcv_raw"] = profile(df, ["symbol", "date"], "ohlcv")
    df = dedup(df, ["symbol", "date"])
    bad = (df.high < df.low) | (df.close <= 0) | (df.open <= 0) | (df.close > df.high * 1.0001) | (df.close < df.low * 0.9999) | (df.open > df.high * 1.0001) | (df.open < df.low * 0.9999)
    rep["ohlcv_clean"] = {"rows": int(len(df)), "ohlc_violations": int(bad.sum()), "zero_volume_rows": int((df.volume.fillna(0) <= 0).sum()),
                          "min_date": str(df.date.min().date()), "max_date": str(df.date.max().date()), "sessions": int(df.date.nunique()),
                          "symbols": int(df.symbol.nunique()), "symbols_without_sector": int(df[df.sector_id.isna()].symbol.nunique()),
                          "ohlc_violations_share_before_2019": float((df.date[bad] < "2019-01-01").mean()) if bad.any() else 0.0,
                          "ohlc_violations_share_index_rows": float(df.symbol[bad].isin(INDEX_SYMBOLS).mean()) if bad.any() else 0.0}
    df.to_pickle(OUT / "ohlcv_adjusted.pkl"); return df

def fundamentals(rep):
    t, _ = parse_dump(find("*fundamental_data*.sql")); df = t["new_fundamentals"]
    rep["fundamentals_raw"] = profile(df, ["symbol", "fiscal_year", "quarter"], "new_fundamentals")
    df = dedup(df, ["symbol", "fiscal_year", "quarter"])
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
    df["quarter_end"] = [quarter_end(f, q) for f, q in zip(df.fiscal_year, df.quarter)]
    df["available_date"] = df.quarter_end + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    lead = (df["quarter_end"] - df["date"]).dt.days
    rep["fundamentals_clean"] = {
        "rows": int(len(df)), "symbols": int(df.symbol.nunique()), "fiscal_years": sorted(df.fiscal_year.dropna().unique().tolist()),
        "unparseable_fiscal_period": int(df.quarter_end.isna().sum()),
        "date_column_days_before_quarter_end": {"median": float(lead.median()), "min": float(lead.min()), "max": float(lead.max()),
                                                "share_before_quarter_end": float((lead > 0).mean())},
        "publication_lag_days_used": PUBLICATION_LAG_DAYS,
        "note": "The table's `date` column falls before the quarter even ends, so it cannot be the publication date. "
                "Features use available_date = quarter end + publication lag instead."}
    daily_path = WORK / "daily_floorsheet.csv.gz"
    if daily_path.exists():
        d = pd.read_csv(daily_path, usecols=["symbol", "trade_date", "close"], parse_dates=["trade_date"]).sort_values("trade_date")
        x = df[(df.pe_ratio > 0) & (df.eps > 0)].assign(price_ref=lambda z: z.pe_ratio * z.eps)
        chk = {}
        for name, col in (("table_date", "date"), ("quarter_end", "quarter_end"), ("available_date", "available_date")):
            y = x.dropna(subset=[col]).sort_values(col)
            m = pd.merge_asof(y, d.rename(columns={"trade_date": col, "close": "raw"}), on=col, by="symbol", direction="backward",
                              tolerance=pd.Timedelta(days=10)).dropna(subset=["raw"])
            chk[name] = {"n": int(len(m)), "share_within_2pct": float((np.abs(m.price_ref / m.raw - 1) < 0.02).mean())}
        rep["fundamentals_clean"]["pe_price_reference_check"] = chk
    keep = ["symbol", "sector_id", "fiscal_year", "quarter", "quarter_end", "available_date", "date", "eps", "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "revenue_growth"]
    df[keep].to_pickle(OUT / "fundamentals_pit.pkl"); return df

def snapshots(rep):
    rep["snapshot_tables"] = {}
    for f in sorted(AUX_DIR.glob("*fundamental_*.sql")):
        if "fundamental_data" in f.name: continue
        t, _ = parse_dump(f)
        for name, df in t.items():
            p = profile(df, ["symbol", "fiscal_year", "quarter"], name)
            per = df.groupby("symbol")[["fiscal_year", "quarter"]].nunique().max(axis=1)
            p.update({"distinct_fiscal_periods": int(df[["fiscal_year", "quarter"]].drop_duplicates().dropna().shape[0]),
                      "periods": df.assign(p=df.fiscal_year.astype(str) + " Q" + df.quarter.astype(str)).p.value_counts().head(5).to_dict(),
                      "max_periods_per_symbol": int(per.max()) if len(per) else 0,
                      "created_at_range": [str(df.created_at.min()), str(df.created_at.max())]})
            p["usable_as_history"] = p["max_periods_per_symbol"] > 1
            rep["snapshot_tables"][name] = p
            dedup(df, ["symbol", "fiscal_year", "quarter"]).to_pickle(OUT / f"{name}.pkl")

def featurestore(rep, fund, ohl):
    t, _ = parse_dump(find("*featurestore*.sql")); df = t["nepse_prediction_featurestore"]; df["target_date"] = pd.to_datetime(df.target_date)
    rep["featurestore_raw"] = profile(df, ["symbol", "target_date"], "featurestore")
    df = dedup(df, ["symbol", "target_date"]).sort_values(["symbol", "target_date"])
    A = {"rows": int(len(df)), "symbols": int(df.symbol.nunique()), "min_date": str(df.target_date.min().date()), "max_date": str(df.target_date.max().date())}
    # 1) target alignment: target_return(t) should equal return_lag_1(t+1)
    nxt = df.groupby("symbol").return_lag_1.shift(-1); m = nxt.notna() & df.target_return.notna()
    A["target_equals_next_row_return_lag_1"] = float((np.abs(df.target_return[m] - nxt[m]) < 1e-9).mean())
    # 2) target vs adjusted close-to-close return from the OHLCV table
    o = ohl[["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy(); o["fwd1"] = o.groupby("symbol").close.shift(-1) / o.close - 1
    j = df.merge(o, left_on=["symbol", "target_date"], right_on=["symbol", "date"], how="inner")
    A["target_vs_ohlcv_next_close_return"] = {"matched_rows": int(len(j)), "share_abs_diff_lt_1e-4": float((np.abs(j.target_return - j.fwd1) < 1e-4).mean()) if len(j) else None}
    # 3) fundamentals timing: when does fundamental_eps change, relative to the quarter end of the report it switches to?
    fe = df[["symbol", "target_date", "fundamental_eps"]].dropna()
    ch = fe[fe.groupby("symbol").fundamental_eps.diff().fillna(0) != 0]
    f2 = fund[["symbol", "eps", "quarter_end", "date"]].dropna(subset=["eps"])
    mm = ch.merge(f2, left_on=["symbol", "fundamental_eps"], right_on=["symbol", "eps"], how="inner")
    mm = mm.sort_values("quarter_end").drop_duplicates(["symbol", "target_date"], keep="last")
    d = (mm.target_date - mm.quarter_end).dt.days
    A["fundamental_eps_switch_timing"] = {"switches_matched": int(len(mm)), "median_days_after_quarter_end": float(d.median()) if len(d) else None,
                                          "share_before_quarter_end": float((d < 0).mean()) if len(d) else None,
                                          "share_before_quarter_end_plus_30d": float((d < 30).mean()) if len(d) else None,
                                          "median_days_after_table_date_column": float((mm.target_date - mm.date).dt.days.median()) if len(mm) else None}
    # 4) single-feature check: how much does each feature know about target_return? (rank correlation)
    num = [c for c in df.columns if c not in ("id", "symbol", "target_date", "sector_id", "target_return", "created_at", "updated_at") and pd.api.types.is_numeric_dtype(df[c])]
    ic = {c: float(df[[c, "target_return"]].dropna().corr(method="spearman").iloc[0, 1]) for c in num if df[c].notna().sum() > 1000}
    A["top_abs_rank_corr_with_target"] = dict(sorted(ic.items(), key=lambda kv: -abs(kv[1]))[:8])
    A["columns"] = num
    rep["featurestore_audit"] = A
    df.to_pickle(OUT / "featurestore.pkl")

def run():
    rep = {}
    ohl = ohlcv(rep); fund = fundamentals(rep); snapshots(rep); featurestore(rep, fund, ohl)
    save_json(rep, REPORTS / "stage01b_aux_report.json"); log("aux sources done -> stage01b_aux_report.json")

if __name__ == "__main__":
    run()
