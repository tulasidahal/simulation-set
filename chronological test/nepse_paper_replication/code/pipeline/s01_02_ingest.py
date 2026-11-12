"""
Stage 01  Validate source datasets: inventory, schema, checksums, per-column missingness.
Stage 02  Clean, optimize and de-duplicate the floorsheets, then aggregate to symbol-sessions.

Two resumable phases (safe to kill and re-run; finished work is never redone):

  Phase A  parse   Stream each zip -> drop invalid rows -> write compact binary trade records (32 bytes/trade)
                   into one bucket file per month of the transaction number (sn). A checkpoint after every batch
                   records the byte size of each bucket; on resume, buckets are truncated back to the last
                   checkpoint, so an interrupted batch is never written twice.
  Phase B  clean   For each month bucket (about 1M trades, small memory): sort by sn, remove duplicate sn
                   (classified as exact or conflicting), run reconciliation checks, save the cleaned trades
                   as clean/YYYYMM.npy, and aggregate daily OHLCV + broker features. Months already done are skipped.

Because sn starts with the trade date (YYYYMMDD...), bucketing by sn month puts every copy of a transaction
in the same bucket: duplicate removal per bucket is a global duplicate removal across all three parts.

  python s01_02_ingest.py              # run / resume both phases
  python s01_02_ingest.py --restart    # wipe intermediate files and start over
"""
import argparse, hashlib, io, os, re, shutil, zipfile
import numpy as np, pandas as pd
from common import *

COLS = ["sn", "symbol", "buyer", "seller", "quantity", "rate", "amount", "trade_date", "trade_time"]
CHUNK_BYTES = 150_000_000
REC = np.dtype([("sn", "<i8"), ("date", "<i4"), ("sym", "<i2"), ("buyer", "<i2"), ("seller", "<i2"),
                ("qty", "<f4"), ("rate", "<f8"), ("src", "i1"), ("pad", "i1")])          # 32 bytes per trade
RAW, CLEAN, DAYS = WORK / "trades_raw", WORK / "trades_clean", WORK / "daily_parts"
PROG = WORK / "ingest_progress.json"

def atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp"); save_json(obj, tmp); os.replace(tmp, path)

def sha256(path, blk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(blk): h.update(b)
    return h.hexdigest()

# ----------------------------------------------------------------------------- stage 01
def inventory():
    inv = []
    for zp in ZIPS:
        with zipfile.ZipFile(zp) as z:
            bad = z.testzip()                                            # CRC check of every member
            members = [{"name": i.filename, "uncompressed_bytes": i.file_size, "crc32": f"{i.CRC:08x}"} for i in z.infolist()]
            with z.open(z.namelist()[0]) as f: first = f.readline(2_000_000)[:800].decode("utf8", "replace")
        m = re.search(r"INSERT INTO (\S+) \(([^)]*)\)", first)
        cols = [c.strip(" `") for c in m.group(2).split(",")] if m else None
        inv.append({"file": zp.name, "zip_bytes": zp.stat().st_size, "sha256": sha256(zp), "crc_ok": bad is None,
                    "members": members, "target_table": m.group(1) if m else None, "schema_columns": cols})
        log(f"inventory {zp.name}: {inv[-1]['zip_bytes']/1e6:.0f} MB, crc_ok={bad is None}, schema_ok={cols == COLS}")
        assert cols == COLS, f"unexpected schema in {zp.name}: {cols}"
        assert bad is None, f"corrupt member {bad} in {zp.name}"
    return inv

# ----------------------------------------------------------------------------- phase A
def iter_batches(zip_path):
    """Yield CSV byte blocks (one trade per line) from a dump of multi-row INSERT statements."""
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        buf, size = [], 0
        for line in f:
            line = line.rstrip(b"\r\n")
            if not line.startswith(b"INSERT"): continue
            body = line[line.find(b"VALUES (") + 8:]
            body = body[:-2] if body.endswith(b");") else body[:-1] if body.endswith(b")") else body
            body = body.replace(b"), (", b"\n"); buf.append(body); size += len(body)
            if size >= CHUNK_BYTES: yield b"\n".join(buf); buf, size = [], 0
        if buf: yield b"\n".join(buf)

def new_progress(inv):
    return {"inventory": inv, "symbols": {}, "batches_done": {zp.name: 0 for zp in ZIPS}, "bucket_bytes": {}, "phase_a_done": False,
            "stats": {"rows_in": 0, "rows_valid": 0, "nulls": {c: 0 for c in COLS}, "dropped_required_null": 0, "dropped_nonpositive": 0,
                      "dropped_bad_broker": 0, "dropped_bad_date": 0, "amount_mismatch": 0, "sn_date_mismatch": 0, "fractional_qty": 0,
                      "per_part": {zp.name: {"rows_in": 0, "rows_valid": 0, "min_date": None, "max_date": None} for zp in ZIPS}}}

ALPHA_BASE = 10000      # non-numeric broker codes (e.g. 'D01', a dealer account) are real trades: give them IDs >= 10000

def broker_codes(col, prog):
    """Numeric broker IDs stay as-is; alphanumeric codes are mapped to stable IDs (ALPHA_BASE + k) and recorded."""
    num = pd.to_numeric(col, errors="coerce")
    alpha = num.isna() & col.notna() & (col.str.strip() != "")
    if alpha.any():
        amap = prog.setdefault("broker_alpha_codes", {})
        for code in pd.unique(col[alpha].str.strip()):
            if code not in amap: amap[code] = ALPHA_BASE + len(amap)
        num = num.where(~alpha, col.str.strip().map(amap))
        prog["stats"]["alpha_broker_fields"] = prog["stats"].get("alpha_broker_fields", 0) + int(alpha.sum())
    return num.where(num.between(0, 32000))

def to_records(ch, part_no, prog):
    S = prog["stats"]; P = S["per_part"][ZIPS[part_no - 1].name]
    S["rows_in"] += len(ch); P["rows_in"] += len(ch)
    for c in COLS: S["nulls"][c] += int(ch[c].isna().sum())
    n0 = len(ch); ch = ch.dropna(subset=COLS[:8]); S["dropped_required_null"] += n0 - len(ch)
    n0 = len(ch); ch = ch[(ch.quantity > 0) & (ch.rate > 0) & (ch.amount > 0)]; S["dropped_nonpositive"] += n0 - len(ch)
    b, s = broker_codes(ch.buyer, prog), broker_codes(ch.seller, prog)
    ok = b.notna() & s.notna(); S["dropped_bad_broker"] += int((~ok).sum()); ch, b, s = ch[ok], b[ok], s[ok]
    d = pd.to_datetime(ch.trade_date, format="%Y-%m-%d", errors="coerce"); ok = d.notna(); S["dropped_bad_date"] += int((~ok).sum())
    ch, b, s, d = ch[ok], b[ok], s[ok], d[ok]
    dint = (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).values.astype("int32")
    S["amount_mismatch"] += int((np.abs(ch.amount.values - ch.quantity.values * ch.rate.values) > 1.0).sum())
    S["sn_date_mismatch"] += int((ch.sn.values // 10**8 != dint).sum())
    S["fractional_qty"] += int((ch.quantity.values % 1 != 0).sum())
    smap = prog["symbols"]
    for sym in pd.unique(ch.symbol):
        if sym not in smap: smap[sym] = len(smap)
    r = np.empty(len(ch), REC)
    r["sn"], r["date"] = ch.sn.values, dint
    r["sym"] = ch.symbol.map(smap).values.astype("int16"); r["buyer"], r["seller"] = b.values.astype("int16"), s.values.astype("int16")
    r["qty"], r["rate"], r["src"], r["pad"] = ch.quantity.values.astype("float32"), ch.rate.values, part_no, 0
    S["rows_valid"] += len(r); P["rows_valid"] += len(r)
    if len(r):
        mn, mx = str(d.min().date()), str(d.max().date())
        P["min_date"] = mn if P["min_date"] is None or mn < P["min_date"] else P["min_date"]
        P["max_date"] = mx if P["max_date"] is None or mx > P["max_date"] else P["max_date"]
    return r

def phase_a(prog):
    RAW.mkdir(parents=True, exist_ok=True)
    # roll every bucket back to the last checkpoint (drops a half-written batch)
    for f in RAW.glob("*.bin"):
        keep = prog["bucket_bytes"].get(f.stem, 0)
        if f.stat().st_size != keep:
            log(f"resume: truncating {f.name} {f.stat().st_size:,} -> {keep:,} bytes")
            with open(f, "r+b") as fh: fh.truncate(keep)
    t0 = time.time()
    for part_no, zp in enumerate(ZIPS, 1):
        done = prog["batches_done"][zp.name]
        for i, blob in enumerate(iter_batches(zp)):
            if i < done: continue
            ch = pd.read_csv(io.BytesIO(blob), names=COLS, quotechar="'", skipinitialspace=True, na_values=["NULL"], on_bad_lines="skip",
                             dtype={"sn": "int64", "symbol": "string", "buyer": "string", "seller": "string", "quantity": "float64",
                                    "rate": "float64", "amount": "float64", "trade_date": "string", "trade_time": "string"}, engine="c")
            del blob
            r = to_records(ch, part_no, prog); del ch
            month = r["sn"] // 10**10                                    # YYYYMM taken from the transaction number
            bad = (month < 200001) | (month > 210012); month[bad] = 0       # malformed sn -> bucket 000000
            order = np.argsort(month, kind="stable"); r, month = r[order], month[order]
            keys, starts = np.unique(month, return_index=True); ends = np.append(starts[1:], len(r))
            for k, a, e in zip(keys, starts, ends):
                name = f"{int(k):06d}"
                with open(RAW / f"{name}.bin", "ab") as fh: r[a:e].tofile(fh)
                prog["bucket_bytes"][name] = prog["bucket_bytes"].get(name, 0) + (e - a) * REC.itemsize
            prog["batches_done"][zp.name] = i + 1; atomic_json(prog, PROG)        # checkpoint
            S = prog["stats"]
            log(f"A part{part_no} batch{i}: rows_in={S['rows_in']:,} valid={S['rows_valid']:,} t={time.time()-t0:.0f}s")
    prog["phase_a_done"] = True; atomic_json(prog, PROG)

# ----------------------------------------------------------------------------- phase B
def broker_features(df, col, side):
    B = df.groupby(["sym", "date", col], sort=False).qty.sum().reset_index()
    B["share"] = B.qty / B.groupby(["sym", "date"], sort=False).qty.transform("sum")
    B = B.sort_values(["sym", "date", "qty"], ascending=[True, True, False]); rk = B.groupby(["sym", "date"], sort=False).cumcount().values
    B["sh2"] = B.share ** 2; B["top1"] = np.where(rk == 0, B.share, 0.0); B["top5"] = np.where(rk < 5, B.share, 0.0)
    return B.groupby(["sym", "date"], sort=False).agg(**{f"{side}_hhi": ("sh2", "sum"), f"{side}_top1": ("top1", "sum"),
                                                        f"{side}_top5": ("top5", "sum"), f"n_{side}s": (col, "size")})

def clean_month(name):
    r = np.fromfile(RAW / f"{name}.bin", dtype=REC)
    st = {"month": name, "rows": int(len(r))}
    r = r[np.argsort(r["sn"], kind="stable")]                            # stable: earlier part/batch wins ties
    dup = np.zeros(len(r), bool); dup[1:] = r["sn"][1:] == r["sn"][:-1]
    st["dup_rows"] = int(dup.sum())
    if dup.any():
        keep_idx = np.where(dup)[0] - 1
        while True:                                                       # point each dropped row at the first copy
            m = dup[keep_idx]
            if not m.any(): break
            keep_idx[m] -= 1
        a, b = r[keep_idx], r[dup]
        same = (a["date"] == b["date"]) & (a["sym"] == b["sym"]) & (a["buyer"] == b["buyer"]) & (a["seller"] == b["seller"]) & (a["qty"] == b["qty"]) & (np.abs(a["rate"] - b["rate"]) < 1e-6)
        st["dup_exact"], st["dup_conflicting"] = int(same.sum()), int((~same).sum())
        pairs = pd.Series(list(zip(a["src"].tolist(), b["src"].tolist()))).value_counts()
        st["dup_by_parts"] = {f"part{x}->part{y}": int(c) for (x, y), c in pairs.items()}
    else:
        st["dup_exact"] = st["dup_conflicting"] = 0; st["dup_by_parts"] = {}
    r = r[~dup]; st["rows_clean"] = int(len(r))
    np.save(CLEAN / f"{name}.npy", r)
    df = pd.DataFrame({k: r[k] for k in ("sn", "date", "sym", "buyer", "seller", "qty", "rate")})
    df["amt"] = df.qty.astype("float64") * df.rate; df["q2"] = df.qty.astype("float64") ** 2
    df["self_q"] = np.where(df.buyer == df.seller, df.qty, 0.0); df["large_q"] = np.where(df.qty >= LARGE_TRADE, df.qty, 0.0)
    g = df.groupby(["sym", "date"], sort=False)                           # rows are in sn order -> first/last = open/close
    day = g.agg(n=("sn", "size"), vol=("qty", "sum"), amt=("amt", "sum"), open=("rate", "first"), high=("rate", "max"), low=("rate", "min"),
                close=("rate", "last"), q2=("q2", "sum"), self_q=("self_q", "sum"), large_q=("large_q", "sum"))
    day = day.join(broker_features(df, "buyer", "buyer")).join(broker_features(df, "seller", "seller")).reset_index()
    day.to_pickle(DAYS / f"{name}.pkl")
    return st

def phase_b(prog):
    CLEAN.mkdir(exist_ok=True); DAYS.mkdir(exist_ok=True)
    bmeta = WORK / "dedup_months.json"; done = load_json(bmeta) if bmeta.exists() else {}
    months = sorted(f.stem for f in RAW.glob("*.bin")); t0 = time.time()
    for k, name in enumerate(months):
        if name in done and (DAYS / f"{name}.pkl").exists(): continue
        done[name] = clean_month(name); atomic_json(done, bmeta)
        d = done[name]
        if d["dup_rows"] or k % 12 == 0:
            log(f"B {name}: rows={d['rows']:,} dups={d['dup_rows']:,} (exact {d['dup_exact']:,}, conflicting {d['dup_conflicting']:,}) t={time.time()-t0:.0f}s")
    return done

def finalize(prog, months):
    inv_map = {v: k for k, v in prog["symbols"].items()}
    daily = pd.concat([pd.read_pickle(DAYS / f"{m}.pkl") for m in sorted(months)], ignore_index=True)
    assert not daily.duplicated(["sym", "date"]).any(), "a symbol-session spans two buckets"
    daily["symbol"] = daily.sym.map(inv_map); daily["trade_date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
    daily["vwap"] = daily.amt / daily.vol
    daily = daily.drop(columns=["sym", "date"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    S = dict(prog["stats"]); M = months
    S["dedup"] = {"rows_before": sum(m["rows"] for m in M.values()), "rows_after": sum(m["rows_clean"] for m in M.values()),
                  "duplicates_removed": sum(m["dup_rows"] for m in M.values()), "exact": sum(m["dup_exact"] for m in M.values()),
                  "conflicting": sum(m["dup_conflicting"] for m in M.values()),
                  "months_with_duplicates": {k: v["dup_rows"] for k, v in M.items() if v["dup_rows"]}, "by_parts": {}}
    for v in M.values():
        for k, c in v["dup_by_parts"].items(): S["dedup"]["by_parts"][k] = S["dedup"]["by_parts"].get(k, 0) + c
    S["malformed_sn_rows"] = M.get("000000", {}).get("rows", 0)
    S["rows_kept"] = S["dedup"]["rows_after"]; S["rows_dropped_total"] = S["rows_in"] - S["rows_kept"]
    S["daily_rows"] = len(daily); S["symbols"] = int(daily.symbol.nunique()); S["sessions"] = int(daily.trade_date.nunique())
    S["min_date"], S["max_date"] = str(daily.trade_date.min().date()), str(daily.trade_date.max().date())
    S["ohlc_violations"] = int(((daily.high < daily.low) | (daily.open > daily.high) | (daily.open < daily.low) | (daily.close > daily.high) | (daily.close < daily.low)).sum())
    S["trades_reconcile"] = int(daily.n.sum()) == S["rows_kept"]
    S["clean_trades_bytes"] = sum(f.stat().st_size for f in CLEAN.glob("*.npy"))
    daily.to_csv(WORK / "daily_floorsheet.csv.gz", index=False)
    save_json({"inventory": prog["inventory"], "cleaning": S, "broker_alpha_codes": prog.get("broker_alpha_codes", {})}, REPORTS / "stage01_02_ingest_report.json")
    log(f"DONE trades in={S['rows_in']:,} clean={S['rows_kept']:,} dups removed={S['dedup']['duplicates_removed']:,} | "
        f"daily rows={len(daily):,} symbols={S['symbols']} sessions={S['sessions']} | trades reconcile={S['trades_reconcile']}")

def run(restart=False):
    if restart:
        for p in (RAW, CLEAN, DAYS): shutil.rmtree(p, ignore_errors=True)
        for p in (PROG, WORK / "dedup_months.json"): p.unlink(missing_ok=True)
    prog = load_json(PROG) if PROG.exists() else new_progress(inventory())
    if not PROG.exists(): atomic_json(prog, PROG)
    if not prog["phase_a_done"]: phase_a(prog)
    else: log("phase A already complete")
    months = phase_b(prog)
    finalize(prog, months)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--restart", action="store_true"); run(ap.parse_args().restart)
