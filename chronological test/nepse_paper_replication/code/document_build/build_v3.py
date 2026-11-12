#!/usr/bin/env python3
"""Build the recovered-data report (three arms), reusing the technical documentation's styles."""
import json, re, shutil, zipfile
from pathlib import Path

SC = Path("/tmp/claude-0/-home-claude/f7ea644d-6066-5e86-aa53-2f9a26b2ec03/scratchpad/td")
TPL = SC / "x"
OUTD = SC / "v3"; OUT = SC / "v3.docx"
A_ = Path("/mnt/user-data/outputs/nepse_results")       # original zips, gap kept
B_ = Path("/home/claude/work_nogap/reports")            # original zips, gap excluded
C_ = Path("/home/claude/work_supp/reports")             # recovered files, remainder excluded

def L(d, f):
    with open(d / f) as fh: return json.load(fh)

T12 = {"A": L(A_, "stage12_final_test.json"), "B": L(B_, "stage12_final_test.json"), "C": L(C_, "stage12_final_test.json")}
T13 = {"A": L(A_, "stage13_backtest.json"), "B": L(B_, "stage13_backtest.json"), "C": L(C_, "stage13_backtest.json")}
P34A, P34C = L(A_, "stage03_04_panel_report.json"), L(C_, "stage03_04_panel_report.json")
P34B = L(B_, "stage03_04_panel_report.json")
P05C = L(C_, "stage05_audit_report.json")
SUP = L(C_, "stage02b_supplement_report.json")
THIN = P34C["bars"]["thin_floorsheet"]
ARMS = [("A", "original"), ("B", "excluded"), ("C", "recovered")]

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))

TOK = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")

def runs(text):
    o = []
    for part in TOK.split(str(text)):
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            o.append('<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part[2:-2]))
        elif part.startswith("`") and part.endswith("`"):
            o.append('<w:r><w:rPr><w:rStyle w:val="CodeChar"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part[1:-1]))
        else:
            o.append('<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part))
    return "".join(o)

def h(l, t): return '<w:p><w:pPr><w:pStyle w:val="Heading%d"/></w:pPr><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % (l, esc(t))
def ti(t):   return '<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % esc(t)
def p(t):    return "<w:p><w:pPr></w:pPr>%s</w:p>" % runs(t)
def li(t, n=10): return ('<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/>'
                         '<w:numId w:val="%d"/></w:numPr></w:pPr>%s</w:p>' % (n, runs(t)))

def table(w, rows, al=None, aligns=None):
    al = aligns or al
    al = al or ["left"] * len(w)
    g = "".join('<w:gridCol w:w="%d"/>' % x for x in w)
    o = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/><w:tblLayout w:type="fixed"/>'
         '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
         '</w:tblPr><w:tblGrid>%s</w:tblGrid>' % g]
    for ri, row in enumerate(rows):
        hd = ri == 0
        o.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>' if hd else "<w:tr>")
        for ci, c in enumerate(row):
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>' if hd else ""
            jc = '<w:jc w:val="right"/>' if al[ci] == "right" and not hd else ""
            body = ('<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(str(c))) if hd else runs(c)
            o.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s</w:tcPr><w:p><w:pPr>%s</w:pPr>%s</w:p></w:tc>' % (w[ci], shd, jc, body))
        o.append("</w:tr>")
    return "".join(o) + "</w:tbl>"

pc = lambda x, d=1: "-" if x is None else f"{100*x:.{d}f}%"
R, LF = "right", "left"

B = [ti("NEPSE Forecasting Study — Recovered Floorsheet Days and Their Effect")]
B.append(p("The three original floorsheet archives were missing or near-empty on 280 of 1,858 sessions. "
           "Eighty-eight recovered SQL day files have now been merged in. This report records what they "
           "contained, what is still missing, and what the recovery does to every headline result. Three "
           "runs are compared on an identical, untouched final test: the original archives with the thin "
           "sessions kept, the original archives with them excluded, and the supplemented archives with "
           "only the still-missing sessions excluded."))

B.append(h(2, "Summary"))
B.append(p("**The recovery works.** Excluding the thin sessions had cost accuracy at 3, 5 and 10 sessions; "
           "feeding the real data back recovers almost all of it while keeping the clean design. At 10 "
           "sessions accuracy goes 55.3% → 50.5% → 54.8% across the three runs, and the ranker's rank-IC "
           "returns to where it was with the full original sample."))
B.append(p("**The 1-session result is stable across all three.** 59.0%, 58.6% and 58.5% against a 55.3% "
           "majority baseline, each with a bootstrap probability below 0.001 of being a baseline-level "
           "result. Its calibration is best in the recovered run (expected calibration error 2.50% against "
           "3.23% and 3.15%), and the 70%-confidence operating point covers more of the test than either "
           "earlier run: 301 predictions at 80.4% accuracy, against 243 at 81.1% and 180 at 83.9%."))
B.append(p("**The recovered files are clean.** 3,687,558 trades parsed with no unparsed lines, no rows "
           "dropped for nulls, non-positive values or bad broker codes, no duplicate transaction numbers, "
           "no amount mismatches, no serial-number/date mismatches and no OHLC violations. They are a "
           "strict superset of what the originals held for those dates: no bar had to be kept from the old "
           "data, and not one replaced bar had fewer trades than before."))
B.append(p("**One block is still missing**: 2022-07-28 to 2023-03-30, 154 sessions. With its 19-session "
           "buffer and the isolated thin days, 173 sessions remain excluded from training, down from 317."))

B.append(h(2, "What the recovered files contained"))
B.append(table([3400, 2980, 2980], [
    ["", "Before", "After"],
    ["Trades in the floorsheet", f"{SUP['trades_before']:,}", f"{SUP['trades_after']:,}"],
    ["Daily symbol-session bars", f"{SUP['daily_rows_before']:,}", f"{SUP['daily_rows_after']:,}"],
    ["Trades on the 97 recovered sessions", f"{SUP['trades_before_on_recovered_dates']:,}", f"{SUP['recovered_trades']:,}"],
    ["Sessions below 50% coverage", "280", "183"],
    ["Panel floorsheet coverage", pc(P34A["floorsheet_coverage_eligible"]), pc(P34C["floorsheet_coverage_eligible"])],
], [LF, R, R]))
B.append(p(f"The merge replaced {SUP['bars_replaced']:,} existing bars and added {SUP['bars_added']:,} new "
           f"ones across {SUP['recovered_sessions']} sessions. {SUP['parse']['alpha_broker_fields']:,} broker "
           "fields carried the alphabetic dealer code `D01`, mapped to the same identifier the original "
           "ingest used."))

B.append(h(2, "Coverage, block by block"))
B.append(table([2950, 1150, 1500, 3760], [
    ["Block", "Sessions", "Status", "Detail"],
    ["2022-02-10 → 2022-02-28", "13", "**recovered**", "All 13 sessions supplied in full"],
    ["2022-03-31 → 2022-07-27", "84", "**recovered**", "Supplied except 10 Fridays (see below)"],
    ["2022-07-28 → 2023-03-30", "154", "still missing", "0–21% coverage; the one remaining structural block"],
    ["2024-03-17 → 2024-03-31", "10", "**recovered**", "All 10 sessions, 550,999 trades"],
], aligns=[LF, R, LF, LF]))
B.append(p("**Still worth chasing.** The 154-session block above; ten Fridays inside the recovered stretch "
           "— 2022-05-20, 05-27, 06-03, 06-10, 06-17, 06-24, 07-01, 07-08, 07-15 and 07-22; and the "
           "isolated days 2018-11-11, 2018-12-31, 2019-01-14, 2019-01-22, 2019-03-09, 2019-09-12, "
           "2020-07-26, 2020-08-26, 2020-08-30, 2021-03-08, 2021-10-05, 2025-04-01, 2025-07-22, "
           "2026-03-02, 2026-08-28, 2026-09-01, 2026-09-04, 2026-09-08 and 2026-10-23."))

B.append(h(2, "The exclusion rule, applied to what is still missing"))
B.append(p(f"The rule is unchanged: a session is thin when fewer than {pc(THIN['min_coverage'],0)} of its "
           f"eligible stocks appear in the floorsheet; runs of at least {THIN['min_block']} consecutive thin "
           f"sessions are dropped along with the {THIN['buffer_sessions']} sessions that follow, so no "
           "rolling or sequence window reads a thin session. Because the data is now there, the rule simply "
           "has less to remove."))
B.append(table([3400, 2980, 2980], [
    ["", "Original archives", "Supplemented"],
    ["Sessions excluded", "317", f"{THIN['excluded_sessions']}"],
    ["Structural blocks found", "3", f"{len(THIN['blocks'])}"],
    ["Eligible stock-sessions", f"{P34B['rows_eligible']:,}", f"{P34C['rows_eligible']:,}"],
    ["Sessions in the panel", f"{P34B['sessions_eligible']:,}", f"{P34C['sessions_eligible']:,}"],
    ["Floorsheet coverage after exclusion", pc(P34B["floorsheet_coverage_eligible"]), pc(P34C["floorsheet_coverage_eligible"])],
], aligns=[LF, R, R]))
B.append(p(f"Training recovers {P34C['rows_eligible'] - P34B['rows_eligible']:,} stock-days against the "
           "excluded run. The walk-forward design keeps eight folds, now with 109-session validation blocks "
           "instead of 91. The leakage and split audit passes all ten checks: rebuilding every feature from "
           "sources truncated at 2024-10-28 reproduces the stored values exactly (max |difference| 0.0, no "
           "missing-data-pattern mismatches, 60 features × 278,781 rows)."))
B.append(p(f"**The final test window is identical in all three runs** — {T12['C']['test_dates'][0]} to "
           f"{T12['C']['test_dates'][1]}, 250 sessions, and exactly "
           f"{T12['C']['horizons']['1d']['n']:,} scored rows at 1 session, "
           f"{T12['C']['horizons']['10d']['n']:,} at 10. Every number below is measured on the same rows."))

B.append(h(2, "Final-test accuracy"))
rows = [["H", "Run", "Candidate", "Base", "Acc", "95% interval", "AUC", "Logloss", "ECE"]]
for hh in ("1d", "3d", "5d", "10d"):
    for k, nm in ARMS:
        ft = T12[k]; sel = ft["selection"][hh]["primary"]; c = ft["horizons"][hh]["candidates"][sel]
        ci = c["ci"]["accuracy"]
        rows.append([hh if k == "A" else "", nm, sel.replace("__", " + "), pc(c["majority_baseline_acc"]),
                     pc(c["accuracy"]), f"{pc(ci[0])[:-1]}–{pc(ci[1])}", f"{c['auc']:.3f}",
                     f"{c['logloss']:.4f}", pc(c["ece"], 2)])
B.append(table([700, 1150, 1250, 850, 850, 1450, 760, 1000, 800], rows, aligns=[LF, LF, LF, R, R, R, R, R, R]))
B.append(p("`original` = the three original archives with the thin sessions kept; `excluded` = the same "
           "archives with them dropped; `recovered` = the supplemented archives with only the still-missing "
           "sessions dropped. Each run scores the candidate its own out-of-fold log-loss selected, locked "
           "before the test was touched."))
B.append(p("Measured like for like on the `weighted` ensemble in all three: 1 session 59.0% / 58.6% / "
           "58.5%; 3 sessions 59.2% / 57.5% / 58.2%; 5 sessions 58.3% / 55.2% / 57.2%; 10 sessions 55.3% / "
           "50.5% / 54.8%. The 5-session `stacking + platt` row in the excluded run, with an AUC of 0.473, "
           "is a selection failure rather than a data effect — the `weighted` ensemble scored 0.578 on the "
           "same rows. The recovered run's own selection avoids it."))

B.append(h(2, "Significance, calibration and the ranker"))
rows = [["Horizon", "Run", "p (acc ≤ base)", "DM vs climatology, Holm p", "Rank-IC", "IC-IR", "Spread"]]
for hh in ("1d", "3d", "5d", "10d"):
    for k, nm in ARMS:
        ft = T12[k]; sel = ft["selection"][hh]["primary"]; c = ft["horizons"][hh]["candidates"][sel]
        H = ft["horizons"][hh]; r = H["ranker"]
        dm = H["dm"].get(f"{sel}_vs_climatology") or list(H["dm"].values())[0]
        pg = c.get("p_acc_gt_majority")
        rows.append([hh if k == "A" else "", nm, "< 0.001" if pg == 0 else f"{pg:.3f}",
                     f"{dm['p_holm']:.3g}" if dm["p_holm"] < 1 else "1.0",
                     f"{r['mean_ic']:.3f}", f"{r['ic_ir']:.2f}", pc(r["spread"], 2)])
B.append(table([1000, 1150, 1600, 2000, 1000, 850, 1060], rows, aligns=[LF, LF, R, R, R, R, R]))
B.append(p("The ranker is the clearest signal that the recovery worked. Exclusion had pulled mean rank-IC "
           "down at every horizon (0.270 → 0.259 at 3 sessions, 0.248 → 0.234 at 5, 0.213 → 0.195 at 10); "
           "with the real data back it returns to 0.272, 0.250 and 0.214 — at or slightly above the "
           "original full-sample run, on a cleaner panel."))

B.append(h(2, "Selective prediction at 70% confidence"))
rows = [["Horizon", "Run", "Coverage", "Predictions", "Accuracy"]]
for hh in ("1d", "3d"):
    for k, nm in ARMS:
        ft = T12[k]; sel = ft["selection"][hh]["primary"]
        s = ft["horizons"][hh]["candidates"][sel].get("sel_0.70", {})
        rows.append([hh if k == "A" else "", nm, pc(s.get("coverage", 0), 2), f"{s.get('n', 0):,}",
                     pc(s["accuracy"]) if s.get("accuracy") is not None else "—"])
B.append(table([1300, 1600, 1800, 2200, 2460], rows, aligns=[LF, LF, R, R, R]))
B.append(p("The 70% operating point remains a narrow one. At 1 session the recovered run makes 301 calls "
           "out of 60,526 rows — half a percent of the test — at 80.4% accuracy. That is the most coverage "
           "of the three runs, but it is still 301 predictions, and the claim that belongs in a paper is "
           "the selective one with its coverage stated, not a headline accuracy."))

B.append(h(2, "Backtest"))
B.append(p("Long-only, entry at the next session's open, 10 bp slippage a side plus commission and the "
           "7.5% short-term capital-gains tax."))
rows = [["Strategy", "Run", "Trades", "Mean net / trade", "95% interval", "P(mean ≤ 0)"]]
for hz, key, lab in (("1d", "thr_0.55", "1 session, p ≥ 0.55"), ("3d", "thr_0.60", "3 sessions, p ≥ 0.60"),
                     ("10d", "thr_0.60", "10 sessions, p ≥ 0.60"), ("1d", "ranker_top10", "1 session, ranker top 10")):
    for k, nm in ARMS:
        t = T13[k]["test"].get(hz, {}).get(key)
        if not t or "mean_net_per_trade" not in t: continue
        ci = t.get("ci_mean_net")
        rows.append([lab if k == "A" else "", nm, f"{t.get('trades',0):,}", pc(t["mean_net_per_trade"], 3),
                     f"{pc(ci[0],3)[:-1]} to {pc(ci[1],3)}" if ci else "—",
                     f"{t.get('p_mean_net_le_0', float('nan')):.3f}"])
B.append(table([2500, 1150, 950, 1650, 2100, 960], rows, aligns=[LF, LF, R, R, R, R]))
B.append(p("Unchanged in character across all three runs: no strategy earns a net return distinguishable "
           "from zero after costs, and the ranker's top-10 portfolio loses about half a percent a trade "
           "whatever the training data. Better features did not become a tradable edge."))

B.append(h(2, "What to report"))
B.append(p("**Report the recovered run as the primary result.** It has the cleanest provenance of the "
           "three: every training row carries real floorsheet data, the one period that does not is "
           "excluded by a measured rule, and the exclusion is now small enough that it costs almost "
           "nothing. The headline is 58.5% at 1 session against a 55.3% baseline on 60,526 held-out rows, "
           "with calibration error of 2.5%."))
B.append(p("**Keep the other two as the sensitivity analysis.** They answer the obvious reviewer question "
           "— what happens if you drop the period where the microstructure features are missing, and what "
           "happens if you keep it — and they show the answer is small at 1 session and material at 10."))
B.append(p("**Say what is still missing.** 154 sessions from 2022-07-28 to 2023-03-30 are absent from the "
           "archive, and the paper should name them rather than let a reader infer continuous coverage "
           "from 2018 to 2026."))
B.append(p("**Both the models and the stand-in caveat are unchanged.** CatBoost, LightGBM, XGBoost, the "
           "Transformer and the TFT could not be installed in this environment; all three runs use "
           "scikit-learn stand-ins. If the native lite-panel run is repeated, it should start from the "
           "supplemented floorsheet and apply the same rule."))

B.append(h(2, "Reproduction"))
B.append(p("The recovered files are parsed by a new stage, `s02b_supplement.py`, which applies the same "
           "validation, de-duplication and daily aggregation as stage 02 and merges the result into "
           "`daily_floorsheet.csv.gz`, replacing a bar only when the recovered archive contains that "
           "symbol-session. Its report is `stage02b_supplement_report.json`."))
B.append(p("`python s02b_supplement.py --src <dir>` then `NEPSE_EXCLUDE_GAP=1 FAMILY=sklearn START=3 bash "
           "run_pipeline.sh`. Stages 1–2 (the 91-million-trade ingest) are not rerun; the cleaned daily "
           "floorsheet is reused and supplemented. The rebuild was verified against the earlier run before "
           "any change: panel rows 405,491, eligible 360,943, 362 symbols, 1,855 sessions and all four "
           "label counts reproduced exactly."))

BODY = "".join(B)

if OUTD.exists(): shutil.rmtree(OUTD)
shutil.copytree(TPL, OUTD)
doc = (OUTD / "word/document.xml").read_text(encoding="utf-8")
head = doc[:doc.index("<w:body>") + len("<w:body>")]
sect = doc[doc.rindex("<w:sectPr"):]
(OUTD / "word/document.xml").write_text(head + BODY + sect, encoding="utf-8")
rels = (OUTD / "word/_rels/document.xml.rels").read_text(encoding="utf-8")
for m in list((OUTD / "word/media").glob("*")): m.unlink()
(OUTD / "word/_rels/document.xml.rels").write_text(re.sub(r'<Relationship[^>]*image[^>]*/>', '', rels), encoding="utf-8")
ct = (OUTD / "[Content_Types].xml").read_text(encoding="utf-8")
(OUTD / "[Content_Types].xml").write_text(re.sub(r'<Default Extension="png"[^>]*/>', '', ct), encoding="utf-8")
if OUT.exists(): OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUTD.rglob("*")):
        if f.is_file(): z.write(f, f.relative_to(OUTD).as_posix())
print("built", OUT, OUT.stat().st_size)
