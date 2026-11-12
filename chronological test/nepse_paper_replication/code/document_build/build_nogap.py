#!/usr/bin/env python3
"""Build the thin-floorsheet exclusion report, reusing the technical documentation's styles."""
import json, os, re, shutil, zipfile
from pathlib import Path

SC = Path("/tmp/claude-0/-home-claude/f7ea644d-6066-5e86-aa53-2f9a26b2ec03/scratchpad/td")
TPL = SC / "x"                       # unpacked technical documentation (styles, numbering)
OUTD = SC / "ng"; OUT = SC / "nogap.docx"
CTRL = Path("/mnt/user-data/outputs/nepse_results")
TRT = Path("/home/claude/work_nogap/reports")

def L(d, f):
    with open(d / f) as fh: return json.load(fh)

P12a, P12b = L(CTRL, "stage12_final_test.json"), L(TRT, "stage12_final_test.json")
P34a, P34b = L(CTRL, "stage03_04_panel_report.json"), L(TRT, "stage03_04_panel_report.json")
P05b = L(TRT, "stage05_audit_report.json")
P13a, P13b = L(CTRL, "stage13_backtest.json"), L(TRT, "stage13_backtest.json")
THIN = P34b["bars"]["thin_floorsheet"]

# ----------------------------------------------------------------- xml helpers
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))

TOK = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")

def runs(text):
    out = []
    for part in TOK.split(text):
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            out.append('<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part[2:-2]))
        elif part.startswith("`") and part.endswith("`"):
            out.append('<w:r><w:rPr><w:rStyle w:val="CodeChar"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part[1:-1]))
        else:
            out.append('<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part))
    return "".join(out)

def h(lvl, t): return '<w:p><w:pPr><w:pStyle w:val="Heading%d"/></w:pPr><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % (lvl, esc(t))
def ti(t):     return '<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % esc(t)
def p(t):      return "<w:p><w:pPr></w:pPr>%s</w:p>" % runs(t)
def li(t, n=10): return ('<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/>'
                         '<w:numId w:val="%d"/></w:numPr></w:pPr>%s</w:p>' % (n, runs(t)))

def table(widths, rows, aligns=None):
    aligns = aligns or ["left"] * len(widths)
    g = "".join('<w:gridCol w:w="%d"/>' % w for w in widths)
    o = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/><w:tblLayout w:type="fixed"/>'
         '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
         '</w:tblPr><w:tblGrid>%s</w:tblGrid>' % g]
    for ri, row in enumerate(rows):
        head = ri == 0
        o.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>' if head else "<w:tr>")
        for ci, cell in enumerate(row):
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>' if head else ""
            jc = '<w:jc w:val="right"/>' if aligns[ci] == "right" and not head else ""
            body = ('<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(cell)) if head else runs(str(cell))
            o.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s</w:tcPr><w:p><w:pPr>%s</w:pPr>%s</w:p></w:tc>' % (widths[ci], shd, jc, body))
        o.append("</w:tr>")
    o.append("</w:tbl>")
    return "".join(o)

pc = lambda x, d=1: "-" if x is None else f"{100*x:.{d}f}%"
R, LF = "right", "left"

# ----------------------------------------------------------------- content
B = [ti("NEPSE Forecasting Study — Excluding the Thin-Floorsheet Sessions")]
B.append(p("This report records a change to the modelling design: the sessions in which the floorsheet is "
           "largely missing are no longer used to train or evaluate any model. It states the rule, what it "
           "removes, and what it does to every headline result, measured against the earlier run on the same "
           "untouched final test window. Both runs use the same scikit-learn stand-in models (CatBoost, "
           "LightGBM, XGBoost, the Transformer and the TFT could not be installed here), the same 60 "
           "predictors and the same code; the only difference is which sessions enter the panel."))

B.append(h(2, "Summary"))
B.append(p("**Excluding the gap does not improve accuracy.** On the untouched final test it costs a little at "
           "every horizon, and a lot at 10 sessions. The 1-session result survives: 58.6% against a 55.3% "
           "majority baseline (95% interval 56.5–60.6%), with the Diebold–Mariano test against climatology "
           "slightly stronger than before (Holm p = 1.2 × 10⁻⁷ versus 3.5 × 10⁻⁷)."))
B.append(p("**That is the useful finding.** The 1-session skill is not an artefact of the degraded period: it "
           "holds after 15% of the sample, including the whole stretch where broker features are missing, is "
           "taken away. At 3, 5 and 10 sessions nothing is distinguishable from the baseline in either run, so "
           "nothing is lost there that was real to begin with."))
B.append(p("**The 10-session result degrades materially**, from 55.3% to 50.5% against a 56.8% baseline. Its "
           "AUC is unchanged (0.584 → 0.588), so the ranking information is still there; what fails is "
           "calibration (expected calibration error 7.3% → 10.0%), which moves the 0.5 cut-off to the wrong "
           "place. With 15% fewer sessions the 10-session labels overlap more heavily within a smaller "
           "effective sample, and the calibration layer has less to learn from."))
B.append(p("**The backtest is unchanged in character.** No strategy earns a net return distinguishable from "
           "zero after costs in either run, and the ranker's top-10 portfolio still loses money at every "
           "horizon."))

B.append(h(2, "The exclusion rule"))
B.append(p("The rule is measured, not assumed. For each session, coverage is the share of that session's "
           "eligible stocks that appear in the floorsheet. The distribution is sharply bimodal — 1,577 of "
           "1,855 sessions above 95% and 261 below 30%, with almost nothing between — so the threshold is not "
           "a judgement call."))
B.append(p("A session is **thin** when coverage is below "
           f"{pc(THIN['min_coverage'],0)}. Thin sessions that form a run of at least {THIN['min_block']} are "
           "treated as a structural gap and dropped from the modelling panel, and so are the "
           f"{THIN['buffer_sessions']} sessions that follow each run. Isolated thin days elsewhere are kept: "
           "one missing observation inside a 20-session window is noise, not a gap."))
B.append(p("**Why the buffer matters.** `amihud20`, `turnover_trend`, `trades_ratio20` and `buyer_hhi_chg5` are "
           "rolling windows over floorsheet turnover and trade counts, and the sequence model reads a "
           "20-session window of floorsheet columns. Without the buffer, a row in April 2023 would sit outside "
           "the gap but still read gap sessions through its own window. With it, no model input is computed "
           "from, or reads, a thin session."))
B.append(table([2300, 2300, 1200, 3560], [
    ["Block", "With buffer", "Sessions", "What happens in it"],
    [f"{THIN['blocks'][0]['from']} → {THIN['blocks'][0]['to']}", f"to {THIN['blocks'][0]['buffer_to']}",
     str(THIN['blocks'][0]['sessions']), "Floorsheet coverage falls to about 43% for the month"],
    [f"{THIN['blocks'][1]['from']} → {THIN['blocks'][1]['to']}", f"to {THIN['blocks'][1]['buffer_to']}",
     str(THIN['blocks'][1]['sessions']), "The long outage: 9–14% coverage, 16–26 stocks a session"],
    [f"{THIN['blocks'][2]['from']} → {THIN['blocks'][2]['to']}", f"to {THIN['blocks'][2]['buffer_to']}",
     str(THIN['blocks'][2]['sessions']), "Coverage falls to about 54% for the second half of the month"],
], aligns=[LF, LF, R, LF]))
B.append(p(f"{THIN['thin_sessions']} sessions are thin in total; {THIN['excluded_sessions']} are excluded once "
           "the blocks and their buffers are counted. The rule is implemented in the panel builder and is "
           "switched on with `NEPSE_EXCLUDE_GAP=1`, so both arms come from one code path."))

B.append(h(2, "What it removes"))
B.append(table([3000, 3180, 3180], [
    ["", "With the gap", "Without the gap"],
    ["Eligible stock-sessions", f"{P34a['rows_eligible']:,}", f"{P34b['rows_eligible']:,}"],
    ["Sessions", f"{P34a['sessions_eligible']:,}", f"{P34b['sessions_eligible']:,}"],
    ["Symbols", f"{P34a['symbols_eligible']}", f"{P34b['symbols_eligible']}"],
    ["Floorsheet coverage", pc(P34a['floorsheet_coverage_eligible']), pc(P34b['floorsheet_coverage_eligible'])],
    ["Coverage in 2022", pc(P34a['floorsheet_coverage_by_year']['2022']), pc(P34b['floorsheet_coverage_by_year']['2022'])],
    ["Coverage in 2023", pc(P34a['floorsheet_coverage_by_year']['2023']), pc(P34b['floorsheet_coverage_by_year']['2023'])],
    ["1-session labelled rows", f"{P34a['targets']['1d']['labeled']:,}", f"{P34b['targets']['1d']['labeled']:,}"],
    ["1-session up-rate", pc(P34a['targets']['1d']['up_rate']), pc(P34b['targets']['1d']['up_rate'])],
], aligns=[LF, R, R]))
B.append(p("The excluded rows are 15.4% of the sample. Coverage across the surviving training data rises "
           "from 88.5% to 99.2%, and the up-rate rises slightly at every horizon: all three excluded blocks "
           "were falling markets — the NEPSE index lost 6.8%, 25.0% and 2.5% across them, against +114% over "
           "the study as a whole — so removing them removes a disproportionate share of \u201cdown\u201d outcomes."))
B.append(p(f"**The final test window does not move.** It is the last 250 sessions, "
           f"{P12b['test_dates'][0]} to {P12b['test_dates'][1]}, "
           f"{P12b['horizons']['1d']['n']:,} scored rows at 1 session in both runs — the gap ends in May 2024, "
           "well before the test begins. Every comparison below is therefore on identical test rows."))

B.append(h(2, "Walk-forward folds after the exclusion"))
B.append(p("The folds are rebuilt over the surviving sessions, so the eight validation blocks are 91 sessions "
           "each instead of about 100. Folds 3 and 4 of the earlier run fell almost entirely inside the gap; "
           "they no longer exist in that form."))
rows = [["Fold", "Train sessions", "Validation dates", "Train rows", "Val rows"]]
for f in P05b["folds"]:
    rows.append([str(f["fold"]), str(f["train_sessions"]), f"{f['val_dates'][0]} → {f['val_dates'][1]}",
                 f"{f['train_rows']:,}", f"{f['val_rows']:,}"])
B.append(table([800, 1700, 3260, 1800, 1800], rows, aligns=[R, R, LF, R, R]))
B.append(p("The leakage and split audit passes all ten checks on the rebuilt panel, including the "
           "truncation-recompute test: rebuilding every feature from sources truncated at 2024-10-28 "
           "reproduces the stored values exactly (max |difference| 0.0, no missing-data-pattern mismatches, "
           "60 features × 286,755 rows)."))

B.append(h(2, "Final-test accuracy"))
B.append(p("Both arms score the candidate their own out-of-fold log-loss selected, locked before the test was "
           "touched. At 5 sessions the no-gap run selected a stacked, Platt-scaled candidate that scored badly; "
           "the like-for-like comparison on the same `weighted` ensemble follows the table."))
rows = [["H", "Arm", "Candidate", "Base", "Acc", "95% interval", "AUC", "Logloss", "ECE"]]
for hh in ("1d", "3d", "5d", "10d"):
    for nm, ft in (("with gap", P12a), ("no gap", P12b)):
        sel = ft["selection"][hh]["primary"]; c = ft["horizons"][hh]["candidates"][sel]
        ci = c["ci"]["accuracy"]
        rows.append([hh if nm == "with gap" else "", nm, sel.replace("__", " + "), pc(c["majority_baseline_acc"]),
                     pc(c["accuracy"]), f"{pc(ci[0])[:-1]}–{pc(ci[1])}", f"{c['auc']:.3f}",
                     f"{c['logloss']:.4f}", pc(c["ece"], 2)])
B.append(table([700, 1000, 1450, 850, 850, 1500, 760, 1000, 800], rows,
               aligns=[LF, LF, LF, R, R, R, R, R, R]))
B.append(p("**Like for like, on the `weighted` ensemble in both arms:** 1 session 59.0% → 58.6%; 3 sessions "
           "59.2% → 57.5%; 5 sessions 58.3% → 55.2%; 10 sessions 55.3% → 50.5%. The 5-session AUC of 0.473 in "
           "the table above belongs to the selected stacked candidate, not to the ensemble; the `weighted` "
           "ensemble gives 0.578 there. That a selection rule validated out of fold can pick a candidate this "
           "poor is itself worth reporting."))
B.append(p("The intervals are wide and overlap at every horizon, so the 1-, 3- and 5-session differences are "
           "within sampling error. The 10-session drop is not: 50.5% sits below its own 56.8% baseline, and "
           "the bootstrap probability that it beats that baseline is 0.013."))

B.append(h(2, "Significance and the ranker"))
rows = [["Horizon", "Arm", "p (acc ≤ base)", "DM vs climatology, Holm p", "Rank-IC", "IC-IR", "Spread"]]
for hh in ("1d", "3d", "5d", "10d"):
    for nm, ft in (("with gap", P12a), ("no gap", P12b)):
        sel = ft["selection"][hh]["primary"]; c = ft["horizons"][hh]["candidates"][sel]; H = ft["horizons"][hh]
        dm = H["dm"].get(f"{sel}_vs_climatology") or list(H["dm"].values())[0]
        r = H["ranker"]
        pg = c.get("p_acc_gt_majority")
        rows.append([hh if nm == "with gap" else "", nm, "< 0.001" if pg == 0 else f"{pg:.3f}",
                     f"{dm['p_holm']:.3g}" if dm["p_holm"] < 1 else "1.0",
                     f"{r['mean_ic']:.3f}", f"{r['ic_ir']:.2f}", pc(r["spread"], 2)])
B.append(table([1000, 1000, 1700, 2100, 1050, 900, 1060], rows, aligns=[LF, LF, R, R, R, R, R]))
B.append(p("The cross-sectional ranker is almost untouched: mean rank-IC 0.284 → 0.281 at 1 session and "
           "0.213 → 0.195 at 10, with the share of sessions whose IC is positive unchanged at 98.3% and 93%. "
           "Whatever the ranker is reading, the gap period was neither feeding it nor fooling it."))

B.append(h(2, "Backtest"))
B.append(p("Long-only, entry at the next session's open, 10 bp slippage a side plus commission and the 7.5% "
           "short-term capital-gains tax. Mean net return per trade, with its session-block bootstrap interval:"))
def bt(d, hz, key):
    t = d["test"].get(hz, {})
    return t.get(key)
rows = [["Strategy", "Arm", "Trades", "Mean net / trade", "95% interval", "P(mean ≤ 0)"]]
for hz, key, label in (("1d", "thr_0.55", "1 session, p ≥ 0.55"), ("3d", "thr_0.60", "3 sessions, p ≥ 0.60"),
                       ("1d", "ranker_top10", "1 session, ranker top 10"), ("10d", "ranker_top10", "10 sessions, ranker top 10")):
    for nm, d in (("with gap", P13a), ("no gap", P13b)):
        r = bt(d, hz, key)
        if not r or "mean_net_per_trade" not in r: continue
        ci = r.get("ci_mean_net")
        rows.append([label if nm == "with gap" else "", nm, f"{r.get('trades',0):,}",
                     pc(r["mean_net_per_trade"], 3),
                     f"{pc(ci[0],3)[:-1]} to {pc(ci[1],3)}" if ci else "-",
                     f"{r.get('p_mean_net_le_0', float('nan')):.3f}"])
B.append(table([2700, 900, 1000, 1700, 2100, 960], rows, aligns=[LF, LF, R, R, R, R]))
B.append(p("Nothing changes qualitatively. The only strategies with a positive point estimate in either run "
           "are the 3-session and 10-session p ≥ 0.60 filters, and both intervals comfortably straddle zero."))

B.append(h(2, "What to report"))
B.append(p("**Report the no-gap run as the primary result.** It is the cleaner design: no model is trained on "
           "sessions where nine stocks in ten have no broker, trade-count or turnover data, and no feature "
           "window reads one. That the 1-session result survives the change, with a slightly stronger "
           "Diebold–Mariano statistic, is a better claim than the marginally higher number obtained by "
           "training on the degraded period."))
B.append(p("**Report the with-gap run beside it as a sensitivity check**, and say plainly that excluding the "
           "gap costs rather than gains accuracy. A reviewer who asks \"what happens if you drop the period "
           "where your microstructure features are missing?\" should find the answer already in the paper."))
B.append(p("**Do not claim the exclusion improved anything.** It did not. What it bought is a claim that does "
           "not depend on a period of degraded data, and a 10-session result that is now visibly worse than "
           "its baseline rather than quietly indistinguishable from it."))
B.append(p("**Both runs still use scikit-learn stand-ins.** CatBoost, LightGBM, XGBoost, the Transformer and "
           "the TFT could not be installed in this environment. If the native lite-panel run is repeated, the "
           "same exclusion should be applied there before its numbers are compared with anything here."))

B.append(h(2, "Reproduction"))
B.append(p("Stages 1–2 (the 91-million-trade ingest) were not rerun; the cleaned daily floorsheet from the "
           "earlier run was reused, and stage 01b onwards was rebuilt from the original SQL dumps. The "
           "rebuild was verified against the earlier run before the exclusion was switched on: panel rows "
           "405,491, eligible 360,943, 362 symbols, 1,855 sessions and all four label counts reproduced "
           "exactly, so the two arms differ only by the rule."))
B.append(p("`NEPSE_EXCLUDE_GAP=1 FAMILY=sklearn START=3 bash run_pipeline.sh` rebuilds the panel with the "
           "exclusion and reruns stages 3–14. The thresholds are `GAP_MIN_COVERAGE`, `GAP_MIN_BLOCK` and "
           "`GAP_BUFFER` in `common.py`. Reports for this run are in `nepse_results_nogap.zip`; the patched "
           "source is in `nepse_pipeline_nogap.zip`."))

BODY = "".join(B)

# ----------------------------------------------------------------- assemble
if OUTD.exists(): shutil.rmtree(OUTD)
shutil.copytree(TPL, OUTD)
doc = (OUTD / "word/document.xml").read_text(encoding="utf-8")
head = doc[:doc.index("<w:body>") + len("<w:body>")]
sect = doc[doc.rindex("<w:sectPr"):]
(OUTD / "word/document.xml").write_text(head + BODY + sect, encoding="utf-8")
# drop images (not referenced here)
rels = (OUTD / "word/_rels/document.xml.rels").read_text(encoding="utf-8")
for m in list(Path(OUTD / "word/media").glob("*")): m.unlink()
rels = re.sub(r'<Relationship[^>]*image[^>]*/>', '', rels)
(OUTD / "word/_rels/document.xml.rels").write_text(rels, encoding="utf-8")
ct = (OUTD / "[Content_Types].xml").read_text(encoding="utf-8")
(OUTD / "[Content_Types].xml").write_text(re.sub(r'<Default Extension="png"[^>]*/>', '', ct), encoding="utf-8")

if OUT.exists(): OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUTD.rglob("*")):
        if f.is_file(): z.write(f, f.relative_to(OUTD).as_posix())
print("built", OUT, OUT.stat().st_size, "bytes")
