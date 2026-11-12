#!/usr/bin/env python3
"""Fold the 9 Nov 2026 execution-file register into the earlier technical documentation."""
import re, shutil, zipfile, os, sys

X = "/tmp/claude-0/-home-claude/f7ea644d-6066-5e86-aa53-2f9a26b2ec03/scratchpad/td/x"
DOC = os.path.join(X, "word/document.xml")
NUM = os.path.join(X, "word/numbering.xml")

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))

TOK = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")

def runs(text):
    out = []
    for part in TOK.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append('<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part[2:-2]))
        elif part.startswith("`") and part.endswith("`"):
            out.append('<w:r><w:rPr><w:rStyle w:val="CodeChar"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part[1:-1]))
        else:
            out.append('<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part))
    return "".join(out)

def h(level, text):
    return '<w:p><w:pPr><w:pStyle w:val="Heading%d"/></w:pPr><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % (level, esc(text))

def p(text):
    return "<w:p><w:pPr></w:pPr>%s</w:p>" % runs(text)

def li(text, numid):
    return ('<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/>'
            '<w:numId w:val="%d"/></w:numPr></w:pPr>%s</w:p>' % (numid, runs(text)))

def table(widths, rows, aligns=None):
    aligns = aligns or ["left"] * len(widths)
    g = "".join('<w:gridCol w:w="%d"/>' % w for w in widths)
    out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
           '<w:tblLayout w:type="fixed"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" '
           'w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
           '<w:tblGrid>%s</w:tblGrid>' % g]
    for ri, row in enumerate(rows):
        head = ri == 0
        out.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>' if head else "<w:tr>")
        for ci, cell in enumerate(row):
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>' if head else ""
            jc = ('<w:jc w:val="right"/>' if aligns[ci] == "right" and not head else "")
            body = ('<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(cell)) if head else runs(cell)
            out.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s</w:tcPr>'
                       '<w:p><w:pPr>%s</w:pPr>%s</w:p></w:tc>' % (widths[ci], shd, jc, body))
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)

BUL, NUMD = 10, 11

# ---------------------------------------------------------------- new section
sec = []
sec.append(h(2, "Experiment register and run lineage"))
sec.append(p("The five native libraries that could not be installed when this study was run — XGBoost, "
             "LightGBM, CatBoost, the Transformer and the TFT — have since been run in a separate "
             "environment on a reduced panel. A file register compiled on 9 November 2026 inventories "
             "everything produced so far: 555 records, made up of 208 files inside the six execution "
             "workspaces, 33 loose files and 314 members of ZIP archives. This section records which run "
             "produced what, so that no figure is read against the wrong experiment."))

sec.append(h(3, "Run lineage"))
for t in [
 "**Full-panel scikit-learn baseline — this document.** `nepse_pipeline/`, `nepse_pipeline_resume/`, "
 "`nepse_pipeline.zip` and `nepse_results.zip`. Five scikit-learn classifiers and one ranker over the "
 "full 362-stock panel. Every accuracy, calibration, significance, driver and backtest figure in this "
 "document comes from this run and from no other.",
 "**Early native lite work.** `nepse_native_lite_work/` holds a seven-family out-of-fold object for "
 "folds 1–8 with early ensemble and statistical outputs. Superseded by the combined run below.",
 "**Native tree branch.** `nepse_native_tree_lite_run/` holds the XGBoost, LightGBM, CatBoost and "
 "logistic out-of-fold outputs, the training log, the panel, the split manifest and stage reports.",
 "**Native sequence and ranker branch.** `nepse_native_seq_ranker_lite_run/` holds the Transformer, "
 "TFT-style and LambdaRank out-of-fold and final-test outputs, saved models and feature diagnostics.",
 "**Combined native seven-model run.** `nepse_native_all_models_lite_run/` merges all seven native "
 "families into one set of out-of-fold predictions, with the selected ensemble, calibration, "
 "statistics, final-test scoring, saved final models, the source panel, the scripts and its own report.",
 "**Standalone refits.** The CatBoost and XGBoost/LightGBM bundles preserve per-horizon refit results, "
 "logs, summary tables and training scripts from before the combined run.",
]:
    sec.append(li(t, NUMD))

sec.append(p("**The native runs are a different experiment, not an update to this one.** They use a "
             "100-stock lite panel; the results in this document use the full 362-stock panel. The model "
             "families differ and the universe differs, so rows from the two must never appear in one "
             "table without naming the run that produced each. No native figure is quoted anywhere in "
             "this document."))

sec.append(h(3, "Execution workspaces"))
sec.append(table([3780, 700, 1080, 3800], [
 ["Workspace", "Files", "Stored size", "Contents"],
 ["`nepse_native_all_models_lite_run/`", "88", "158.5 MB",
  "Latest merged native seven-model lite-panel run: out-of-fold predictions, ensemble and calibration, "
  "statistics, final-test metrics, saved final models, scripts, source panel and research report."],
 ["`nepse_native_seq_ranker_lite_run/`", "60", "129.7 MB",
  "Earlier Transformer / TFT-style / LambdaRank branch: sequence and ranker outputs, models, feature "
  "diagnostics, reports and scripts; also some stale process-ID marker files."],
 ["`nepse_native_tree_lite_run/`", "8", "67.8 MB",
  "Earlier tree branch: XGBoost, LightGBM, CatBoost and logistic out-of-fold outputs, training log, "
  "panel, split manifest and stage reports."],
 ["`nepse_native_lite_work/`", "11", "130.8 MB",
  "First combined native attempt. Its out-of-fold object already covers all seven families, but it "
  "predates the merged bundle."],
 ["`nepse_pipeline/`", "21", "212 KB",
  "Source of the pipeline described in this document: ingestion, auxiliary sources, panel, audit, "
  "models, ensemble, statistics, final test, backtest export and feature importance."],
 ["`nepse_pipeline_resume/`", "20", "165 KB",
  "Resumed copy of the same source tree, including the model, backtest and reporting scripts."],
], aligns=["left", "right", "right", "left"]))

sec.append(h(3, "Packages and reports"))
sec.append(table([3860, 5500], [
 ["Package or document", "What it is"],
 ["`nepse_native_all_models_lite_bundle.zip` (82.3 MB, 88 entries)",
  "Complete combined native seven-model lite-panel bundle."],
 ["`nepse_native_all_models_lite_results.zip` (195 KB, 34 entries)",
  "Compact reports, metrics and logs for the same run."],
 ["`research_report_native_all_models_lite.md` (12.8 KB)",
  "Narrative report for the combined native run, with its own addendum file."],
 ["`nepse_native_sequence_ranker_lite_bundle.zip` (69.3 MB)",
  "Earlier sequence and ranker bundle, with a compact results package beside it."],
 ["`nepse_catboost_completed_bundle.zip` (24.3 KB, 17 entries)",
  "Standalone CatBoost refit: per-horizon results, logs and the training script."],
 ["`nepse_xgb_lightgbm_run_bundle.zip` / `…_spec_refit_bundle.zip`",
  "XGBoost and LightGBM first run and specification-aligned refit, with summary CSVs."],
 ["`nepse_results.zip` (268 KB) and `nepse_pipeline.zip` (57.5 KB)",
  "Stage reports, figures and source for the baseline run documented here."],
 ["`NEPSE_COMPLETE_MODEL_EXECUTION_FILE_REGISTER.csv`",
  "The register itself: 555 rows of record type, container, path, byte size and description."],
]))

sec.append(h(3, "What the combined native run holds"))
for t in [
 "`oof.pkl` — merged out-of-fold predictions for XGBoost, LightGBM, CatBoost, logistic regression, the "
 "Transformer, the TFT-style model and LambdaRank; `ens_oof.pkl` — ensemble and calibration outputs; "
 "`test_predictions.pkl` — final-test predictions with row and session metadata.",
 "`models/` — saved final models per horizon in each library's own format (`.json` for XGBoost, `.cbm` "
 "for CatBoost, `.txt` for LightGBM and the ranker, `.pt` for the Transformer and TFT, `.joblib` for "
 "logistic regression and the second-level models).",
 "`reports/` — stage reports, candidate metric tables for out-of-fold and final test, bootstrap and "
 "Holm results, ranker statistics and feature-group diagnostics.",
 "`scripts/`, `source/TRAINING_SPEC.md` and `source/nepse_model_panel_lite.csv.gz` — the code and the "
 "exact panel the run read, so the run can be repeated without rebuilding the panel.",
 "`splits.json`, `selection.json`, `final_test.lock`, `environment.txt`, `RUN_STATUS.json` and "
 "`CHECKSUMS.sha256` — the fold manifest, the pre-test selection and its hash, the dependency snapshot, "
 "the completion record and the integrity manifest.",
]:
    sec.append(li(t, BUL))

sec.append(h(3, "Reading the register"))
for t in [
 "**Duplicates are deliberate.** The same artifact is copied into several bundles; a repeated path is a "
 "package copy, not a separate run.",
 "**Integrity.** The combined run's `CHECKSUMS.sha256` was verified for all 87 payload files when that "
 "bundle was built. The register covers workspaces and archives beyond that bundle, which carry no such "
 "manifest.",
 "**Stale markers.** The older sequence and ranker workspace still contains `.pid` files. They record "
 "process IDs from finished runs and are not restart instructions.",
 "**An inventory is not a result.** Listing a run does not make it complete, statistically independent "
 "of the others, or deployable; the runs share a panel and overlap heavily in their folds.",
]:
    sec.append(li(t, BUL))

NEW_SECTION = "".join(sec)

# ---------------------------------------------------------------- apply
d = open(DOC, encoding="utf-8").read()
orig = d
edits = 0

def sub1(old, new):
    global d, edits
    if old not in d:
        sys.exit("NOT FOUND: " + old[:90])
    if d.count(old) != 1:
        sys.exit("NOT UNIQUE: " + old[:90])
    d = d.replace(old, new)
    edits += 1

# 1. insert the new section before "Limitations and threats to validity"
anchor = '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t xml:space="preserve">Limitations and threats to validity</w:t></w:r></w:p>'
sub1(anchor, NEW_SECTION + anchor)

# 2. Summary closing bullet
sub1("It cannot claim 70% accuracy, and it should say that the numbers come from scikit-learn stand-ins, "
     "not from CatBoost, LightGBM, XGBoost, the Transformer or the TFT, which could not be installed here.",
     "It cannot claim 70% accuracy, and it should say that the numbers come from scikit-learn stand-ins, "
     "not from CatBoost, LightGBM, XGBoost, the Transformer or the TFT, which could not be installed here. "
     "Those five families have since been run on a separate 100-stock lite panel; that is a different "
     "experiment and none of its figures appear in this document. See &quot;Experiment register and run "
     "lineage&quot;.")

# 3. Models section intro
sub1(") but could not be installed in this environment, so they have not been run; their "
     "results must not be inferred from these.",
     ") but could not be installed in this environment, so they did not produce any "
     "number in this document. They were later run on a reduced 100-stock panel in a separate workspace, "
     "recorded under &quot;Experiment register and run lineage&quot;; results from that run belong to it, "
     "not to the tables here.")

# 4. Limitations stand-in bullet
sub1(" reruns stages 6–13 with the named libraries.",
     " reruns stages 6–13 with the named libraries, and that is how the separate native lite-panel run "
     "was produced; because that run uses 100 stocks rather than 362, it does not replace the figures "
     "here.")

# 5. Reproducibility: register paragraph after the stage-report sentence
sub1('<w:p><w:pPr></w:pPr><w:r><w:t xml:space="preserve">Each stage writes a JSON report to ',
     p("Every run is inventoried in the 9 November 2026 execution register "
     "(`NEPSE_COMPLETE_MODEL_EXECUTION_FILE_REGISTER.csv`, 555 rows), which names each workspace, bundle "
     "and archive member, its size and its role. Each bundle carries its own `environment.txt` and, for "
     "the combined native run, a verified `CHECKSUMS.sha256`.")
     + '<w:p><w:pPr></w:pPr><w:r><w:t xml:space="preserve">Each stage writes a JSON report to ')

open(DOC, "w", encoding="utf-8").write(d)

# numbering: add a bullet list (abstract 0) and a decimal list (abstract 1)
n = open(NUM, encoding="utf-8").read()
m = re.search(r'<w:abstractNum w:abstractNumId="1">.*?</w:abstractNum>', n, re.S)
fresh = m.group(0).replace('w:abstractNumId="1"', 'w:abstractNumId="7"', 1)
n = n.replace(m.group(0), m.group(0) + fresh)
add = ('<w:num w:numId="10"><w:abstractNumId w:val="0"/></w:num>'
       '<w:num w:numId="11"><w:abstractNumId w:val="7"/></w:num>')
n = n.replace("</w:numbering>", add + "</w:numbering>")
open(NUM, "w", encoding="utf-8").write(n)

print("edits applied:", edits, "| document.xml", len(orig), "->", len(d))
