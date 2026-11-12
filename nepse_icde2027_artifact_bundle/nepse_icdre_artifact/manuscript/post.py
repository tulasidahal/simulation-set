from docx import Document
from docx.shared import Pt, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy
d = Document("raw.docx")
body = d.element.body
# final section: 2 columns, continuous, margins
sp = body.find(qn('w:sectPr'))
for ch in list(sp):
    if ch.tag in (qn('w:cols'), qn('w:type'), qn('w:pgSz'), qn('w:pgMar')): sp.remove(ch)
def add(tag, **a):
    e = OxmlElement(tag)
    for k, v in a.items(): e.set(qn('w:'+k), str(v))
    sp.insert(0, e) if False else sp.append(e); return e
# order matters: type, pgSz, pgMar, cols must precede docGrid etc.; insert at proper spot
refs = [x for x in sp if x.tag in (qn('w:footerReference'), qn('w:headerReference'))]
for x in list(sp): sp.remove(x)
for x in refs: sp.append(x)
add('w:type', val='continuous'); add('w:pgSz', w=12240, h=15840)
add('w:pgMar', top=1080, right=900, bottom=1440, left=900, header=720, footer=540, gutter=0)
add('w:cols', num=2, space=360)
# footer page number on first section
sec = d.sections[0]
fp = sec.footer.paragraphs[0]; fp.alignment = 1
r = fp.add_run()
for t, txt in (('begin', None), (None, 'PAGE'), ('end', None)):
    if t:
        e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), t); r._r.append(e)
    else:
        e = OxmlElement('w:instrText'); e.set(qn('xml:space'), 'preserve'); e.text = txt; r._r.append(e)
r.font.size = Pt(9); r.font.name = "Times New Roman"
# tables: compact, three-line borders, full width
def border(tc_pr_parent, edges):
    b = OxmlElement('w:tblBorders')
    for e in ('top','left','bottom','right','insideH','insideV'):
        x = OxmlElement('w:'+e)
        if e in edges: x.set(qn('w:val'),'single'); x.set(qn('w:sz'), str(edges[e])); x.set(qn('w:color'),'000000')
        else: x.set(qn('w:val'),'nil')
        b.append(x)
    return b
for t in d.tables:
    tblPr = t._tbl.tblPr
    for x in tblPr.findall(qn('w:tblBorders')): tblPr.remove(x)
    tblPr.append(border(tblPr, {'top':8,'bottom':8}))
    for x in tblPr.findall(qn('w:tblW')): tblPr.remove(x)
    w = OxmlElement('w:tblW'); w.set(qn('w:type'),'pct'); w.set(qn('w:w'),'5000'); tblPr.append(w)
    for x in tblPr.findall(qn('w:tblLayout')): tblPr.remove(x)
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'),'autofit'); tblPr.append(lay)
    cm = OxmlElement('w:tblCellMar')
    for e, v in (('top',10),('left',40),('bottom',10),('right',40)):
        x = OxmlElement('w:'+e); x.set(qn('w:w'),str(v)); x.set(qn('w:type'),'dxa'); cm.append(x)
    tblPr.append(cm)
    for ri, row in enumerate(t.rows):
        for c in row.cells:
            tcPr = c._tc.get_or_add_tcPr()
            if ri == 0:
                b = OxmlElement('w:tcBorders'); x = OxmlElement('w:bottom'); x.set(qn('w:val'),'single'); x.set(qn('w:sz'),'4'); x.set(qn('w:color'),'000000'); b.append(x); tcPr.append(b)
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(8); r.font.name = "Times New Roman"
                    if ri == 0: r.font.bold = True
# column widths: proportional to content length, wide table (index 1 = Table II) spans the page
def widths(t, total):
    n = len(t.columns); mx = []
    for j in range(n):
        L = max(max((len(w) for w in c.text.split()), default=1) for c in t.columns[j].cells)
        S = max(len(c.text) for c in t.columns[j].cells)
        mx.append(max(L * 1.0, min(S, 40) ** 0.6 * 1.2))
    s = sum(mx); return [int(total * m / s) for m in mx]
for ti, t in enumerate(d.tables):
    total = 10440 if t.rows[0].cells[0].text.strip() == 'ID' else 5040
    ws = widths(t, total)
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn('w:gridCol')), ws): gc.set(qn('w:w'), str(w))
    for row in t.rows:
        for c, w in zip(row.cells, ws):
            tcPr = c._tc.get_or_add_tcPr()
            for x in tcPr.findall(qn('w:tcW')): tcPr.remove(x)
            e = OxmlElement('w:tcW'); e.set(qn('w:w'), str(w)); e.set(qn('w:type'), 'dxa'); tcPr.insert(0, e)
    tp = t._tbl.tblPr
    for x in tp.findall(qn('w:tblW')): tp.remove(x)
    e = OxmlElement('w:tblW'); e.set(qn('w:w'), str(total)); e.set(qn('w:type'), 'dxa'); tp.append(e)
    for x in tp.findall(qn('w:tblLayout')): tp.remove(x)
    e = OxmlElement('w:tblLayout'); e.set(qn('w:type'), 'fixed'); tp.append(e)
# keep each table together on one page
for t in d.tables:
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr(); trPr.append(OxmlElement('w:cantSplit'))
        for c in row.cells:
            for pp in c.paragraphs: pp.paragraph_format.keep_with_next = ri < len(t.rows) - 1
d.core_properties.title='[Experiment, Analysis, and Benchmark] Leakage-Safe Point-in-Time Data Preparation and Evaluation for Direction Prediction from Financial Time Series'; d.core_properties.author='Tulasi Dahal, Kishor Basyal'
d.save("NEPSE_ICDE2027_paper.docx"); print("saved")
