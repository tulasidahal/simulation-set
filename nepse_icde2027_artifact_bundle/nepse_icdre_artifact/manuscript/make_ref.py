from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
d = Document("ref0.docx")
S = d.styles
F = "Times New Roman"
def setfont(st, size, bold=None, italic=None, caps=None, color=RGBColor(0,0,0)):
    f = st.font; f.name = F; f.size = Pt(size); f.color.rgb = color
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    if caps is not None: f.all_caps = caps
    rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn('w:rFonts'))
    if rf is None: rf = OxmlElement('w:rFonts'); rpr.append(rf)
    for a in ('w:ascii','w:hAnsi','w:cs','w:eastAsia'): rf.set(qn(a), F)
    for a in ('w:asciiTheme','w:hAnsiTheme','w:cstheme','w:eastAsiaTheme'):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
def para(st, align=None, before=0, after=0, line=None, first=None, left=None, hanging=None, keep=None):
    p = st.paragraph_format
    if align is not None: p.alignment = align
    p.space_before = Pt(before); p.space_after = Pt(after)
    if line: p.line_spacing = line
    if first is not None: p.first_line_indent = Inches(first)
    if left is not None: p.left_indent = Inches(left)
    if hanging is not None: p.left_indent = Inches(hanging); p.first_line_indent = Inches(-hanging)
    if keep is not None: p.keep_with_next = keep
def find(name):
    for x in S:
        if x.name == name: return x
def get(name, typ=WD_STYLE_TYPE.PARAGRAPH, base="Normal"):
    x = find(name)
    if x is not None: return x
    if True:
        st = S.add_style(name, typ)
        if typ == WD_STYLE_TYPE.PARAGRAPH: st.base_style = find(base)
        return st
J = WD_ALIGN_PARAGRAPH
for n in ["Normal","Body Text","First Paragraph","Compact"]:
    st = get(n); setfont(st, 10); para(st, J.JUSTIFY, 0, 0, 1.0, first=0.0)
find("Body Text").paragraph_format.first_line_indent = Inches(0.2)
find("First Paragraph").paragraph_format.first_line_indent = Inches(0.2)
setfont(find("Compact"), 10); para(find("Compact"), J.JUSTIFY, 0, 0, 1.0)
st = get("Title"); setfont(st, 24, bold=False); para(st, J.CENTER, 0, 6, 1.0)
st = get("Author"); setfont(st, 11); para(st, J.CENTER, 6, 2)
st = get("Affil"); setfont(st, 10, italic=True); para(st, J.CENTER, 0, 10)
st = get("Abstract"); setfont(st, 9, bold=True); para(st, J.JUSTIFY, 0, 0, left=0.0)
st = get("Keywords"); setfont(st, 9, bold=True); para(st, J.JUSTIFY, 0, 6)
for n in ["Heading 1"]:
    st = find(n); setfont(st, 10, bold=False, caps=False); st.font.small_caps = True; para(st, J.CENTER, 10, 4, keep=True)
for n in ["Heading 2","Heading 3"]:
    st = find(n); setfont(st, 10, bold=False, italic=True); para(st, J.LEFT, 6, 3, keep=True)
st = get("TabCap"); setfont(st, 8, caps=False); st.font.small_caps = True; para(st, J.CENTER, 8, 4, keep=True)
st = get("FigureWide"); setfont(st, 8); para(st, J.CENTER, 4, 4)
st = get("FigureCol"); setfont(st, 8); para(st, J.CENTER, 4, 4)
st = get("Reference"); setfont(st, 9); para(st, J.JUSTIFY, 0, 2, hanging=0.3)
for n in ["Image Caption","Captioned Figure"]:
    st = get(n); setfont(st, 8, italic=False); para(st, J.CENTER, 2, 4)
for n in ["Table Caption","Figure"]:
    st = find(n)
    if st is not None: setfont(st, 8); para(st, J.CENTER, 2, 4)
st = get("Placeholder", WD_STYLE_TYPE.CHARACTER)
rpr = st.element.get_or_add_rPr(); h = OxmlElement('w:highlight'); h.set(qn('w:val'),'yellow'); rpr.append(h)
for n in ["Hyperlink","Verbatim Char"]:
    pass
d.save("ref.docx"); print("ok", [s.name for s in S if 'aption' in s.name])
