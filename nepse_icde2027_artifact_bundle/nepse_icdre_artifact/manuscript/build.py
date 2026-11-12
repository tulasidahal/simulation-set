import re, subprocess, sys
sys.path.insert(0, ".")
import tables as T
from refs import B
tpl = open("paper_tpl.md").read()
ARCH = "/home/claude/manuscript2/arch_generic.png"
PROT = T.R + "figures/protocols.png"
rep = {"@@ARCH@@": ARCH, "@@PROT@@": PROT, "@@T_MAIN@@": T.t_main(), "@@T_PERCLASS@@": T.t_perclass(),
       "@@T_BLOCKS@@": T.t_blocks(), "@@T_PROTOCOL@@": T.t_protocol(), "@@T_XGB@@": T.t_xgb(),
       "@@T_STACKABL@@": T.t_stackabl(), "@@T_REGIME@@": T.t_regime(), "@@T_EFF@@": T.t_eff(),
       "@@INFL@@": f"{T.values()['infl']*100:.0f}"}
for k, v in rep.items():
    assert k in tpl, k
    tpl = tpl.replace(k, v)
# yellow placeholders
tpl = re.sub(r"\[\[(.+?)\]\]", lambda m: '[' + m.group(1) + ']{custom-style="Placeholder"}', tpl, flags=re.S)
# citations numbered by first appearance
order = []
def cite(m):
    nums = []
    for k in m.group(1).split(","):
        k = k.strip(); assert k in B, k
        if k not in order: order.append(k)
        nums.append(order.index(k) + 1)
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j+1] == nums[j] + 1: j += 1
        out.append(str(nums[i]) if j == i else (f"{nums[i]}, {nums[j]}" if j == i+1 else f"{nums[i]}–{nums[j]}")); i = j + 1
    return "\\[" + ", ".join(out) + "\\]"
body = tpl.split("@@REFS@@")[0]
body = re.sub(r"\{\{c:([^}]+)\}\}", cite, body)
refs = "\n".join(f'::: {{custom-style="Reference"}}\n\\[{i+1}\\] {B[k]}\n:::\n' for i, k in enumerate(order))
unused = [k for k in B if k not in order]
open("paper.md", "w").write(body + refs)
print("refs", len(order), "unused", unused)
subprocess.run(["pandoc", "paper.md", "-f", "markdown+tex_math_dollars+smart", "-o", "raw.docx", "--reference-doc", "ref.docx"], check=True)
