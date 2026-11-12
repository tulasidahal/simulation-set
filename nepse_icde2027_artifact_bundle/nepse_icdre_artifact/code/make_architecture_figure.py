#!/usr/bin/env python3
"""
Vector architecture figure for the paper (IEEE double-column width).
Edit DATASET below once your final counts are fixed. COMBINER selects the
final layer: "average" (probability averaging) or "logistic" (learned stacker).
    python make_architecture_figure.py              -> averaging version
    python make_architecture_figure.py logistic     -> logistic-stacker version
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DATASET = "N observations · M entities · T dates · d numeric features + categorical identifiers"
COMBINER = sys.argv[1] if len(sys.argv) > 1 else "average"

INK, MUTED = "#1F2A37", "#5B6573"
PAL = {"store": ("#EAF1F8", "#2E6F9E"), "cat": ("#EAF4FB", "#2C7FB8"),
       "lgb": ("#EAF5EE", "#2E8B57"), "xgb": ("#FDF1E6", "#C8691C"),
       "tf": ("#F2EEF9", "#6A4C9C"), "layer": ("#F5F6F8", "#7A8494"),
       "stack": ("#EEF2F8", "#1F3A5F"), "out": ("#EAF5EE", "#2E7D4F")}

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.16, 3.75))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis("off")


def box(x, y, w, h, key, title, lines, title_size=7.6, lw=1.0, badge=None):
    fc, ec = PAL[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    ty = y + h - 2.6
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=ec)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, ty - 2.5 - i * 2.15, ln, ha="center", va="center",
                fontsize=5.6, color=INK if i == 0 else MUTED)
    if badge:
        ax.add_patch(FancyBboxPatch((x + w - 10.5, y + h - 1.6), 10, 2.6,
                                    boxstyle="round,pad=0,rounding_size=1.2", fc=ec, ec=ec))
        ax.text(x + w - 5.5, y + h - 0.3, badge, ha="center", va="center", fontsize=5.0,
                color="white", fontweight="bold")


def arrow(x0, y0, x1, y1, color="#8A94A3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=color, shrinkA=0, shrinkB=0))


ax.text(50, 67.6, "Four-model ensemble prediction architecture", ha="center", fontsize=10,
        fontweight="bold", color=INK)
ax.text(50, 65.3, "Point-in-time data, calibrated base learners, leakage-safe walk-forward evaluation",
        ha="center", fontsize=6.5, color=MUTED)

box(6, 54.5, 88, 9, "store", "Point-in-time feature store",
    [DATASET, "Event/price adjustment audit  ·  truncation-invariance test  ·  IC leakage scan  ·  "
              "within-date rank normalisation"], lw=1.2)

cols = [(4, "cat", "CatBoost", ["Ordered boosting, categoricals",
                                "≤2000 iter · depth 7 · lr 0.04", "early stop on embargoed tail"], None),
        (28, "lgb", "LightGBM", ["Leaf-wise gradient boosting",
                                 "≤2000 iter · 31 leaves · lr 0.035", "early stop on embargoed tail"], None),
        (52, "xgb", "XGBoost", ["Hist trees, categorical partitions",
                                "softmax or focal loss (γ)", "depth 4 · lr 0.035 · min_child 20",
                                "λ 5 · α 0.5 · row/col subsampling"], "CUSTOMISED"),
        (76, "tf", "Transformer", ["L × d sequential input",
                                   "2 layers · 4 heads · d 64", "FFN 96 · dropout 0.2"], None)]
for x, key, title, lines, badge in cols:
    arrow(50, 54.5, x + 10, 47.6)
    box(x, 34.0, 20, 13.5, key, title, lines, lw=1.6 if badge else 1.0, badge=badge)

box(4, 25.0, 92, 6.5, "layer", "Shared training recipe for every base learner",
    ["class-balanced × time-decayed sample weights  ·  per-model temperature scaling  "
     "→  calibrated log-probabilities (4 × 3)"], title_size=6.6)
for x, *_ in cols:
    arrow(x + 10, 34.0, x + 10, 31.5)

arrow(50, 25.0, 50, 22.6, INK)
if COMBINER == "logistic":
    box(14, 13.6, 72, 9, "stack", "Multinomial logistic stacker on log-probabilities",
        ["C = 0.1  ·  learned per-model, per-class coefficients",
         "trained on out-of-fold blocks 0…k−1  ·  purge h + 25-trading-day embargo"], lw=1.4)
else:
    box(14, 13.6, 72, 9, "stack", "Probability averaging of calibrated base learners",
        ["equal weights  ·  no meta-learner to drift under regime shift",
         "learned logistic stacker kept as an ablation  ·  purge h + 25-trading-day embargo"],
        lw=1.4)

arrow(50, 13.6, 50, 11.2, INK)
box(14, 2.2, 72, 9, "out", "Predicted direction and confidence",
    ["UP / FLAT / DOWN  ·  confidence = maxₖ pₖ  ·  accuracy on the top 10 / 20 / 30 % most confident",
     "evaluated by walk-forward blocks, 5 seeds, DM / McNemar / block bootstrap"], lw=1.2)

out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "architecture.pdf")
fig.savefig(out / "architecture.png", dpi=220)
print("wrote figures/architecture.pdf and .png")
