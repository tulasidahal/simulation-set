"""Tables (CSV + LaTeX booktabs) and figures for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
COL_W = 3.5      # IEEE single column width in inches


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str))


def mean_std_table(df: pd.DataFrame, by: str, cols: list[str], digits: int = 3) -> pd.DataFrame:
    g = df.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std(ddof=1).fillna(0)
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.{digits}f} ± {b:.{digits}f}" for a, b in zip(m[c], s[c])]
    return out


def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, bold_best: dict | None = None):
    """Minimal booktabs table; bold_best = {column: 'max'|'min'} on the numeric means."""
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("±", r"$\pm$").replace("%", r"\%")
    body = df.astype(str).map(esc)
    if bold_best:
        for c, how in bold_best.items():
            if c not in df:
                continue
            vals = pd.to_numeric(df[c].astype(str).str.split(" ").str[0], errors="coerce")
            if vals.notna().any():
                best = vals.idxmax() if how == "max" else vals.idxmin()
                body.loc[best, c] = r"\textbf{" + body.loc[best, c] + "}"
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join(["Method"] + [esc(c) for c in cols]) + r" \\", r"\midrule"]
    for idx, row in body.iterrows():
        lines.append(" & ".join([esc(idx)] + list(row)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def reliability_figure(curves: dict, path: Path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.8))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.5", label="Perfect calibration")
    for name, (y, P) in curves.items():
        conf = P.max(1)
        corr = (P.argmax(1) == y)
        edges = np.linspace(conf.min(), 1, n_bins + 1)
        mids, accs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.sum() >= 30:
                mids.append(conf[m].mean())
                accs.append(corr[m].mean())
        ax.plot(mids, accs, marker="o", ms=3, lw=1.2, label=name)
    ax.set_xlabel("Predicted confidence (max class probability)")
    ax.set_ylabel("Observed accuracy")
    ax.legend(frameon=False)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def risk_coverage_figure(curves: dict, path: Path):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.7))
    for name, df in curves.items():
        ax.plot(df["coverage"], df["accuracy"], lw=1.2, label=name)
    ax.set_xlabel("Coverage (share of predictions kept, most confident first)")
    ax.set_ylabel("Accuracy on kept predictions")
    ax.legend(frameon=False)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def equity_figure(eq: pd.DataFrame, path: Path, label: str):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))
    ax.plot(eq["date"], eq["strategy_curve"], lw=1.2, label=label)
    ax.plot(eq["date"], eq["benchmark_curve"], lw=1.0, ls="--", label="Equal-weight universe")
    ax.set_ylabel("Growth of 1 (net of costs)")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def protocol_figure(tab: pd.DataFrame, metric: str, path: Path):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.7))
    vals = tab[metric]
    colors = ["#B5563A" if ("random" in i or "no_purge" in i) else
              "#888780" if "shuffled" in i else "#2E6F9E" for i in tab.index]
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([i.replace("_", " ") for i in tab.index])
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " "))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
