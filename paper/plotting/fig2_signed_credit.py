"""Figure 2 (core diagnostic): the sign of memory credit is set by role, not age.

Every one of the 200 controlled prefixes contributes one measurement per
historical block. The strip shows all 200 per-prefix values; the bold marker and
whisker show the mean and a bootstrap 95% interval over prefixes.

The scientific job is polarity, so the encoding is the documented diverging pair
(blue = the original block is preferable to its matched replacement, red = the
replacement is preferable). Neutral grey marks the two matched references.
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import style
from style import HARMFUL, INK, INK2, MUTED, NEUTRAL, USEFUL, AXIS, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# top-to-bottom reading order
ORDER = ["latest", "reference_2", "superseded", "reference_1", "initial"]
LABEL = {
    "latest": "Latest update",
    "reference_2": "Reference 2",
    "superseded": "Superseded update",
    "reference_1": "Reference 1",
    "initial": "Initial record",
}
COLOR = {
    "latest": USEFUL,
    "reference_2": NEUTRAL,
    "superseded": HARMFUL,
    "reference_1": NEUTRAL,
    "initial": NEUTRAL,
}


def main():
    style.apply_style()
    with open(os.path.join(DATA, "credit200.csv")) as fh:
        rows = list(csv.DictReader(fh))
    with open(os.path.join(DATA, "credit200_by_role.csv")) as fh:
        summ = {r["role"]: r for r in csv.DictReader(fh)}

    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(style.TEXTWIDTH, 2.55))
    fig.subplots_adjust(left=0.155, right=0.995, top=0.80, bottom=0.215)

    n = len(ORDER)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    # tint the two update rows so the contrast with the references reads at a
    # glance; the two reference rows and the initial record stay on white
    for i, role in enumerate(ORDER):
        if role in ("superseded", "latest"):
            ax.axhspan(n - 1.5 - i, n - 0.5 - i, color="#f4f6f7", zorder=0)

    for i, role in enumerate(ORDER):
        y = n - 1 - i
        vals = np.array([float(r["memory_advantage"]) for r in rows
                         if r["role"] == role])
        c = COLOR[role]
        jit = rng.uniform(-0.20, 0.20, len(vals))
        ax.scatter(vals, y + jit, s=3.0, color=c, alpha=0.30,
                   linewidths=0, zorder=3)
        s = summ[role]
        lo, hi, m = float(s["ci_lo"]), float(s["ci_hi"]), float(s["mean"])
        ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="butt",
                zorder=5)
        ax.plot([m], [y], marker="o", ms=6.0, color=c, mec="white", mew=0.9,
                zorder=6)
        # the dominant-sign fraction sits in its own right-hand column so it can
        # never collide with the strip it describes
        frac = float(s["pos_frac"])
        if frac >= 0.5:
            txt = f"{frac:.1%} positive"
        else:
            txt = f"{1 - frac:.1%} negative"
        ax.text(0.0292, y, txt, ha="right", va="center", fontsize=6.7,
                fontweight="bold", color=c, zorder=7)

    ax.set_yticks(range(n))
    # rows are drawn top-down as reversed(ORDER) (index 0 = lowest y), so the
    # tick labels must use the same reversed order
    ax.set_yticklabels([LABEL[r] for r in reversed(ORDER)], fontsize=7.4)
    for t, r in zip(ax.get_yticklabels(), reversed(ORDER)):
        if r in ("superseded", "latest"):
            t.set_color(COLOR[r])
            t.set_fontweight("bold")
    ax.set_xlim(-0.0205, 0.0295)
    ax.set_xticks([-0.02, -0.01, 0, 0.01])
    ax.set_xticklabels(["−.02", "−.01", "0", "+.01"])
    ax.set_xlabel("signed memory credit  $c^{\\mathrm{val}}$   "
                  "(candidate-value units)", fontsize=7.4, labelpad=2)
    ax.set_ylim(-0.62, n - 0.38)
    ax.grid(axis="y", visible=False)
    style.despine(ax)
    # a last vertical tick marks where the annotation column begins
    ax.axvline(0.0140, color=GRID, lw=0.7, zorder=0)

    # sign convention: positive means the original block beats its matched
    # replacement, i.e. the original deserves to keep influencing the decision.
    fig.text(0.5, 0.012,
             "negative: the matched replacement is preferred   ·   "
             "positive: the original block is preferred",
             fontsize=6.4, color=MUTED, ha="center", va="bottom")

    fig.text(0.008, 0.985,
             "The sign is set by role, not by age.",
             fontsize=8.6, fontweight="bold", color=INK, va="top")
    fig.text(0.008, 0.895,
             "200 controlled prefixes · 1,000 matched-block measurements · "
             "frozen Qwen2.5-VL-3B, decoder K, layers 12–23",
             fontsize=6.6, color=INK2, va="top")

    out = os.path.join(ROOT, "figures", "fig2_signed_credit.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
