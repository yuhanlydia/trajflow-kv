"""Figure 4: confidence is not a memory diagnostic.

The same four-hundred paired MiniWoB cases, scored two ways. Left: the mean
top-1 candidate margin the training objective actually optimises. Right: the
number of cases the agent completes.

Both panels share the method axis, so the divergence is read across rows rather
than through a second y-scale. Panels are dot plots, not bars: the quantities
have non-zero baselines and a truncated bar would misstate them.
"""

import csv
import os

import matplotlib.pyplot as plt

import style
from style import HARMFUL, INK, INK2, MUTED, NEUTRAL, USEFUL, AXIS, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# top-to-bottom rows
ROWS = ["base", "shared_warm_start", "return_weighted",
        "successful_only_action_ce", "shuffled_return"]
LABEL = {
    "base": "Frozen base",
    "shared_warm_start": "Shared warm start",
    "return_weighted": "Return-weighted",
    "successful_only_action_ce": "Successful-only CE",
    "shuffled_return": "Shuffled return",
}
ACCENT = {
    "return_weighted": "#eb6834",     # categorical slot 2
    "successful_only_action_ce": USEFUL,  # categorical slot 1
}


def main():
    style.apply_style()
    with open(os.path.join(DATA, "miniwob.csv")) as fh:
        by = {r["method"]: r for r in csv.DictReader(fh)}
    base = by["base"]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(style.TEXTWIDTH, 2.25),
        gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(left=0.152, right=0.988, top=0.745, bottom=0.185,
                        wspace=0.16)

    n = len(ROWS)
    ys = {m: n - 1 - i for i, m in enumerate(ROWS)}

    # ---- (a) margin ----
    bm = float(base["mean_top1_margin"])
    axA.axvline(bm, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    for m in ROWS:
        r = by[m]
        y, v = ys[m], float(r["mean_top1_margin"])
        c = ACCENT.get(m, NEUTRAL)
        axA.plot([bm, v], [y, y], color=c, lw=1.1, alpha=0.55, zorder=2)
        axA.plot([v], [y], marker="o", ms=5.4, color=c, zorder=4,
                 mec="white", mew=0.7)
        axA.text(v, y + 0.30, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color=c, zorder=5)
    axA.set_yticks(range(n))
    axA.set_yticklabels([LABEL[m] for m in reversed(ROWS)], fontsize=7.2)
    for t, m in zip(axA.get_yticklabels(), reversed(ROWS)):
        if m in ACCENT:
            t.set_color(ACCENT[m])
            t.set_fontweight("bold")
    axA.set_xlim(0.1435, 0.1755)
    axA.set_xticks([0.15, 0.16, 0.17])
    axA.set_xlabel("mean top-1 candidate margin", fontsize=7.2, labelpad=2)
    axA.set_title("(a)  what the objective optimises", fontsize=7.5, pad=5)
    axA.grid(axis="y", visible=False)
    style.despine(axA)
    axA.set_ylim(-0.62, n - 0.20)
    axA.text(bm, -0.50, "frozen base", ha="center", va="center",
             fontsize=6.2, color=MUTED)

    # ---- (b) success ----
    bs = float(base["successes"])
    axB.axvline(bs, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    for m in ROWS:
        r = by[m]
        y, v = ys[m], float(r["successes"])
        c = ACCENT.get(m, NEUTRAL)
        axB.plot([bs, v], [y, y], color=c, lw=1.1, alpha=0.55, zorder=2)
        axB.plot([v], [y], marker="o", ms=5.4, color=c, zorder=4,
                 mec="white", mew=0.7)
        d = v - bs
        txt = f"{v:.0f}" + ("" if d == 0 else f"  ({d:+.0f})")
        axB.text(v, y + 0.30, txt, ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color=c, zorder=5)
    axB.set_yticks(range(n))
    axB.set_yticklabels([])
    axB.set_xlim(325.5, 335.5)
    axB.set_xticks([328, 330, 332, 334])
    axB.set_xlabel("MiniWoB tasks completed (of 400)", fontsize=7.2, labelpad=2)
    axB.set_title("(b)  what the agent actually does", fontsize=7.5, pad=5)
    axB.grid(axis="y", visible=False)
    style.despine(axB, keep=("bottom",))
    axB.set_ylim(-0.62, n - 0.20)
    axB.text(bs, -0.50, "frozen base", ha="center", va="center",
             fontsize=6.2, color=MUTED)

    fig.text(0.008, 0.985,
             "Sharper scores, same behaviour.",
             fontsize=8.6, fontweight="bold", color=INK, va="top")
    fig.text(0.008, 0.878,
             "400 paired MiniWoB cases, identical histories and action space",
             fontsize=6.6, color=INK2, va="top")

    out = os.path.join(ROOT, "figures", "fig4_divergence.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
