"""Figure 1 (hero): one historical key span decides the action.

Two aligned panels share the y-axis (the five historical blocks), so the reader
sees behaviour and measured credit side by side on the same rows. Quantitative
production route: plotting code, not illustrative art.

Left  panel - critical-decision accuracy after replacing one block.
Right panel - the same replacement's signed effect on the correct-action score.
"""

import csv
import os

import matplotlib.pyplot as plt

import style
from style import HARMFUL, INK, INK2, MUTED, NEUTRAL, USEFUL, AXIS, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

ORDER = ["initial", "reference_1", "superseded", "reference_2", "latest"]
LABEL = {
    "initial": "Initial record",
    "reference_1": "Reference 1",
    "reference_2": "Reference 2",
    "superseded": "Superseded update",
    "latest": "Latest update",
}
COLOR = {
    "initial": NEUTRAL,
    "reference_1": NEUTRAL,
    "reference_2": NEUTRAL,
    "superseded": HARMFUL,
    "latest": USEFUL,
}


def main():
    style.apply_style()
    with open(os.path.join(DATA, "pilot10.csv")) as fh:
        rows = list(csv.DictReader(fh))
    by = {r["role"]: r for r in rows}
    baseline = float(rows[0]["baseline_accuracy"])

    # y order: latest at top
    ys = {r: i for i, r in enumerate(reversed(ORDER))}
    n = len(ORDER)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(style.TEXTWIDTH, 2.18),
        gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(left=0.148, right=0.988, top=0.795, bottom=0.185,
                        wspace=0.30)

    # ---------------- left: accuracy ----------------
    axL.axvline(baseline, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    axL.annotate(f"unmodified {baseline:.0%}", xy=(baseline, -0.62),
                 ha="center", va="center", fontsize=6.3, color=INK2)
    for role in ORDER:
        r = by[role]
        y = ys[role]
        acc = float(r["patched_accuracy"])
        c = COLOR[role]
        axL.annotate("", xy=(acc, y), xytext=(baseline, y),
                     arrowprops=dict(arrowstyle="-|>,head_width=.16,head_length=.34",
                                     color=c, lw=1.5, shrinkA=0, shrinkB=0),
                     zorder=3)
        axL.plot([baseline], [y], marker="o", ms=2.6, mfc="white",
                 mec=MUTED, mew=0.8, zorder=4)
        axL.plot([acc], [y], marker="o", ms=5.0, color=c, zorder=5)
        # keep the value label clear of the dashed baseline it would otherwise
        # sit on for the two neutral references
        if acc < 0.15 or acc >= baseline:
            lx, ha = acc + 0.035, "left"
        else:
            lx, ha = acc - 0.035, "right"
        axL.text(lx, y + 0.26, f"{acc:.0%}", ha=ha, va="bottom",
                 fontsize=7.0, fontweight="bold", color=c, zorder=6)
    axL.set_yticks(range(n))
    axL.set_yticklabels([LABEL[r] for r in reversed(ORDER)], fontsize=7.2)
    for t, r in zip(axL.get_yticklabels(), reversed(ORDER)):
        if r in ("superseded", "latest"):
            t.set_color(COLOR[r])
            t.set_fontweight("bold")
    axL.set_xlim(-0.08, 1.22)
    axL.set_xticks([0, 0.5, 1.0])
    axL.set_xticklabels(["0%", "50%", "100%"])
    axL.set_xlabel("accuracy", fontsize=7.2, labelpad=1.5)
    axL.set_title("(a)  critical-decision accuracy", fontsize=7.6, pad=4)
    axL.grid(axis="y", visible=False)
    style.despine(axL)
    axL.set_ylim(-0.92, n - 0.20)

    # ---------------- right: signed effect ----------------
    axR.axvline(0, color=AXIS, lw=0.8, zorder=1)
    for role in ORDER:
        r = by[role]
        y = ys[role]
        e = float(r["correct_score_effect"])
        c = COLOR[role]
        axR.barh(y, e, height=0.42, color=c, zorder=3,
                 edgecolor="white", linewidth=0.8)
        off = 0.008 if e > 0 else -0.008
        axR.text(e + off, y, f"{e:+.3f}", ha="left" if e > 0 else "right",
                 va="center", fontsize=6.7, fontweight="bold", color=c, zorder=4)
    axR.set_yticks([])
    axR.set_xlim(-0.152, 0.335)
    axR.set_xticks([-0.1, 0, 0.1, 0.2])
    axR.set_xlabel("effect on correct-action score", fontsize=7.2, labelpad=1.5)
    axR.set_title("(b)  signed influence", fontsize=7.6, pad=4)
    axR.grid(axis="y", visible=False)
    style.despine(axR, keep=("bottom",))
    axR.set_ylim(-0.92, n - 0.20)
    axR.text(-0.145, -0.62, "replacement helps", fontsize=6.3, color=MUTED,
             ha="left", va="center")
    axR.text(0.328, -0.62, "replacement hurts", fontsize=6.3, color=MUTED,
             ha="right", va="center")

    fig.text(0.008, 0.975, "Same screenshots. Opposite effects.",
             fontsize=8.8, fontweight="bold", color=INK, va="top")

    out = os.path.join(ROOT, "figures", "fig1_intervention.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
