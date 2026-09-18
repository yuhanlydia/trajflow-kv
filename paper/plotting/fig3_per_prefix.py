"""Figure 3: the opposing pattern holds per prefix, not only on average.

Each point is one controlled prefix, placed by its superseded-block credit (x)
and its latest-block credit (y). A role-driven account predicts the lower-right
quadrant. Shading marks that quadrant; the legend counts how many prefixes land
in it, in the one-sign-only band, and in neither.
"""

import csv
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import style
from style import HARMFUL, INK, INK2, MUTED, NEUTRAL, USEFUL, AXIS, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# axis windows fitted to the observed range so the cloud fills the panel
XLIM = (-0.0270, 0.0075)
YLIM = (-0.0045, 0.0345)


def main():
    style.apply_style()
    with open(os.path.join(DATA, "credit200_per_prefix.csv")) as fh:
        rows = list(csv.DictReader(fh))
    with open(os.path.join(DATA, "credit200_stats.json")) as fh:
        stats = json.load(fh)

    fig, ax = plt.subplots(figsize=(3.45, 2.42))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.985, bottom=0.145)

    # expected quadrant: superseded < 0 (replacement wins) and latest > 0
    ax.add_patch(Rectangle((XLIM[0], 0), -XLIM[0], YLIM[1],
                           facecolor=USEFUL, alpha=0.06,
                           edgecolor="none", zorder=0))
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)

    both, one, neither = [], [], []
    for r in rows:
        s, l = float(r["superseded"]), float(r["latest"])
        (both if (s < 0 and l > 0) else one if (s < 0 or l > 0) else neither).append((s, l))

    for pts, c, lab, a, sz in (
            (both, USEFUL, f"both signs as predicted  ({len(both)}/{len(rows)})", 0.72, 12),
            (one, NEUTRAL, f"one sign only  ({len(one)}/{len(rows)})", 0.85, 16),
            (neither, HARMFUL, f"neither  ({len(neither)}/{len(rows)})", 0.9, 16)):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=sz, color=c, alpha=a, linewidths=0.4,
                   edgecolors="white" if c == NEUTRAL else "none",
                   zorder=4 if c == NEUTRAL else 3, label=lab)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([-0.02, -0.01, 0])
    ax.set_xticklabels(["−.02", "−.01", "0"])
    ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.set_yticklabels(["0", "+.01", "+.02", "+.03"])
    ax.set_xlabel("superseded-update credit", fontsize=7.3, labelpad=2)
    ax.set_ylabel("latest-update credit", fontsize=7.3, labelpad=2)
    style.despine(ax)

    ax.text(XLIM[1] - 0.0012, YLIM[1] - 0.0012,
            f"AUROC {stats['auroc_superseded_vs_rest']:.2f} / "
            f"{stats['auroc_latest_vs_rest']:.2f}\n"
            "separating each update block\nfrom the reference blocks",
            fontsize=6.2, color=INK2, ha="right", va="top", linespacing=1.35,
            zorder=6)

    leg = ax.legend(loc="lower left", bbox_to_anchor=(-0.015, -0.02),
                    fontsize=6.2, handletextpad=0.45, borderpad=0.25,
                    labelspacing=0.32, markerscale=0.9)
    for t in leg.get_texts():
        t.set_color(INK2)

    out = os.path.join(ROOT, "figures", "fig3_per_prefix.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
