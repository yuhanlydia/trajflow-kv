"""Shared visual system for the manuscript figures.

Palette is the dataviz reference instance, validated with
    node scripts/validate_palette.js "#2a78d6,#e34948" --mode light --surface "#ffffff" --pairs all
    -> ALL CHECKS PASS (worst all-pairs CVD dE 21.6 protan; normal-vision dE 32.3)

Roles used here:
  diverging poles  useful = #2a78d6 (blue), harmful = #e34948 (red)
  neutral midpoint #898781 (documented muted ink; a deliberate diverging
                   midpoint, so the categorical chroma floor does not apply)
Figure surface is white to match the printed page.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette ---------------------------------------------------------------
USEFUL = "#2a78d6"   # positive signed credit
HARMFUL = "#e34948"  # negative signed credit
NEUTRAL = "#898781"  # near-zero / reference
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# 3-slot categorical (blue/orange/aqua) -- validated all-pairs, aqua carries the
# relief rule (visible direct labels) because its contrast is 2.82:1.
CAT3 = ["#2a78d6", "#eb6834", "#1baf7a"]

SURFACE = "#ffffff"

TEXTWIDTH = 5.5  # inches; ICLR text block. Verified against the compiled PDF.
COLWIDTH = 2.65


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def sign_color(v):
    return USEFUL if v > 0 else HARMFUL


def savefig(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print("wrote", path)
