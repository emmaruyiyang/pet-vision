#!/usr/bin/env python3
"""Checklist item 12: what the dichromat transform does to urban target colours.

The transform itself lives in dog_vision.py; this draws the figure and puts a
number on it. The number is the point. A picture of a red sign going brown
invites the reader to judge by eye, and by eye a dichromat render still looks
like a picture of a sign. What matters for H2 is not how it looks, it is
whether the target still SEPARATES from what is behind it.

So each target colour is paired with the background it is actually seen
against, and the separation is measured before and after, as Euclidean distance
in CIE Lab -- roughly, how many just-noticeable steps apart they are. Under
about 10 the two are hard to tell apart at a glance.

AS ALWAYS WITH THIS TRANSFORM: it is a model of which colours get CONFUSED, not
of what a dog perceives. Deuteranope is the default because the canine
long-wavelength cone (555 nm) sits near the human L cone (560) rather than the
M cone (530); the protanope column shows how much that choice moves things, and
it moves reds a lot.

Writes aoi/10_colour_collapse.png.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aoi_plots import AOI_DIR, INK, MUTED, style  # noqa: E402
from dog_vision import dichromat  # noqa: E402

OUT = AOI_DIR

#  (name, target sRGB, background it is seen against, background name)
CASES = [
    ("red fire hydrant", (200, 35, 35), (60, 150, 60), "grass"),
    ("red stop sign", (196, 30, 36), (120, 120, 120), "asphalt"),
    ("orange cone", (240, 110, 30), (150, 148, 140), "pavement"),
    ("yellow tactile paving", (235, 190, 40), (150, 148, 140), "pavement"),
    ("green exit sign", (0, 122, 61), (235, 235, 230), "white wall"),
    ("blue parking sign", (0, 90, 160), (235, 235, 230), "white wall"),
    ("brown timber bollard", (120, 80, 45), (60, 150, 60), "grass"),
    ("A-frame sign, black on white", (30, 30, 30), (240, 240, 235), "white board"),
]


def to_lab(rgb_u8):
    """sRGB -> CIE Lab (D65). Small, exact, and avoids a scikit-image dependency."""
    c = np.asarray(rgb_u8, dtype=float).reshape(-1, 3) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def render(rgb, kind):
    px = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    return dichromat(px, kind=kind)[0, 0]


def sep(a, b):
    la, lb = to_lab([a, b])
    return float(np.linalg.norm(la - lb))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="Lab distance below which two colours read as the same")
    args = ap.parse_args()

    rows = []
    for name, tgt, bg, bgname in CASES:
        r = {"name": name, "bgname": bgname,
             "orig": (tgt, bg, sep(tgt, bg))}
        for kind, key in (("deuteranopia", "deut"), ("protanopia", "prot")):
            t2, b2 = render(tgt, kind), render(bg, kind)
            r[key] = (t2, b2, sep(t2, b2))
        rows.append(r)
    rows.sort(key=lambda r: r["deut"][2] / max(r["orig"][2], 1e-6))

    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.62 * n + 2.2), dpi=150,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax, ax2 = axes
    ax.set_xlim(0, 3.4)
    ax.set_ylim(-0.6, n)
    ax.axis("off")
    for col, label in ((0.35, "original"), (1.45, "deuteranope"), (2.55, "protanope")):
        ax.text(col + 0.35, n - 0.25, label, ha="center", color=MUTED, fontsize=9)
    for i, r in enumerate(rows):
        y = n - 1.1 - i
        ax.text(0.28, y + 0.18, r["name"], ha="right", color=INK, fontsize=9)
        ax.text(0.28, y - 0.12, f"on {r['bgname']}", ha="right",
                color=MUTED, fontsize=7.5)
        for col, key in ((0.35, "orig"), (1.45, "deut"), (2.55, "prot")):
            tgt, bg, dist = r[key]
            ax.add_patch(plt.Rectangle((col, y - 0.3), 0.7, 0.62,
                                       color=np.array(bg) / 255.0))
            ax.add_patch(plt.Rectangle((col + 0.17, y - 0.17), 0.36, 0.36,
                                       color=np.array(tgt) / 255.0))
            ax.text(col + 0.78, y, f"{dist:.0f}", va="center",
                    color=INK if dist >= args.threshold else MUTED, fontsize=8)

    style(ax2)
    y = np.arange(n)
    keep = np.array([r["deut"][2] / max(r["orig"][2], 1e-6) for r in rows])
    colour = ["var" for _ in rows]
    bars = ax2.barh(y, keep * 100, height=0.6,
                    color=["#c05621" if k < 0.5 else "#8a8a8a" for k in keep])
    ax2.axvline(100, color=MUTED, lw=0.8, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels([r["name"] for r in rows], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("separation kept, deuteranope vs original (%)",
                   color=MUTED, fontsize=9)
    for i, (bar, r) in enumerate(zip(bars, rows)):
        ax2.text(bar.get_width() + 2, i,
                 f"{r['orig'][2]:.0f} → {r['deut'][2]:.0f}",
                 va="center", color=MUTED, fontsize=8)
    worst = rows[0]
    ax2.set_title(f"Worst case: {worst['name']} on {worst['bgname']}, "
                  f"{worst['orig'][2]:.0f} → {worst['deut'][2]:.0f}",
                  color=INK, fontsize=10, loc="left")

    fig.suptitle("Target-background separation under a dichromat transform\n"
                 "CIE Lab distance; under about 10 the two read as the same "
                 "colour at a glance",
                 color=INK, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "10_colour_collapse.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"-> {out}\n")
    print(f"{'target':<30}{'on':<14}{'orig':>7}{'deut':>7}{'prot':>7}{'kept':>8}")
    for r in rows:
        print(f"{r['name']:<30}{r['bgname']:<14}{r['orig'][2]:>7.1f}"
              f"{r['deut'][2]:>7.1f}{r['prot'][2]:>7.1f}"
              f"{r['deut'][2] / max(r['orig'][2], 1e-6):>7.0%}")


if __name__ == "__main__":
    main()
