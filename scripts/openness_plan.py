#!/usr/bin/env python3
"""The same measure seen from above: the forward wedge and the two thresholds.

The side elevation (openness_diagram.py) shows how a pixel's distance is
computed. This is the orthogonal view -- looking straight down -- and it shows
the two things the side view cannot:

  the field is a forward 90 degree WEDGE, not a hemisphere, so neither measure
  here is a sky view factor;

  "ten camera heights" is a circle on the ground whose radius depends on who
  is looking: 16.0 m for the human camera, 4.5 m for the dog's. The band
  between the two arcs is ground the dog counts as open and the human does
  not, from the same spot at the same instant.

Sky does not appear in a plan view at all -- it is above this plane, and both
measures count it.

Writes aoi/25_openness_plan.png.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from aoi_plots import AOI_DIR, INK, MUTED, HUMAN, DOG  # noqa: E402

HFOV = 90.0
RMAX = 22.0
THR = {"dog": 4.5, "human": 16.0}
#  matplotlib measures wedge angles from the +x axis; the camera looks up the
#  page, so the forward wedge is centred on 90 degrees
T1, T2 = 90 - HFOV / 2, 90 + HFOV / 2
NEAR, MID, FAR = "#d8d2c8", "#f0d9c4", "#e4edf2"


def main():
    fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=160)

    ax.add_patch(Wedge((0, 0), RMAX, T1, T2, facecolor=FAR, lw=0))
    ax.add_patch(Wedge((0, 0), THR["human"], T1, T2, facecolor=MID, lw=0))
    ax.add_patch(Wedge((0, 0), THR["dog"], T1, T2, facecolor=NEAR, lw=0))

    #  arc labels go outside the wedge on the left, where nothing else sits
    for r, name, colour in ((THR["dog"], "dog", DOG),
                            (THR["human"], "human", HUMAN)):
        a = np.radians(np.linspace(T1, T2, 200))
        ax.plot(r * np.cos(a), r * np.sin(a), color=colour, lw=1.8)
        edge = r / np.sqrt(2)
        ax.plot([-edge - 1.0, -edge], [edge, edge], color=colour, lw=0.7)
        ax.annotate(f"10 \u00d7 {name} height = {r:.1f} m",
                    xy=(-edge - 1.3, edge), ha="right", va="center",
                    color=colour, fontsize=9.5)

    for t in (T1, T2):
        a = np.radians(t)
        ax.plot([0, RMAX * np.cos(a)], [0, RMAX * np.sin(a)],
                color=INK, lw=1.0)
    ax.plot([0], [0], marker="o", ms=10, color=INK, zorder=5)
    ax.annotate("camera", xy=(0, -1.0), ha="center", va="top", color=INK,
                fontsize=9.5)

    #  zone labels: the two wide bands take a label on the centre line, the
    #  small near sector is too narrow for one and gets a leader instead
    ax.plot([-1.1, -4.6], [1.1, 1.1], color="#6b6253", lw=0.7)
    ax.annotate("neither counts\nthis as open", xy=(-4.9, 1.1), ha="right",
                va="center", color="#6b6253", fontsize=9, linespacing=1.5)
    ax.annotate("the dog counts this as open,\nthe human does not",
                xy=(0, 8.8), ha="center", va="center", color="#9c5a28",
                fontsize=9.5, linespacing=1.5)
    ax.annotate("both count this as open", xy=(0, 19.2), ha="center",
                va="center", color="#3d6b8e", fontsize=9.5)
    ax.annotate("forward 90\u00b0 field \u2014 this is why neither measure is "
                "a sky view factor",
                xy=(0, -3.0), ha="center", va="top", color=MUTED, fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlim(-RMAX * 0.95, RMAX * 0.82)
    ax.set_ylim(-5.2, RMAX + 0.6)
    ax.axis("off")
    fig.suptitle("The same measure from above: one wedge, two thresholds",
                 color=INK, fontsize=12.5, y=0.97)
    fig.text(0.5, 0.025,
             "Plan view from directly overhead, both viewpoints drawn at the "
             "same scale from the same spot.\nSky does not appear here — it "
             "is above this plane, and both measures count all of it.",
             ha="center", va="top", color=MUTED, fontsize=8.5, linespacing=1.6)
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.04, right=0.96)
    out = AOI_DIR / "25_openness_plan.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
