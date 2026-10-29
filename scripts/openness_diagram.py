#!/usr/bin/env python3
"""What `open_share` counts, drawn in side elevation.

The measure itself is one line of depth_frames.py:

    open_share = (sky | (horizontal_distance / camera_height > 10)).mean()

-- the fraction of the frame that is either sky or further away than ten
times the camera's own height. The threshold is in camera heights rather
than metres so that both viewpoints are asked the same question: how much
of what I can see is too far away to matter to a body my size. The cost is
that the two thresholds are different distances on the ground, 16.0 m for
the human camera and 4.5 m for the dog's, which is exactly what this
drawing is for.

Writes aoi/24_openness_diagram.png.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from aoi_plots import AOI_DIR, INK, MUTED, HUMAN, DOG  # noqa: E402

VIEWS = [("human", 1.60, HUMAN), ("dog", 0.45, DOG)]
XMAX, YMAX = 20.0, 3.6
SKY = "#cfe3f2"
OPEN = "#e8eef3"
NEAR = "#d8d2c8"


def panel(ax, name, h, colour, show_ray=False):
    thr = 10 * h

    #  sky: everything above the horizon, which sits at the camera's height
    ax.add_patch(Rectangle((0, h), XMAX, YMAX - h, color=SKY, lw=0))
    #  ground beyond the threshold counts as open; nearer than it does not
    ax.add_patch(Rectangle((thr, 0), XMAX - thr, h, color=OPEN, lw=0))
    ax.add_patch(Rectangle((0, 0), thr, h, color=NEAR, lw=0))
    #  something to be blocked by, so the near band reads as a place
    ax.add_patch(Rectangle((thr * 0.45, 0), thr * 0.07, h * 0.75,
                           color="#9c9384", lw=0))

    ax.axhline(0, color=INK, lw=1.4)
    ax.plot([0, XMAX], [h, h], color=MUTED, lw=0.9, ls=(0, (2, 3)))
    ax.plot([0, 0], [0, h], color=colour, lw=2.0)
    ax.plot([0], [h], marker="o", ms=9, color=colour, zorder=5)

    ax.axvline(thr, color=INK, lw=1.3, ls=(0, (4, 3)))
    #  the label goes on whichever side of the line has room
    right = thr < XMAX * 0.6
    ax.annotate(f"10 \u00d7 camera height = {thr:.1f} m",
                xy=(thr + (0.4 if right else -0.4), h + (YMAX - h) * 0.52),
                ha="left" if right else "right",
                color=INK, fontsize=9.5)
    ax.annotate("openness adds the ground beyond here",
                xy=(thr + (0.4 if right else -0.4), h + (YMAX - h) * 0.3),
                ha="left" if right else "right",
                color=MUTED, fontsize=8.5)

    #  one ray, so the drawing shows what is actually computed: the depth
    #  model gives distance ALONG the ray, and the pixel's elevation angle
    #  turns it into the horizontal distance the threshold is applied to
    if show_ray:
        x0 = min(thr * 0.7, XMAX * 0.55)
        ax.plot([0, x0], [h, 0], color=INK, lw=1.2)
        th = np.degrees(np.arctan2(h, x0))
        ax.annotate("\u03b8", xy=(x0 * 0.085, h - 0.1), color=INK,
                    fontsize=10, va="top")
        ax.annotate("d", xy=(x0 * 0.55, h * 0.47), color=INK, fontsize=9.5,
                    rotation=-th * 2.4)
        ax.annotate("", xy=(x0, -0.42), xytext=(0, -0.42),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
        ax.annotate("d\u00b7cos\u03b8, the horizontal distance",
                    xy=(x0 + 0.4, -0.42), va="center", color=INK, fontsize=8.5)

    #  keep the sky label off the threshold line, whichever side has room
    sky_right = thr < XMAX * 0.4
    ax.annotate("sky \u2014 this band alone is the sky share",
                xy=(XMAX - 0.4 if sky_right else 0.3, YMAX - 0.3),
                ha="right" if sky_right else "left",
                va="top", color="#3d6b8e", fontsize=8.5)
    ax.annotate("near field \u2014 not open", xy=(thr * 0.5, -0.28),
                ha="center", va="top", color="#6b6253",
                fontsize=8.5) if not show_ray else None
    ax.annotate(f"{h:.2f} m", xy=(-0.25, h / 2), ha="right", va="center",
                color=colour, fontsize=9)

    ax.set_xlim(-1.6, XMAX)
    ax.set_ylim(-1.0, YMAX)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, XMAX + 1, 5))
    ax.set_xlabel("horizontal distance from the camera (m)", color=MUTED,
                  fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"{name} \u00b7 camera at {h:.2f} m", color=colour,
                 fontsize=11, loc="left")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.2), dpi=160, sharex=True)
    for i, (ax, (name, h, colour)) in enumerate(zip(axes, VIEWS)):
        panel(ax, name, h, colour, show_ray=(i == 0))
    axes[0].set_xlabel("")
    fig.suptitle("What the two measures count: the sky band, and the ground "
                 "beyond ten camera heights", color=INK, fontsize=12.5)
    fig.text(0.5, 0.015,
             "Sky share counts the blue band only and needs just the "
             "segmenter. Openness counts that band plus any ground further "
             "than 10 \u00d7 the camera's own height.\nThe threshold is in body "
             "heights so both viewpoints answer the same question, which puts "
             "it at 16.0 m for the human camera and 4.5 m for the dog's \u2014 "
             "the asymmetry behind the indoor result.\nThe distance compared "
             "against it is horizontal: the depth model gives d along the ray, "
             "and the pixel's elevation angle \u03b8 turns it into "
             "d\u00b7cos\u03b8.",
             ha="center", va="top", color=MUTED, fontsize=8.5, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    out = AOI_DIR / "24_openness_diagram.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
