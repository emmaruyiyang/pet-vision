#!/usr/bin/env python3
"""Checklist items 7 and 10: the paired difference over time, and the 2x2.

Both read the per-second CSVs that aoi_frames.py and depth_frames.py already
wrote; neither re-runs a model.

ITEM 7, the paired difference curve. Every second has a human panel and a dog
panel of the SAME instant, so the difference is paired by construction and the
scene cancels out of it. What is left is the height. The curve is that
difference second by second; the band behind it is the mean with a 95%
moving-block bootstrap interval, 30-second blocks, because consecutive seconds
of a walk are nearly the same picture and treating them as independent gives
an interval several times too narrow.

ITEM 10, the 2x2. Species x indoor/outdoor, as the checklist asks.

  READ THE INDOOR HALF WITH CARE. There is exactly one indoor stretch in this
  recording -- the 9.6-minute sit, 19.0-28.6 min -- and it is one fixed scene.
  The width of an indoor violin is therefore the spread of ONE scene over ten
  minutes, not the spread of indoor scenes. The outdoor violins are a real
  sample of scenes; the indoor ones are a sample of seconds. They do not mean
  the same thing and the figure says so on its face. A second indoor location
  would fix this and costs one short shoot.

Writes aoi/16_paired_delta.png and aoi/17_two_by_two.png.

Usage
-----
    python3 scripts/walk_contrasts.py
    python3 scripts/walk_contrasts.py --metric ground_share
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aoi_plots import (AOI_DIR, INK, MUTED, HUMAN, DOG, SEDENTARY,
                       load, block_boot, style, smooth)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = AOI_DIR
#  the four composition groups worth a row, plus openness from the depth side
DELTA_ROWS = ["ground", "vegetation", "structure", "sky"]


KEYS = ("open_share", "ground_share", "obstacle_share", "sky_share")


def load_depth():
    """Per-second depth metrics for both views, on the pairs BOTH views have.

    depth_frames.py skips a frame it cannot read, and it does not always skip
    the same one on both sides: the human view has 2033 seconds here and the
    dog view 1996. Subtracting those two arrays as they come would silently
    compare second i of one walk with second i+37 of the other, so the two are
    intersected on pair_index first.
    """
    with (AOI_DIR / "depth_metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for view in ("human", "dog"):
        by[view] = {int(x["pair_index"]): x for x in rows if x["view"] == view}
    shared = sorted(set(by["human"]) & set(by["dog"]))
    dropped = len(set(by["human"]) | set(by["dog"])) - len(shared)
    if dropped:
        print(f"  depth: {len(shared)} paired seconds, {dropped} unpaired dropped")
    out = {}
    for view in ("human", "dog"):
        r = [by[view][i] for i in shared]
        out[view] = (np.array([float(x["t_s"]) for x in r]),
                     {k: np.array([float(x[k]) for x in r]) for k in KEYS})
    return out


def fig_delta(names, data, depth, out):
    th, mh = data["human"]
    _, md = data["dog"]
    walking = (th < SEDENTARY[0]) | (th > SEDENTARY[1])

    rows = [(g, md[:, names.index(g)] - mh[:, names.index(g)], th)
            for g in DELTA_ROWS]
    dt, dm = depth["human"][0], depth["dog"][1]["open_share"] - depth["human"][1]["open_share"]
    rows.append(("openness", dm, dt))

    fig, axes = plt.subplots(len(rows), 1, figsize=(11.5, 1.75 * len(rows)),
                             dpi=150, sharex=True)
    for ax, (name, d, t) in zip(axes, rows):
        keep = (t < SEDENTARY[0]) | (t > SEDENTARY[1])
        lo, hi = block_boot(d[keep] * 100)
        mean = d[keep].mean() * 100
        ax.axhspan(lo, hi, color="#c9d8e8", alpha=0.75, zorder=1)
        ax.axhline(mean, color=HUMAN, lw=1.2, zorder=2)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
        ax.axvspan(SEDENTARY[0] / 60, SEDENTARY[1] / 60, color="#ededed",
                   zorder=0)
        ax.plot(t / 60, smooth(d * 100, 15), color=DOG, lw=1.0, zorder=3)
        style(ax)
        ax.set_ylabel(name, color=INK, fontsize=9)
        crosses = "crosses zero" if lo <= 0 <= hi else "excludes zero"
        ax.set_title(f"mean {mean:+.1f} pts, 95% CI [{lo:+.1f}, {hi:+.1f}] "
                     f"· {crosses}", color=MUTED, fontsize=8, loc="right")
    axes[-1].set_xlabel("minutes into the walk", color=MUTED)
    fig.suptitle("Item 7: dog minus human, second by second, paired on the "
                 "same instant\norange = 15 s moving average  ·  blue band "
                 "= mean with its 95% block-bootstrap CI  ·  grey = the "
                 "indoor sit, left out of every mean",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


#  sky_share first and open_share second, deliberately. sky_share comes from
#  the segmenter alone; open_share's ten-camera-height threshold rides on the
#  monocular depth model, which this project's own audit failed (asked how
#  high the camera is, it answers ~3.1 m for both views). The two agree, so
#  the one that does not need the depth model leads.
PANELS = [("sky_share", "sky share of the forward 90\u00b0 field"),
          ("open_share", "openness: sky, or beyond 10 camera heights")]


def fig_2x2(depth, _metric, out):
    t = depth["human"][0]
    indoor = (t >= SEDENTARY[0]) & (t <= SEDENTARY[1])
    cells, labels, colours, hollow = [], [], [], []
    for view, col in (("human", HUMAN), ("dog", DOG)):
        for place, m in (("outdoor", ~indoor), ("indoor", indoor)):
            labels.append(f"{view} \u00b7 {place}\n{m.sum()} s")
            colours.append(col)
            hollow.append(place == "indoor")
            cells.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=150, sharex=True)
    for ax, (key, title) in zip(axes, PANELS):
        vals = [depth[lab.split(" \u00b7 ")[0]][1][key][m] * 100
                for lab, m in zip(labels, cells)]
        parts = ax.violinplot(vals, showextrema=False, widths=0.8)
        for body, c, hol in zip(parts["bodies"], colours, hollow):
            body.set_facecolor(c)
            #  the indoor cells are drawn hollow: one scene, not a sample
            body.set_alpha(0.22 if hol else 0.65)
            body.set_edgecolor(c)
            body.set_linewidth(1.2)
        for i, v in enumerate(vals, start=1):
            ax.scatter([i], [np.median(v)], s=38, color=INK, zorder=3)
            ax.plot([i, i], np.percentile(v, [25, 75]), color=INK, lw=2,
                    zorder=3)
        ax.set_xticks(range(1, len(vals) + 1))
        ax.set_xticklabels(labels, fontsize=8.5, color=INK)
        ax.set_ylabel("% of the frame", color=MUTED)
        ax.set_ylim(0, None)
        style(ax)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")

        print(f"\n{key}, % of frame")
        for lab, v in zip(labels, vals):
            print(f"  {lab.replace(chr(10), ' '):<22} median {np.median(v):5.1f}"
                  f"   IQR {np.percentile(v, 25):5.1f}-{np.percentile(v, 75):5.1f}")

    fig.text(0.5, 0.035,
             "The indoor cells are ONE fixed scene sampled every second, not a "
             "sample of indoor scenes \u2014 their width is measurement spread, "
             "not scene variety.\nSky share needs only the segmenter; openness "
             "also needs the depth model. They agree.",
             ha="center", color=MUTED, fontsize=8.5, linespacing=1.6)
    fig.suptitle("Viewpoint \u00d7 setting: going indoors costs the lower "
                 "viewpoint much more\ndot = median, bar = interquartile range",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.90))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="open_share",
                    choices=["open_share", "ground_share", "obstacle_share"],
                    help="the quantity the 2x2 violins show")
    args = ap.parse_args()

    names, data = load()
    depth = load_depth()
    p1 = fig_delta(names, data, depth, OUT / "16_paired_delta.png")
    p2 = fig_2x2(depth, args.metric, OUT / "17_two_by_two.png")
    print(f"\n-> {p1}\n-> {p2}")


if __name__ == "__main__":
    main()
