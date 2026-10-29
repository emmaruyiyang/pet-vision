#!/usr/bin/env python3
"""Checklist item 4: the binocular wedge against the monocular wings.

WHAT IS MEASURED AND WHAT IS ASSUMED. The field angles are not ours. A single
camera cannot measure binocular overlap, and nothing in this project does.
They are literature values, and the figure says so on its face:

  human   about 200 deg total, about 120 deg of it binocular
  dog     about 240 deg total, about 30-60 deg of it binocular; the narrow
          end is the brachycephalic case, the wide end the long-nosed one

  Clean Run, January 2009, p. 62 (240-270 deg total, 30-60 binocular; human
  200/140); Read 2021 (human overlap about 120 deg, after Spector 1990).
  Sources disagree on the human figure -- 120 and 140 are both in print --
  so the human binocular edge is drawn as a band, not a line.

What IS measured is the composition: the paired footage is already binned by
azimuth in aoi/elev_azim_shares.csv, and its bin edges fall exactly on +-30
and +-45, so the dog's binocular wedge is the four central bins and its
monocular wings are the two outer ones, with nothing to interpolate.

THE COMPARISON ONLY RUNS ONE WAY. Our panels are 90 deg wide. The human's
binocular zone is about 120 deg, so the whole frame sits inside it and there
are no human monocular wings to measure -- the honest figure is therefore not
"human wings versus dog wings" but "what falls in the dog's wings", beside a
schematic that shows why the other half of the question is out of frame.

Bins are weighted by their true pixel area, f^2 (tan a2 - tan a1)(tan e2 -
tan e1) for a pinhole camera, not treated as equal: at 90 deg HFOV the outer
azimuth bins are far wider in pixels than the central ones, and an unweighted
mean would understate the wings by about a third.

Writes aoi/19_binocular_field.png.

Usage
-----
    python3 scripts/binocular_field.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aoi_plots import AOI_DIR, INK, MUTED, HUMAN, DOG, style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = AOI_DIR
#  the two viewpoints' fields, in degrees, from the sources in the docstring
FIELDS = {"human": {"total": 200.0, "binoc": (120.0, 140.0)},
          "dog": {"total": 240.0, "binoc": (30.0, 60.0)}}
DOG_BINOC_HALF = 30.0        # the wide end of the dog range, half-angle
CAM_HFOV = 90.0
SHOWN = ["ground", "vegetation", "furniture", "human", "vehicle", "structure",
         "sky", "signage"]


def load():
    with (AOI_DIR / "elev_azim_shares.csv").open() as fh:
        return list(csv.DictReader(fh))


def cell_weight(r):
    """Pixel area of one elevation x azimuth cell in a pinhole image."""
    t = np.tan(np.radians([float(r["azim_lo"]), float(r["azim_hi"]),
                           float(r["elev_lo"]), float(r["elev_hi"])]))
    return (t[1] - t[0]) * (t[3] - t[2])


def composition(rows, view, inner):
    """Area-weighted AOI shares over the cells inside (or outside) +-30 deg."""
    acc = np.zeros(len(SHOWN))
    tot = 0.0
    for r in rows:
        if r["view"] != view:
            continue
        mid = (float(r["azim_lo"]) + float(r["azim_hi"])) / 2
        if (abs(mid) <= DOG_BINOC_HALF) != inner:
            continue
        w = cell_weight(r)
        #  drop the dog's own body before renormalising, as everywhere else
        keep = np.array([float(r[n]) for n in SHOWN])
        denom = 1.0 - float(r["animal"])
        acc += w * keep / max(denom, 1e-9)
        tot += w
    return acc / max(tot, 1e-9), tot


def fan(ax, half_total, binoc_lo, binoc_hi, colour, label):
    """A plan-view visual field: the animal looks up the page."""
    for lo, hi, c, a in ((-half_total, half_total, colour, 0.18),
                         (-binoc_hi / 2, binoc_hi / 2, colour, 0.30),
                         (-binoc_lo / 2, binoc_lo / 2, colour, 0.55)):
        th = np.radians(np.linspace(lo, hi, 200)) + np.pi / 2
        ax.fill(np.concatenate([[0], np.cos(th)]),
                np.concatenate([[0], np.sin(th)]), color=c, alpha=a, lw=0)
    th = np.radians(np.linspace(-CAM_HFOV / 2, CAM_HFOV / 2, 2)) + np.pi / 2
    for t in th:
        ax.plot([0, np.cos(t) * 1.08], [0, np.sin(t) * 1.08], color=INK,
                lw=1.1, ls=(0, (4, 3)))
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    ax.set_title(label, color=INK, fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    rows = load()

    fig = plt.figure(figsize=(12.2, 5.4), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.1], wspace=0.25)

    for i, (view, col) in enumerate((("human", HUMAN), ("dog", DOG))):
        f = FIELDS[view]
        ax = fig.add_subplot(gs[0, i])
        fan(ax, f["total"] / 2, f["binoc"][0], f["binoc"][1], col,
            f"{view}\n{f['total']:.0f}° total, "
            f"{f['binoc'][0]:.0f}–{f['binoc'][1]:.0f}° binocular")

    #  what sits in the dog's monocular wings, against its binocular wedge
    ax = fig.add_subplot(gs[0, 2])
    inner, w_in = composition(rows, "dog", True)
    outer, w_out = composition(rows, "dog", False)
    y = np.arange(len(SHOWN))
    ax.barh(y + 0.2, inner * 100, 0.38, color=DOG, alpha=0.9,
            label=f"binocular wedge, |azimuth| ≤ 30°")
    ax.barh(y - 0.2, outer * 100, 0.38, color=DOG, alpha=0.38,
            label=f"monocular wings, 30–45°")
    ax.set_yticks(y)
    ax.set_yticklabels(SHOWN, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of that part of the dog's frame", color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    wings = w_out / (w_in + w_out)
    ax.set_title(f"The wings are {wings:.0%} of the dog panel's pixels",
                 color=INK, fontsize=10, loc="left")

    fig.text(0.5, 0.045,
             "Field angles are literature values (Clean Run 2009 p.62; Read "
             "2021), not measured here — dashed lines mark our 90° "
             "camera frame.\nIt covers the whole human binocular zone, so "
             "there are no human monocular wings in frame; only the dog's "
             "half of this comparison is measurable from this footage.",
             ha="center", color=MUTED, fontsize=8.5)
    fig.suptitle("Item 4: binocular wedge against monocular wings",
                 color=INK, fontsize=12.5)
    #  not tight_layout: the fans are equal-aspect axes it cannot size, and it
    #  lays the caption over the bar chart's x label
    fig.subplots_adjust(top=0.86, bottom=0.26, left=0.03, right=0.98)
    out = OUT / "19_binocular_field.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"dog panel: binocular wedge {1 - wings:.1%} of pixels, "
          f"monocular wings {wings:.1%}")
    print(f"\n{'class':<12}{'wedge %':>10}{'wings %':>10}{'wings-wedge':>13}")
    for i, n in enumerate(SHOWN):
        print(f"{n:<12}{inner[i] * 100:>10.1f}{outer[i] * 100:>10.1f}"
              f"{(outer[i] - inner[i]) * 100:>13.1f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
