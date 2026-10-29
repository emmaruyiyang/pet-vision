#!/usr/bin/env python3
"""Checklist item 1, cut down to what this footage can actually support.

An isovist is the set of points visible from where you stand (Benedikt 1979).
The classic drawing is a PLAN: rays fired all round, each stopping at the first
thing it hits, the outline joined up. Its known flaw is that it has no height,
which is exactly the flaw this project exists to attack -- a plan isovist is
identical for a person and a dog standing on the same spot.

WHAT THIS DRAWS, AND WHY IT IS NOT THE SPHERICAL VERSION. A spherical isovist
needs a ray in every direction, so it needs equirectangular frames. These
panels are rectilinear 90 x 59 deg cuts from the combined export, so only a
quarter-turn wedge in front of the walker was recorded in a form this pipeline
can read. What is drawn is therefore a HORIZONTAL SLICE of that wedge: a fan,
in plan, from the viewpoint out to the first obstruction in each direction.
Two of those -- one per camera height, at the same instant -- put the height
difference on a plan drawing, which is the thing a plan isovist cannot do.

DISTANCES ARE IN CAMERA HEIGHTS, like the rest of depth_frames.py, and for the
same reason: the geometry scales with the camera, so metres would mostly plot
that ratio back at you. And the depth they rest on is near-field only -- see
depth_frames.py for the audit. Rays that end in sky have no depth at all and
are drawn as open, to the edge of the fan, rather than given a number.

Writes aoi/11_isovist.png.

Usage
-----
    python3 isovist.py                       # four moments
    python3 isovist.py --at 533 615 950
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from aoi_frames import Segmenter, NAMES, DEFAULT_MODEL, load as load_img  # noqa: E402
from aoi_plots import AOI_DIR, INK, MUTED, HUMAN, DOG, style  # noqa: E402
from depth_frames import (Depth, geometry, CAM, CALIB_MAX_ELEV, W, H,
                          PAIRED)  # noqa: E402

OUT = AOI_DIR
SLICE_DEG = 4.0      # half-height of the horizontal band the slice is taken from
MAX_R = 12.0         # camera heights; an open ray is drawn out to here
COLOUR = {"human": HUMAN, "dog": DOG}


def slice_one(view, depth, lab, theta, sin_t, cos_t):
    """One horizontal slice: (azimuth rad, radius in camera heights, open?)."""
    cam = CAM[view]
    st = np.broadcast_to(sin_t, lab.shape)
    gmask = (lab == NAMES.index("ground")) & (st < np.sin(np.radians(CALIB_MAX_ELEV)))
    if gmask.sum() < 500:
        return None
    s = cam / np.median(depth[gmask] * np.abs(st[gmask]))

    band = np.abs(np.degrees(theta[:, 0])) < SLICE_DEG
    d = depth[band] * s / cam                      # camera heights, along the ray
    sky = (lab[band] == NAMES.index("sky"))

    #  the first obstruction per column: the nearest non-sky return in the band
    r = np.where(sky, np.inf, d)
    radius = r.min(axis=0)
    open_ray = ~np.isfinite(radius)
    radius = np.where(open_ray, MAX_R, np.minimum(radius, MAX_R))

    u = (np.arange(W) + 0.5) / W
    azim = np.arctan2((u - 0.5) * W, W / 2.0)      # +ve to the right
    return azim, radius, open_ray


def draw(ax, azim, radius, open_ray, colour, label):
    x = radius * np.sin(azim)
    y = radius * np.cos(azim)
    ax.fill(np.r_[0, x, 0], np.r_[0, y, 0], color=colour, alpha=0.22, lw=0)
    seg = np.where(open_ray, np.nan, 1.0)
    ax.plot(x * seg, y * seg, color=colour, lw=1.4, label=label)
    ax.plot([0], [0], marker="o", ms=4, color=colour)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--at", type=float, nargs="*",
                    #  955.5, not 950.5: at 950 the handler's hand is across
                    #  the human lens, which does not just look bad -- it
                    #  shortens that panel's reach and manufactures most of
                    #  the dog's advantage at this moment (4.2 vs 8.5 with the
                    #  hand, 9.2 vs 10.6 five seconds later, same place)
                    default=[186.5, 533.5, 615.5, 955.5, 409.5, 1906.5,
                             88.5, 1759.5],
                    help="candidate seconds; the first four that calibrate "
                         "on BOTH panels are drawn")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with (PAIRED / "manifest.csv").open() as fh:
        rows = {round(float(r["t_real_s"]), 1): r for r in csv.DictReader(fh)}
    picks = []
    for t in args.at:
        key = min(rows, key=lambda k: abs(k - t))
        picks.append(rows[key])

    import torch
    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    seg = Segmenter(DEFAULT_MODEL, dev)
    dep = Depth(dev)
    theta, sin_t, cos_t = geometry()

    #  a moment is only drawable if BOTH panels show enough ground to anchor
    #  the scale; without that there is no radius, and half a pair is worse
    #  than none -- a reader would compare one fan against nothing.
    ready = []
    for rec in picks:
        slices = {}
        for view in ("human", "dog"):
            img = load_img(PAIRED / rec[f"{view}_file"])
            d = dep([img])[0]
            lab = np.asarray(Image.fromarray(
                seg.labels([img])[0].astype(np.uint8)).resize((W, H), Image.NEAREST))
            got = slice_one(view, d, lab, theta, sin_t, cos_t)
            if got is not None:
                slices[view] = got
        if len(slices) == 2:
            ready.append((rec, slices))
        else:
            print(f"  t={float(rec['t_real_s']):.0f}s skipped: too little "
                  f"ground to anchor the scale")
        if len(ready) == args.count:
            break

    #  The fan on its own is unreadable as evidence: a lopsided outline could be
    #  a wall, a hedge, a parked van or a bad depth estimate, and nobody can
    #  tell which from the plan alone. Each column therefore carries the two
    #  frames the fan was computed from, directly above it.
    n = len(ready)
    fig, axes_all = plt.subplots(3, n, figsize=(3.2 * n, 7.6), dpi=150,
                                 gridspec_kw={"height_ratios": [1, 1, 2.2]})
    axes_all = axes_all.reshape(3, n)
    axes = axes_all[2]
    stats = []
    for col, (ax, (rec, slices)) in enumerate(zip(axes, ready)):
        for r, view in enumerate(("human", "dog")):
            a = axes_all[r, col]
            a.imshow(Image.open(PAIRED / rec[f"{view}_file"]))
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color(COLOUR[view])
                sp.set_linewidth(1.6)
            if col == 0:
                a.set_ylabel("human 1.6 m" if view == "human" else "dog 0.45 m",
                             color=COLOUR[view], fontsize=9)
        for view in ("human", "dog"):
            azim, radius, open_ray = slices[view]
            draw(ax, azim, radius, open_ray, COLOUR[view],
                 "human 1.6 m" if view == "human" else "dog 0.45 m")
            stats.append((rec["t_real_s"], view, float(radius.mean()),
                          float(open_ray.mean())))
        t = float(rec["t_real_s"])
        mh = float(slices["human"][1].mean())
        md = float(slices["dog"][1].mean())
        axes_all[0, col].set_title(f"{int(t) // 60:d}:{int(t) % 60:02d}",
                                   color=INK, fontsize=11)
        ax.set_title(f"mean reach {mh:.1f} vs {md:.1f}",
                     color=INK, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-MAX_R * 0.78, MAX_R * 0.78)
        ax.set_ylim(-0.6, MAX_R * 1.04)
        ax.axis("off")
        for r in (2, 5, 10):
            a = np.linspace(-np.pi / 4, np.pi / 4, 80)
            ax.plot(r * np.sin(a), r * np.cos(a), color="#e2e2e2", lw=0.7, zorder=0)
            ax.text(0, r, f"{r}", color=MUTED, fontsize=7, ha="center",
                    va="bottom", zorder=1)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
                     loc="lower center", bbox_to_anchor=(0.5, -0.03))
    for x in leg.get_texts():
        x.set_color(MUTED)
    fig.suptitle("Isovist slice: how far the view runs in each direction\n"
                 "plan view of the forward 90° wedge, radius in camera "
                 "heights, clipped at 12; a straight edge is that clip",
                 color=INK, fontsize=12, y=1.015)
    fig.tight_layout()
    out = OUT / "11_isovist.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"-> {out}\n")
    print(f"{'t_s':>8}{'view':>8}{'mean radius':>13}{'open rays':>11}")
    for t, v, r, o in stats:
        print(f"{float(t):>8.1f}{v:>8}{r:>13.2f}{o:>11.0%}")


if __name__ == "__main__":
    main()
