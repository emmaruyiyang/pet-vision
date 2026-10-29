#!/usr/bin/env python3
"""Checklist items 2 and 3, from elevation_frames.py output.

Item 3, the elevation profile: what each horizontal slice of the visual field
is made of, at each height. This is the figure the frame-average cannot give.
Averaged over a whole frame the two viewpoints look almost alike -- that is the
headline of 01_mean_composition.png -- but the average hides where things sit.
Sky and canopy can leave the top of one frame and arrive in the top of the
other without changing either total by much.

Item 2, the disagreement map: the same data on a 2D grid of elevation by
azimuth, plotted as dog minus human, so the question becomes "in which
direction does each class gain or lose". It is the forward 90 x 59 deg window,
not a full sphere: these panels are rectilinear cuts from the combined export,
and nothing outside that window was recorded in a form this pipeline can read.

READ A BAND AS A DIRECTION, NOT A SHELF. An elevation band is where you are
looking; a height band (B0-B3 in the work log) is how high the thing is off the
ground. The low bands hold the pavement at your feet and the pavement fifty
metres away alike. Going from one to the other needs distance, so it needs
depth estimation.

Walking frames only: the nine-minute indoor sit is one fixed scene and would
otherwise contribute a quarter of every band.

Writes aoi/05_elevation_profile.png and aoi/06_disagreement_map.png.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aoi_plots import AOI_DIR, SEDENTARY, INK, MUTED, HUMAN, DOG, style  # noqa: E402

OUT = AOI_DIR
#  the groups worth drawing; `animal` is the dog's own fur and `other` is the
#  leftovers, and both are dropped after renormalising (see aoi_frames.py)
DROP = ("animal", "other")
#  Hues stay semantic -- brown ground, green vegetation, blue sky -- because a
#  reader who glances at the legend once should not have to keep glancing.
#  What changed is lightness: the old set put ground, structure, vegetation and
#  furniture at nearly the same value, so the stack read as one grey mass and
#  the boundary between two segments was invisible at slide size. Now the big
#  three alternate light and dark, `structure` is a cool slate so it cannot be
#  mistaken for sky, and `human` is a clear red rather than a second brown.
PALETTE = {
    "ground": "#c99a63", "vegetation": "#4e9d62", "sky": "#9fcbe8",
    "structure": "#7f8b9c", "human": "#dd5f43", "furniture": "#e3c48d",
    "vehicle": "#8073b8", "signage": "#f0c233", "door": "#36617f",
}

#  Fixed, earth-to-sky, instead of sorted by the dog's own shares. With a fixed
#  order every row starts its `ground` block at zero, so the ground wedge can
#  be compared across bands and across the two panels by eye; with a data
#  driven order the same class starts at a different x in every row.
STACK = ["ground", "vegetation", "furniture", "vehicle", "human",
         "structure", "door", "signage", "sky"]


def load_bands(keep_sedentary=False):
    """-> (names, {view: {(lo, hi): mean share vector}}, n_frames)"""
    path = AOI_DIR / "elevation_shares.csv"
    with path.open() as fh:
        rdr = csv.DictReader(fh)
        names = [k for k in rdr.fieldnames
                 if k not in ("pair_index", "t_s", "view", "elev_lo", "elev_hi")]
        acc = defaultdict(lambda: np.zeros(len(names)))
        cnt = defaultdict(int)
        seen = set()
        for r in rdr:
            t = float(r["t_s"])
            if not keep_sedentary and SEDENTARY[0] <= t <= SEDENTARY[1]:
                continue
            key = (r["view"], (float(r["elev_lo"]), float(r["elev_hi"])))
            acc[key] += np.array([float(r[n]) for n in names])
            cnt[key] += 1
            seen.add(r["pair_index"])
    out = defaultdict(dict)
    for (view, band), v in acc.items():
        out[view][band] = v / cnt[(view, band)]
    return names, out, len(seen)


def load_grid(keep_sedentary=False):
    path = AOI_DIR / "elev_azim_shares.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    names = [k for k in rows[0]
             if k not in ("view", "elev_lo", "elev_hi", "azim_lo", "azim_hi",
                          "n_frames")]
    elev = sorted({(float(r["elev_lo"]), float(r["elev_hi"])) for r in rows})
    azim = sorted({(float(r["azim_lo"]), float(r["azim_hi"])) for r in rows})
    g = {v: np.zeros((len(elev), len(azim), len(names)))
         for v in ("human", "dog")}
    for r in rows:
        b = elev.index((float(r["elev_lo"]), float(r["elev_hi"])))
        a = azim.index((float(r["azim_lo"]), float(r["azim_hi"])))
        g[r["view"]][b, a] = [float(r[n]) for n in names]
    return names, g, elev, azim


def renorm(vec, names):
    keep = [i for i, n in enumerate(names) if n not in DROP]
    v = vec[..., keep]
    return v / np.maximum(v.sum(-1, keepdims=True), 1e-9), [names[i] for i in keep]


def fig_profile(names, bands, n_frames):
    kept = [n for n in names if n not in DROP]
    order = ([n for n in STACK if n in kept]
             + [n for n in kept if n not in STACK])
    #  ascending, so barh's y=0 (the bottom of the axes) is the lowest band and
    #  the picture reads the way the field of view does: ground at the bottom.
    levels = sorted(bands["human"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, (view, colour) in zip(axes, (("human", HUMAN), ("dog", DOG))):
        style(ax)
        ax.grid(axis="y", color="white")
        y = np.arange(len(levels))
        left = np.zeros(len(levels))
        for n in order:
            vals = np.array([renorm(bands[view][b], names)[0][kept.index(n)]
                             for b in levels])
            ax.barh(y, vals, left=left, height=0.78, color=PALETTE.get(n, "#cccccc"),
                    edgecolor="white", linewidth=0.8,
                    label=n if view == "human" else None)
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels([f"{int(lo):+d} to {int(hi):+d}°" for lo, hi in levels],
                           fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of that band", color=MUTED, fontsize=9)
        ax.set_title("human 1.6 m" if view == "human" else "dog 0.45 m",
                     color=colour, fontsize=11, loc="left")
    axes[0].set_ylabel("elevation band (0° = straight ahead)",
                       color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(labels),
                     loc="lower center", bbox_to_anchor=(0.5, -0.06))
    for t in leg.get_texts():
        t.set_color(MUTED)
    fig.suptitle(f"What fills each elevation band, {n_frames} walking seconds",
                 color=INK, fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT / "05_elevation_profile.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out, order, levels


def fig_map(names, grid, elev, azim, classes):
    gh, kept = renorm(grid["human"], names)
    gd, _ = renorm(grid["dog"], names)
    diff = gd - gh

    show = [c for c in classes if c in kept][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), dpi=150)
    vmax = max(abs(diff[..., kept.index(c)]).max() for c in show)
    for ax, c in zip(axes.ravel(), show):
        #  grid row 0 is the LOWEST elevation band, and imshow's origin="upper"
        #  draws row 0 at the top, so flip to put the sky above the pavement.
        d = diff[::-1, :, kept.index(c)]
        im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="upper")
        ax.set_title(c, color=INK, fontsize=10, loc="left")
        ax.set_xticks(range(len(azim)))
        ax.set_xticklabels([f"{int((lo + hi) / 2):+d}" for lo, hi in azim],
                           color=MUTED, fontsize=7)
        ax.set_yticks(range(len(elev)))
        ax.set_yticklabels([f"{int((lo + hi) / 2):+d}" for lo, hi in reversed(elev)],
                           color=MUTED, fontsize=7)
        for s in ax.spines.values():
            s.set_color(MUTED)
    for ax in axes[1]:
        ax.set_xlabel("azimuth (deg)", color=MUTED, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("elevation (deg)", color=MUTED, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("dog share − human share", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(MUTED)
    fig.suptitle("Where each class moves between the two heights\n"
                 "red = more of it in the dog's view, blue = more in the "
                 "human's; forward 90 × 59° window",
                 color=INK, fontsize=12, y=1.04)
    out = OUT / "06_disagreement_map.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-sedentary", action="store_true")
    args = ap.parse_args()

    names, bands, n = load_bands(args.keep_sedentary)
    p1, order, levels = fig_profile(names, bands, n)
    gnames, grid, elev, azim = load_grid(args.keep_sedentary)
    p2 = fig_map(gnames, grid, elev, azim, order)
    print(f"-> {p1}\n-> {p2}")

    kept = [x for x in names if x not in DROP]
    print(f"\n{'band':>16} " + " ".join(f"{c:>11}" for c in order[:5]))
    for b in sorted(levels, reverse=True):
        h = renorm(bands["human"][b], names)[0]
        d = renorm(bands["dog"][b], names)[0]
        cells = " ".join(f"{d[kept.index(c)] - h[kept.index(c)]:+11.3f}"
                         for c in order[:5])
        print(f"{int(b[0]):+4d} to {int(b[1]):+4d}° {cells}")
    print("\n(dog minus human, share of that band)")


if __name__ == "__main__":
    main()
