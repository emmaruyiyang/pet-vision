#!/usr/bin/env python3
"""Checklist items 6 and 8, from depth_frames.py output.

Item 6, openness over time: how much of the field is sky or far enough away to
be scenery rather than surroundings. Item 8, obstacles: how much of the field
they cover and how close the nearest of them is.

Distances are in CAMERA HEIGHTS wherever the two panels are compared. That is
not a cosmetic choice. The viewing geometry scales with the camera: the nearest
visible ground sits at cam/tan(29.36 deg) whatever the camera height is, which
is 2.84 m for the human panel and 0.80 m for the dog. Plotting metres would
show that ratio and nothing else. Plotting camera heights takes it out, so
what is left is the thing worth looking at -- whether the dog's world is more
enclosed than the geometry alone would make it.

Walking frames only. See depth_frames.py for why nothing here is a height band.

Writes aoi/07_openness.png and aoi/08_obstacles.png.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aoi_plots import AOI_DIR, SEDENTARY, INK, MUTED, HUMAN, DOG, style, smooth, block_boot  # noqa: E402

OUT = AOI_DIR
COLOUR = {"human": HUMAN, "dog": DOG}
LABEL = {"human": "human 1.6 m", "dog": "dog 0.45 m"}


def smooth_gapped(x, w):
    """smooth(), but a NaN gap stays a gap instead of smearing across it."""
    out = np.full_like(x, np.nan, dtype=float)
    ok = ~np.isnan(x)
    edges = np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(x)]):
        if ok[a]:
            out[a:b] = smooth(x[a:b], w)
    return out


def load(keep_sedentary=False):
    with (AOI_DIR / "depth_metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for view in ("human", "dog"):
        r = [x for x in rows if x["view"] == view]
        r.sort(key=lambda x: int(x["pair_index"]))
        if not keep_sedentary:
            r = [x for x in r
                 if not (SEDENTARY[0] <= float(x["t_s"]) <= SEDENTARY[1])]
        cols = {k: np.array([float(x[k]) if x[k] != "" else np.nan for x in r])
                for k in r[0] if k not in ("view",)}
        out[view] = cols
    return out


def paired(data, key):
    """Values for both panels on the pairs where both produced a number."""
    ih = {int(p): i for i, p in enumerate(data["human"]["pair_index"])}
    idd = {int(p): i for i, p in enumerate(data["dog"]["pair_index"])}
    keys = sorted(set(ih) & set(idd))
    h = data["human"][key][[ih[k] for k in keys]]
    d = data["dog"][key][[idd[k] for k in keys]]
    t = data["human"]["t_s"][[ih[k] for k in keys]]
    ok = ~(np.isnan(h) | np.isnan(d))
    return t[ok], h[ok], d[ok]


def break_gap(t, *series, gap=5.0):
    """Insert a NaN wherever the series jumps a gap, so lines do not bridge it.

    The indoor block is dropped from the data, which leaves a nine-minute hole
    in the middle of the walk. Plotted as-is, matplotlib draws a straight line
    across it, and a flat line for nine minutes reads as a measurement.
    """
    jump = np.flatnonzero(np.diff(t) > gap) + 1
    t2 = np.insert(t.astype(float), jump, np.nan)
    return (t2, *[np.insert(s.astype(float), jump, np.nan) for s in series])


def fig_openness(data):
    t, h, d = paired(data, "open_share")
    diff = d - h
    lo, hi = block_boot(diff, 30)
    tg, hg, dg, diffg = break_gap(t, h, d, diff)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), dpi=150,
                             gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    style(axes[0]); style(axes[1])
    m = tg / 60.0
    w = 15
    for ax in axes:
        ax.axvspan(SEDENTARY[0] / 60, SEDENTARY[1] / 60, color="#ededed", zorder=0)
    axes[0].text((SEDENTARY[0] + SEDENTARY[1]) / 120, 0.02,
                 "indoor block, dropped", ha="center", color=MUTED, fontsize=8)
    axes[0].plot(m, smooth_gapped(hg, w), color=HUMAN, lw=1.2, label=LABEL["human"])
    axes[0].plot(m, smooth_gapped(dg, w), color=DOG, lw=1.2, label=LABEL["dog"])
    axes[0].set_ylabel("open share of the field", color=MUTED, fontsize=9)
    axes[0].set_title(
        f"Openness over the walk: the dog's field is "
        f"{abs(diff.mean()) * 100:.1f} points "
        f"{'less' if diff.mean() < 0 else 'more'} open on average "
        f"(95% CI {lo * 100:+.1f} to {hi * 100:+.1f})",
        color=INK, fontsize=11, loc="left")
    leg = axes[0].legend(frameon=False, fontsize=8)
    for x in leg.get_texts():
        x.set_color(MUTED)

    axes[1].axhline(0, color=MUTED, lw=0.8)
    sd = smooth_gapped(diffg, w)
    #  colour by WHOSE field is more open, so the fill agrees with the legend
    axes[1].fill_between(m, 0, sd, where=sd >= 0, color=DOG, alpha=0.35)
    axes[1].fill_between(m, 0, sd, where=sd < 0, color=HUMAN, alpha=0.35)
    axes[1].set_ylabel("dog − human", color=MUTED, fontsize=9)
    axes[1].set_xlabel("walk time (min)", color=MUTED, fontsize=9)
    axes[1].set_title("below zero = the dog sees less open field, 15 s smoothing",
                      color=INK, fontsize=10, loc="left")

    fig.tight_layout()
    out = OUT / "07_openness.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out, diff.mean(), (lo, hi)


def fig_obstacles(data):
    t, hs, ds = paired(data, "obstacle_share")
    _, hn, dn = paired(data, "ground_near_cam")
    _, hm, dm = paired(data, "ground_median_cam")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
    for ax in axes:
        style(ax)

    ds_diff = ds - hs
    lo, hi = block_boot(ds_diff, 30)
    axes[0].hist(hs, bins=np.linspace(0, 1, 26), color=HUMAN, alpha=0.6,
                 label=LABEL["human"])
    axes[0].hist(ds, bins=np.linspace(0, 1, 26), color=DOG, alpha=0.6,
                 label=LABEL["dog"])
    axes[0].set_xlabel("obstacle share of the field", color=MUTED, fontsize=9)
    axes[0].set_ylabel("walking seconds", color=MUTED, fontsize=9)
    axes[0].set_title(f"Obstacle cover: {ds_diff.mean() * 100:+.1f} pts\n"
                      f"95% CI {lo * 100:+.1f} to {hi * 100:+.1f}",
                      color=INK, fontsize=10, loc="left")
    leg = axes[0].legend(frameon=False, fontsize=8)
    for x in leg.get_texts():
        x.set_color(MUTED)

    for vals, colour, label in ((hn, HUMAN, LABEL["human"]),
                                (dn, DOG, LABEL["dog"])):
        axes[1].hist(vals, bins=np.linspace(0.5, 3.5, 46), color=colour, alpha=0.6,
                     label=label)
    axes[1].axvline(1 / np.tan(np.radians(29.36)), color=MUTED, lw=1.0, ls="--")
    axes[1].set_xlabel("nearest visible ground (camera heights)",
                       color=MUTED, fontsize=9)
    axes[1].set_title("Nearest ground, with the\ngeometric floor dashed",
                      color=INK, fontsize=10, loc="left")

    for vals, colour in ((hm, HUMAN), (dm, DOG)):
        axes[2].hist(vals, bins=np.linspace(0, 9, 46), color=colour, alpha=0.6)
    axes[2].set_xlabel("median visible ground (camera heights)",
                       color=MUTED, fontsize=9)
    axes[2].set_title(f"How far the ground runs\n"
                      f"median {np.median(hm):.1f} vs {np.median(dm):.1f} "
                      f"camera heights",
                      color=INK, fontsize=10, loc="left")

    fig.tight_layout()
    out = OUT / "08_obstacles.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out, ds_diff.mean(), (lo, hi), (np.median(hn), np.median(dn))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-sedentary", action="store_true")
    args = ap.parse_args()

    data = load(args.keep_sedentary)
    p1, od, oci = fig_openness(data)
    p2, bd, bci, near = fig_obstacles(data)
    print(f"-> {p1}\n-> {p2}")
    print(f"\nopen share      dog - human {od * 100:+.1f} pts  "
          f"95% CI [{oci[0] * 100:+.1f}, {oci[1] * 100:+.1f}]")
    print(f"obstacle share  dog - human {bd * 100:+.1f} pts  "
          f"95% CI [{bci[0] * 100:+.1f}, {bci[1] * 100:+.1f}]")
    print(f"nearest ground  human {near[0]:.2f} cam heights, "
          f"dog {near[1]:.2f}  (geometry: {1 / np.tan(np.radians(29.36)):.2f})")


if __name__ == "__main__":
    main()
