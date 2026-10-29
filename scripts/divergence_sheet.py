#!/usr/bin/env python3
"""Checklist item 26: a contact sheet of the moments the two views disagree most.

The mean composition figure says the two heights see almost the same mix of
things over a whole walk. That average is true and also misleading: it is an
average over a walk that is mostly open pavement, where height changes little.
The disagreement is concentrated in particular places -- a sign that clears the
dog's frame, a planter that swallows the far half of the street -- and this
sheet pulls those moments out so they can be looked at rather than averaged.

HOW MOMENTS ARE RANKED. For each paired second, both AOI share vectors are
renormalised without the `animal` group (the dog's own fur, see aoi_frames.py)
and compared with total variation distance,

    TV = 0.5 * sum_k |human_k - dog_k|

which is 0 when the two frames are made of the same stuff in the same
proportions and 1 when they share nothing. The indoor sedentary block is
excluded: it is one fixed scene repeated for nine minutes and it would fill the
sheet with near-duplicates. Picks are also spaced at least --min-gap seconds
apart, for the same reason.

The sheet is a selection, not a measurement: it shows where to look, and the
caption under each row gives the number so a reader can tell a large gap from
a small one.

Usage
-----
    python3 divergence_sheet.py                 # top 8
    python3 divergence_sheet.py -n 12 --min-gap 45
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from aoi_plots import AOI_DIR, SEDENTARY, INK, MUTED, HUMAN, DOG, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
PAIRED = ROOT / "frames_paired"
OUT = ROOT / "aoi"


def pick(names, data, n, min_gap, keep_sedentary, bad):
    th, mh = data["human"]
    td, md = data["dog"]
    assert np.allclose(th, td)
    tv = 0.5 * np.abs(mh - md).sum(1)
    order = np.argsort(-tv)

    chosen = []
    for i in order:
        t = th[i]
        if i in bad:
            continue
        if not keep_sedentary and SEDENTARY[0] <= t <= SEDENTARY[1]:
            continue
        if any(abs(t - th[j]) < min_gap for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) == n:
            break
    return chosen, tv, th, mh, md


def label(names, h, d):
    """The two classes that move most between the views, as readable text."""
    diff = d - h
    up = int(np.argmax(diff))
    down = int(np.argmin(diff))
    return (f"dog +{diff[up]:.0%} {names[up]}, "
            f"−{abs(diff[down]):.0%} {names[down]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--count", type=int, default=8)
    ap.add_argument("--min-gap", type=float, default=30.0,
                    help="seconds between picks, so one scene cannot repeat")
    ap.add_argument("--keep-sedentary", action="store_true")
    ap.add_argument("--min-brightness", type=float, default=12.0,
                    help="drop pairs where either panel is darker than this")
    ap.add_argument("--max-self", type=float, default=0.25,
                    help="drop pairs where own body or handler covers more "
                         "than this share of a panel")
    args = ap.parse_args()

    names, data = load(exclude_self=True)
    with (PAIRED / "manifest.csv").open() as fh:
        man = {int(r["pair_index"]): r for r in csv.DictReader(fh)}

    #  The human export has two stretches of black frames (t 19-78 s and
    #  1820-1829 s; see the work log). A black panel segments as one solid
    #  class, which makes it the most "divergent" second in the whole walk --
    #  the top of the ranking is otherwise entirely export damage. Catch them
    #  by brightness rather than by hardcoded times, so a future export with a
    #  different fault is caught too.
    bad = {i for i, r in man.items()
           if min(float(r["human_brightness"]),
                  float(r["dog_brightness"])) < args.min_brightness}
    print(f"{len(bad)} pairs dropped as too dark "
          f"(brightness < {args.min_brightness:g})")

    #  The other way to top this ranking is to put something on the lens. The
    #  dog camera rides on the dog's back, so its own fur fills a corner of
    #  most frames; the human camera is hand-held, so a hand crosses it now and
    #  then; and the handler's legs are a metre from the dog lens. Any of these
    #  swamps a frame and reads as a huge height effect. Unfiltered, six of the
    #  top eight seconds were self-occlusion. They are a real and reportable
    #  property of this rig, but they are not what the sheet is for, so the
    #  near-field classes are capped here and the count is printed.
    raw_names, raw = load(exclude_self=False)
    ai = raw_names.index("animal")
    hi = raw_names.index("human")
    occluded = set()
    for view in ("human", "dog"):
        _, m = raw[view]
        over = (m[:, ai] > args.max_self) | (m[:, hi] > args.max_self)
        occluded |= {int(j) for j in np.flatnonzero(over)}
    print(f"{len(occluded)} pairs dropped as self-occluded "
          f"(own body or handler over {args.max_self:.0%} of a panel)")
    bad |= occluded

    chosen, tv, th, mh, md = pick(names, data, args.count, args.min_gap,
                                  args.keep_sedentary, bad)
    rows = len(chosen)

    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.1 * rows), dpi=150)
    axes = np.atleast_2d(axes)
    for r, i in enumerate(chosen):
        rec = man[i]
        for c, (view, colour) in enumerate((("human", HUMAN), ("dog", DOG))):
            ax = axes[r, c]
            ax.imshow(Image.open(PAIRED / rec[f"{view}_file"]))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(colour)
                s.set_linewidth(1.6)
            if r == 0:
                ax.set_title("human 1.6 m" if view == "human" else "dog 0.45 m",
                             color=colour, fontsize=10, pad=6)
        t = th[i]
        axes[r, 0].set_ylabel(f"{int(t) // 60:d}:{int(t) % 60:02d}",
                              color=MUTED, fontsize=9)
        axes[r, 1].set_xlabel(f"TV = {tv[i]:.2f}   ·   "
                              f"{label(names, mh[i], md[i])}",
                              color=MUTED, fontsize=8, loc="left")

    fig.suptitle("Where the two heights disagree most\n"
                 "paired seconds ranked by total variation distance between "
                 "their AOI composition; indoor block excluded",
                 color=INK, fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = OUT / "04_divergence_sheet.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    #  the ranking itself is worth keeping, not just the eight frames it picked:
    #  it is a per-second series like the others, and any later figure that
    #  wants "when did the two views disagree" reads it from here.
    series = OUT / "divergence_tv.csv"
    with series.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["pair_index", "t_s", "tv", "dropped_dark_or_occluded",
                     "sedentary", "picked"])
        picked = set(chosen)
        for i, t in enumerate(th):
            wr.writerow([i, round(float(t), 1), round(float(tv[i]), 5),
                         int(i in bad),
                         int(SEDENTARY[0] <= t <= SEDENTARY[1]),
                         int(i in picked)])

    print(f"-> {out}\n-> {series}")
    print(f"TV over the walk: median {np.median(tv):.3f}, "
          f"90th pct {np.percentile(tv, 90):.3f}, max {tv.max():.3f}")
    for i in chosen:
        print(f"  t={th[i]:7.1f}s  TV={tv[i]:.3f}  {label(names, mh[i], md[i])}")


if __name__ == "__main__":
    main()
