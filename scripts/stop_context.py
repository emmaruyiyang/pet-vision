#!/usr/bin/env python3
"""Checklist item 16, as far as this rig can honestly take it.

The plan was a sniffing-event map: lowered head plus near-zero speed. Half of
that is not measurable here. The camera is on a back harness, so a lowered nose
barely moves it -- imu_plots.py measured the difference between paused and
moving body pitch at +0.7 deg, which is nothing. The head half needs a
head-mounted camera.

The other half IS measurable, and it is the half that carries the spatial
question anyway: WHERE the dog stopped, and what was in front of it when it
did. A stop is not proof of a sniff, but "the dog stopped here, and here is
what filled its field while it was stopped" is a real claim, and it is the one
the route design needs.

NO MAP, NO ROUTE AXIS. There is no GPS, and the yaw rate cannot be integrated
into a heading (it drifts 145,000 deg over the walk -- see imu_attitude.py), so
there is no way to lay these stops out in space. The axis is time. Saying
otherwise would be drawing a route that was never measured.

Writes aoi/stops_context.csv -- one row per stop, with its duration and the
mean AOI composition of both panels while it lasted -- and aoi/09_stops.png.

Usage
-----
    python3 stop_context.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aoi_plots import AOI_DIR, SEDENTARY, INK, MUTED, HUMAN, DOG, style, load as load_aoi  # noqa: E402
from imu_plots import RATE, load as load_imu  # noqa: E402
from imu_attitude import segments  # noqa: E402

OUT = AOI_DIR
#  the classes a stop could plausibly be ABOUT; `structure` and `sky` are
#  everywhere and say nothing about why a walk paused
INTEREST = ("vegetation", "ground", "human", "furniture", "vehicle")


def stop_runs(pause_mask):
    """Contiguous True runs of the pause mask, as (start_s, end_s)."""
    runs, i, n = [], 0, len(pause_mask)
    while i < n:
        if pause_mask[i]:
            j = i
            while j < n and pause_mask[j]:
                j += 1
            runs.append((i / RATE, j / RATE))
            i = j
        else:
            i += 1
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-stop", type=float, default=3.0)
    args = ap.parse_args()

    d = load_imu()
    sed, pause, active, thr = segments(d)
    stops = [(a, b) for a, b in stop_runs(pause) if b - a >= args.min_stop]

    names, aoi = load_aoi(exclude_self=True)
    th, mh = aoi["human"]
    td, md = aoi["dog"]

    rows = []
    for k, (a, b) in enumerate(stops, 1):
        if SEDENTARY[0] <= a <= SEDENTARY[1]:
            continue
        m = (th >= a) & (th <= b)
        if m.sum() == 0:
            continue
        row = {"stop": k, "start_s": round(a, 1), "end_s": round(b, 1),
               "duration_s": round(b - a, 1), "n_frames": int(m.sum())}
        for i, n in enumerate(names):
            row[f"human_{n}"] = round(float(mh[m, i].mean()), 4)
            row[f"dog_{n}"] = round(float(md[m, i].mean()), 4)
        rows.append(row)

    out_csv = OUT / "stops_context.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    #  baseline: the same classes over all walking seconds that are NOT a stop
    walking = ~((th >= SEDENTARY[0]) & (th <= SEDENTARY[1]))
    in_stop = np.zeros(len(th), bool)
    for r in rows:
        in_stop |= (th >= r["start_s"]) & (th <= r["end_s"])
    moving = walking & ~in_stop

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150,
                             gridspec_kw={"width_ratios": [2, 1]})
    style(axes[0]); style(axes[1])

    t = np.array([r["start_s"] for r in rows]) / 60.0
    dur = np.array([r["duration_s"] for r in rows])
    axes[0].axvspan(SEDENTARY[0] / 60, SEDENTARY[1] / 60, color="#ededed", zorder=0)
    axes[0].vlines(t, 0, dur, color=DOG, lw=2)
    axes[0].scatter(t, dur, s=18, color=DOG, zorder=3)
    axes[0].set_xlabel("walk time (min)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("stop duration (s)", color=MUTED, fontsize=9)
    axes[0].set_title(f"{len(rows)} stops of {args.min_stop:g} s or more while "
                      f"walking; the grey band is the indoor block",
                      color=INK, fontsize=11, loc="left")

    y = np.arange(len(INTEREST))
    width = 0.38
    for off, (view, mat, colour, label) in enumerate(
            ((("human"), mh, HUMAN, "human 1.6 m"),
             (("dog"), md, DOG, "dog 0.45 m"))):
        stop_mean = [mat[in_stop, names.index(c)].mean() for c in INTEREST]
        move_mean = [mat[moving, names.index(c)].mean() for c in INTEREST]
        diff = np.array(stop_mean) - np.array(move_mean)
        axes[1].barh(y + (off - 0.5) * width, diff, height=width,
                     color=colour, label=label)
    axes[1].axvline(0, color=MUTED, lw=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(INTEREST, fontsize=9)
    axes[1].set_xlabel("share while stopped − share while moving",
                       color=MUTED, fontsize=9)
    axes[1].set_title("What is in front during a stop",
                      color=INK, fontsize=11, loc="left")
    leg = axes[1].legend(frameon=False, fontsize=8)
    for x in leg.get_texts():
        x.set_color(MUTED)

    fig.tight_layout()
    out_png = OUT / "09_stops.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"-> {out_csv}\n-> {out_png}")
    print(f"{len(rows)} walking stops, median {np.median(dur):.1f} s, "
          f"total {dur.sum() / 60:.1f} min")
    print(f"\n{'class':<12}{'human diff':>12}{'dog diff':>10}")
    for c in INTEREST:
        i = names.index(c)
        print(f"{c:<12}{mh[in_stop, i].mean() - mh[moving, i].mean():>+12.3f}"
              f"{md[in_stop, i].mean() - md[moving, i].mean():>+10.3f}")


if __name__ == "__main__":
    main()
