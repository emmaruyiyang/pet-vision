#!/usr/bin/env python3
"""Figures from the dog-side IMU: motion, stops, and body attitude.

Reads imu/imu.csv (written by imu_extract.py) and writes four PNGs.

Read them as BODY measures, not head measures. The camera is on a harness on
the dog's back, so a lowered head does not show up directly. What shows up is
the body: when it moved, when it stopped, how it was tilted. Stops are where
sniffing usually happens, which is the honest way to phrase that claim.

Pitch and roll are relative to the session median, because the harness mount is
tilted by an unknown amount.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
IMU = ROOT / "imu"
RATE = 20.0
INK, MUTED, ACCENT, WARM = "#1a1a1a", "#8a8a8a", "#2b6cb0", "#c05621"


def load():
    with (IMU / "imu.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    d = {k: np.array([float(r[k]) for r in rows])
         for k in ("t_total_s", "pitch_rel_deg", "roll_rel_deg",
                   "turn_dps", "motion")}
    d["clip"] = np.array([r["clip"] for r in rows])
    return d


def find_runs(motion, thr, min_s, max_s=None):
    """Contiguous runs below `thr` lasting at least `min_s` (and under `max_s`).

    Two very different things hide in "the dog was not moving": brief pauses
    during a walk, and long sedentary stretches -- this session has eleven
    minutes of sitting indoors. Lumping them together makes the sedentary block
    dominate every statistic, so they are separated by duration here.
    """
    quiet = motion < thr
    stops, i, n = [], 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            dur = (j - i) / RATE
            if dur >= min_s and (max_s is None or dur < max_s):
                stops.append((i / RATE, j / RATE))
            i = j
        else:
            i += 1
    return stops


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)


def fig_motion(d, stops, thr):
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=150)
    t = d["t_total_s"] / 60
    ax.fill_between(t, 0, d["motion"], color=ACCENT, alpha=0.25, lw=0)
    ax.plot(t, d["motion"], color=ACCENT, lw=0.4)
    for a, b in d.get("sedentary", []):
        ax.axvspan(a / 60, b / 60, color=MUTED, alpha=0.22, lw=0)
    for a, b in stops:
        ax.axvspan(a / 60, b / 60, color=WARM, alpha=0.45, lw=0)
    ax.axhline(thr, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, np.percentile(d["motion"], 99.5))
    ax.set_xlabel("walk time (min)", color=MUTED, fontsize=9)
    ax.set_ylabel("motion", color=MUTED, fontsize=9)
    sed = sum(b - a for b_, b in [] for a in []) or sum(
        b - a for a, b in d.get("sedentary", []))
    ax.set_title(f"{len(stops)} pauses while walking; separately, "
                 f"{sed / 60:.0f} min sitting indoors",
                 color=INK, fontsize=11, fontweight="600", loc="left")
    style(ax)
    fig.text(0.125, -0.02, "orange = pause while walking   grey = sedentary block (60 s or longer)",
             color=MUTED, fontsize=8)
    fig.tight_layout(); fig.savefig(IMU / "01_motion_and_stops.png",
                                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_stop_hist(stops):
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=150)
    dur = np.array([b - a for a, b in stops])
    ax.hist(dur, bins=np.logspace(np.log10(3), np.log10(max(dur.max(), 4)), 26),
            color=WARM, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("stop duration (s)", color=MUTED, fontsize=9)
    ax.set_ylabel("count", color=MUTED, fontsize=9)
    ax.set_title(f"Most stops are short; the median is {np.median(dur):.0f} s",
                 color=INK, fontsize=11, fontweight="600", loc="left")
    style(ax)
    fig.tight_layout(); fig.savefig(IMU / "02_stop_durations.png",
                                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_pitch_time(d, stops):
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=150)
    t = d["t_total_s"] / 60
    ax.plot(t, d["pitch_rel_deg"], color=ACCENT, lw=0.4)
    ax.axhline(0, color=MUTED, lw=0.8)
    for a, b in stops:
        ax.axvspan(a / 60, b / 60, color=WARM, alpha=0.22, lw=0)
    lo, hi = np.percentile(d["pitch_rel_deg"], [0.5, 99.5])
    ax.set_xlim(0, t[-1]); ax.set_ylim(lo, hi)
    ax.set_xlabel("walk time (min)", color=MUTED, fontsize=9)
    ax.set_ylabel("body pitch, relative (deg)", color=MUTED, fontsize=9)
    ax.set_title("Body pitch over the walk, against the session median",
                 color=INK, fontsize=11, fontweight="600", loc="left")
    style(ax)
    fig.tight_layout(); fig.savefig(IMU / "03_pitch_over_time.png",
                                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_attitude(d, stops):
    """Pitch distribution, moving versus stopped."""
    sed = d["sed_mask"]
    paused = np.zeros(len(d["motion"]), bool)
    for a, b in stops:
        paused[int(a * RATE):int(b * RATE)] = True
    moving = (~paused) & (~sed)
    paused = paused & (~sed)
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=150)
    bins = np.linspace(*np.percentile(d["pitch_rel_deg"], [0.5, 99.5]), 48)
    ax.hist(d["pitch_rel_deg"][moving], bins=bins, density=True,
            color=ACCENT, alpha=0.55, label="moving")
    ax.hist(d["pitch_rel_deg"][paused], bins=bins, density=True,
            color=WARM, alpha=0.55, label="paused")
    med_m = np.median(d["pitch_rel_deg"][moving])
    med_s = np.median(d["pitch_rel_deg"][paused])
    ax.axvline(med_m, color=ACCENT, lw=1.2)
    ax.axvline(med_s, color=WARM, lw=1.2)
    ax.set_xlabel("body pitch, relative (deg)", color=MUTED, fontsize=9)
    ax.set_ylabel("density", color=MUTED, fontsize=9)
    ax.set_title(f"Paused vs moving: {med_s - med_m:+.1f} deg, "
                 f"sedentary block excluded",
                 color=INK, fontsize=11, fontweight="600", loc="left")
    leg = ax.legend(frameon=False, fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(MUTED)
    style(ax)
    fig.tight_layout(); fig.savefig(IMU / "04_pitch_moving_vs_stopped.png",
                                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return med_m, med_s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--still-pct", type=float, default=25.0,
                    help="motion percentile that counts as still (default 25)")
    ap.add_argument("--min-stop", type=float, default=3.0,
                    help="shortest run that counts as a pause, seconds")
    ap.add_argument("--sedentary", type=float, default=60.0,
                    help="a still run this long or longer is sedentary, not a "
                         "pause (seconds)")
    args = ap.parse_args()

    d = load()
    total = d["t_total_s"][-1]

    # Sedentary blocks first, so they can be taken out of everything else.
    # A long sit is not a long run below threshold -- it is peppered with small
    # spikes (the dog shifts, someone walks past) that chop any such run into
    # pieces. So judge it on a 60 s rolling median instead, which ignores them.
    w = int(args.sedentary * RATE)
    k = np.ones(w) / w
    slow = np.convolve(np.pad(d["motion"], w // 2, mode="edge"), k, "same")[
        w // 2:w // 2 + len(d["motion"])]
    thr_all = float(np.percentile(d["motion"], args.still_pct))
    sedentary = find_runs(slow, thr_all * 2, args.sedentary)
    sed_mask = np.zeros(len(d["motion"]), bool)
    for a, b in sedentary:
        sed_mask[int(a * RATE):int(b * RATE)] = True
    sed_s = sed_mask.sum() / RATE

    # pauses are judged against how the dog moves WHILE WALKING
    walk = d["motion"][~sed_mask]
    thr = float(np.percentile(walk, args.still_pct))
    stops = [(a, b) for a, b in
             find_runs(d["motion"], thr, args.min_stop, args.sedentary)
             if not sed_mask[int(a * RATE)]]
    stopped = sum(b - a for a, b in stops)

    print(f"{total / 60:.1f} min total")
    print(f"  {len(sedentary)} sedentary block(s) of {args.sedentary:g} s or "
          f"more: {sed_s / 60:.1f} min ({sed_s / total:.0%}) -- "
          f"the indoor sit, excluded below")
    print(f"  walking: {(total - sed_s) / 60:.1f} min, "
          f"pause threshold {thr:.2f}")
    print(f"  {len(stops)} pauses of {args.min_stop:g}-{args.sedentary:g} s, "
          f"{stopped / 60:.1f} min ({stopped / (total - sed_s):.0%} of walking)")
    dur = np.array([b - a for a, b in stops])
    print(f"  pause duration: median {np.median(dur):.1f} s, "
          f"longest {dur.max():.0f} s")
    d["sed_mask"] = sed_mask

    d["sedentary"] = sedentary
    fig_motion(d, stops, thr)
    fig_stop_hist(stops)
    fig_pitch_time(d, stops)
    med_m, med_s = fig_attitude(d, stops)
    print(f"body pitch: moving median {med_m:+.1f} deg, "
          f"stopped {med_s:+.1f} deg, difference {med_s - med_m:+.1f} deg")

    with (IMU / "stops.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s", "duration_s"])
        for a, b in stops:
            w.writerow([round(a, 2), round(b, 2), round(b - a, 2)])
    print(f"\n4 figures and stops.csv -> {IMU}")


if __name__ == "__main__":
    main()
