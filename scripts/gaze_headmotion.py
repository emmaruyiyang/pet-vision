#!/usr/bin/env python3
"""Checklist item 22: how much the head moved during each trial.

The Neon's own IMU sits in the glasses, so it measures the HEAD, not the
scene. During a seated screen task the head barely moves, so the honest
expectation here is a null: if the three conditions differ in head motion,
something other than the stimulus is driving it (fatigue, posture, the
experimenter talking), and that is worth knowing before any gaze measure is
read as an effect of the manipulation.

Two measures per trial, both from the gyroscope, which needs no integration
and therefore does not drift:

  ANGULAR SPEED   the magnitude of the gyro vector, deg/s, median and 95th
                  percentile. How fast the head is turning at a typical
                  instant, and in the worst moments.
  TRAVEL          that speed integrated over the trial and divided by its
                  duration, deg/s again, i.e. the mean. Over a short trial
                  this is close to the median; a gap between them means a few
                  large movements rather than steady motion.

Writes aoi/18_head_motion.png and aoi/head_motion.csv.

Usage
-----
    python3 scripts/gaze_headmotion.py
"""

import argparse
import ast
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi_plots import AOI_DIR, INK, MUTED, style  # noqa: E402
from gaze_on_stimulus import load_trials, STIM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "pupil_eye_tracking"
OUT = AOI_DIR
CONDS = ["human", "dog", "dog_dichromat"]
COLS = {"human": "#2b6cb0", "dog": "#c05621", "dog_dichromat": "#7c5cbf"}


def read_dtype(path):
    """A .dtype file is a Python list of (name, type) pairs. Parse, don't eval."""
    pairs = ast.literal_eval(path.read_text())
    return np.dtype([(n, "<" + np.dtype(t).str[1:]) for n, t in pairs])


def load_imu(session):
    """-> (seconds from the scene camera's first frame, |gyro| in deg/s).

    The IMU keeps its own clock, so it is put on the SCENE camera's timeline,
    the same zero the trial table was marked against. Without that the two are
    offset by however long the sensors took to start.
    """
    d = REC / session
    raw, dt = d / "imu ps1.raw", d / "imu.dtype"
    tim = d / "imu ps1.time"
    if not raw.exists():
        raise SystemExit(f"no IMU stream in {d}")
    rec = np.fromfile(raw, dtype=read_dtype(dt))
    t0 = int(np.fromfile(d / "Neon Scene Camera v1 ps1.time", dtype="<i8")[0])
    ts = (np.fromfile(tim, dtype="<i8") if tim.exists()
          else rec["timestamp_ns"])
    t = (ts - t0) / 1e9
    g = np.sqrt(rec["gyro_x"] ** 2 + rec["gyro_y"] ** 2 + rec["gyro_z"] ** 2)
    return t, g.astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", default="P01")
    args = ap.parse_args()

    trials = load_trials(STIM / f"{args.participant}_trials.csv")
    named = {t.get("session") for t in trials if t.get("session")}
    if len(named) != 1:
        raise SystemExit(f"the trial table must name one session, got {named}")
    session = named.pop()
    t, g = load_imu(session)
    rate = len(t) / (t[-1] - t[0])
    print(f"session {session}: {len(g)} IMU samples, {rate:.0f} Hz, "
          f"{t[0]:.1f}-{t[-1]:.1f} s")

    rows = []
    for tr in trials:
        a, b = float(tr["onset_s"]), float(tr["offset_s"])
        m = (t >= a) & (t <= b)
        if m.sum() < 10:
            print(f"  {tr['slug']} {tr['condition']}: only {m.sum()} IMU "
                  f"samples, left out")
            continue
        v = g[m]
        rows.append({"trial": tr, "n": int(m.sum()), "dur": b - a,
                     "median": float(np.median(v)),
                     "mean": float(v.mean()),
                     "p95": float(np.percentile(v, 95))})

    with (OUT / "head_motion.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["presented", "slug", "condition", "duration_s",
                    "imu_samples", "gyro_median_dps", "gyro_mean_dps",
                    "gyro_p95_dps"])
        for r in rows:
            tr = r["trial"]
            w.writerow([tr.get("presented", tr["trial"]), tr["slug"],
                        tr["condition"], f"{r['dur']:.2f}", r["n"],
                        f"{r['median']:.2f}", f"{r['mean']:.2f}",
                        f"{r['p95']:.2f}"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150,
                             gridspec_kw={"width_ratios": [1.5, 1]})

    #  left: every trial in presentation order, so drift over the session shows
    ax = axes[0]
    for r in rows:
        c = r["trial"]["condition"]
        x = int(r["trial"].get("presented", 0) or 0)
        ax.plot([x, x], [r["median"], r["p95"]], color=COLS[c], lw=2.2,
                alpha=0.55, solid_capstyle="round")
        ax.scatter([x], [r["median"]], s=34, color=COLS[c], zorder=3)
    style(ax)
    ax.set_xlabel("trial, in the order it was shown", color=MUTED)
    ax.set_ylabel("head angular speed, deg/s", color=MUTED)
    ax.set_title("dot = median within the trial, bar up to its 95th percentile",
                 color=INK, fontsize=10, loc="left")
    for c in CONDS:
        ax.scatter([], [], color=COLS[c], label=c)
    ax.legend(frameon=False, fontsize=8)

    #  right: the same medians grouped by condition
    ax = axes[1]
    for i, c in enumerate(CONDS):
        v = [r["median"] for r in rows if r["trial"]["condition"] == c]
        if not v:
            continue
        ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v,
                   s=34, color=COLS[c], alpha=0.8, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color=INK, lw=2.2,
                zorder=4)
        print(f"  {c:<16} median of trial medians {np.median(v):5.2f} deg/s "
              f"(n={len(v)} trials, range {min(v):.2f}-{max(v):.2f})")
    style(ax)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(CONDS, fontsize=9)
    ax.set_ylabel("head angular speed, deg/s", color=MUTED)
    ax.set_title("black bar = median across trials", color=INK, fontsize=10,
                 loc="left")

    fig.suptitle("Item 22: head movement during each trial, from the Neon's "
                 "own gyroscope\nseated screen task, n = 1 — this figure "
                 "is here to rule head motion OUT as an explanation, not to "
                 "find an effect in it", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = OUT / "18_head_motion.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n-> {out}\n-> {OUT / 'head_motion.csv'}")


if __name__ == "__main__":
    main()
