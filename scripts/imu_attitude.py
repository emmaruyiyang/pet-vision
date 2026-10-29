#!/usr/bin/env python3
"""Two more IMU figures: where the body pointed, and how hard it was moving.

Checklist items 15 (attitude rose) and 14 (motion time series). Both read
imu/imu.csv and neither needs segmentation, the reshoot, or eye tracking.

WHAT THESE ARE NOT. The camera rides on a back harness, so every angle here is
the BODY, not the head. A dog lowering its nose to the pavement barely moves
its back; imu_plots.py measured that directly and found +0.7 deg between paused
and moving. So this is posture and gait, not gaze.

NEITHER HEADING NOR SPEED SURVIVES INTEGRATION, so neither is plotted. There
is no magnetometer and no GPS to correct the drift, and the harness mount is
violent: the median |yaw rate| while walking is 115 deg/s, so integrating it
over 26 minutes accumulates 145,000 deg -- four hundred full turns of pure
drift. Integrating the accelerometer for speed fails the same way within
seconds. That is also why checklist item 13, the trajectory map, is struck out.

What IS drift-free is the direction of gravity in the body frame. The rose
below is therefore a TILT rose: which way the body leaned and by how much,
taken straight from the accelerometer, with no integration anywhere. The
motion figure plots acceleration magnitude, a proxy for gait vigour, and the
instantaneous turn rate -- both raw, neither integrated.

Writes imu/05_attitude_rose.png and imu/06_motion_profile.png.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from imu_plots import IMU, RATE, INK, MUTED, ACCENT, WARM, load, find_runs, style

SEDENTARY_MIN = 60.0          # a quiet run this long is sitting, not a pause
PAUSE_MIN = 3.0


def segments(d, still_pct=25.0):
    """Masks for the sedentary block, walking pauses, and active walking.

    Same rule as imu_plots.py, and for the same reason: a long sit is not one
    long run below threshold, it is peppered with small spikes (the dog shifts,
    someone walks past) that chop any such run into pieces. Judge it on a 60 s
    rolling median, then judge pauses against how the dog moves WHILE WALKING.
    """
    motion = d["motion"]
    n = len(motion)
    w = int(SEDENTARY_MIN * RATE)
    k = np.ones(w) / w
    slow = np.convolve(np.pad(motion, w // 2, mode="edge"), k, "same")[
        w // 2:w // 2 + n]
    thr_all = float(np.percentile(motion, still_pct))

    sed = np.zeros(n, bool)
    for a, b in find_runs(slow, thr_all * 2, SEDENTARY_MIN):
        sed[int(a * RATE):int(b * RATE)] = True

    thr = float(np.percentile(motion[~sed], still_pct))
    pause = np.zeros(n, bool)
    for a, b in find_runs(motion, thr, PAUSE_MIN, SEDENTARY_MIN):
        if not sed[int(a * RATE)]:
            pause[int(a * RATE):int(b * RATE)] = True
    return sed, pause, ~sed & ~pause, thr


def tilt(d):
    """Direction and magnitude of body lean, from gravity. No integration."""
    pitch, roll = d["pitch_rel_deg"], d["roll_rel_deg"]
    direction = np.degrees(np.arctan2(roll, pitch)) % 360.0   # 0 = nose up
    return direction, np.hypot(pitch, roll)


def fig_rose(d, sed, active):
    """Item 15: which way the body leaned, and how its pitch was distributed."""
    direction, magnitude = tilt(d)
    pitch = d["pitch_rel_deg"]

    fig = plt.figure(figsize=(11, 4.4), dpi=150)
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    bins = np.arange(0, 361, 10)
    theta = np.radians(bins[:-1] + 5)
    for mask, colour, label in ((active, WARM, "walking"),
                                (sed, ACCENT, "sitting indoors")):
        counts, _ = np.histogram(direction[mask], bins=bins,
                                 weights=magnitude[mask])
        share = counts / max(counts.sum(), 1e-9)
        ax.bar(theta, share, width=np.radians(10), color=colour,
               edgecolor="white", linewidth=0.4, alpha=0.75, label=label)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["nose up", "roll right", "nose down", "roll left"],
                       color=MUTED, fontsize=8)
    ax.set_yticklabels([])
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color="#e8e8e8", lw=0.6)
    ax.set_title("Which way the body leaned\n"
                 "(share of total tilt, from gravity, no integration)",
                 color=INK, fontsize=10, pad=18)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower left",
                    bbox_to_anchor=(-0.12, -0.08))
    for t in leg.get_texts():
        t.set_color(MUTED)

    ax2 = fig.add_subplot(1, 2, 2)
    style(ax2)
    edges = np.arange(-45, 46, 2.5)
    ax2.hist(pitch[active], bins=edges, color=WARM, alpha=0.85,
             label=f"walking ({active.sum() / RATE / 60:.1f} min)")
    ax2.hist(pitch[sed], bins=edges, color=ACCENT, alpha=0.6,
             label=f"sitting indoors ({sed.sum() / RATE / 60:.1f} min)")
    ax2.axvline(np.median(pitch[active]), color=WARM, lw=1.2, ls="--")
    ax2.axvline(np.median(pitch[sed]), color=ACCENT, lw=1.2, ls="--")
    ax2.set_xlabel("body pitch, relative to session median (deg)",
                   color=MUTED, fontsize=9)
    ax2.set_ylabel("samples (20 Hz)", color=MUTED, fontsize=9)
    ax2.set_title("Body pitch: the indoor block sits "
                  f"{np.median(pitch[sed]) - np.median(pitch[active]):+.1f}° "
                  "off the walking posture",
                  color=INK, fontsize=10, loc="left")
    leg = ax2.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(MUTED)

    fig.tight_layout()
    out = IMU / "05_attitude_rose.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_motion(d, sed, pause, thr):
    """Item 14: motion magnitude and turn rate over the walk."""
    t = d["t_total_s"] / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), dpi=150, sharex=True)

    for ax in axes:
        style(ax)
        ax.axvspan(t[sed].min(), t[sed].max(), color="#ededed", zorder=0)

    w = int(RATE)  # 1 s
    k = np.ones(w) / w
    mot = np.convolve(d["motion"], k, "same")
    turn = np.convolve(np.abs(d["turn_dps"]), k, "same")

    axes[0].plot(t, mot, color=WARM, lw=0.6)
    axes[0].axhline(thr, color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylabel("motion magnitude (g)", color=MUTED, fontsize=9)
    axes[0].set_title("Gait vigour over the walk; the grey band is the indoor "
                      "block, the dashed line the stop threshold",
                      color=INK, fontsize=10, loc="left")

    axes[1].plot(t, turn, color=ACCENT, lw=0.6)
    axes[1].set_ylabel("|turn rate| (deg/s)", color=MUTED, fontsize=9)
    axes[1].set_xlabel("walk time (min)", color=MUTED, fontsize=9)
    axes[1].set_title("Turning, same time axis", color=INK, fontsize=10, loc="left")

    fig.tight_layout()
    out = IMU / "06_motion_profile.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    active = ~sed & ~pause
    return out, {
        "walking_min": active.sum() / RATE / 60.0,
        "sitting_min": sed.sum() / RATE / 60.0,
        "pause_min": pause.sum() / RATE / 60.0,
        "motion_median_walking": float(np.median(d["motion"][active])),
        "motion_median_sitting": float(np.median(d["motion"][sed])),
        "turn_median_walking": float(np.median(np.abs(d["turn_dps"][active]))),
        "tilt_median_walking": float(np.median(tilt(d)[1][active])),
        "tilt_median_sitting": float(np.median(tilt(d)[1][sed])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--still-pct", type=float, default=25.0,
                    help="percentile of motion magnitude taken as 'still'")
    args = ap.parse_args()

    d = load()
    sed, pause, active, thr = segments(d, args.still_pct)
    p1 = fig_rose(d, sed, active)
    p2, stats = fig_motion(d, sed, pause, thr)
    print(f"-> {p1}\n-> {p2}")
    for k, v in stats.items():
        print(f"{k:26s} {v:.3f}")


if __name__ == "__main__":
    main()
