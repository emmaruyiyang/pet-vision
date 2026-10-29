#!/usr/bin/env python3
"""Pull the dog-side IMU out of the Insta360 clips and resample it.

The gyro and accelerometer live in a proprietary trailer at the end of each
.insv, which ffmpeg does not expose. telemetry-parser reads it.

What the signal actually is, so the plots are not over-read: the camera is on a
harness on the dog's BACK. It measures the body, not the head. A dog can drop
its head to sniff while the body stays level, so this cannot detect sniffing
directly -- what it detects is stopping, which is when sniffing usually happens.

The mount is tilted (gravity sits at roughly 32 degrees off the camera's
vertical), so absolute pitch is not readable without calibrating the mount.
Pitch here is therefore relative to the session median, which is what the
"did the body tilt down" question needs anyway.

Writes one CSV at --rate Hz with time, raw gyro and accel, body pitch and roll
relative to the session median, and the two motion measures the plots use.

Usage
-----
    python3 imu_extract.py            # both dog clips, 20 Hz
    python3 imu_extract.py --rate 50
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

DOG = Path("/Users/ruyiyang/Library/Mobile Documents/com~apple~CloudDocs/"
           "insta-360/dog")
CLIPS = [("013", DOG / "VID_20261025_084340_00_013.insv"),
         ("014", DOG / "VID_20261025_084340_00_014.insv")]
OUT = Path(__file__).resolve().parent.parent  / "imu"


def read_clip(path):
    import telemetry_parser as tp
    imu = tp.Parser(str(path)).normalized_imu()
    t = np.array([s["timestamp_ms"] for s in imu], dtype=np.float64) / 1000.0
    gyro = np.array([s["gyro"] for s in imu], dtype=np.float64)
    accl = np.array([s["accl"] for s in imu], dtype=np.float64)
    return t, gyro, accl


def resample(t, y, grid):
    """Linear interpolation onto a uniform grid, one column at a time."""
    if y.ndim == 1:
        return np.interp(grid, t, y)
    return np.stack([np.interp(grid, t, y[:, i]) for i in range(y.shape[1])], 1)


def smooth(x, n):
    """Centred moving average; n in samples."""
    if n < 2:
        return x
    k = np.ones(n) / n
    pad = n // 2
    return np.convolve(np.pad(x, pad, mode="edge"), k, "same")[pad:pad + len(x)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=20.0,
                    help="output sample rate in Hz (default 20)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    rows = []
    offset = 0.0
    for name, path in CLIPS:
        if not path.exists():
            sys.exit(f"missing: {path}")
        print(f"{name}  {path.name}")
        t, gyro, accl = read_clip(path)
        # the parser starts a little before the video; clip to t >= 0 so the
        # two clips butt together on the video timeline, not the sensor's
        keep = t >= 0
        t, gyro, accl = t[keep], gyro[keep], accl[keep]
        dur = t[-1] - t[0]
        print(f"  {len(t)} samples, {dur:.1f} s, {len(t) / dur:.0f} Hz")

        grid = np.arange(0, dur, 1.0 / args.rate)
        g = resample(t - t[0], gyro, grid)
        a = resample(t - t[0], accl, grid)

        # gravity direction, from accel smoothed over 2 s so walking cancels
        win = int(2.0 * args.rate)
        a_lp = np.stack([smooth(a[:, i], win) for i in range(3)], 1)
        down = a_lp / np.linalg.norm(a_lp, axis=1, keepdims=True)
        # mount is tilted, so report angles against the session's own median
        pitch = np.degrees(np.arctan2(-down[:, 2], down[:, 1]))
        roll = np.degrees(np.arctan2(down[:, 0], down[:, 1]))
        pitch -= np.median(pitch)
        roll -= np.median(roll)

        turn = np.linalg.norm(g, axis=1)                 # deg/s
        jolt = np.abs(np.linalg.norm(a, axis=1) - 9.81)  # dynamic accel, m/s^2
        motion = smooth(turn, int(0.5 * args.rate)) / 100.0 + \
            smooth(jolt, int(0.5 * args.rate))

        for i, tt in enumerate(grid):
            rows.append({
                "clip": name,
                "t_clip_s": round(float(tt), 3),
                "t_total_s": round(float(tt + offset), 3),
                "gx": round(float(g[i, 0]), 3), "gy": round(float(g[i, 1]), 3),
                "gz": round(float(g[i, 2]), 3),
                "ax": round(float(a[i, 0]), 3), "ay": round(float(a[i, 1]), 3),
                "az": round(float(a[i, 2]), 3),
                "pitch_rel_deg": round(float(pitch[i]), 2),
                "roll_rel_deg": round(float(roll[i]), 2),
                "turn_dps": round(float(turn[i]), 2),
                "motion": round(float(motion[i]), 3),
            })
        offset += dur
        print(f"  -> {len(grid)} rows at {args.rate:g} Hz")

    path = OUT / "imu.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows, {offset:.1f} s total -> {path}")

    m = np.array([r["motion"] for r in rows])
    still = m < np.percentile(m, 20)
    print(f"motion: p20 {np.percentile(m, 20):.2f}  median {np.median(m):.2f}  "
          f"p95 {np.percentile(m, 95):.2f}")
    print(f"quietest fifth of the time = {still.sum() / args.rate:.0f} s")


if __name__ == "__main__":
    main()
