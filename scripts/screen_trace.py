#!/usr/bin/env python3
"""A hint layer for marking trial boundaries by hand in the viewer.

The automatic attempts failed, and the work log says why: the scene camera
exposed for a dark room, so the monitor is blown out and its CONTENT barely
survives. Matching stimulus images against it does not work, and neither does
change-point detection on the raw frames, because head motion moves everything
by as much as a stimulus switch does.

This does not try again. A person watching the video can see every switch, so
the boundaries get marked by hand; all this produces is a trace that makes them
easy to aim at.

WHAT IT MEASURES. Each sampled frame is reduced to the brightest pixels, which
on this footage is the monitor, and two numbers come out:

  screen  -- mean brightness of that bright region. A switch to a darker or
             lighter stimulus steps it, even when the picture itself is washed
             out. trial_lists.csv records each stimulus's own mean grey, so the
             steps and the list can be read against each other.
  change  -- mean absolute difference from the previous sample, inside that
             region only. It spikes at a switch; it ALSO spikes when the head
             moves, which is exactly why this is a hint and not a detector.

Writes viewer/data/<session>.trace.json beside the session's own JSON.

Usage
-----
    python3 scripts/screen_trace.py 2026-10-27-04-12-39
    python3 scripts/screen_trace.py 2026-10-27-04-12-39 --hz 4
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "pupil_eye_tracking"
OUT = ROOT / "viewer" / "data"
W, H = 160, 120            # enough to see a whole-screen change, cheap to decode
BRIGHT_PCT = 88            # pixels above this percentile count as "the screen"


def frames(video, hz):
    """Decode the video to small greyscale frames at `hz` samples a second."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"fps={hz},scale={W}:{H}", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W).astype(np.float32)
    p.stdout.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--hz", type=float, default=4.0)
    args = ap.parse_args()

    video = SESSIONS / args.session / "Neon Scene Camera v1 ps1.mp4"
    if not video.exists():
        raise SystemExit(f"no scene video at {video}")

    screen, change, prev = [], [], None
    for f in frames(video, args.hz):
        thr = np.percentile(f, BRIGHT_PCT)
        mask = f >= thr
        screen.append(float(f[mask].mean()))
        if prev is None:
            change.append(0.0)
        else:
            both = mask | (prev >= np.percentile(prev, BRIGHT_PCT))
            change.append(float(np.abs(f[both] - prev[both]).mean()))
        prev = f

    screen = np.array(screen)
    change = np.array(change)
    t = np.arange(len(screen)) / args.hz

    #  Candidates come from the SCREEN BRIGHTNESS step, not from the change
    #  trace. Both spike at a switch, but the head also moves, and that moves
    #  every pixel while leaving the screen's own mean brightness alone. So the
    #  brightness step is the one signal here that a switch produces and head
    #  motion does not.
    #
    #  Deliberately generous: this is a hint, so a false candidate costs a
    #  glance and a missed one costs a boundary.
    k = max(1, int(round(args.hz)))            # 1 s smoothing, to kill decode noise
    sm = np.convolve(screen, np.ones(k) / k, mode="same")
    step = np.abs(np.diff(sm, prepend=sm[0]))
    floor = np.percentile(step, 90)
    peaks = []
    for i in range(1, len(step) - 1):
        if step[i] > floor and step[i] >= step[i - 1] and step[i] > step[i + 1]:
            if not peaks or t[i] - peaks[-1] > 3.0:
                peaks.append(round(float(t[i]), 2))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.session}.trace.json"
    out.write_text(json.dumps({
        "hz": args.hz,
        "screen": [round(float(v), 1) for v in screen],
        "change": [round(float(v), 2) for v in change],
        "candidates": peaks,
        "threshold": round(float(floor), 2),
    }, separators=(",", ":")))
    print(f"-> {out}")
    print(f"{len(screen)} samples at {args.hz} Hz, {len(peaks)} candidate switches")
    print("screen brightness: median %.0f, range %.0f-%.0f"
          % (np.median(screen), screen.min(), screen.max()))


if __name__ == "__main__":
    main()
