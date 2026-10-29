#!/usr/bin/env python3
"""Demo video: the two viewpoints side by side, with a live AOI bar chart below.

Output is 1920x1080 --

    +---------------------+---------------------+
    |   human 1.6 m       |   dog 0.45 m        |   960x540 each
    +---------------------+---------------------+
    |      grouped AOI bars, updating live       |   1920x540
    +--------------------------------------------+

The source is the combined export (1920x2160, human on top, dog below, already
time-synchronised). The bars come from aoi/aoi_shares.csv, which is sampled at
1 Hz, so they are interpolated up to --chart-fps and smoothed over --smooth
seconds. Raw 1 Hz bars jump about so much that the real differences are hard to
see; the smoothing window is stated on the chart so nobody reads it as raw.

ALIGNMENT. Frame i of the CSV was sampled at source t = i + 0.5 s. The chart is
offset by that half second, so bars and picture refer to the same instant to
within the 1 Hz sampling.

THE DOG'S OWN BODY is dropped and both panels renormalised, same as
aoi_plots.py -- the fur on the lens is the mount, not the scene.

Usage
-----
    python3 aoi_video.py --start 600 --duration 60      # a 1 min excerpt
    python3 aoi_video.py                                # the whole 35 min
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the on-screen caption is Chinese; DejaVu has no CJK and would draw boxes
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
AOI_DIR = ROOT / "aoi"
SRC = Path("/Users/ruyiyang/Library/Mobile Documents/com~apple~CloudDocs/"
           "insta-360/export/pet-vision-combined.mp4")
W, H_CHART = 1920, 540
INK, MUTED = "#1a1a1a", "#8a8a8a"
HUMAN, DOG = "#2b6cb0", "#c05621"
SAMPLE_OFFSET = 0.5


def load():
    with (AOI_DIR / "aoi_shares.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    names = [k for k in rows[0] if k not in ("pair_index", "t_s", "view")]
    out = {}
    for view in ("human", "dog"):
        r = sorted((x for x in rows if x["view"] == view),
                   key=lambda x: int(x["pair_index"]))
        out[view] = (np.array([float(x["t_s"]) for x in r]),
                     np.array([[float(x[n]) for n in names] for x in r]))
    keep = [i for i, n in enumerate(names) if n != "animal"]
    names = [names[i] for i in keep]
    for v in out:
        t, m = out[v]
        m = m[:, keep]
        out[v] = (t, m / np.maximum(m.sum(1, keepdims=True), 1e-9))
    return names, out


def smooth(x, w):
    if w < 2:
        return x
    k = np.ones(w) / w
    p = w // 2
    return np.convolve(np.pad(x, p, mode="edge"), k, "same")[p:p + len(x)]


class Chart:
    """One reused figure; each call returns the next frame as raw RGB bytes."""

    def __init__(self, groups, ymax):
        self.fig, self.ax = plt.subplots(figsize=(W / 150, H_CHART / 150), dpi=150)
        x = np.arange(len(groups))
        self.bh = self.ax.bar(x - 0.19, np.zeros(len(groups)), 0.36,
                              color=HUMAN, label="human 1.6 m")
        self.bd = self.ax.bar(x + 0.19, np.zeros(len(groups)), 0.36,
                              color=DOG, label="dog 0.45 m")
        self.labels = [
            self.ax.text(xx + dx, 0, "", ha="center", va="bottom",
                         fontsize=9, color=c)
            for xx in x for dx, c in ((-0.19, HUMAN), (0.19, DOG))]
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(groups, fontsize=12, color=INK)
        self.ax.set_ylim(0, ymax)
        self.ax.set_ylabel("share of frame", color=MUTED, fontsize=10)
        for s in ("top", "right"):
            self.ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            self.ax.spines[s].set_color(MUTED)
        self.ax.tick_params(colors=MUTED, labelsize=9)
        self.ax.grid(axis="y", color="#e8e8e8", lw=0.6)
        self.ax.set_axisbelow(True)
        leg = self.ax.legend(frameon=False, fontsize=11, loc="upper right",
                             ncol=2)
        for t in leg.get_texts():
            t.set_color(MUTED)
        self.title = self.ax.set_title("", color=INK, fontsize=13,
                                       fontweight="600", loc="left", pad=12)
        # not tight_layout: the figure is exactly 540 px tall and tight_layout
        # crops the title against the video panel above it
        self.fig.subplots_adjust(left=0.045, right=0.99, top=0.84, bottom=0.11)

    def frame(self, hv, dv, text):
        for b, v in zip(self.bh, hv):
            b.set_height(v)
        for b, v in zip(self.bd, dv):
            b.set_height(v)
        vals = [v for pair in zip(hv, dv) for v in pair]
        for lab, v in zip(self.labels, vals):
            lab.set_y(v + 0.006)
            lab.set_text(f"{v:.0%}")
        self.title.set_text(text)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=AOI_DIR / "aoi_demo.mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = all")
    ap.add_argument("--chart-fps", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=3, help="seconds")
    ap.add_argument("--groups", type=int, default=5)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"missing source video: {args.src}")
    names, data = load()
    th, mh = data["human"]
    _, md = data["dog"]
    both = np.concatenate([mh, md])
    order = [names[i] for i in np.argsort(-both.mean(0))[:args.groups]]
    cols = [names.index(g) for g in order]

    hs = np.stack([smooth(mh[:, c], args.smooth) for c in cols], 1)
    ds = np.stack([smooth(md[:, c], args.smooth) for c in cols], 1)

    end = th[-1] + 0.5 if args.duration <= 0 else args.start + args.duration
    n = int(round((end - args.start) * args.chart_fps))
    grid = args.start + np.arange(n) / args.chart_fps
    src_t = th - SAMPLE_OFFSET            # so bars lead the picture by nothing
    hi = np.stack([np.interp(grid, src_t, hs[:, i]) for i in range(len(order))], 1)
    di = np.stack([np.interp(grid, src_t, ds[:, i]) for i in range(len(order))], 1)

    ymax = float(max(hi.max(), di.max())) * 1.22
    print(f"{n} chart frames at {args.chart_fps} fps "
          f"({args.start:.0f}-{end:.0f} s), groups: {', '.join(order)}")

    chart = AOI_DIR / "_chart.mp4"
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H_CHART}", "-r", str(args.chart_fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(chart)],
        stdin=subprocess.PIPE)
    c = Chart(order, ymax)
    for i in range(n):
        t = grid[i]
        enc.stdin.write(c.frame(
            hi[i], di[i],
            f"画面构成占比  |  {int(t) // 60:02d}:{int(t) % 60:02d}  |  "
            f"{args.smooth} s 平滑，已剔除狗自身身体"))
        if (i + 1) % (args.chart_fps * 60) == 0:
            print(f"  {i + 1}/{n}", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("chart encode failed")

    trim = ["-ss", str(args.start)] + (
        ["-t", str(args.duration)] if args.duration > 0 else [])
    fc = ("[0:v]crop=1920:1080:0:0,scale=960:540[h];"
          "[0:v]crop=1920:1080:0:1080,scale=960:540[d];"
          "[h][d]hstack=inputs=2[top];"
          "[1:v]fps=30[ch];"
          "[top][ch]vstack=inputs=2[v]")
    cmd = (["ffmpeg", "-v", "error", "-y"] + trim + ["-i", str(args.src),
           "-i", str(chart), "-filter_complex", fc, "-map", "[v]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-shortest", str(args.out)])
    print("composing...")
    subprocess.run(cmd, check=True)
    chart.unlink()
    size = args.out.stat().st_size / 1e6
    print(f"-> {args.out}  ({size:.0f} MB)")


if __name__ == "__main__":
    main()
