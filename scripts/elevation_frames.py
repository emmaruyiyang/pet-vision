#!/usr/bin/env python3
"""Where in the visual field each AOI sits, not just how much of it there is.

aoi_frames.py collapses a frame to one number per class. That throws away the
only thing the height hypothesis is about: a sign at 1.6 m and a sign at 0.45 m
can cover the same area and still sit in completely different parts of the
field. This script keeps the position. Every paired frame is segmented, then
the pixel grid is binned by ELEVATION (and by azimuth) before the shares are
taken, so the output is a class profile per direction rather than per frame.

GEOMETRY. The panels are rectilinear 1280x720 cuts with a 90 deg horizontal
field of view, so the focal length is 640 px and the vertical field is
2*atan(360/640) = 58.7 deg. For a row at fractional height v (0 = top of
frame),

    elevation = atan((0.5 - v) * 720 / 640)
    azimuth   = atan((u - 0.5) * 1280 / 640)

The camera is assumed level; the IMU says body pitch wanders about +-10 deg
around its median, so treat the bins as nominal, not surveyed.

ELEVATION BAND IS NOT HEIGHT BAND. A band here is a direction you are looking
in, not a shelf in the world: a low elevation is the ground nearby AND the
ground far away. Converting to the B0-B3 height bands needs distance, which
needs depth estimation. See the work log.

Output: aoi/elevation_shares.csv, one row per (pair_index, view, elev_bin)
with the share of that band covered by each AOI group, and
aoi/elev_azim_shares.csv, the same aggregated over frames on a 2D grid.

Usage
-----
    python3 elevation_frames.py --limit 16
    python3 elevation_frames.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi_frames import NAMES, Segmenter, DEFAULT_MODEL, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
PAIRED = ROOT / "frames_paired"
OUT = ROOT / "aoi"

FRAME_W, FRAME_H = 1280, 720
FOCAL = FRAME_W / 2.0                      # 90 deg horizontal
ELEV_EDGES = np.array([-30, -20, -10, 0, 10, 20, 30], dtype=float)
AZIM_EDGES = np.array([-45, -30, -15, 0, 15, 30, 45], dtype=float)


def band_index(n_px, edges, extent_px):
    """Bin index per pixel row/column; -1 outside the outermost edge."""
    frac = (np.arange(n_px) + 0.5) / n_px          # 0..1 across the axis
    centred = (0.5 - frac) * extent_px             # +ve above / left of centre
    angle = np.degrees(np.arctan2(centred, FOCAL))
    idx = np.digitize(angle, edges) - 1
    idx[(angle < edges[0]) | (angle >= edges[-1])] = -1
    return idx, angle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with (PAIRED / "manifest.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    rows = rows[::args.every]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} pairs")

    seg = Segmenter(args.model, args.device)
    print(f"{args.model} on {seg.device}")

    OUT.mkdir(exist_ok=True)
    n_el, n_az, n_cl = len(ELEV_EDGES) - 1, len(AZIM_EDGES) - 1, len(NAMES)
    grid = {v: np.zeros((n_el, n_az, n_cl)) for v in ("human", "dog")}
    grid_n = {v: 0 for v in ("human", "dog")}
    rbin = cbin = None

    out = OUT / "elevation_shares.csv"
    t0 = time.time()
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_index", "t_s", "view", "elev_lo", "elev_hi"] + NAMES)
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            for view in ("human", "dog"):
                imgs = [load(PAIRED / r[f"{view}_file"]) for r in chunk]
                ids = seg.labels(imgs)                     # (n, h, w) AOI indices
                n, h, wpx = ids.shape
                if rbin is None or len(rbin) != h:
                    rbin, _ = band_index(h, ELEV_EDGES, FRAME_H)
                    cbin, _ = band_index(wpx, AZIM_EDGES, FRAME_W)
                    # azimuth sign: band_index gives +ve to the LEFT, flip it
                    cbin = cbin[::-1]
                for r, lab in zip(chunk, ids):
                    for b in range(n_el):
                        m = lab[rbin == b]
                        if m.size == 0:
                            continue
                        cnt = np.bincount(m.ravel(), minlength=n_cl) / m.size
                        w.writerow([r["pair_index"], r["t_real_s"], view,
                                    ELEV_EDGES[b], ELEV_EDGES[b + 1]]
                                   + [round(float(x), 5) for x in cnt])
                    for b in range(n_el):
                        rm = rbin == b
                        if not rm.any():
                            continue
                        sub = lab[rm]
                        for a in range(n_az):
                            cm = cbin == a
                            if not cm.any():
                                continue
                            cell = sub[:, cm]
                            grid[view][b, a] += (
                                np.bincount(cell.ravel(), minlength=n_cl) / cell.size)
                    grid_n[view] += 1
            done = i + len(chunk)
            if done % 80 == 0 or done == len(rows):
                el = time.time() - t0
                print(f"  {done}/{len(rows)} pairs  {el:.0f}s  "
                      f"eta {el / done * (len(rows) - done):.0f}s", flush=True)

    grid_out = OUT / "elev_azim_shares.csv"
    with grid_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "elev_lo", "elev_hi", "azim_lo", "azim_hi", "n_frames"]
                   + NAMES)
        for view in ("human", "dog"):
            g = grid[view] / max(grid_n[view], 1)
            for b in range(n_el):
                for a in range(n_az):
                    w.writerow([view, ELEV_EDGES[b], ELEV_EDGES[b + 1],
                                AZIM_EDGES[a], AZIM_EDGES[a + 1], grid_n[view]]
                               + [round(float(x), 5) for x in g[b, a]])

    print(f"\n-> {out}\n-> {grid_out}")


if __name__ == "__main__":
    main()
