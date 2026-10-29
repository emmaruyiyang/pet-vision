#!/usr/bin/env python3
"""Export the per-second indicators for one excerpt of the walk, for the
video+chart viewer in docs/live-aoi/.

Everything is keyed on SOURCE time (seconds into the combined export). The
viewer adds the clip offset itself, so the JSON stays honest about where the
numbers come from.

Row i of aoi_shares.csv was sampled at t = i + 0.5 s; that half second is kept
in `t_s` rather than rounded away, and the viewer interpolates between rows.

The dog's own fur (`animal`) is dropped and both panels renormalised, same as
every figure in aoi/.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AOI = ROOT / "aoi"
OUT = ROOT / "docs" / "live-aoi"

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=float, default=300.0)
ap.add_argument("--duration", type=float, default=300.0)
ap.add_argument("--out", default="data.json")
a = ap.parse_args()
lo, hi = a.start, a.start + a.duration


def window(rows, key="t_s"):
    return [r for r in rows if lo <= float(r[key]) < hi]


# ---- composition -----------------------------------------------------------
shares = list(csv.DictReader((AOI / "aoi_shares.csv").open()))
names = [k for k in shares[0] if k not in ("pair_index", "t_s", "view")]
keep = [n for n in names if n != "animal"]

comp = {}
for view in ("human", "dog"):
    r = sorted((x for x in window(shares) if x["view"] == view),
               key=lambda x: float(x["t_s"]))
    t = [float(x["t_s"]) for x in r]
    m = np.array([[float(x[n]) for n in keep] for x in r])
    m = m / np.maximum(m.sum(1, keepdims=True), 1e-9)
    comp[view] = {"t": t,
                  "series": {n: [round(v, 5) for v in m[:, i]]
                             for i, n in enumerate(keep)}}

# ---- depth / openness ------------------------------------------------------
depth = list(csv.DictReader((AOI / "depth_metrics.csv").open()))
dep = {}
for view in ("human", "dog"):
    r = sorted((x for x in window(depth) if x["view"] == view),
               key=lambda x: float(x["t_s"]))
    dep[view] = {
        "t": [float(x["t_s"]) for x in r],
        "open_share": [round(float(x["open_share"]), 4) for x in r],
        "enclosed_share": [round(float(x["enclosed_share"]), 4) for x in r],
        "obstacle_share": [round(float(x["obstacle_share"]), 4) for x in r],
        "ground_near_cam": [round(float(x["ground_near_cam"]), 3) for x in r],
    }

# ---- divergence ------------------------------------------------------------
div = sorted(window(list(csv.DictReader((AOI / "divergence_tv.csv").open()))),
             key=lambda x: float(x["t_s"]))
diverg = {"t": [float(x["t_s"]) for x in div],
          "tv": [round(float(x["tv"]), 4) for x in div],
          "dropped": [int(x["dropped_dark_or_occluded"]) for x in div]}

# ---- IMU, averaged into the same 1 Hz bins ---------------------------------
it, imot, iturn = [], [], []
with (ROOT / "imu" / "imu.csv").open() as fh:
    for r in csv.DictReader(fh):
        t = float(r["t_total_s"])
        if lo <= t < hi:
            it.append(t); imot.append(float(r["motion"])); iturn.append(abs(float(r["turn_dps"])))
it = np.array(it); imot = np.array(imot); iturn = np.array(iturn)
bins = np.arange(int(lo), int(hi) + 1)
idx = np.clip(np.digitize(it, bins) - 1, 0, len(bins) - 2)
motion, turn = [], []
for b in range(len(bins) - 1):
    m = idx == b
    motion.append(round(float(imot[m].mean()), 4) if m.any() else None)
    turn.append(round(float(iturn[m].mean()), 3) if m.any() else None)
imu = {"t": [float(b) + 0.5 for b in bins[:-1]], "motion": motion, "turn_dps": turn}

stops = [{"start_s": float(r["start_s"]), "end_s": float(r["end_s"])}
         for r in csv.DictReader((ROOT / "imu" / "stops.csv").open())
         if float(r["end_s"]) > lo and float(r["start_s"]) < hi]

json.dump({"start_s": lo, "duration_s": a.duration, "groups": keep,
           "composition": comp, "depth": dep, "divergence": diverg,
           "imu": imu, "stops": stops},
          (OUT / a.out).open("w"), separators=(",", ":"))
print(f"{lo:.0f}-{hi:.0f}s -> {OUT / a.out}")
