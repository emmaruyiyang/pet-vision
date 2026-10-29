#!/usr/bin/env python3
"""Export the n=1 stimulus gaze into a web-playable bundle.

Writes docs/gaze-by-scene/data.json plus a downscaled copy of each stimulus into
docs/gaze-by-scene/stim/. Coordinates are kept in stimulus pixels (1920x1080) and
normalised in the browser, so the image copy can be any size.

The homography error is tens of stimulus pixels; the viewer says so on screen.
"""
import csv, json, shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "gaze-by-scene"
STIMOUT = OUT / "stim"
STIM_W, STIM_H = 1920, 1080
TARGET_HZ = 60.0

STIMOUT.mkdir(parents=True, exist_ok=True)

trials = list(csv.DictReader((ROOT / "eyetracking_n1" / "P01_trials.csv").open()))
sessions = sorted({t["session"] for t in trials})

gaze = {}
for s in sessions:
    p = ROOT / "viewer" / "data" / f"{s}.gaze_px.csv"
    rows = []
    for r in csv.DictReader(p.open()):
        if r["on_screen"] != "1" or not r["stim_x"]:
            continue
        rows.append((float(r["t_s"]), float(r["stim_x"]), float(r["stim_y"])))
    rows.sort()
    gaze[s] = rows

out_trials = []
for t in trials:
    on, off = float(t["onset_s"]), float(t["offset_s"])
    rows = [r for r in gaze[t["session"]] if on <= r[0] < off]
    # decimate to ~TARGET_HZ so the JSON stays small
    kept, last = [], -1e9
    for ts, x, y in rows:
        if ts - last < 1.0 / TARGET_HZ:
            continue
        last = ts
        kept.append((round(ts - on, 3), round(x, 1), round(y, 1)))

    src = ROOT / "eyetracking_n1" / t["file"]
    name = Path(t["file"]).name
    dst = STIMOUT / name
    if not dst.exists():
        im = Image.open(src).convert("RGB")
        im.thumbnail((1280, 720), Image.LANCZOS)
        im.save(dst, quality=85, optimize=True)

    #  the aggregate heat map for this same trial, rendered separately; the
    #  page lets one plate show either the live trace or this
    heat = f"{t['slug']}__{t['condition']}.jpg"
    has_heat = (OUT / "heat" / heat).exists()

    out_trials.append({
        "presented": int(t["presented"]),
        "trial": t["trial"],
        "scene": t["scene"],
        "slug": t["slug"],
        "condition": t["condition"],
        "img": f"stim/{name}",
        "heat": f"heat/{heat}" if has_heat else None,
        "onset_s": on,
        "duration_s": round(off - on, 3),
        "n": len(kept),
        "t": [p[0] for p in kept],
        "x": [p[1] for p in kept],
        "y": [p[2] for p in kept],
    })

# per-trial AOI dwell, for the side readout
dwell = {}
for r in csv.DictReader((ROOT / "aoi" / "gaze_aoi_shares.csv").open()):
    key = int(r["presented"])
    dwell[key] = {k[len("dwell_"):]: float(v) for k, v in r.items()
                  if k.startswith("dwell_")}
    dwell[key + 10000] = {k[len("area_"):]: float(v) for k, v in r.items()
                          if k.startswith("area_")}

for tr in out_trials:
    tr["dwell"] = dwell.get(tr["presented"], {})
    tr["area"] = dwell.get(tr["presented"] + 10000, {})

#  the homography's own report: the rectified screen comes out at 1.43 where
#  the stimulus is 1.78, so vertical gaze positions are compressed ~20%. The
#  viewer states it rather than silently drawing a trace that is 20% too short.
surf = ROOT / "viewer" / "data" / f"{sessions[0]}.surface.json"
aspect = json.load(surf.open())["aspect_median"] if surf.exists() else None

json.dump({"stim_w": STIM_W, "stim_h": STIM_H,
           "aspect_median": aspect, "aspect_nominal": STIM_W / STIM_H,
           "trials": out_trials},
          (OUT / "data.json").open("w"), separators=(",", ":"))
print(f"{len(out_trials)} trials, {sum(t['n'] for t in out_trials)} samples")
print(OUT / "data.json")
