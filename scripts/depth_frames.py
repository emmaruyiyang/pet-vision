#!/usr/bin/env python3
"""Depth over the paired frames: openness, enclosure, and how far the ground runs.

Checklist items 6 (openness over time) and 8 (obstacle share and distance
distribution), plus the data behind the height-band attempt. Runs Depth
Anything V2 beside the ADE20K segmenter on every pair and writes one row per
panel to aoi/depth_metrics.csv.

WHAT THE DEPTH MODEL CAN AND CANNOT DO HERE -- read this before using the
columns, it decides which ones are safe.

The "metric" checkpoint is not metric at this viewpoint. Asked how high the
camera is -- which the ground plane answers on its own, since a ground pixel at
elevation theta and distance d puts the camera d*sin|theta| above it -- it
returns about 3.1 m for BOTH panels, where the truth is 1.60 and 0.45. It is
anchored to its training prior (a car roof), not to the picture.

So the scale is solved per frame instead, from the one thing that is known:

    s = camera_height / median over ground pixels of (d * |sin theta|)

After that anchor the NEAR field is good. The nearest visible ground comes out
at 2.70 m for the human panel and 0.77 m for the dog panel, against 2.84 and
0.80 from pure geometry -- within 5% in both, independently.

The FAR field is not. Heights of the same class at the same instant disagree
between the panels by roughly the ratio of the two camera heights: structure
reads 2.1 m from the human panel and 0.6 m from the dog panel. The model is
reproducing the anchor, not the scene. THEREFORE: no column here converts an
elevation band into a height band (B0-B3), and no column reports the height of
anything above the near field. The work log's warning stands, and `--audit`
prints the numbers behind it.

The columns that survive are ratios and near-field distances, in units of the
camera's own height where that makes them comparable between the panels.

Usage
-----
    python3 depth_frames.py --limit 16          # speed check
    python3 depth_frames.py --audit             # the cross-view height check
    python3 depth_frames.py --every 2           # every other second
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi_frames import Segmenter, NAMES, DEFAULT_MODEL, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
PAIRED = ROOT / "frames_paired"
OUT = ROOT / "aoi"

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
W, H, FOCAL = 1280, 720, 640.0
CAM = {"human": 1.60, "dog": 0.45}
#  rays steep enough that d*sin(theta) is well conditioned; near the horizon a
#  small depth error moves the implied camera height by metres
CALIB_MAX_ELEV = -15.0
#  "obstacle" = the things that stop a walk, as opposed to the surfaces it
#  happens on and the sky above it
OBSTACLE = ("structure", "furniture", "vehicle", "vegetation", "human", "door")


class Depth:
    def __init__(self, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
        self.model.to(device).eval()

    def __call__(self, images):
        inputs = self.proc(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            pred = self.model(**inputs).predicted_depth
        up = self.torch.nn.functional.interpolate(
            pred[:, None], size=(H, W), mode="bicubic", align_corners=False)
        return up[:, 0].cpu().numpy()


def geometry():
    v = (np.arange(H) + 0.5) / H
    theta = np.arctan2((0.5 - v) * H, FOCAL)[:, None]
    return theta, np.sin(theta), np.cos(theta)


def metrics(view, depth, lab, theta, sin_t, cos_t):
    """One panel -> dict of columns, or None if the ground is not visible."""
    cam = CAM[view]
    st = np.broadcast_to(sin_t, lab.shape)
    ct = np.broadcast_to(cos_t, lab.shape)
    gi = NAMES.index("ground")

    gmask = (lab == gi) & (st < np.sin(np.radians(CALIB_MAX_ELEV)))
    if gmask.sum() < 500:
        return None
    s = cam / np.median(depth[gmask] * np.abs(st[gmask]))

    dist = s * depth                       # metres along the ray, near field only
    horiz = dist * ct                      # horizontal distance
    ground = (lab == gi)
    obst = np.isin(lab, [NAMES.index(c) for c in OBSTACLE])
    sky = lab == NAMES.index("sky")

    #  in camera heights, so the two panels are on the same footing: the
    #  geometry says the nearest visible ground sits at cam/tan(29.36 deg)
    #  whatever the camera height is
    hz_cam = horiz / cam
    near = hz_cam[ground & (st < np.sin(np.radians(-5)))]

    row = {
        "scale": round(float(s), 4),
        "sky_share": round(float(sky.mean()), 4),
        "ground_share": round(float(ground.mean()), 4),
        "obstacle_share": round(float(obst.mean()), 4),
        "ground_near_cam": round(float(np.percentile(near, 2)), 3) if near.size else "",
        "ground_median_cam": round(float(np.median(near)), 3) if near.size else "",
        "ground_far_cam": round(float(np.percentile(near, 98)), 3) if near.size else "",
        "obstacle_median_m": round(float(np.median(horiz[obst])), 3) if obst.any() else "",
        "obstacle_p10_m": round(float(np.percentile(horiz[obst], 10)), 3) if obst.any() else "",
        #  openness: the share of the field that is either sky or further than
        #  ten camera heights. Both panels then ask the same question -- how
        #  much of what I can see is NOT close enough to matter.
        "open_share": round(float((sky | (hz_cam > 10)).mean()), 4),
        "enclosed_share": round(float(((~sky) & (hz_cam < 2)).mean()), 4),
    }
    return row


COLUMNS = ["pair_index", "t_s", "view", "scale", "sky_share", "ground_share",
           "obstacle_share", "ground_near_cam", "ground_median_cam",
           "ground_far_cam", "obstacle_median_m", "obstacle_p10_m",
           "open_share", "enclosed_share"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--audit", action="store_true",
                    help="print the cross-view height check and stop")
    args = ap.parse_args()

    with (PAIRED / "manifest.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    rows = rows[::args.every]
    if args.limit:
        rows = rows[:args.limit]

    import torch
    dev = args.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    seg = Segmenter(DEFAULT_MODEL, dev)
    dep = Depth(dev)
    theta, sin_t, cos_t = geometry()
    print(f"{len(rows)} pairs on {dev}")

    if args.audit:
        audit(rows, seg, dep, theta, sin_t, cos_t)
        return

    OUT.mkdir(exist_ok=True)
    out = OUT / "depth_metrics.csv"
    t0 = time.time()
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            for view in ("human", "dog"):
                imgs = [load(PAIRED / r[f"{view}_file"]) for r in chunk]
                depths = dep(imgs)
                labs = seg.labels(imgs)
                for r, d, lab in zip(chunk, depths, labs):
                    lab = np.asarray(_resize(lab))
                    m = metrics(view, d, lab, theta, sin_t, cos_t)
                    if m is None:
                        continue
                    w.writerow({"pair_index": r["pair_index"],
                                "t_s": r["t_real_s"], "view": view, **m})
            done = i + len(chunk)
            if done % 40 == 0 or done == len(rows):
                el = time.time() - t0
                print(f"  {done}/{len(rows)} pairs  {el:.0f}s  "
                      f"eta {el / done * (len(rows) - done):.0f}s", flush=True)
    print(f"\n-> {out}")


def _resize(lab):
    from PIL import Image
    return Image.fromarray(lab.astype(np.uint8)).resize((W, H), Image.NEAREST)


def audit(rows, seg, dep, theta, sin_t, cos_t):
    """Do the two panels agree on how high the same things are? (They do not.)"""
    classes = ("structure", "vegetation", "signage", "vehicle")
    got = {c: [] for c in classes}
    near = {v: [] for v in CAM}
    for r in rows:
        heights = {}
        for view in ("human", "dog"):
            img = load(PAIRED / r[f"{view}_file"])
            d = dep([img])[0]
            lab = np.asarray(_resize(seg.labels([img])[0]))
            st = np.broadcast_to(sin_t, lab.shape)
            gm = (lab == NAMES.index("ground")) & (st < np.sin(np.radians(CALIB_MAX_ELEV)))
            if gm.sum() < 500:
                break
            s = CAM[view] / np.median(d[gm] * np.abs(st[gm]))
            hgt = CAM[view] + s * d * st
            heights[view] = {c: np.median(hgt[lab == NAMES.index(c)])
                             for c in classes
                             if (lab == NAMES.index(c)).sum() > 200}
            hz = s * d * np.broadcast_to(cos_t, lab.shape)
            gn = hz[(lab == NAMES.index("ground")) & (st < np.sin(np.radians(-5)))]
            if gn.size:
                near[view].append(np.percentile(gn, 2))
        if len(heights) == 2:
            for c in classes:
                if c in heights["human"] and c in heights["dog"]:
                    got[c].append((heights["human"][c], heights["dog"][c]))

    print("\nNEAR FIELD -- anchored scale against pure geometry")
    print(f"{'view':>6} {'measured':>10} {'predicted':>10} {'n':>5}")
    for v in ("human", "dog"):
        pred = CAM[v] / np.tan(np.radians(29.36))
        print(f"{v:>6} {np.median(near[v]):>10.2f} {pred:>10.2f} {len(near[v]):>5}")

    print("\nCROSS-VIEW HEIGHTS -- the same thing, at the same instant")
    print(f"{'class':<12}{'human (m)':>11}{'dog (m)':>10}{'ratio':>8}{'pairs':>7}")
    for c in classes:
        if not got[c]:
            continue
        a = np.array(got[c])
        print(f"{c:<12}{np.median(a[:, 0]):>11.2f}{np.median(a[:, 1]):>10.2f}"
              f"{np.median(a[:, 0] / a[:, 1]):>8.2f}{len(a):>7}")
    print(f"\nthe two camera heights are in the ratio "
          f"{CAM['human'] / CAM['dog']:.2f}; a ratio near that means the model "
          f"is\nreproducing the anchor rather than measuring the scene.")


if __name__ == "__main__":
    main()
