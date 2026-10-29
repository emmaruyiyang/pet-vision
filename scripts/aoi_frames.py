#!/usr/bin/env python3
"""AOI composition of the paired frames, with no eye tracking involved.

This answers "what is IN the picture at each height", not "where did anyone
look". Every frame of frames_paired/ is segmented with an ADE20K model, the
150 classes are collapsed to a handful of AOI groups, and the share of the
frame each group covers is written out. Human and dog panels are the same
instant, so every row pairs with its counterpart by pair_index.

The frames are rectilinear 1280x720 panels cut from the combined export, NOT
equirectangular, so there is no cubemap split here and no cos weighting --
that machinery (sphere_geometry.py, segment_frames.py) is for 360 sources.
Pixel shares are computed on the model's square 512x512 input; a uniform
rescale preserves area RATIOS, so the shares need no correction.

ONE THING TO KNOW BEFORE READING THE NUMBERS. The dog camera rides on a
harness on the dog's own back, so the dog's fur fills a corner of most dog
frames, and the handler's legs appear in many. Both are real parts of that
viewpoint, but neither is scene content. `animal` is kept as its own group
and `--exclude-self` renormalises the dog panel without it, which is the
comparison worth reporting.

Usage
-----
    python3 aoi_frames.py --limit 16          # speed check
    python3 aoi_frames.py                     # all 2092 pairs
    python3 aoi_frames.py --every 2           # every other second
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_frames import AOI as BASE_AOI, build_lookup  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
PAIRED = ROOT / "frames_paired"
OUT = ROOT / "aoi"

# the dog's own body and other animals; see the module docstring
AOI = dict(BASE_AOI)
AOI["animal"] = ["animal"]
NAMES = list(AOI) + ["other"]
OTHER = len(NAMES) - 1

DEFAULT_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"


def build_lut(id2label):
    by_name = {n: NAMES.index(g) for g, names in AOI.items() for n in names}
    lut = np.full(max(id2label) + 1, OTHER, dtype=np.int64)
    for cid, label in id2label.items():
        head = label.split(",")[0].strip().lower()
        if head in by_name:
            lut[cid] = by_name[head]
    return lut


class Segmenter:
    def __init__(self, model_name, device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.torch, self.device = torch, device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.lut = build_lut(self.id2label)

    def labels(self, images):
        """images: list of HxWx3 uint8 -> (n, h, w) int AOI-group index per pixel."""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            ids = self.model(**inputs).logits.argmax(dim=1).cpu().numpy()
        return self.lut[ids]

    def shares(self, images):
        """images: list of HxWx3 uint8 -> (n, len(NAMES)) float shares per frame."""
        groups = self.labels(images)                 # (n, h, w)
        n, h, w = groups.shape
        flat = groups.reshape(n, -1)
        counts = np.stack([np.bincount(row, minlength=len(NAMES))
                           for row in flat])
        return counts / float(h * w)


def load(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth pair")
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
    missing = sorted({v.split(',')[0].strip().lower()
                      for k, v in seg.id2label.items() if seg.lut[k] == OTHER})
    print(f"{len(missing)} of {len(seg.id2label)} classes fall in 'other'")

    OUT.mkdir(exist_ok=True)
    out = OUT / "aoi_shares.csv"
    t0 = time.time()
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_index", "t_s", "view"] + NAMES)
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            for view in ("human", "dog"):
                imgs = [load(PAIRED / r[f"{view}_file"]) for r in chunk]
                sh = seg.shares(imgs)
                for r, s in zip(chunk, sh):
                    w.writerow([r["pair_index"], r["t_real_s"], view]
                               + [round(float(x), 5) for x in s])
            done = i + len(chunk)
            if done % 80 == 0 or done == len(rows):
                el = time.time() - t0
                print(f"  {done}/{len(rows)} pairs  {el:.0f}s  "
                      f"eta {el / done * (len(rows) - done):.0f}s", flush=True)

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
