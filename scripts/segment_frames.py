#!/usr/bin/env python3
"""Segment equirectangular frames and report composition by elevation band.

Step 2 of the metric pipeline, after extract_frames.py. For every frame:

    equirect -> 6 cube faces -> ADE20K segmentation -> back onto the sphere
    -> collapse 150 classes to AOI groups -> cos-weighted area per band

What comes out is one table. Every figure in the visualisation list is a
different plot of it: the elevation/azimuth heat map, the banded composition
chart, green-view and ground ratios over time, floor-material breakdown.

Bands are ELEVATION bands, not the B0-B3 world-height bands -- per-pixel
segmentation gives direction, not position. Converting needs depth.

Usage
-----
    python3 segment_frames.py --selftest            # geometry, no model
    python3 segment_frames.py frames/ --dry-run     # plumbing, no model
    python3 segment_frames.py frames/ --limit 10    # real run, 10 frames
    python3 segment_frames.py frames/               # all of them

Needs numpy and ffmpeg for --dry-run; torch, transformers and Pillow for real
work:  uv pip install torch transformers pillow
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under python -I
import sphere_geometry as sg  # noqa: E402

# ADE20K class names per AOI group. Names are matched exactly against the
# model's own id2label, so a model whose label set differs is caught, not
# silently mis-grouped.
AOI = {
    "ground": [
        "floor", "road", "sidewalk", "earth", "path", "land", "field", "rug",
        "dirt track", "runway", "sand", "stairs", "stairway", "step",
        "escalator",
    ],
    "vegetation": ["tree", "grass", "plant", "flower", "palm"],
    "furniture": [
        "cabinet", "bed", "table", "chair", "sofa", "shelf", "armchair",
        "seat", "desk", "wardrobe", "chest of drawers", "counter",
        "countertop", "case", "pool table", "coffee table", "bookcase",
        "bench", "kitchen island", "swivel chair", "stool", "ottoman",
        "buffet", "bar", "booth", "stage",
    ],
    "human": ["person"],
    "vehicle": [
        "car", "bus", "truck", "van", "boat", "ship", "airplane", "bicycle",
        "minibike",
    ],
    "structure": [
        "wall", "building", "house", "skyscraper", "ceiling", "column",
        "windowpane", "fence", "railing", "bannister", "hovel", "tower",
        "bridge", "grandstand", "canopy", "awning", "pier", "curtain",
        "blind",
    ],
    "sky": ["sky"],
    "signage": [
        "signboard", "poster", "bulletin board", "screen", "crt screen",
        "monitor", "television receiver", "trade name", "flag",
        "traffic light",
    ],
    "door": ["door", "screen door"],
}
AOI_NAMES = list(AOI) + ["other"]
OTHER = len(AOI_NAMES) - 1

DEFAULT_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"


def build_lookup(id2label):
    """Map every model class id to an AOI index, reporting what fell through."""
    by_name = {}
    for group, names in AOI.items():
        for n in names:
            by_name[n] = AOI_NAMES.index(group)

    lut = np.full(max(id2label) + 1, OTHER, dtype=np.int64)
    unmapped = []
    for cid, label in id2label.items():
        # HF ADE20K labels are comma-separated synonyms, sometimes padded
        head = label.split(",")[0].strip().lower()
        if head in by_name:
            lut[cid] = by_name[head]
        else:
            unmapped.append(head)
    return lut, unmapped


class Segmenter:
    """ADE20K semantic segmentation over a batch of cube faces."""

    def __init__(self, model_name, device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def __call__(self, faces):
        """faces: (n, size, size, 3) uint8 -> (n, size, size) int64 class ids."""
        size = faces.shape[1]
        inputs = self.processor(images=list(faces), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        logits = self.torch.nn.functional.interpolate(
            logits, size=(size, size), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1).cpu().numpy()


class FakeSegmenter:
    """Stand-in for --dry-run: labels by elevation so the plumbing is testable."""

    id2label = {0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "tree"}

    def __call__(self, faces):
        n, size = faces.shape[0], faces.shape[1]
        rows = np.linspace(0, 1, size)[None, :, None]
        out = np.where(rows < 0.4, 2, np.where(rows < 0.7, 4, 3))
        return np.broadcast_to(out, (n, size, size)).astype(np.int64)


def load_equirect(path):
    """Read an image as RGB. Uses Pillow when installed, else ffmpeg."""
    try:
        from PIL import Image
    except ImportError:
        return _load_via_ffmpeg(path)
    return np.asarray(Image.open(path).convert("RGB"))


def _load_via_ffmpeg(path):
    import json
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    s = json.loads(probe.stdout)["streams"][0]
    w, h = s["width"], s["height"]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8)[: w * h * 3].reshape(h, w, 3)


def render_faces(img, fmap, size):
    """Cut the six cube faces out of one equirectangular frame."""
    faces = np.empty((6, size, size, 3), dtype=img.dtype)
    for i, name in enumerate(sg.FACE_ORDER):
        sy, sx = fmap[name]
        faces[i] = img[sy, sx]
    return faces


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="?", type=Path,
                    help="directory from extract_frames.py (holds manifest.csv)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bands", type=int, default=12,
                    help="elevation bands from -90 to +90 (default 12 = 15 deg)")
    ap.add_argument("--face-size", type=int, default=512)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only")
    ap.add_argument("--heatmap-size", type=int, default=180,
                    help="rows of the accumulated sphere grid")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake the model, to check the plumbing")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(sg.selftest())
    if args.frames is None:
        ap.error("give a frames directory, or --selftest")

    manifest = args.frames / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"no manifest.csv in {args.frames} -- run extract_frames.py first")
    with manifest.open() as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("manifest is empty")

    out_dir = args.out or args.frames / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = FakeSegmenter() if args.dry_run else Segmenter(args.model)
    lut, unmapped = build_lookup(seg.id2label)
    if unmapped and not args.dry_run:
        print(f"{len(unmapped)} classes fell into 'other', e.g. "
              f"{', '.join(sorted(set(unmapped))[:8])}")

    first = load_equirect(args.frames / rows[0]["file"])
    h, w = first.shape[:2]
    if abs(w / h - 2) > 0.02:
        print(f"warning: {w}x{h} is not 2:1 -- is this really equirectangular?")
    print(f"{len(rows)} frames at {w}x{h}, {args.bands} bands, "
          f"{args.face_size}px faces, device "
          f"{getattr(seg, 'device', 'none (dry run)')}")

    fmap = sg.face_to_equirect_map(w, h, args.face_size)
    face_idx, fy, fx = sg.equirect_to_face_map(w, h, args.face_size)
    weights = sg.pixel_weights(w, h)
    band_of_row = sg.band_index(h, args.bands)
    edges_deg = np.degrees(sg.band_edges(args.bands))

    hm_h = args.heatmap_size
    hm_w = hm_h * 2
    hm_y = (np.arange(h) * hm_h // h)
    hm_x = (np.arange(w) * hm_w // w)
    heat = np.zeros((len(AOI_NAMES), hm_h, hm_w))

    records = []
    for n, row in enumerate(rows, 1):
        img = first if n == 1 else load_equirect(args.frames / row["file"])
        faces = render_faces(img, fmap, args.face_size)
        labels = seg(faces)                              # (6, size, size)
        sphere = lut[labels[face_idx, fy, fx]]           # (h, w) AOI indices

        comp = sg.composition(sphere, weights, band_of_row,
                              args.bands, len(AOI_NAMES))
        for b in range(args.bands):
            rec = {
                "file": row["file"],
                "t_real_s": row["t_real_s"],
                "band": b,
                "lo_deg": round(float(edges_deg[b]), 1),
                "hi_deg": round(float(edges_deg[b + 1]), 1),
            }
            for j, name in enumerate(AOI_NAMES):
                rec[name] = round(float(comp[b, j]), 6)
            records.append(rec)

        np.add.at(heat, (sphere, hm_y[:, None], hm_x[None, :]), weights)

        if n % 25 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}", flush=True)

    csv_path = out_dir / "composition.csv"
    fields = ["file", "t_real_s", "band", "lo_deg", "hi_deg"] + AOI_NAMES
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    heat /= len(rows)
    np.savez_compressed(out_dir / "sphere_heatmap.npz",
                        heat=heat, names=np.array(AOI_NAMES),
                        band_edges_deg=edges_deg)

    print(f"\n{csv_path}")
    print(f"{out_dir / 'sphere_heatmap.npz'}  "
          f"({len(AOI_NAMES)} x {hm_h} x {hm_w})")

    share = np.zeros(len(AOI_NAMES))
    for rec in records:
        for j, name in enumerate(AOI_NAMES):
            share[j] += rec[name]
    share /= len(rows)
    print("\nwhole-sphere share, averaged over frames:")
    for j in np.argsort(-share):
        if share[j] > 0.001:
            print(f"  {AOI_NAMES[j]:12} {share[j]:6.1%}")
    total = share.sum()
    if abs(total - 1) > 1e-3:
        print(f"  warning: shares sum to {total:.4f}, not 1")


if __name__ == "__main__":
    main()
