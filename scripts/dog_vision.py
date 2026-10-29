#!/usr/bin/env python3
"""Render images the way a dichromat sees them, as a stand-in for dog colour vision.

WHAT THIS IS, AND WHAT IT IS NOT

This is a human-dichromat simulation borrowed as an approximation of dog colour
vision. It is the common approach and it is defensible, but it is not a measured
model of what a dog sees, and it should not be captioned as one.

Dogs are dichromats with cones near 429 nm and 555 nm (Neitz, Geist & Jacobs
1989). Human cones sit near 420 (S), 530 (M) and 560 (L) nm, so a deuteranope --
S plus L, no M -- is the closest human pair to a dog's, which is why it is the
default here. A protanope (S plus M) is the usual alternative; run `--type both`
to see how much the choice moves the image, because on some scenes it moves it a
lot.

Where the approximation is weakest:

- The neutral point, the wavelength a dichromat sees as grey, is reported around
  480 nm in dogs and around 492-498 nm in human dichromats. Cyans and blue-greens
  are therefore the colours this renders least reliably.
- Brettel/Vienot methods preserve which colours get CONFUSED; the hues they
  render are a human convention. Nobody knows a dog's subjective colour.
- Dog ocular media, the tapetum, and a rod-dominated retina all shift brightness
  and contrast, and none of that is modelled here.

So treat the output as a DISCRIMINABILITY model, not an appearance model: it
answers "would this target still stand out from its background", which is the
question H2 and RQ3 actually need. It does not answer "what colour is this to a
dog". Say it that way in the write-up.

The transform runs in LINEAR RGB. Applying a colour matrix straight to gamma-
encoded sRGB, which is the usual shortcut, darkens and desaturates everything by
a few percent and is wrong in a way that is easy to miss by eye.

Optional and off by default: an acuity blur. Dog acuity is roughly 20/75 against
a human 20/20, so about 3.75 arcmin against 1. Pass the image's horizontal field
of view and the blur is derived from it. Keep it off while colour is the only
variable under test -- turning both on at once confounds them.

Usage
-----
    python3 dog_vision.py frame.jpg                       # one file
    python3 dog_vision.py pairs/ --out pairs_dog/         # a folder
    python3 dog_vision.py frame.jpg --side-by-side        # original | filtered
    python3 dog_vision.py frame.jpg --acuity-fov 100      # also blur

Needs numpy and ffmpeg. No image library.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# linear sRGB <-> LMS (Hunt-Pointer-Estevez, as used by Brettel/Vienot)
RGB2LMS = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709],
])
LMS2RGB = np.linalg.inv(RGB2LMS)

# project onto the surviving dichromat plane
PROJECT = {
    # no M cone: rebuild M from L and S
    "deuteranopia": np.array([
        [1.0, 0.0, 0.0],
        [0.494207, 0.0, 1.24827],
        [0.0, 0.0, 1.0],
    ]),
    # no L cone: rebuild L from M and S
    "protanopia": np.array([
        [0.0, 2.02344, -2.52581],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]),
}


def srgb_to_linear(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    out = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def dichromat(rgb_u8, kind="deuteranopia", severity=1.0):
    """rgb_u8: (H, W, 3) uint8 sRGB -> same, seen by a dichromat."""
    lin = srgb_to_linear(rgb_u8.astype(np.float64))
    m = LMS2RGB @ PROJECT[kind] @ RGB2LMS
    out = lin @ m.T
    if severity < 1.0:
        out = lin * (1 - severity) + out * severity
    return linear_to_srgb(out)


def gaussian_blur(img, sigma):
    """Separable Gaussian, reflect-padded. sigma in pixels."""
    if sigma <= 0:
        return img
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    x = img.astype(np.float64)
    for axis in (0, 1):
        x = np.apply_along_axis(
            lambda row: np.convolve(np.pad(row, r, mode="reflect"), k, "valid"),
            axis, x)
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def acuity_sigma(width_px, hfov_deg, dog_arcmin=3.75, human_arcmin=1.0):
    """Blur that takes human acuity down to a dog's, for this image geometry."""
    px_per_deg = width_px / hfov_deg
    extra_arcmin = (dog_arcmin ** 2 - human_arcmin ** 2) ** 0.5
    return extra_arcmin / 60.0 * px_per_deg / 2.0


def read_image(path):
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
    return np.frombuffer(raw, np.uint8)[: w * h * 3].reshape(h, w, 3)


def write_image(path, img, quality=2):
    h, w = img.shape[:2]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-i", "-", "-q:v", str(quality), "-y", str(path)],
        input=img.tobytes(), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="image files or folders")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: alongside, suffixed)")
    ap.add_argument("--type", choices=sorted(PROJECT) + ["both"],
                    default="deuteranopia",
                    help="'both' writes the two side by side, to show how much "
                         "the choice of human analogue matters")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="0 = unchanged, 1 = full dichromacy")
    ap.add_argument("--acuity-fov", type=float, default=0.0, metavar="DEG",
                    help="horizontal FOV of the image; enables the acuity blur")
    ap.add_argument("--side-by-side", action="store_true",
                    help="write original | filtered instead of filtered alone")
    ap.add_argument("--quality", type=int, default=2)
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        if p.is_dir():
            files += sorted(q for q in p.iterdir()
                            if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
        elif p.exists():
            files.append(p)
        else:
            sys.exit(f"no such file: {p}")
    if not files:
        sys.exit("nothing to do")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    print(f"{len(files)} image(s), {args.type}, severity {args.severity}"
          + (f", acuity blur at {args.acuity_fov} deg HFOV"
             if args.acuity_fov else ", no acuity blur"))

    for f in files:
        img = read_image(f)
        kinds = sorted(PROJECT) if args.type == "both" else [args.type]
        panels = []
        for kind in kinds:
            p = dichromat(img, kind, args.severity)
            if args.acuity_fov:
                p = gaussian_blur(p, acuity_sigma(img.shape[1], args.acuity_fov))
            panels.append(p)
        out = np.concatenate(panels, axis=1)
        if args.side_by_side:
            out = np.concatenate([img, out], axis=1)
        suffix = "_sbs" if args.side_by_side else ""
        name = f"{f.stem}_{args.type[:5]}{suffix}{f.suffix}"
        dst = (args.out / name) if args.out else f.with_name(name)
        write_image(dst, out, args.quality)
        print(f"  {f.name} -> {dst.name}")


if __name__ == "__main__":
    main()
