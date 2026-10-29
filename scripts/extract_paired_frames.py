#!/usr/bin/env python3
"""Split a vertically-stacked dual-viewpoint video into PAIRED frame sets.

The source is 1920x2160: top half = human / high viewpoint, bottom half =
dog / low viewpoint, time-synchronised. This samples 1 frame per real second
from BOTH panels at the SAME source timestamp, so frames_paired/human/NNNNN.jpg
and frames_paired/dog/NNNNN.jpg are always the same instant.

Sampling: a FIXED offset each second (default t = 0.5, 1.5, 2.5 ...), the
same timestamp for both panels, so pairing is correct by construction. There
is deliberately NO sharpest-candidate selection: keeping the sharpest frame of
each second is a non-random sample favouring moments the camera moved slowly
(the dog pausing), which are exactly the moments whose scene composition
differs — that would bias the composition metrics downstream. Sharpness and
mean brightness are still MEASURED for every kept frame and written to the
manifest, for later filtering or reporting.

One decode pass over the source produces both panels plus the small grayscale
copies used for sharpness/brightness, so the 6.8 GB file is never re-encoded
wholesale.

Only ffmpeg and numpy are needed.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from extract_frames import PROBE_W, PROBE_H, laplacian_variance, probe, run

PANELS = {"human": 0, "dog": 1080}  # name -> crop y offset


def decode(src, work, offset, max_width, quality):
    """Single pass: 1 fps JPEGs for both panels + gray probe streams."""
    chains, maps = [], []
    for name, y in PANELS.items():
        d = work / f"{name}_raw"
        d.mkdir(parents=True, exist_ok=True)
        chains.append(
            f"[v]crop=1920:1080:0:{y},split=2[{name}a][{name}b];"
            f"[{name}a]scale={max_width}:-2[{name}big];"
            f"[{name}b]scale={PROBE_W}:{PROBE_H},format=gray[{name}s]"
        )
        maps += ["-map", f"[{name}big]", "-q:v", str(quality),
                 str(d / "c_%06d.jpg")]
        maps += ["-map", f"[{name}s]", "-f", "rawvideo", "-pix_fmt", "gray",
                 str(work / f"{name}.raw")]

    # trim then re-base PTS: fps=1 now lands on source t = offset, offset+1, ...
    # (fps's own start_time pads by duplicating frame 0 instead of offsetting)
    pre = f"trim=start={offset},setpts=PTS-STARTPTS," if offset > 0 else ""
    fc = (f"[0:v]{pre}fps=1,split={len(PANELS)}"
          + "".join(f"[v{i}]" for i in range(len(PANELS))) + ";")
    for i, c in enumerate(chains):
        fc += c.replace("[v]", f"[v{i}]", 1) + ";"
    fc = fc.rstrip(";")

    run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-filter_complex", fc] + maps)


def load_probe(path):
    g = np.fromfile(path, dtype=np.uint8)
    n = g.size // (PROBE_W * PROBE_H)
    g = g[: n * PROBE_W * PROBE_H].reshape(n, PROBE_H, PROBE_W)
    return laplacian_variance(g), g.reshape(n, -1).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--offset", type=float, default=0.5,
                    help="fixed within-second sampling offset, seconds")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--quality", type=int, default=3)
    ap.add_argument("--skip-decode", action="store_true")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg must be on PATH")
    info = probe(args.video)
    print(f"source {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
          f"{info['duration']:.1f} s")
    if (info["width"], info["height"]) != (1920, 2160):
        sys.exit("expected a 1920x2160 stacked source")

    args.work.mkdir(parents=True, exist_ok=True)
    if not args.skip_decode:
        decode(args.video, args.work, args.offset, args.max_width, args.quality)

    files = {n: sorted((args.work / f"{n}_raw").glob("c_*.jpg")) for n in PANELS}
    sharp, bright = {}, {}
    for n in PANELS:
        sharp[n], bright[n] = load_probe(args.work / f"{n}.raw")

    counts = {f"{n}_jpg": len(files[n]) for n in PANELS}
    counts.update({f"{n}_probe": len(sharp[n]) for n in PANELS})
    n_keep = min(counts.values())
    if len(set(counts.values())) != 1:
        print(f"WARNING: unequal stream lengths {counts}; truncating to {n_keep}")
    print(f"frames per panel: {n_keep}")

    for n in PANELS:
        (args.out / n).mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(n_keep):
        i = idx
        t = args.offset + idx
        row = {"pair_index": idx, "t_real_s": round(t, 3)}
        for n in PANELS:
            dst = args.out / n / f"{n}_{idx:05d}.jpg"
            files[n][i].replace(dst)
            row[f"{n}_file"] = str(dst.relative_to(args.out))
            row[f"{n}_sharpness"] = round(float(sharp[n][i]), 2)
            row[f"{n}_brightness"] = round(float(bright[n][i]), 2)
        rows.append(row)

    cols = ["pair_index", "t_real_s",
            "human_file", "dog_file",
            "human_sharpness", "dog_sharpness",
            "human_brightness", "dog_brightness"]
    man = args.out / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    for n in PANELS:
        shutil.rmtree(args.work / f"{n}_raw", ignore_errors=True)
        (args.work / f"{n}.raw").unlink(missing_ok=True)

    print(f"\n{len(rows)} pairs -> {man}")
    for n in PANELS:
        v = np.array([r[f"{n}_sharpness"] for r in rows])
        b = np.array([r[f"{n}_brightness"] for r in rows])
        print(f"{n:6s} sharp p5 {np.percentile(v,5):8.1f} med "
              f"{np.median(v):8.1f} p95 {np.percentile(v,95):8.1f} | "
              f"bright med {np.median(b):6.2f}")


if __name__ == "__main__":
    main()
