#!/usr/bin/env python3
"""Sample 1 frame per second of REAL time from Insta360 Studio exports.

Handles sped-up exports: pass --speed N when the export was rendered N times
faster than real time, and real-world timestamps come out correct anyway.

For each real second the script decodes a few candidate frames and keeps the
sharpest one, which is how motion blur from the dog-height footage gets dropped
instead of silently polluting the segmentation input.

Usage
-----
    python3 extract_frames.py dog-01.mp4 --speed 10 --out frames/

    # several clips of one walk, timestamps continuous across them
    python3 extract_frames.py clip1.mp4 clip2.mp4 --speed 10 --continuous

Only ffmpeg and numpy are needed. Writes JPEGs plus a manifest.csv holding the
real-time stamp of every kept frame, so the video can be deleted afterwards.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROBE_W, PROBE_H = 320, 180  # size of the grayscale copy used for sharpness


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc


def probe(path):
    out = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": float(num) / float(den),
        "duration": float(info["format"]["duration"]),
    }


def laplacian_variance(frames):
    """Variance of the 4-neighbour Laplacian — the usual blur score."""
    f = frames.astype(np.float32)
    lap = (
        -4.0 * f[:, 1:-1, 1:-1]
        + f[:, :-2, 1:-1]
        + f[:, 2:, 1:-1]
        + f[:, 1:-1, :-2]
        + f[:, 1:-1, 2:]
    )
    return lap.reshape(len(f), -1).var(axis=1)


def extract(path, out_dir, speed, candidates, quality, max_width):
    """Decode once, writing full-size candidates and a small gray copy."""
    info = probe(path)
    cand_fps = candidates * speed
    if cand_fps > info["fps"] + 1e-6:
        cand_fps = info["fps"]
        candidates = max(1, int(round(cand_fps / speed)))
        print(
            f"  note: {info['fps']:.2f} fps source caps candidates at "
            f"{candidates}/real second"
        )

    cand_dir = out_dir / "_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    probe_raw = out_dir / "_sharpness.raw"

    scale = f",scale={max_width}:-2" if max_width else ""
    run([
        "ffmpeg", "-v", "error", "-y", "-i", str(path),
        "-filter_complex",
        f"[0:v]fps={cand_fps}[c];[c]split=2[full][small];"
        f"[full]{('null' if not scale else 'scale=%d:-2' % max_width)}[big];"
        f"[small]scale={PROBE_W}:{PROBE_H},format=gray[s]",
        "-map", "[big]", "-q:v", str(quality), str(cand_dir / "c_%06d.jpg"),
        "-map", "[s]", "-f", "rawvideo", "-pix_fmt", "gray", str(probe_raw),
    ])

    gray = np.fromfile(probe_raw, dtype=np.uint8)
    n_probe = gray.size // (PROBE_W * PROBE_H)
    gray = gray[: n_probe * PROBE_W * PROBE_H].reshape(n_probe, PROBE_H, PROBE_W)
    sharp = laplacian_variance(gray)
    probe_raw.unlink()

    files = sorted(cand_dir.glob("c_*.jpg"))
    n = min(len(files), len(sharp))
    return info, files[:n], sharp[:n], candidates


def pick_sharpest(files, sharp, candidates, t0):
    """One winner per real second; t_real = index / candidates + t0."""
    buckets = {}
    for i, (f, s) in enumerate(zip(files, sharp)):
        t_real = i / candidates
        key = int(t_real)
        if key not in buckets or s > buckets[key][1]:
            buckets[key] = (f, s, t_real + t0)
    return [buckets[k] for k in sorted(buckets)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("frames"))
    ap.add_argument("--speed", type=float, default=1.0,
                    help="export speed factor: 10 means 10x faster than real time")
    ap.add_argument("--candidates", type=int, default=3,
                    help="frames decoded per real second; the sharpest is kept")
    ap.add_argument("--quality", type=int, default=2,
                    help="JPEG quality for ffmpeg -q:v, 2 is near-lossless")
    ap.add_argument("--max-width", type=int, default=0,
                    help="downscale kept frames to this width, 0 keeps native")
    ap.add_argument("--continuous", action="store_true",
                    help="timestamps run on across clips instead of restarting at 0")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg and ffprobe must be on PATH")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = 0.0

    for video in args.videos:
        if not video.exists():
            sys.exit(f"no such file: {video}")
        print(f"{video.name}")
        stem = video.stem
        clip_dir = args.out / stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        info, files, sharp, candidates = extract(
            video, clip_dir, args.speed, args.candidates,
            args.quality, args.max_width,
        )
        kept = pick_sharpest(files, sharp, candidates, t0)

        for src, score, t_real in kept:
            dst = clip_dir / f"{stem}_t{t_real:08.2f}.jpg"
            src.replace(dst)
            rows.append({
                "file": str(dst.relative_to(args.out)),
                "source": video.name,
                "t_real_s": round(t_real, 2),
                "t_video_s": round((t_real - t0) / args.speed, 3),
                "sharpness": round(float(score), 1),
            })

        shutil.rmtree(clip_dir / "_candidates", ignore_errors=True)
        real_dur = info["duration"] * args.speed
        print(f"  {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
              f"{info['duration']:.1f} s video = {real_dur:.1f} s real")
        print(f"  kept {len(kept)} frames from {len(files)} candidates")

        if args.continuous:
            t0 += real_dur

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["file", "source", "t_real_s", "t_video_s", "sharpness"])
        writer.writeheader()
        writer.writerows(rows)

    scores = np.array([r["sharpness"] for r in rows]) if rows else np.array([0.0])
    size_mb = sum(f.stat().st_size for f in args.out.rglob("*.jpg")) / 1e6
    print(f"\n{len(rows)} frames, {size_mb:.0f} MB -> {manifest}")
    print(f"sharpness: min {scores.min():.0f}  median {np.median(scores):.0f}  "
          f"max {scores.max():.0f}")
    if scores.min() < np.median(scores) * 0.2:
        print("the softest frames are far below the median — look at the bottom "
              "of the manifest before trusting those seconds")


if __name__ == "__main__":
    main()
