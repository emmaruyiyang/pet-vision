#!/usr/bin/env python3
"""Spherical geometry for equirectangular frames: cubemap split and area weights.

Two jobs that the segmentation pipeline depends on and that are easy to get
silently wrong:

1. Segmentation models are trained on perspective images. Run them straight on
   an equirectangular frame and the poles, stretched across the full width, get
   mis-labelled wholesale. So the sphere is split into six 90-degree cube faces,
   each face segmented on its own, and the labels mapped back.

2. In an equirectangular image a pixel's solid angle is proportional to
   cos(latitude). Counting raw pixels therefore inflates the zenith and nadir
   enormously -- a green-view ratio can come out several times too large. Every
   area here is weighted by cos(latitude).

Run `python3 sphere_geometry.py --selftest` to check both against closed-form
values. No model or image library needed for that.
"""

import argparse
import sys

import numpy as np

# face name -> (right, up, forward) basis vectors
FACES = {
    "front": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "back": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "right": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "left": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "up": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "down": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
}
FACE_ORDER = list(FACES)


def equirect_dirs(width, height):
    """Unit direction for every pixel of an equirectangular image.

    Returns (height, width, 3) with +Y up and +Z forward, and the latitude of
    each row in radians.
    """
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    cos_lat = np.cos(lat)[:, None]
    dirs = np.empty((height, width, 3), dtype=np.float64)
    dirs[..., 0] = cos_lat * np.sin(lon)[None, :]
    dirs[..., 1] = np.sin(lat)[:, None]
    dirs[..., 2] = cos_lat * np.cos(lon)[None, :]
    return dirs, lat


def face_dirs(face, size):
    """Unit direction for every pixel of one cube face."""
    right, up, forward = (np.array(v, dtype=np.float64) for v in FACES[face])
    s = (np.arange(size) + 0.5) / size * 2 - 1          # left to right
    t = 1 - (np.arange(size) + 0.5) / size * 2          # top to bottom
    d = (forward[None, None, :]
         + s[None, :, None] * right[None, None, :]
         + t[:, None, None] * up[None, None, :])
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def equirect_to_face_map(width, height, size):
    """For every equirectangular pixel: which cube face, and where on it.

    Returns (face_index, face_y, face_x), each (height, width) integer arrays,
    for mapping a face's labels back onto the sphere.
    """
    dirs, _ = equirect_dirs(width, height)
    face_idx = np.full((height, width), -1, dtype=np.int8)
    fy = np.zeros((height, width), dtype=np.int32)
    fx = np.zeros((height, width), dtype=np.int32)

    for i, name in enumerate(FACE_ORDER):
        right, up, forward = (np.array(v, dtype=np.float64) for v in FACES[name])
        df = dirs @ forward
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (dirs @ right) / df
            t = (dirs @ up) / df
        on = (df > 0) & (np.abs(s) <= 1 + 1e-12) & (np.abs(t) <= 1 + 1e-12)
        on &= face_idx < 0                      # first face wins on the seams
        if not on.any():
            continue
        px = np.clip(((s[on] + 1) / 2 * size).astype(np.int32), 0, size - 1)
        py = np.clip(((1 - t[on]) / 2 * size).astype(np.int32), 0, size - 1)
        face_idx[on], fx[on], fy[on] = i, px, py

    if (face_idx < 0).any():
        raise RuntimeError("some directions matched no cube face")
    return face_idx, fy, fx


def face_to_equirect_map(width, height, size):
    """For every cube-face pixel: where to read it from the equirect image.

    Returns {face_name: (src_y, src_x)} for rendering the faces to segment.
    """
    out = {}
    for name in FACE_ORDER:
        d = face_dirs(name, size)
        lat = np.arcsin(np.clip(d[..., 1], -1, 1))
        lon = np.arctan2(d[..., 0], d[..., 2])
        x = (lon + np.pi) / (2 * np.pi) * width - 0.5
        y = (np.pi / 2 - lat) / np.pi * height - 0.5
        out[name] = (
            np.clip(np.rint(y), 0, height - 1).astype(np.int32),
            (np.rint(x).astype(np.int32) % width),
        )
    return out


def pixel_weights(width, height):
    """Solid angle of each equirect pixel, normalised to sum to 1."""
    _, lat = equirect_dirs(width, height)
    w = np.repeat(np.cos(lat)[:, None], width, axis=1)
    return w / w.sum()


def band_edges(n_bands):
    """Elevation band edges in radians, equal angular steps from -90 to +90."""
    return np.linspace(-np.pi / 2, np.pi / 2, n_bands + 1)


def row_latitudes(height):
    """Latitude of each equirect row, in radians, top row first."""
    return np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi


def band_index(height, n_bands):
    """Band each equirect row falls in, low elevation first.

    A band edge generally falls inside a row, so a band's discrete area is off
    by up to half a row -- about pi/(2*height). At 1024 rows that is under 0.1%,
    far below segmentation error; at 256 it is visible, so keep frames large.
    """
    lat = row_latitudes(height)
    return np.clip(np.digitize(lat, band_edges(n_bands)[1:-1]), 0, n_bands - 1)


def band_solid_angle(n_bands):
    """Closed-form fraction of the sphere in each elevation band."""
    e = band_edges(n_bands)
    return (np.sin(e[1:]) - np.sin(e[:-1])) / 2


def composition(labels, weights, rows, n_bands, n_classes):
    """Weighted area fraction of every class in every elevation band.

    `labels` is an equirect label image, `rows` the band index per row. The
    result sums to 1 across the whole array, so a band's row sums give that
    band's share of the sphere.
    """
    flat = (rows[:, None] * n_classes + labels).ravel()
    counts = np.bincount(flat, weights=weights.ravel(),
                         minlength=n_bands * n_classes)
    return counts.reshape(n_bands, n_classes)


def selftest():
    """Check the weights and the band split against closed-form values."""
    ok = True
    w, h, size, n_bands = 2048, 1024, 256, 12

    weights = pixel_weights(w, h)
    rows = band_index(h, n_bands)
    got = np.bincount(rows, weights=weights.sum(axis=1), minlength=n_bands)
    want = band_solid_angle(n_bands)
    err = np.abs(got - want).max()
    bound = np.pi / (2 * h)
    print(f"band solid angle      max error {err:.2e} (bound {bound:.2e})", end="  ")
    ok &= err < bound
    print("ok" if err < bound else "FAIL")

    upper = weights[rows >= n_bands // 2].sum()
    print(f"upper hemisphere      {upper:.6f} (want 0.500000)", end="  ")
    ok &= abs(upper - 0.5) < 1e-6
    print("ok" if abs(upper - 0.5) < 1e-6 else "FAIL")

    zenith = weights[rows == n_bands - 1].sum()
    naive = (rows == n_bands - 1).sum() / h
    print(f"  top 15 deg band is {zenith:.4f} of the sphere; counting raw pixels "
          f"gives {naive:.4f}, {naive / zenith:.1f}x too much")

    face_idx, fy, fx = equirect_to_face_map(w, h, size)
    share = np.bincount(face_idx.ravel(), weights=weights.ravel(), minlength=6)
    err = np.abs(share - 1 / 6).max()
    print(f"cube faces equal area max error {err:.2e}", end="  ")
    ok &= err < 2e-3
    print("ok" if err < 2e-3 else "FAIL")

    # a face rendered from equirect, then read back, must land on itself
    back = face_to_equirect_map(w, h, size)
    bad = 0
    for i, name in enumerate(FACE_ORDER):
        sy, sx = back[name]
        bad += int((face_idx[sy, sx] != i).mean() > 0.02)
    print(f"round trip            {6 - bad}/6 faces consistent", end="  ")
    ok &= bad == 0
    print("ok" if bad == 0 else "FAIL")

    # a synthetic sky/ground split must come out at exactly half each
    labels = np.broadcast_to(
        (rows[:, None] >= n_bands // 2).astype(np.int64), (h, w))
    comp = composition(labels, weights, rows, n_bands, 2)
    print(f"synthetic sky share   {comp[:, 1].sum():.6f} (want 0.500000)", end="  ")
    ok &= abs(comp[:, 1].sum() - 0.5) < 1e-6
    print("ok" if abs(comp[:, 1].sum() - 0.5) < 1e-6 else "FAIL")

    print("\nall good" if ok else "\nsomething is off")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    ap.print_help()
