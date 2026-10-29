#!/usr/bin/env python3
"""Find the monitor in each scene frame and map gaze into stimulus pixels.

This is the step the whole screen-based arm has been missing. The Neon gives
gaze in SCENE-CAMERA pixels, which move with the head; the hypotheses are about
where on the STIMULUS someone looked, which does not move at all. Nothing can
be said about an AOI until gaze is expressed in the stimulus's own coordinates.

Pupil's own answer is surface tracking with AprilTags on the screen corners.
This recording has none, so the corners are found from the picture instead: in
a dark room the monitor is the one large bright region, and its outline is a
quadrilateral under perspective. Four corners and the stimulus's own size give
a homography, and the homography carries gaze across.

HOW THE CORNERS ARE FOUND, and where each step can fail:

1. Threshold for bright pixels, then keep the largest connected component.
   Fails if something else in frame is as bright and as large (a window, a
   second screen). The laptop in session 04-06-27 is exactly that case.
2. Convex hull of that component, then the maximum-area quadrilateral whose
   corners are hull vertices. The hull is used rather than the raw outline
   because the dog's fur and the wearer's hands bite chunks out of the edge.
3. Order the corners, solve the 8x8 DLT system for H.
4. Median-filter each corner over a short window. A single frame's corners
   jitter by a few pixels; the monitor does not move, so most of that jitter is
   estimation noise and a median over a second removes it without smearing real
   head motion.

WHAT COMES OUT IS AN ESTIMATE, not surface tracking. Two numbers say how much
to trust it: the share of frames where a quad was found at all, and the
rectified aspect ratio, which has to land near the stimulus's own 16:9 if the
geometry is right. Both are printed, and the per-frame values are written out
so a bad stretch can be excluded rather than silently averaged in.

Outputs
-------
    viewer/data/<session>.surface.json   per-frame corners + quality
    viewer/data/<session>.gaze_px.csv    gaze in stimulus pixels, per sample

Usage
-----
    python3 scripts/screen_homography.py 2026-10-27-04-12-39
    python3 scripts/screen_homography.py 2026-10-27-04-12-39 --montage 8
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "pupil_eye_tracking"
OUT = ROOT / "viewer" / "data"

SCENE_W, SCENE_H = 1600, 1200      # the Neon scene camera
DET_W, DET_H = 400, 300            # detection resolution; corners scale back up
STIM_W, STIM_H = 1920, 1080        # the stimulus images
BRIGHT = 170                       # a monitor in a dark room clears this easily
MIN_AREA = 0.04                    # of the frame, below which it is not the screen
#  a 16:9 panel seen off-axis still lands inside this band; a screen merged
#  with the wall it lights does not
ASPECT_LO, ASPECT_HI = 1.25, 2.35
RECT_MIN = 0.55                    # opposite sides within ~half of each other
#  The stimulus monitor is the thing the wearer is facing, so it sits near the
#  middle of the scene camera. The laptop that shows the same slides sits low
#  and to one side, and the wall the monitor lights up spreads to an edge.
#  Centrality is what tells those apart when brightness and shape cannot.
CENTRE_WEIGHT = 2.2


# ---------------------------------------------------------------- components

def otsu(g):
    """Threshold between the dark room and the lit screen, per frame.

    A fixed threshold fails across sessions: when the monitor is closer or
    brighter it also lights the wall above it, the two merge into one blob, and
    the "screen" that comes out is screen-plus-wall. Otsu follows the frame's
    own two populations instead.
    """
    hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256).astype(float)
    total = hist.sum()
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-9
    between = (mu_t * omega - mu) ** 2 / denom
    return float(np.argmax(between))


def label(mask):
    """4-connected labelling. -> (labels int array, sizes per label)."""
    h, w = mask.shape
    parent = np.full(h * w, -1, dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    idx = np.flatnonzero(mask.ravel())
    parent[idx] = idx
    m = mask
    #  link right and down neighbours
    ys, xs = np.nonzero(m[:, :-1] & m[:, 1:])
    for y, x in zip(ys, xs):
        union(y * w + x, y * w + x + 1)
    ys, xs = np.nonzero(m[:-1, :] & m[1:, :])
    for y, x in zip(ys, xs):
        union(y * w + x, (y + 1) * w + x)

    roots = np.array([find(i) for i in idx])
    if not len(roots):
        return None, None
    vals, counts = np.unique(roots, return_counts=True)
    lab = np.zeros(h * w, dtype=np.int32)
    order = np.argsort(-counts)
    for rank, oi in enumerate(order, start=1):
        lab[idx[roots == vals[oi]]] = rank
    return lab.reshape(h, w), counts[order]


def largest_component(mask):
    """Kept for the diagnostics: the biggest region only."""
    lab, sizes = label(mask)
    return None if lab is None else (lab == 1)


# ---------------------------------------------------------------- geometry

def convex_hull(pts):
    """Andrew's monotone chain. pts: (n,2) float. Returns hull in order."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(p) < 3:
        return p

    def half(points):
        out = []
        for q in points:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def max_area_quad(hull):
    """The 4 hull vertices enclosing the most area. O(n^2) with a small hull."""
    n = len(hull)
    if n < 4:
        return None
    if n > 40:                      # thin the hull; corners survive, noise does not
        hull = hull[np.linspace(0, n - 1, 40).astype(int)]
        n = len(hull)

    def tri(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2

    best, quad = 0.0, None
    for i in range(n):
        for k in range(i + 2, n):
            #  for the diagonal i-k, the best j and l are found independently
            j = max(range(i + 1, k), key=lambda t: tri(hull[i], hull[t], hull[k]),
                    default=None)
            lo = list(range(k + 1, n)) + list(range(0, i))
            l = max(lo, key=lambda t: tri(hull[i], hull[t], hull[k]), default=None)
            if j is None or l is None:
                continue
            a = tri(hull[i], hull[j], hull[k]) + tri(hull[i], hull[l], hull[k])
            if a > best:
                best, quad = a, np.array([hull[i], hull[j], hull[k], hull[l]])
    return quad


def order_corners(q):
    """-> [top-left, top-right, bottom-right, bottom-left]."""
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    start = np.argmin(q[:, 0] + q[:, 1])      # top-left has the smallest x+y
    return np.roll(q, -start, axis=0)


def homography(src, dst):
    """DLT for 4 correspondences. src, dst: (4,2). -> 3x3."""
    #  u = (h11 x + h12 y + h13) / (h31 x + h32 y + h33), rearranged to
    #  h11 x + h12 y + h13 - u(h31 x + h32 y + h33) = 0, and the same for v.
    #  The final term carries the MINUS; dropping it silently returns a
    #  homography that maps the corners to mirrored, oversized coordinates.
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    a = np.array(a, dtype=float)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h, pts):
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------- per frame

def quad_shape(q):
    """(aspect, rectangularity) of an ordered quad."""
    top = np.linalg.norm(q[1] - q[0])
    bot = np.linalg.norm(q[2] - q[3])
    lef = np.linalg.norm(q[3] - q[0])
    rig = np.linalg.norm(q[2] - q[1])
    w = (top + bot) / 2
    h = (lef + rig) / 2
    if h <= 1 or w <= 1:
        return 0.0, 0.0
    #  a rectangle has equal opposite sides; a blob that merged with the wall
    #  does not
    rect = min(top, bot) / max(top, bot) * min(lef, rig) / max(lef, rig)
    return w / h, rect


def detect(gray):
    """-> corners in DETECTION pixels, or None.

    The biggest bright region is not reliably the screen: a second monitor, a
    laptop, or the wall the screen itself lights up can all be bigger or merge
    with it. So several candidates are scored on the two things a monitor
    showing a 16:9 stimulus must satisfy -- the quad has to be rectangular, and
    its aspect has to be near 16:9 -- and the best one wins.
    """
    thr = max(BRIGHT * 0.6, otsu(gray))
    mask = gray >= thr
    if mask.mean() < MIN_AREA:
        return None, False
    lab, sizes = label(mask)
    if lab is None:
        return None, False

    #  Shape is a FILTER, not a score. Scoring on shape picks the laptop in the
    #  corner, which is a cleaner rectangle than the monitor and shows the same
    #  slides. Among the candidates that could be a 16:9 screen at all, the
    #  stimulus monitor is simply the biggest one in frame.
    h, w = gray.shape
    target = STIM_W / STIM_H
    cands = []
    for rank in range(1, min(6, len(sizes)) + 1):
        comp = lab == rank
        frac = comp.mean()
        if frac < MIN_AREA * 0.5:
            break
        ys, xs = np.nonzero(comp)
        quad = max_area_quad(convex_hull(np.c_[xs, ys].astype(float)))
        if quad is None:
            continue
        q = order_corners(quad)
        aspect, rect = quad_shape(q)
        if aspect <= 0:
            continue
        #  1 at the frame centre, 0 at a corner
        cx, cy = xs.mean() / w, ys.mean() / h
        centre = 1.0 - min(1.0, np.hypot((cx - 0.5) * 2, (cy - 0.5) * 2))
        cands.append((frac, aspect, rect, centre, q))
    if not cands:
        return None, False

    #  Score every candidate rather than filtering first: on a dark stimulus
    #  the monitor's bright area breaks up and fails a shape test, and a hard
    #  filter would then leave only the laptop to choose from.
    def score(c):
        frac, aspect, rect, centre, _ = c
        return (np.log(max(frac, 1e-4))
                + CENTRE_WEIGHT * centre
                + 1.0 * rect
                - 2.0 * abs(np.log(aspect / target)))

    best = max(cands, key=score)
    frac, aspect, rect, centre, q = best
    #  Trust it only if it looks like a screen AND sits where the screen is.
    #  An untrusted frame is not discarded: the caller fills it from the
    #  trusted frames around it, because the monitor barely moves.
    trusted = (ASPECT_LO <= aspect <= ASPECT_HI and rect >= RECT_MIN
               and centre >= 0.5 and frac >= MIN_AREA)
    return q, bool(trusted)


def frames(video, hz):
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"fps={hz},scale={DET_W}:{DET_H}", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = DET_W * DET_H
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(DET_H, DET_W).astype(np.float32)
    p.stdout.close()
    p.wait()


def fill_from_trusted(c, trust, win):
    """Every frame's corners = the median of the TRUSTED frames around it.

    The monitor does not move; the head does, slowly. So a frame whose own
    detection was rejected -- a dark stimulus, a hand across the screen -- is
    better served by its neighbours' corners than by its own, and a frame whose
    detection was accepted still benefits from the median, which removes the
    few-pixel jitter of the corner estimate.
    """
    out = np.full_like(c, np.nan)
    n = len(c)
    half = max(1, win // 2)
    idx = np.flatnonzero(trust)
    if not len(idx):
        return out
    for i in range(n):
        near = idx[np.abs(idx - i) <= half]
        if len(near) < 2:                      # widen until something is near
            k = np.argsort(np.abs(idx - i))[:5]
            near = idx[k]
        out[i] = np.median(c[near], axis=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--fill", type=float, default=6.0,
                    help="seconds of trusted frames to take the corners from")
    ap.add_argument("--montage", type=int, default=0,
                    help="save N rectified screens to check the geometry by eye")
    args = ap.parse_args()

    sess = SESSIONS / args.session
    video = sess / "Neon Scene Camera v1 ps1.mp4"
    if not video.exists():
        raise SystemExit(f"no scene video at {video}")

    corners, trust = [], []
    for gray in frames(video, args.hz):
        q, ok = detect(gray)
        corners.append(np.full((4, 2), np.nan) if q is None else q.astype(float))
        trust.append(ok)
    corners = np.array(corners)
    trust = np.array(trust)
    n = len(corners)
    print(f"{n} frames at {args.hz} Hz, screen found in "
          f"{np.isfinite(corners[:, 0, 0]).sum()}, trusted in "
          f"{trust.sum()} ({trust.mean():.0%})")
    if not trust.any():
        raise SystemExit("no frame looked like a screen; is this the right session?")

    corners = fill_from_trusted(corners, trust,
                                max(1, int(round(args.fill * args.hz))))

    #  scale detection pixels up to the scene camera's own pixels, which is
    #  where the gaze lives
    sx, sy = SCENE_W / DET_W, SCENE_H / DET_H
    scene_corners = corners * np.array([sx, sy])

    dst = np.array([[0, 0], [STIM_W, 0], [STIM_W, STIM_H], [0, STIM_H]], dtype=float)
    hs, aspects = [], []
    for q in scene_corners:
        if np.isnan(q).any():
            hs.append(None)
            continue
        h = homography(q, dst)
        hs.append(h)
        top = np.linalg.norm(q[1] - q[0])
        bot = np.linalg.norm(q[2] - q[3])
        left = np.linalg.norm(q[3] - q[0])
        right = np.linalg.norm(q[2] - q[1])
        aspects.append(((top + bot) / 2) / max(1e-6, (left + right) / 2))
    aspects = np.array(aspects)
    print(f"rectified aspect ratio: median {np.median(aspects):.2f} "
          f"(the stimulus is {STIM_W/STIM_H:.2f}), 10-90 pct "
          f"{np.percentile(aspects, [10, 90]).round(2)}")

    # ---- gaze -> stimulus pixels
    dt = np.dtype([("x", "<f4"), ("y", "<f4")])
    gaze = np.fromfile(sess / "gaze ps1.raw", dtype=dt)
    gt = np.fromfile(sess / "gaze ps1.time", dtype="<i8")
    scene_t = np.fromfile(sess / "Neon Scene Camera v1 ps1.time", dtype="<i8")
    t0 = int(scene_t[0])
    secs = (gt - t0) / 1e9
    frame_of = np.clip((secs * args.hz).round().astype(int), 0, n - 1)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"{args.session}.gaze_px.csv"
    inside = 0
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "scene_x", "scene_y", "stim_x", "stim_y", "on_screen"])
        for s, gx, gy, fi in zip(secs, gaze["x"], gaze["y"], frame_of):
            h = hs[fi]
            if h is None:
                w.writerow([round(float(s), 3), round(float(gx), 1),
                            round(float(gy), 1), "", "", 0])
                continue
            u, v = apply_h(h, np.array([[gx, gy]]))[0]
            on = 0 <= u <= STIM_W and 0 <= v <= STIM_H
            inside += on
            w.writerow([round(float(s), 3), round(float(gx), 1), round(float(gy), 1),
                        round(float(u), 1), round(float(v), 1), int(on)])
    print(f"{len(gaze)} gaze samples, {inside} ({inside/len(gaze):.0%}) land on the "
          f"stimulus")

    surf = OUT / f"{args.session}.surface.json"
    surf.write_text(json.dumps({
        "hz": args.hz, "scene": [SCENE_W, SCENE_H], "stimulus": [STIM_W, STIM_H],
        "trusted": float(trust.mean()),
        "aspect_median": round(float(np.median(aspects)), 3),
        "corners": [None if np.isnan(q).any() else
                    [[round(float(a), 1), round(float(b), 1)] for a, b in q]
                    for q in scene_corners],
    }, separators=(",", ":")))
    print(f"-> {csv_path}\n-> {surf}")

    if args.montage:
        montage(video, args, hs, n)


def montage(video, args, hs, n):
    """Rectify N frames spread over the recording, so the geometry can be seen."""
    from PIL import Image
    picks = [i for i in np.linspace(0, n - 1, args.montage * 3).astype(int)
             if hs[i] is not None][:args.montage]
    tiles, tw, th = [], 320, 180
    for i in picks:
        t = i / args.hz
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
             "-vf", f"scale={SCENE_W}:{SCENE_H}", "-pix_fmt", "gray",
             "-f", "rawvideo", "-"], capture_output=True).stdout
        if len(raw) < SCENE_W * SCENE_H:
            continue
        img = np.frombuffer(raw, dtype=np.uint8).reshape(SCENE_H, SCENE_W)
        hinv = np.linalg.inv(hs[i])
        uu, vv = np.meshgrid(np.linspace(0, STIM_W, tw), np.linspace(0, STIM_H, th))
        pts = apply_h(hinv, np.c_[uu.ravel(), vv.ravel()])
        xs = np.clip(pts[:, 0].round().astype(int), 0, SCENE_W - 1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, SCENE_H - 1)
        tile = img[ys, xs].reshape(th, tw)
        tiles.append((t, Image.fromarray(tile)))
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("L", (tw * cols, th * rows), 0)
    for k, (t, im) in enumerate(tiles):
        sheet.paste(im, (tw * (k % cols), th * (k // cols)))
    out = OUT.parent / f"{args.session}_rectified.png"
    sheet.save(out)
    print(f"-> {out}   times: {[round(t) for t, _ in tiles]}")


if __name__ == "__main__":
    main()
