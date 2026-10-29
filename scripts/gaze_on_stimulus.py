#!/usr/bin/env python3
"""Checklist items 18 and 21: heat map and scanpath, on the stimulus itself.

This is the first figure in the project that puts gaze where the hypotheses
are about. It needs two things that only arrived together:

  screen_homography.py  -> gaze in STIMULUS pixels          (the space)
  the viewer's marking  -> which seconds belong to which trial  (the time)

Both are required. Without the homography, gaze is in scene-camera pixels and
moves with the head; without the trial table, there is no way to say which
stimulus a sample belongs to.

ONE PARTICIPANT. Everything here is n=1, so it shows what this person looked
at, not what people look at. Item 20, between-observer dispersion, needs more
participants and is not attempted. Read these as a demonstration that the
pipeline runs end to end, and as a look at whether the manipulation did
anything at all.

READ THE FIGURES WITH THE ERROR IN MIND. The homography is estimated from the
monitor's own outline, not from AprilTags, so a fixation is placed to within
roughly the corner-estimation error -- tens of stimulus pixels, not single
ones. Draw AOIs with that slack; do not score a target a reader could not
separate from its neighbour at this precision.

Writes aoi/12_gaze_heatmaps.png and aoi/13_scanpaths.png.

Usage
-----
    python3 scripts/gaze_on_stimulus.py
    python3 scripts/gaze_on_stimulus.py --trials 1 7 13
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from aoi_plots import AOI_DIR, INK, MUTED  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "eyetracking_n1"
GAZE = ROOT / "viewer" / "data"
OUT = AOI_DIR
STIM_W, STIM_H = 1920, 1080


def load_trials(path):
    if not path.exists():
        raise SystemExit(
            f"no trial table at {path}\n"
            "Mark the boundaries first: python3 viewer/serve.py, open the viewer,\n"
            "turn on 「标试次」, press M at each switch, then 保存 CSV.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_gaze(path):
    if not path.exists():
        raise SystemExit(
            f"no mapped gaze at {path}\n"
            "Run: python3 scripts/screen_homography.py <session>")
    t, x, y = [], [], []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["on_screen"] != "1" or not r["stim_x"]:
                continue
            t.append(float(r["t_s"]))
            x.append(float(r["stim_x"]))
            y.append(float(r["stim_y"]))
    return np.array(t), np.array(x), np.array(y)


def heat(xs, ys, shape, sigma=60.0):
    """Gaussian-splatted density on the stimulus grid, in stimulus pixels."""
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    gy, gx = np.mgrid[0:h, 0:w]
    #  one pass per point is fine at this count; a separable blur of a
    #  histogram would be faster but the sample counts here are small
    hist, _, _ = np.histogram2d(ys, xs, bins=[h, w],
                                range=[[0, STIM_H], [0, STIM_W]])
    k = int(sigma / (STIM_W / w) * 3)
    if k < 1:
        return hist
    ax = np.arange(-k, k + 1)
    g = np.exp(-(ax ** 2) / (2 * (sigma / (STIM_W / w)) ** 2))
    g /= g.sum()
    acc = np.apply_along_axis(lambda m: np.convolve(m, g, "same"), 1, hist)
    acc = np.apply_along_axis(lambda m: np.convolve(m, g, "same"), 0, acc)
    return acc


def panel(ax, trial, xs, ys, mode):
    img = Image.open(STIM / trial["file"]).convert("RGB")
    ax.imshow(img, extent=[0, STIM_W, STIM_H, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3a424e")

    if len(xs) < 3:
        ax.text(STIM_W / 2, STIM_H / 2, "no gaze on screen", color="w",
                ha="center", fontsize=9)
    elif mode == "heat":
        hm = heat(xs, ys, (108, 192))
        if hm.max() > 0:
            ax.imshow(hm, extent=[0, STIM_W, STIM_H, 0], cmap="inferno",
                      alpha=0.55, interpolation="bilinear")
    else:
        #  white, not blue: these stimuli are mostly sky, pavement and glass,
        #  and a blue trace disappeared into all three. The thin dark casing
        #  keeps it readable over the bright sky and pavement as well.
        ax.plot(xs, ys, color="#1a1a1a", lw=1.4, alpha=0.45)
        ax.plot(xs, ys, color="#ffffff", lw=0.6, alpha=0.95)
        ax.scatter(xs[0], ys[0], s=30, color="#6cc08a", ec="#102015", lw=0.8,
                   zorder=3)
        ax.scatter(xs[-1], ys[-1], s=30, color="#e06c6c", ec="#2a1010", lw=0.8,
                   zorder=3)
    ax.set_xlim(0, STIM_W)
    ax.set_ylim(STIM_H, 0)



def figure(trials, gt, gx, gy, mode, out):
    """One row per scene, one column per condition, so the same scene lines up.

    The design shows six scenes under three treatments. Laid out in
    presentation order the three versions of a scene land in different rows and
    a reader has to hunt for them; laid out as a grid they sit side by side and
    the comparison is the thing the eye does first. The practice A-frame pair
    gets its own row at the bottom, with the column it has no version for left
    empty rather than filled with something else.
    """
    cols = ["human", "dog", "dog_dichromat"]
    head = ["human 1.6 m", "dog 0.45 m", "dog, dichromat"]
    scenes = []
    for t in trials:
        if t["slug"] not in scenes:
            scenes.append(t["slug"])
    #  the practice scene last, whatever order it was shown in
    x_slugs = [t["slug"] for t in trials if t["scene"] == "X"]
    scenes = [s for s in scenes if s not in x_slugs] + sorted(set(x_slugs))

    by = {}
    for t in trials:
        by[(t["slug"], t["condition"])] = t

    rows = len(scenes)
    fig, axes = plt.subplots(rows, len(cols), figsize=(3.6 * len(cols), 2.3 * rows),
                             dpi=150)
    axes = np.atleast_2d(axes)
    for r, slug in enumerate(scenes):
        for c, cond in enumerate(cols):
            ax = axes[r, c]
            tr = by.get((slug, cond))
            if tr is None:
                ax.axis("off")
                continue
            a, b = float(tr["onset_s"]), float(tr["offset_s"])
            m = (gt >= a) & (gt <= b)
            panel(ax, tr, gx[m], gy[m], mode)
            if r == 0:
                ax.set_xlabel("")
            ax.set_title(f"{tr.get('presented', tr['trial'])}. {len(gx[m])} samples"
                         f"  ·  {b - a:.0f} s", color=MUTED, fontsize=8, loc="left")
        #  the scene's name once, on the left, instead of on every panel
        first = next((axes[r, c] for c, cond in enumerate(cols)
                      if (slug, cond) in by), axes[r, 0])
        first.set_ylabel(slug, color=INK, fontsize=10)
        first.axis("on")
        first.set_xticks([]); first.set_yticks([])
    for c, h in enumerate(head):
        axes[0, c].annotate(h, xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", color=INK, fontsize=12)

    fig.suptitle(
        ("Where this participant looked: same scene across a row"
         if mode == "heat" else
         "Scanpath, same scene across a row; green start, red end")
        + "\nn = 1; gaze placed by a homography from the monitor's outline, "
          "so treat the position as good to tens of pixels, not single ones",
        color=INK, fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None,
                    help="overrides the session named in the trial table")
    ap.add_argument("--participant", default="P01")
    ap.add_argument("--trials", nargs="*",
                    help="only these trial numbers")
    args = ap.parse_args()

    trials = load_trials(STIM / f"{args.participant}_trials.csv")
    #  the table says which recording it was marked against; trusting a default
    #  here is how times get read against the wrong video
    named = {t.get("session") for t in trials if t.get("session")}
    if len(named) > 1:
        raise SystemExit(f"the trial table mixes sessions: {sorted(named)}")
    session = args.session or (named.pop() if named else None)
    if session is None:
        raise SystemExit(
            "the trial table has no session column and none was given.\n"
            "Re-save it from the viewer, or pass --session <id>.")
    args.session = session
    print(f"session {session}")
    if args.trials:
        want = {str(t) for t in args.trials}
        trials = [t for t in trials
                  if t["trial"] in want or t.get("presented") in want]
    gt, gx, gy = load_gaze(GAZE / f"{args.session}.gaze_px.csv")
    print(f"{len(trials)} trials, {len(gt)} gaze samples on the stimulus")

    p1 = figure(trials, gt, gx, gy, "heat", OUT / "12_gaze_heatmaps.png")
    p2 = figure(trials, gt, gx, gy, "path", OUT / "13_scanpaths.png")
    print(f"-> {p1}\n-> {p2}")

    print(f"\n{'trial':>12} {'slug':<18}{'condition':<16}{'samples':>8}{'on-screen':>11}")
    for tr in trials:
        a, b = float(tr["onset_s"]), float(tr["offset_s"])
        m = (gt >= a) & (gt <= b)
        dur = b - a
        print(f"{tr['trial']:>12} {tr['slug']:<18}{tr['condition']:<16}"
              f"{m.sum():>8}{m.sum() / max(dur * 200, 1):>10.0%}")


if __name__ == "__main__":
    main()
