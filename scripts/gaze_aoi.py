#!/usr/bin/env python3
"""What the gaze landed ON: dwell by semantic class, and the bias behind it.

The heat maps say where this person looked. They do not say what he looked
AT, and they cannot be compared across a row by eye, because the dog images
contain more ground than the human ones to begin with. More gaze on ground in
a dog panel is therefore not yet a finding.

So two numbers per class, per trial:

  DWELL SHARE   fraction of this trial's on-screen samples that fell on the
                class. What he looked at.
  BIAS          dwell share minus the class's AREA share of that same image.
                Zero means he spread his gaze over the class exactly as often
                as its size would give by chance; positive means the class
                pulled gaze beyond its size. This is the number that survives
                the fact that the two heights frame different amounts of
                ground, and it is the one to compare across a row.

The stimulus images are segmented with the same ADE20K model and the same AOI
groups as the walking footage (aoi_frames.py), so the classes here mean the
same thing they do there.

ONE PARTICIPANT, six scenes. A scene contributes one paired difference; with
six of them a mean difference is a description of these six, not an estimate
for a population. No p-values are printed for that reason -- the per-scene
column is the evidence, and it is shown.

Writes aoi/gaze_aoi_shares.csv, aoi/14_gaze_aoi_bias.png and
aoi/15_gaze_aoi_stack.png (checklist item 19).

Usage
-----
    python3 scripts/gaze_aoi.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi_frames import NAMES, DEFAULT_MODEL, Segmenter  # noqa: E402
from aoi_plots import AOI_DIR, INK, MUTED  # noqa: E402
from gaze_on_stimulus import load_trials, load_gaze, STIM, GAZE, STIM_W, STIM_H  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = AOI_DIR
CONDS = ["human", "dog", "dog_dichromat"]
#  `animal` cannot appear in a stimulus (these are photographs of streets, not
#  of the dog), and `other` is the model's leftovers; both are dropped from the
#  figures but kept in the CSV so the shares still sum to 1.
SHOWN = [n for n in NAMES if n not in ("animal", "other")]


def segment_stimuli(trials, model_name):
    """AOI group index per pixel for each distinct stimulus image."""
    seg = Segmenter(model_name)
    maps = {}
    for tr in trials:
        f = tr["file"]
        if f in maps:
            continue
        img = Image.open(STIM / f).convert("RGB")
        lab = seg.labels([np.asarray(img)])[0]          # model's own grid
        #  nearest-neighbour back up to stimulus pixels: these are class
        #  indices, so anything that interpolates would invent classes
        maps[f] = np.asarray(Image.fromarray(lab.astype(np.uint8))
                             .resize((STIM_W, STIM_H), Image.NEAREST))
        print(f"  segmented {f}")
    return maps


def per_trial(trials, gt, gx, gy, maps):
    rows = []
    for tr in trials:
        a, b = float(tr["onset_s"]), float(tr["offset_s"])
        m = (gt >= a) & (gt <= b)
        lab = maps[tr["file"]]
        area = np.bincount(lab.ravel(), minlength=len(NAMES)) / lab.size
        xs = np.clip(gx[m].astype(int), 0, STIM_W - 1)
        ys = np.clip(gy[m].astype(int), 0, STIM_H - 1)
        hit = lab[ys, xs]
        dwell = (np.bincount(hit, minlength=len(NAMES)) / max(len(hit), 1)
                 if len(hit) else np.zeros(len(NAMES)))
        rows.append({"trial": tr, "n": len(hit), "dur": b - a,
                     "dwell": dwell, "area": area, "t": gt[m] - a, "hit": hit})
    return rows


def write_csv(rows, path):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["presented", "scene", "slug", "condition", "samples",
                    "duration_s"]
                   + [f"dwell_{n}" for n in NAMES]
                   + [f"area_{n}" for n in NAMES])
        for r in rows:
            tr = r["trial"]
            w.writerow([tr.get("presented", tr["trial"]), tr["scene"],
                        tr["slug"], tr["condition"], r["n"],
                        f"{r['dur']:.2f}"]
                       + [f"{v:.5f}" for v in r["dwell"]]
                       + [f"{v:.5f}" for v in r["area"]])
    return path


def paired(rows, lo, hi):
    """Per-scene (hi minus lo) dwell and bias, for scenes that have both."""
    by = {(r["trial"]["slug"], r["trial"]["condition"]): r for r in rows}
    slugs = [s for s, c in by if c == lo and (s, hi) in by]
    out = []
    for s in sorted(slugs):
        p, q = by[(s, lo)], by[(s, hi)]
        out.append({"slug": s,
                    "dwell": q["dwell"] - p["dwell"],
                    "bias": (q["dwell"] - q["area"]) - (p["dwell"] - p["area"]),
                    "n": min(p["n"], q["n"])})
    return out


def figure_bias(rows, diffs, out):
    idx = [NAMES.index(n) for n in SHOWN]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=150,
                             gridspec_kw={"width_ratios": [1.15, 1]})

    #  left: where gaze went, and where it went beyond the area it was given
    ax = axes[0]
    y = np.arange(len(SHOWN))
    for k, (cond, col) in enumerate(zip(CONDS, ["#6ba3dd", "#e08a4e", "#9b7fd4"])):
        rs = [r for r in rows if r["trial"]["condition"] == cond]
        if not rs:
            continue
        bias = np.mean([r["dwell"] - r["area"] for r in rs], axis=0)[idx]
        ax.barh(y + (1 - k) * 0.26, bias * 100, height=0.25, color=col,
                label=f"{cond}  (n={len(rs)} trials)")
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(SHOWN, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("dwell share minus area share, percentage points", color=MUTED)
    ax.set_title("What pulled gaze beyond its size on screen", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    #  right: the paired dog-minus-human difference, one dot per scene
    ax = axes[1]
    for i, n in enumerate(SHOWN):
        j = NAMES.index(n)
        vals = [d["bias"][j] * 100 for d in diffs]
        ax.scatter(vals, [i] * len(vals), s=26, color="#6ba3dd", alpha=0.75,
                   zorder=3)
        ax.scatter([np.mean(vals)], [i], s=90, marker="|", color="#e08a4e",
                   lw=2.4, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(range(len(SHOWN)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("dog minus human, percentage points of bias", color=MUTED)
    ax.set_title(f"Paired by scene (n = {len(diffs)}); orange = mean",
                 color=INK, fontsize=11, loc="left")

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Gaze by semantic class, corrected for how much of the image "
                 "each class covers\nn = 1 participant; each scene contributes "
                 "one paired difference",
                 color=INK, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_stack(rows, out):
    """Checklist item 19: dwell by class over trial time, stacked."""
    idx = [NAMES.index(n) for n in SHOWN]
    cols = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:len(SHOWN)]
    scenes = []
    for r in rows:
        if r["trial"]["slug"] not in scenes:
            scenes.append(r["trial"]["slug"])
    #  the practice scene last, as in the heat-map grid
    x_slugs = {r["trial"]["slug"] for r in rows if r["trial"]["scene"] == "X"}
    scenes = [s for s in scenes if s not in x_slugs] + sorted(x_slugs)
    fig, axes = plt.subplots(len(scenes), len(CONDS),
                             figsize=(3.5 * len(CONDS), 1.7 * len(scenes)),
                             dpi=150, squeeze=False)
    by = {(r["trial"]["slug"], r["trial"]["condition"]): r for r in rows}
    for i, slug in enumerate(scenes):
        for j, cond in enumerate(CONDS):
            ax = axes[i][j]
            r = by.get((slug, cond))
            if r is None or r["n"] < 20:
                ax.axis("off")
                continue
            #  a 1-second sliding window over the trial, so the area is dwell
            #  per second rather than a count that grows with the trial
            edges = np.arange(0, r["dur"] + 1e-6, 1.0)
            share = np.zeros((len(SHOWN), max(len(edges) - 1, 1)))
            for b in range(len(edges) - 1):
                m = (r["t"] >= edges[b]) & (r["t"] < edges[b + 1])
                if m.sum() < 5:
                    continue
                c = np.bincount(r["hit"][m], minlength=len(NAMES)) / m.sum()
                share[:, b] = c[idx]
            mid = (edges[:-1] + edges[1:]) / 2
            ax.stackplot(mid, share, colors=cols, labels=SHOWN)
            ax.set_xlim(0, r["dur"])
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.set_title(f"{slug} · {cond}", color=MUTED, fontsize=7.5,
                         loc="left")
    #  on the figure, not on axes[0][0]: that panel is switched off whenever
    #  its scene has no human version, and a legend on a hidden axis vanishes
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols],
               labels=SHOWN, frameon=False, fontsize=8, ncol=len(SHOWN),
               loc="upper center", bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Item 19: dwell by class over each trial, 1-second windows\n"
                 "x is seconds into the trial; gaps are seconds with too few "
                 "on-screen samples", color=INK, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", default="P01")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-samples", type=int, default=100,
                    help="trials with fewer on-screen samples are left out")
    args = ap.parse_args()

    trials = load_trials(STIM / f"{args.participant}_trials.csv")
    named = {t.get("session") for t in trials if t.get("session")}
    if len(named) != 1:
        raise SystemExit(f"the trial table must name one session, got {named}")
    session = named.pop()
    gt, gx, gy = load_gaze(GAZE / f"{session}.gaze_px.csv")
    print(f"session {session}: {len(trials)} trials, {len(gt)} mapped samples")

    maps = segment_stimuli(trials, args.model)
    rows = per_trial(trials, gt, gx, gy, maps)
    thin = [r for r in rows if r["n"] < args.min_samples]
    for r in thin:
        print(f"  dropping {r['trial']['slug']} {r['trial']['condition']}: "
              f"only {r['n']} samples on screen")
    rows = [r for r in rows if r["n"] >= args.min_samples]

    p0 = write_csv(rows, OUT / "gaze_aoi_shares.csv")
    diffs = paired(rows, "human", "dog")
    p1 = figure_bias(rows, diffs, OUT / "14_gaze_aoi_bias.png")
    p2 = figure_stack(rows, OUT / "15_gaze_aoi_stack.png")
    print(f"-> {p0}\n-> {p1}\n-> {p2}")

    print("\ndog minus human, paired by scene, percentage points of BIAS")
    print(f"{'class':<12}" + "".join(f"{d['slug'][:9]:>11}" for d in diffs)
          + f"{'mean':>9}")
    for n in SHOWN:
        j = NAMES.index(n)
        vals = [d["bias"][j] * 100 for d in diffs]
        print(f"{n:<12}" + "".join(f"{v:>11.1f}" for v in vals)
              + f"{np.mean(vals):>9.1f}")

    print("\nraw dwell share, mean over trials, %")
    print(f"{'class':<12}" + "".join(f"{c:>16}" for c in CONDS))
    for n in SHOWN:
        j = NAMES.index(n)
        line = f"{n:<12}"
        for c in CONDS:
            rs = [r for r in rows if r["trial"]["condition"] == c]
            line += (f"{np.mean([r['dwell'][j] for r in rs]) * 100:>16.1f}"
                     if rs else f"{'-':>16}")
        print(line)


if __name__ == "__main__":
    main()
