#!/usr/bin/env python3
"""Per scene: how the gaze split between semantic classes in each condition.

14_gaze_aoi_bias.png averages over scenes, which is where the claim lives but
also where a reader loses the thing they can check. This takes the same
per-trial numbers apart again, one row per scene, so the three conditions of
ONE scene sit next to each other and the reader can see which scenes carry the
average and which fight it.

Two outputs, from one pass over the data:

  aoi/21_gaze_by_scene.png   the condensed figure, one row per scene
  aoi/scene_pick/            one heat-map panel per trial, plus scene.json,
                             the material for the interactive picker

Dwell share is used here, not bias. Within one row the image is the same
photograph in all three conditions, so area share is identical across the row
and subtracting it would shift all three bars by the same amount. Bias only
earns its keep when comparing ACROSS scenes, which is what figure 14 does.

Writes aoi/21_gaze_by_scene.png and aoi/scene_pick/.

Usage
-----
    python3 scripts/gaze_by_scene.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi_frames import NAMES, DEFAULT_MODEL  # noqa: E402
from aoi_plots import AOI_DIR, INK, MUTED, style  # noqa: E402
from elevation_plots import PALETTE, STACK  # noqa: E402
from gaze_on_stimulus import (load_trials, load_gaze, heat, STIM, GAZE,
                              STIM_W, STIM_H)  # noqa: E402
from gaze_aoi import segment_stimuli, per_trial, CONDS, SHOWN  # noqa: E402

OUT = AOI_DIR
PICK = AOI_DIR / "scene_pick"
PANEL_W = 520
LABEL = {"human": "human 1.6 m", "dog": "dog 0.45 m",
         "dog_dichromat": "dog, dichromat"}


def scene_order(rows):
    seen = []
    for r in rows:
        if r["trial"]["slug"] not in seen:
            seen.append(r["trial"]["slug"])
    x = {r["trial"]["slug"] for r in rows if r["trial"]["scene"] == "X"}
    return [s for s in seen if s not in x] + sorted(x)


def figure(rows, out):
    order = [n for n in STACK if n in SHOWN] + [n for n in SHOWN if n not in STACK]
    scenes = scene_order(rows)
    by = {(r["trial"]["slug"], r["trial"]["condition"]): r for r in rows}

    fig, axes = plt.subplots(len(scenes), 1, dpi=150,
                             figsize=(9.6, 1.05 * len(scenes) + 1.6),
                             sharex=True)
    for ax, slug in zip(np.atleast_1d(axes), scenes):
        y = np.arange(len(CONDS))
        left = np.zeros(len(CONDS))
        for cls in order:
            j = NAMES.index(cls)
            vals = np.array([by[(slug, c)]["dwell"][j] if (slug, c) in by else 0.0
                             for c in CONDS])
            ax.barh(y, vals, left=left, height=0.72,
                    color=PALETTE.get(cls, "#cccccc"), edgecolor="white",
                    linewidth=0.8, label=cls if slug == scenes[0] else None)
            left += vals
        #  a condition that was not shown gets a word, not an empty bar a
        #  reader would read as "he looked at nothing"
        for i, c in enumerate(CONDS):
            if (slug, c) not in by:
                ax.text(0.01, i, "not shown", color=MUTED, fontsize=8,
                        va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in CONDS], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        style(ax)
        ax.grid(False)
        ax.set_ylabel(slug, color=INK, fontsize=9.5, rotation=0,
                      ha="right", va="center", labelpad=12)
    np.atleast_1d(axes)[-1].set_xlabel("share of this trial's on-screen gaze",
                                       color=MUTED, fontsize=9)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, fontsize=8, ncol=len(l),
                     loc="lower center", bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_color(MUTED)
    fig.suptitle("Where the gaze went in each scene, by condition\n"
                 "n = 1; same photograph down each row, so the three bars "
                 "differ only in viewpoint and colour treatment",
                 color=INK, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def panels(rows, maps):
    """One heat-map image per trial, written small enough to ship."""
    if PICK.exists():
        shutil.rmtree(PICK)
    PICK.mkdir(parents=True)
    data = {"scenes": [], "classes": SHOWN,
            "colours": {c: PALETTE.get(c, "#cccccc") for c in SHOWN}}
    by = {(r["trial"]["slug"], r["trial"]["condition"]): r for r in rows}
    for slug in scene_order(rows):
        entry = {"slug": slug, "conditions": []}
        for cond in CONDS:
            r = by.get((slug, cond))
            if r is None:
                continue
            tr = r["trial"]
            name = f"{slug}__{cond}.jpg"
            draw_panel(tr, r, PICK / name)
            j = [NAMES.index(c) for c in SHOWN]
            entry["conditions"].append({
                "condition": cond, "label": LABEL[cond], "image": name,
                "presented": tr.get("presented", tr["trial"]),
                "samples": r["n"], "seconds": round(r["dur"], 1),
                "dwell": [round(float(v), 4) for v in r["dwell"][j]],
                "area": [round(float(v), 4) for v in r["area"][j]],
            })
        data["scenes"].append(entry)
    (PICK / "scene.json").write_text(json.dumps(data, ensure_ascii=False))
    return data


def draw_panel(trial, row, path):
    img = Image.open(STIM / trial["file"]).convert("RGB")
    fig, ax = plt.subplots(figsize=(4.0, 4.0 * STIM_H / STIM_W), dpi=130)
    ax.imshow(img, extent=[0, STIM_W, STIM_H, 0])
    xs, ys = row["gx"], row["gy"]
    if len(xs) >= 3:
        hm = heat(xs, ys, (108, 192))
        if hm.max() > 0:
            ax.imshow(hm, extent=[0, STIM_W, STIM_H, 0], cmap="inferno",
                      alpha=0.55, interpolation="bilinear")
    ax.set_xlim(0, STIM_W)
    ax.set_ylim(STIM_H, 0)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, facecolor="black", pad_inches=0,
                pil_kwargs={"quality": 82})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", default="P01")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    trials = load_trials(STIM / f"{args.participant}_trials.csv")
    named = {t.get("session") for t in trials if t.get("session")}
    if len(named) != 1:
        raise SystemExit(f"the trial table must name one session, got {named}")
    session = named.pop()
    gt, gx, gy = load_gaze(GAZE / f"{session}.gaze_px.csv")
    maps = segment_stimuli(trials, args.model)
    rows = per_trial(trials, gt, gx, gy, maps)
    #  per_trial keeps the samples; the panel drawer needs them by name
    for r in rows:
        a, b = float(r["trial"]["onset_s"]), float(r["trial"]["offset_s"])
        m = (gt >= a) & (gt <= b)
        r["gx"], r["gy"] = gx[m], gy[m]

    p1 = figure(rows, OUT / "21_gaze_by_scene.png")
    data = panels(rows, maps)
    print(f"-> {p1}")
    print(f"-> {PICK} ({sum(len(s['conditions']) for s in data['scenes'])} "
          f"panels over {len(data['scenes'])} scenes)")


if __name__ == "__main__":
    main()
