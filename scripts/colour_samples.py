#!/usr/bin/env python3
"""A sample image for the colour-collapse claim: what the number looks like.

10_colour_collapse.png gives the separation in Lab units, which is the right
measure and the wrong picture -- a bar chart cannot show someone what "loses
86% of its separation" means. This figure shows it twice over:

  TOP   the eight target/background pairs the separations are computed from,
        each as a patch of target on its background, normal above and under
        the dichromat transform below. These ARE the cases behind the numbers,
        not an illustration of them.
  BOTTOM one real stimulus from the experiment, the hay-bale scene, whose
        pumpkins are warm colours on vegetation -- the same situation as the
        fire hydrant on grass, photographed rather than drawn.

WHAT THIS IS NOT. The transform is a human dichromat model (deuteranopia)
standing in for the dog, because the dog's long-wave cone at 555 nm sits near
the human L cone at 560 nm. It answers "can these two be told apart", not
"what does a dog see". Nobody knows what a dog sees.

Writes aoi/20_colour_samples.png.

Usage
-----
    python3 scripts/colour_samples.py
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from aoi_plots import AOI_DIR, INK, MUTED  # noqa: E402
from dog_vision import dichromat  # noqa: E402
from dog_vision_figure import CASES, to_lab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "eyetracking_n1" / "P01"
#  Two real pairs, one indoor and one out, because the indoor one alone makes
#  the effect look like a furniture-showroom curiosity. Each is (file, crop as
#  a fraction of the stimulus, caption).
#
#  The sofa scene has 7.4% warm pixels on 1.8% green, the most
#  red-on-vegetation-like pairing in the set. The hay-bale scene cropped to
#  the pumpkin row carries the textbook case photographed: a RED FIRE HYDRANT
#  standing in front of a green planter, left of centre, beside an orange
#  pumpkin and autumn foliage -- the first row of the swatch grid, outdoors.
REAL = [
    ("07_b2_S05_sofa-back_dog.jpg", (0.0, 0.0, 1.0, 1.0),
     "indoors\nwarm cushions and\ntimber on a green sofa"),
    ("01_b1_S07_hay-bale_human.jpg", (0.34, 0.40, 0.78, 0.82),
     "outdoors\na red hydrant on a\ngreen planter, a pumpkin"),
]


def patch(target, bg, size=72):
    """A square of background with the target as a disc on it."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :] = bg
    y, x = np.mgrid[0:size, 0:size]
    r = ((x - size / 2) ** 2 + (y - size / 2) ** 2) <= (size * 0.29) ** 2
    img[r] = target
    return img


def sep(a, b):
    return float(np.linalg.norm(to_lab(a)[0] - to_lab(b)[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="deuteranopia")
    args = ap.parse_args()

    n = len(CASES)
    fig = plt.figure(figsize=(1.55 * n, 12.4), dpi=150)
    gs = fig.add_gridspec(2 + len(REAL), n,
                          height_ratios=[1.25, 1.25] + [3.3] * len(REAL),
                          hspace=0.16, wspace=0.12)

    for i, (name, tgt, bg, bgname) in enumerate(CASES):
        t2 = dichromat(np.array(tgt, np.uint8).reshape(1, 1, 3), args.kind)[0, 0]
        b2 = dichromat(np.array(bg, np.uint8).reshape(1, 1, 3), args.kind)[0, 0]
        s0, s1 = sep(tgt, bg), sep(t2, b2)
        for row, (a, b) in enumerate(((tgt, bg), (t2, b2))):
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(patch(a, b))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
            if row == 0:
                ax.set_title(f"{name}\non {bgname}", color=INK, fontsize=7.5,
                             linespacing=1.4)
            else:
                drop = 1 - s1 / max(s0, 1e-9)
                #  two of the eight GAIN a little separation; "--5%" is not a
                #  reading, so say which direction it went
                lab = ("no change" if abs(drop) < 0.005
                       else f"−{drop:.0%}" if drop > 0
                       else f"+{-drop:.0%}")
                ax.set_xlabel(lab, color=INK, fontsize=10)
        fig.text(0, 0, "")

    #  the real pairs, same transform, same pixels
    half = n // 2
    for row, (fname, crop_box, caption) in enumerate(REAL):
        img = Image.open(STIM / fname).convert("RGB")
        w, h = img.size
        box = (int(crop_box[0] * w), int(crop_box[1] * h),
               int(crop_box[2] * w), int(crop_box[3] * h))
        crop = np.asarray(img.crop(box))
        for j, (a, lab) in enumerate(((crop, "as photographed"),
                                      (dichromat(crop, args.kind),
                                       "under the dichromat transform"))):
            ax = fig.add_subplot(gs[2 + row, j * half:(j + 1) * half])
            ax.imshow(a)
            ax.set_xticks([]); ax.set_yticks([])
            title = lab if row == 0 else ""
            if title:
                ax.set_title(title, color=INK, fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(caption, color=MUTED, fontsize=8.5,
                              rotation=0, ha="right", va="center", labelpad=12,
                              linespacing=1.6)

    fig.text(0.5, 0.995,
             "What \"loses 86% of its separation\" looks like",
             ha="center", va="top", color=INK, fontsize=14)
    #  short lines, hard wrapped: one long line forces bbox_inches="tight" to
    #  widen the whole canvas and the panels end up stranded in the middle
    fig.text(0.5, 0.973,
             "top rows: the eight target/background pairs the Lab separations "
             "are computed from,\nnormal above, dichromat below; the number is "
             "how much separation is lost.\n"
             "bottom: two stimuli from the experiment, one indoor and one out "
             "\u2014 the outdoor one carries\nthe first swatch row "
             "photographed, a red hydrant standing in front of a green "
             "planter.\n"
             "Deuteranopia stands in for the dog: it answers \"can these be "
             "told apart\", not \"what does a dog see\".",
             ha="center", va="top", color=MUTED, fontsize=8.5,
             linespacing=1.6)
    fig.subplots_adjust(top=0.865, bottom=0.02, left=0.085, right=0.985)
    out = AOI_DIR / "20_colour_samples.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    #  the outdoor pair on its own as well. The combined figure above already
    #  carries it, but that figure is anchored by a comment thread in the
    #  write-up and replacing it would detach the thread, so the outdoor case
    #  also ships as a plate of its own that can be placed beside it.
    fname, crop_box, _ = REAL[1]
    img = Image.open(STIM / fname).convert("RGB")
    w, h = img.size
    crop = np.asarray(img.crop((int(crop_box[0] * w), int(crop_box[1] * h),
                                int(crop_box[2] * w), int(crop_box[3] * h))))
    f2, ax2 = plt.subplots(1, 2, figsize=(11.6, 3.5), dpi=150)
    for ax, (a, lab) in zip(ax2, ((crop, "as photographed"),
                                  (dichromat(crop, args.kind),
                                   "under the dichromat transform"))):
        ax.imshow(a)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
    f2.text(0.5, 0.03,
            "The red hydrant stands in front of the green planter, left of "
            "centre; the pumpkin is orange on straw.\nBoth sink into their "
            "surroundings; the sky and the blue-grey building do not move.",
            ha="center", va="top", color=MUTED, fontsize=8.5, linespacing=1.6)
    f2.suptitle("The first swatch row, photographed outdoors",
                color=INK, fontsize=12.5)
    f2.tight_layout(rect=(0, 0.11, 1, 0.93))
    out2 = AOI_DIR / "22_colour_outdoor.png"
    f2.savefig(out2, bbox_inches="tight", facecolor="white")
    plt.close(f2)

    for name, tgt, bg, bgname in CASES:
        t2 = dichromat(np.array(tgt, np.uint8).reshape(1, 1, 3), args.kind)[0, 0]
        b2 = dichromat(np.array(bg, np.uint8).reshape(1, 1, 3), args.kind)[0, 0]
        s0, s1 = sep(tgt, bg), sep(t2, b2)
        print(f"{name:<32} {s0:7.1f} -> {s1:6.1f}   {1 - s1 / s0:>6.0%} lost")
    print(f"\n-> {out}\n-> {out2}")


if __name__ == "__main__":
    main()
