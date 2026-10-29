#!/usr/bin/env python3
"""Figures from aoi/aoi_shares.csv: what the two viewpoints actually contain.

No eye tracking here. These are composition measures -- the share of the frame
each AOI group covers -- compared between the human panel and the dog panel at
the same instant.

Two things decide whether the headline number is honest:

1. THE DOG'S OWN BODY. The camera rides on the dog's back, so its fur fills a
   corner of most dog frames. Leaving that in the denominator shrinks every
   other share in the dog panel and would manufacture a difference that is
   really just the mount. --exclude-self (default on) drops the `animal` group
   from BOTH panels and renormalises, so the comparison is scene against scene.

2. AUTOCORRELATION. Consecutive seconds of a walk are nearly the same picture,
   so treating 2092 frames as 2092 independent observations gives a confidence
   interval several times too narrow. The interval here is a moving-block
   bootstrap over 30 s blocks, which keeps the within-block correlation.

Writes three PNGs to aoi/.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
AOI_DIR = ROOT / "aoi"
INK, MUTED = "#1a1a1a", "#8a8a8a"
HUMAN, DOG = "#2b6cb0", "#c05621"


#  The IMU found one 9.4-minute block, 19.0-28.6 min, where the dog sat indoors.
#  That is 28% of the frames and all of them are the SAME fixed scene, so it
#  dominates any mean taken over the whole recording -- and it does: with it in,
#  `ground` reads +0.043 (CI excludes zero), and with it out, +0.008 (it does
#  not). Keep it out by default; --keep-sedentary puts it back.
SEDENTARY = (1140.0, 1716.0)


def load(exclude_self=True):
    with (AOI_DIR / "aoi_shares.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    names = [k for k in rows[0] if k not in ("pair_index", "t_s", "view")]
    out = {}
    for view in ("human", "dog"):
        r = [x for x in rows if x["view"] == view]
        r.sort(key=lambda x: int(x["pair_index"]))
        t = np.array([float(x["t_s"]) for x in r])
        m = np.array([[float(x[n]) for n in names] for x in r])
        out[view] = (t, m)
    if exclude_self:
        keep = [i for i, n in enumerate(names) if n != "animal"]
        names = [names[i] for i in keep]
        for view in out:
            t, m = out[view]
            m = m[:, keep]
            m = m / np.maximum(m.sum(1, keepdims=True), 1e-9)
            out[view] = (t, m)
    return names, out


def block_boot(diff, block=30, n=4000, seed=6384):
    """95% CI for the mean of an autocorrelated series, moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n_obs = len(diff)
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs - block + 1, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :])
    means = diff[idx.reshape(n, -1)[:, :n_obs]].mean(1)
    return np.percentile(means, [2.5, 97.5])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)


def smooth(x, w):
    if w < 2:
        return x
    k = np.ones(w) / w
    p = w // 2
    return np.convolve(np.pad(x, p, mode="edge"), k, "same")[p:p + len(x)]


def fig_means(names, data, order, block, keep):
    th, mh = data["human"]
    td, md = data["dog"]
    mh, md = mh[keep], md[keep]
    n = len(order)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.6), dpi=150, gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(n)
    hv = [mh[:, names.index(g)].mean() for g in order]
    dv = [md[:, names.index(g)].mean() for g in order]
    ax.barh(y + 0.2, hv, 0.38, color=HUMAN, label="human 1.6 m")
    ax.barh(y - 0.2, dv, 0.38, color=DOG, label="dog 0.45 m")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean share of frame", color=MUTED, fontsize=9)
    ax.set_title("What is in the picture, walking only",
                 color=INK, fontsize=11, fontweight="600", loc="left")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(MUTED)
    style(ax)

    lines = []
    for i, g in enumerate(order):
        d = md[:, names.index(g)] - mh[:, names.index(g)]
        lo, hi = block_boot(d, block)
        c = DOG if d.mean() > 0 else HUMAN
        ax2.plot([lo, hi], [i, i], color=c, lw=2.5, solid_capstyle="round")
        ax2.plot([d.mean()], [i], "o", color=c, ms=5)
        lines.append((g, d.mean(), lo, hi))
    ax2.axvline(0, color=MUTED, lw=0.9)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("dog minus human", color=MUTED, fontsize=9)
    ax2.set_title(f"Difference, 95% CI ({block} s blocks)",
                  color=INK, fontsize=11, fontweight="600", loc="left")
    style(ax2)
    fig.tight_layout()
    fig.savefig(AOI_DIR / "01_mean_composition.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return lines


def fig_stacked(names, data, order):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), dpi=150, sharex=True)
    rest = [g for g in names if g not in order]
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, len(order) + 1))
    for ax, view, label in zip(axes, ("human", "dog"),
                               ("human 1.6 m", "dog 0.45 m")):
        t, m = data[view]
        series = [smooth(m[:, names.index(g)], 15) for g in order]
        series.append(smooth(m[:, [names.index(g) for g in rest]].sum(1), 15))
        ax.stackplot(t / 60, *series, labels=order + ["other"], colors=cols)
        ax.set_xlim(0, t[-1] / 60); ax.set_ylim(0, 1)
        ax.set_ylabel(label, color=MUTED, fontsize=9)
        if SEDENTARY:
            ax.axvspan(SEDENTARY[0] / 60, SEDENTARY[1] / 60,
                       color="#1a1a1a", alpha=0.10, lw=0, zorder=5)
        style(ax); ax.grid(False)
    axes[0].set_title("Composition over the walk, 15 s smoothing "
                      "(grey band = the indoor sit, excluded from the means)",
                      color=INK, fontsize=11, fontweight="600", loc="left")
    axes[1].set_xlabel("walk time (min)", color=MUTED, fontsize=9)
    leg = axes[0].legend(frameon=False, fontsize=7, ncol=len(order) + 1,
                         loc="lower center", bbox_to_anchor=(0.5, 1.12))
    for t_ in leg.get_texts():
        t_.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(AOI_DIR / "02_composition_over_time.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_diff(names, data, order):
    th, mh = data["human"]
    td, md = data["dog"]
    pick = order[:4]
    fig, axes = plt.subplots(len(pick), 1, figsize=(11, 1.5 * len(pick)),
                             dpi=150, sharex=True)
    for ax, g in zip(axes, pick):
        d = smooth(md[:, names.index(g)] - mh[:, names.index(g)], 15)
        ax.fill_between(th / 60, 0, d, where=d > 0, color=DOG, alpha=0.5, lw=0)
        ax.fill_between(th / 60, 0, d, where=d <= 0, color=HUMAN, alpha=0.5, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel(g, color=MUTED, fontsize=9)
        ax.set_xlim(0, th[-1] / 60)
        if SEDENTARY:
            ax.axvspan(SEDENTARY[0] / 60, SEDENTARY[1] / 60,
                       color="#1a1a1a", alpha=0.10, lw=0)
        style(ax)
    axes[0].set_title("Dog minus human (orange = more in the dog's view); "
                      "the grey band is one fixed indoor scene, not walking",
                      color=INK, fontsize=11, fontweight="600", loc="left")
    axes[-1].set_xlabel("walk time (min)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(AOI_DIR / "03_difference_over_time.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-self", action="store_true",
                    help="leave the dog's own body in the denominator")
    ap.add_argument("--block", type=int, default=30,
                    help="bootstrap block length in seconds (default 30)")
    ap.add_argument("--groups", type=int, default=6,
                    help="how many AOI groups to plot (default 6, by size)")
    ap.add_argument("--keep-sedentary", action="store_true",
                    help="include the 9.4 min indoor sit in the means "
                         "(it is 28%% of frames and one fixed scene)")
    args = ap.parse_args()

    names, data = load(exclude_self=not args.keep_self)
    th = data["human"][0]
    keep = (np.ones(len(th), bool) if args.keep_sedentary
            else (th < SEDENTARY[0]) | (th > SEDENTARY[1]))
    both = np.concatenate([data["human"][1][keep], data["dog"][1][keep]])
    # "other" is the residual band in the stack plot, so it never competes for
    # a slot of its own -- listing it twice put it in the legend twice
    order = [names[i] for i in np.argsort(-both.mean(0))
             if names[i] != "other"][:args.groups]
    print(f"{len(th)} pairs, {keep.sum()} used for the means "
          f"({'with' if args.keep_sedentary else 'without'} the indoor sit), "
          f"{'with' if args.keep_self else 'without'} the dog's own body")
    print(f"groups plotted: {', '.join(order)}")

    lines = fig_means(names, data, order, args.block, keep)
    fig_stacked(names, data, order)
    fig_diff(names, data, order)

    print(f"\n{'group':<12}{'human':>8}{'dog':>8}{'diff':>9}  95% CI")
    mh = data["human"][1][keep]
    md = data["dog"][1][keep]
    for g, dm, lo, hi in lines:
        h = mh[:, names.index(g)].mean()
        dd = md[:, names.index(g)].mean()
        sig = "" if lo <= 0 <= hi else "  *"
        print(f"{g:<12}{h:>8.3f}{dd:>8.3f}{dm:>+9.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}]{sig}")
    print("\n* = 95% CI excludes zero")
    print(f"3 figures -> {AOI_DIR}")


if __name__ == "__main__":
    main()
