#!/usr/bin/env python3
"""Build the eye-tracking stimulus set: 12 scenes, two counterbalanced lists.

Design constraints this encodes:

- A participant sees each scene ONCE, at one height only. Seeing the same scene
  at both heights turns the second viewing into recall rather than search.
- Height is assigned by list: list A gives odd scenes the dog view, list B gives
  them the human view. Across a pair of participants every scene is seen at both
  heights, so scene and condition are not confounded.
- Trial order is shuffled per participant, with a cap on how many consecutive
  trials may share a condition, so a shuffle never accidentally blocks.

Output:

    eyetracking_stimuli/
      scenes/            master set, scene-named, all three versions each
      practice/          four trials from pairs rejected for the main set
      runs/<run>/        ready to play: filenames sort into presentation order
      optional_2x2/      human frames under the colour filter
      unmatched/         the A-frame sign, dog view only
      trial_lists.csv    every participant's assignment and order

Run it again to rebuild from scratch; it is deterministic for a given seed.
"""

import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dog_vision import (  # noqa: E402
    linear_to_srgb, read_image, srgb_to_linear, write_image,
)

ROOT = Path(__file__).resolve().parent.parent   # the project, not scripts/
PAIRS = ROOT / "stimulus_pairs"
OUT = ROOT / "eyetracking_stimuli"

# scene id -> (source stem, folder, slug, what the pair is for)
SCENES = [
    ("S01", "v06_t1104", "from_video", "starbucks", "storefront sign leaves the dog frame"),
    ("S02", "m01_narrow-passage", "manual", "narrow-passage", "1.2 m grasses blind the dog behind them"),
    ("S03", "m03_louvre-bench", "manual", "poster-louvre", "wall poster unreadable from dog height"),
    ("S04", "v01_t752", "from_video", "steps-handrail", "handrail becomes overhead occlusion"),
    ("S05", "v08_t2007", "from_video", "sofa-back", "sofa back as indoor occluder"),
    ("S06", "v04_t882", "from_video", "crosswalk-post", "crosswalk plus pedestrian post"),
    ("S07", "v03_t865", "from_video", "hay-bale", "hay bale hides pedestrians from the dog"),
    ("S08", "m04_benjerry-sign", "manual", "sandwich-board", "same sign at very different scale"),
    ("S09", "m05_pumpkin-arch", "manual", "pumpkin-arch", "2.5 m temporary installation"),
    ("S10", "m02_crosswalk", "manual", "tactile-paving", "kerb and tactile paving"),
    ("S11", "v09_t2012", "from_video", "counter-notice", "printed notice on an indoor counter"),
    ("S12", "v10_t2080", "from_video", "indoor-stairs", "stair nosing and handrail indoors"),
]

# rejected for the main set, good enough to learn the task on
PRACTICE = [
    ("P1", "v02_t763", "from_video", "park-path"),
    ("P2", "v05_t960", "from_video", "plaza-paving"),
    ("P3", "v11_t1816_aframe", "from_video", "distant-aframe"),
    ("P4", "v07_t1697", "from_video", "patio-backlit"),
]

N_PARTICIPANTS = 24
MAX_RUN = 3          # interleaved only: cap on consecutive trials of one condition
ORDER = "blocked"    # "blocked" = one block per condition, or "interleaved"
CONDITIONS = ["human", "dog", "dog_dichromat"]
SEED = 6384
EQUALISE = False     # leave pixels as captured; brightness goes in as a covariate
SIZE = "1920:1080"


def run(cmd, **kw):
    subprocess.run(cmd, check=True, capture_output=True, **kw)


def normalise(src: Path, dst: Path):
    run(["ffmpeg", "-v", "error", "-i", str(src),
         "-vf", f"scale={SIZE}:flags=lanczos", "-q:v", "2", "-y", str(dst)])


def dichromat(paths, out_dir=None):
    cmd = [sys.executable, "-I", str(Path(__file__).resolve().parent / "dog_vision.py")] + [str(p) for p in paths]
    if out_dir:
        cmd += ["--out", str(out_dir)]
    run(cmd)
    base = Path(out_dir) if out_dir else None
    for p in paths:
        d = (base or Path(p).parent) / f"{Path(p).stem}_deute{Path(p).suffix}"
        d.rename(d.with_name(d.name.replace("_deute", "_dichromat")))


def mean_luminance(img):
    """Mean relative luminance, in linear light."""
    lin = srgb_to_linear(img.astype(np.float64))
    return float((lin @ [0.2126, 0.7152, 0.0722]).mean())


def equalise_pair(human_path, dog_path):
    """Give the two frames of a pair the same mean luminance.

    The proposal asks for matched luminance across conditions, and two of the
    pairs were cut from separate recordings whose exposure differs by enough to
    be an alternative explanation for any detection difference.

    Both frames are brought DOWN to the darker one's luminance. Meeting in the
    middle would mean scaling the darker frame up, and on these daylight scenes
    that blows the sky: a geometric-mean target clipped up to 10 % of pixels on
    some pairs, which destroys exactly the region B3 targets sit against. Only
    attenuating costs some headroom and cannot lose highlight detail.

    Gain is applied in linear light, which is what changing exposure actually
    does; doing it on gamma-encoded values would alter contrast as well as
    level. Returns the gains and how much each frame clipped.
    """
    imgs = {p: read_image(p) for p in (human_path, dog_path)}
    ys = {p: mean_luminance(im) for p, im in imgs.items()}
    target = min(ys.values())

    report = {}
    for p, im in imgs.items():
        gain = target / ys[p]
        lin = srgb_to_linear(im.astype(np.float64)) * gain
        clipped = float((lin > 1.0).mean())
        write_image(p, linear_to_srgb(lin))
        report[p] = (gain, clipped)
    return report


def condition_for(scene_index, list_name):
    """Two-condition interleaved design: list A odd->dog, list B odd->human."""
    odd = scene_index % 2 == 1
    if list_name == "A":
        return "dog" if odd else "human"
    return "human" if odd else "dog"


def blocked_plan(pnum, rng):
    """Three blocks, one condition each, a DIFFERENT scene set in each block.

    Blocking by condition is what was asked for, and it is only sound if a
    participant never meets the same scene twice: seeing a scene again turns
    search into recall, and no amount of spacing within a session undoes that.
    So the 12 scenes split into three sets of four and each set is shown under
    one condition only.

    The cost of blocking is that time on task, practice and fatigue load onto
    whichever condition came last. That is handled across participants: block
    order cycles through all six permutations, and the set-to-block mapping
    rotates, so every (scene set, condition) combination occurs equally often.
    """
    import itertools
    if N_PARTICIPANTS == 1:
        # Block position cannot be counterbalanced with one participant, so the
        # contrast that matters goes where contamination is lowest: human first
        # (naive), dog second, the colour extension last.
        order = [CONDITIONS.index(c) for c in ("human", "dog", "dog_dichromat")]
    else:
        orders = list(itertools.permutations(range(len(CONDITIONS))))   # 6
        order = orders[pnum % len(orders)]
    rot = pnum % len(CONDITIONS)
    if N_PARTICIPANTS == 1:
        sets = [list(range(len(SCENES)))] * len(CONDITIONS)   # every scene, every block
    else:
        sets = [list(range(i, len(SCENES), len(CONDITIONS)))
                for i in range(len(CONDITIONS))]              # disjoint sets

    trials = []
    for block, cond_i in enumerate(order, 1):
        scene_set = sets[(cond_i + rot) % len(sets)]
        shuffled = scene_set[:]
        rng.shuffle(shuffled)
        for scene_i in shuffled:
            trials.append((scene_i, CONDITIONS[cond_i], block))
    return trials


def shuffled_order(conditions, rng):
    """Shuffle trial indices so no more than MAX_RUN in a row share a condition."""
    idx = list(range(len(conditions)))
    for _ in range(10000):
        rng.shuffle(idx)
        longest = best = 1
        for a, b in zip(idx, idx[1:]):
            best = best + 1 if conditions[a] == conditions[b] else 1
            longest = max(longest, best)
        if longest <= MAX_RUN:
            return idx
    raise RuntimeError("could not satisfy the run-length constraint")


def main():
    global OUT, CONDITIONS, N_PARTICIPANTS
    if "--n1" in sys.argv:
        # One participant. The no-repeat rule exists to protect a condition
        # contrast, and with n=1 there is no contrast to protect: every scene
        # would sit in exactly one condition, so scene and condition would be
        # perfectly confounded and nothing could be read off the result.
        #
        # So the rule is deliberately traded away. A smaller set of scenes is
        # shown under all three conditions, which is the one thing a single
        # participant can actually show: the same image, the same eyes, three
        # treatments, heat maps that can be laid on top of each other.
        #
        # The price is that the second and third viewings of a scene are
        # re-viewings, and with one participant that cannot be counterbalanced
        # away. Block 1 is the only uncontaminated one. Say so.
        global SCENES
        OUT = ROOT / "eyetracking_n1"
        N_PARTICIPANTS = 1
        keep = {"S01", "S02", "S03", "S04", "S05", "S07"}
        SCENES = [sc for sc in SCENES if sc[0] in keep]
    elif "--pilot" in sys.argv:
        # A one- or two-person pilot cannot estimate an effect size, so it does
        # not try. All three conditions and all 12 scenes, two runs: across the
        # two participants 24 of the 36 scene-by-condition cells are covered,
        # which is as much as 24 trials can reach.
        OUT = ROOT / "eyetracking_pilot"
        N_PARTICIPANTS = 2

    keep = OUT / "README.md"
    saved = keep.read_text() if keep.exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("_master/scenes", "practice",
                "_master/optional_2x2", "_master/unmatched"):
        (OUT / sub).mkdir(parents=True)
    if saved is not None:          # a rebuild must not eat the hand-written notes
        keep.write_text(saved)

    print(f"{len(SCENES)} scenes, {len(PRACTICE)} practice, "
          f"{N_PARTICIPANTS} participants")

    # master set. Pixels are left as captured by default: equalising means
    # scaling one frame, and on these daylight scenes that either blows the sky
    # or costs headroom. The per-image luminance is measured and carried into
    # trial_lists.csv instead, so brightness can go into the model as a
    # covariate rather than being baked into the stimuli.
    dog_frames, human_frames = [], []
    grey = {}
    print("\n  mean grey per frame"
          + (" (equalising)" if EQUALISE else " (left as captured)") + ":")
    for sid, stem, folder, slug, _ in SCENES:
        paths = {}
        for cond in ("human", "dog"):
            src = PAIRS / folder / f"{stem}_{cond}.jpg"
            if not src.exists():
                sys.exit(f"missing source: {src}")
            dst = OUT / "_master" / "scenes" / f"{sid}_{slug}_{cond}.jpg"
            normalise(src, dst)
            paths[cond] = dst
        if EQUALISE:
            equalise_pair(paths["human"], paths["dog"])
        g = {c: float(read_image(p).mean()) for c, p in paths.items()}
        grey[sid] = g
        delta = abs(g["human"] - g["dog"])
        flag = "   <- large" if delta > 25 else ""
        print(f"    {sid} {slug:<16} human {g['human']:5.1f}   "
              f"dog {g['dog']:5.1f}   Δ {delta:4.1f}{flag}")
        dog_frames.append(paths["dog"])
        human_frames.append(paths["human"])
    dichromat(dog_frames)
    dichromat(human_frames, OUT / "_master" / "optional_2x2")
    print(f"\n  scenes/        {len(list((OUT / '_master' / 'scenes').glob('*.jpg')))} files")

    # practice, order baked in
    for n, (pid, stem, folder, slug) in enumerate(PRACTICE, 1):
        cond = "human" if n % 2 else "dog"
        normalise(PAIRS / folder / f"{stem}_{cond}.jpg",
                  OUT / "practice" / f"{n:02d}_{pid}_{slug}_{cond}.jpg")
    print(f"  practice/      {len(list((OUT / 'practice').glob('*.jpg')))} files")

    # the A-frame, which has no human counterpart
    shot = ("/Users/ruyiyang/Library/Mobile Documents/com~apple~CloudDocs/"
            "insta-360/screenshot/"
            "VID_20261025_083627_00_010_2026-10-27_00-57-30_screenshot.jpg")
    x1 = OUT / "_master" / "unmatched" / "X1_aframe-sign_dog.jpg"
    normalise(Path(shot), x1)
    dichromat([x1])

    # participant runs
    rng = random.Random(SEED)
    rows = []
    for pnum in range(1, N_PARTICIPANTS + 1):
        if ORDER == "blocked":
            plan = blocked_plan(pnum, rng)
            list_name = f"blk{pnum % 6 + 1}"   # recorded, not used in the path
        else:
            lname = "A" if pnum % 2 else "B"
            conds = [condition_for(i, lname) for i in range(1, len(SCENES) + 1)]
            plan = [(i, conds[i], 1) for i in shuffled_order(conds, rng)]
            list_name = lname
        run_dir = OUT / f"P{pnum:02d}"
        run_dir.mkdir()
        for trial, (scene_i, cond, block) in enumerate(plan, 1):
            sid, _, _, slug, _ = SCENES[scene_i]
            src = OUT / "_master" / "scenes" / f"{sid}_{slug}_{cond}.jpg"
            name = f"{trial:02d}_b{block}_{sid}_{slug}_{cond}.jpg"
            shutil.copy2(src, run_dir / name)
            base = "dog" if cond.startswith("dog") else "human"
            rows.append({
                "participant": f"P{pnum:02d}", "list": list_name,
                "block": block, "trial": trial, "scene": sid, "slug": slug,
                "condition": cond, "viewpoint": base,
                "colour": "dichromat" if cond.endswith("dichromat") else "normal",
                "file": f"{run_dir.name}/{name}",
                "mean_grey": round(grey[sid][base], 1),
                "pair_grey_delta": round(
                    abs(grey[sid]["human"] - grey[sid]["dog"]), 1),
            })
    n_trials = len(rows) // N_PARTICIPANTS
    print(f"  runs/          {N_PARTICIPANTS} folder(s) x {n_trials} trials"
          f"  ({ORDER})")

    with (OUT / "trial_lists.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "participant", "list", "block", "trial", "scene", "slug",
            "condition", "viewpoint", "colour", "file",
            "mean_grey", "pair_grey_delta"])
        w.writeheader()
        w.writerows(rows)

    # checks worth having fail loudly
    if N_PARTICIPANTS == 1:
        per_cond = len(SCENES)
    elif ORDER == "blocked":
        per_cond = len(SCENES) // len(CONDITIONS)
    else:
        per_cond = len(SCENES) // 2
    for pnum in range(1, N_PARTICIPANTS + 1):
        mine = [r for r in rows if r["participant"] == f"P{pnum:02d}"]
        if N_PARTICIPANTS > 1:
            assert len({r["scene"] for r in mine}) == len(SCENES), "scene repeated"
        for c in (CONDITIONS if ORDER == "blocked" else ["human", "dog"]):
            n = sum(r["condition"] == c for r in mine)
            assert n == per_cond, f"P{pnum:02d}: {n} trials of {c}"
        if ORDER == "blocked":
            seq = [r["block"] for r in sorted(mine, key=lambda r: r["trial"])]
            assert seq == sorted(seq), "blocks are not contiguous"
    cells = {(r["scene"], r["condition"]) for r in rows}
    full = len(SCENES) * len(CONDITIONS)
    if N_PARTICIPANTS >= 6:
        assert len(cells) == full, "some scene never appears under some condition"
    else:
        # too few participants to reach every cell; say how far it gets
        print(f"  coverage       {len(cells)}/{full} scene x condition cells "
              f"({N_PARTICIPANTS} participants)")
    tail = (f"every scene seen under all {len(CONDITIONS)} conditions"
            if N_PARTICIPANTS >= 6 else
            f"{len(cells)}/{full} cells covered")
    repeat = ("no scene repeats within a participant"
              if N_PARTICIPANTS > 1 else
              "every scene under every condition (scenes repeat by design)")
    print(f"\nchecks passed: {repeat}, "
          f"{per_cond} trials per condition, blocks contiguous, {tail}")
    print(f"{OUT / 'trial_lists.csv'}")


if __name__ == "__main__":
    main()
