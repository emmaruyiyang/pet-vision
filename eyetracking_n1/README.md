# Single participant — 6 scenes, 3 conditions, 18 trials

Rebuild with `python3 build_stimuli.py --n1` from the project root.

## Run it

```
practice/   first, 4 trials
P01/        18 trials
```

Play each folder in filename order. The leading number is the trial number.

## Why this is built differently from the 2-person version

With one participant the no-repeat rule stops being protective and starts being
harmful. If each scene sat in exactly one condition, scene and condition would
be perfectly confounded — any difference between blocks could just as easily be
the difference between those particular images. One person cannot unconfound it.

So the rule is traded away on purpose. Six scenes are shown under all three
conditions, which is the one thing a single participant can actually
demonstrate: **the same image, the same eyes, three treatments, heat maps you
can lay on top of each other.**

**The price, stated plainly: trials 7–18 are re-viewings.** Scene recognition is
durable and twelve intervening images do not erase it. Gaze on a familiar scene
is less exploratory and more targeted. With one participant this cannot be
counterbalanced away.

So:

- **Block 1 (trials 1–6, human) is the only uncontaminated block.**
- Block order is fixed to human → dog → dog_dichromat, putting the primary
  contrast where contamination is lowest and the colour extension last.
- Report block 2 and 3 as illustrative. Do not report a condition difference.

This runs as **one session**, so the re-viewing is at its strongest: the three
viewings of a scene are about 50 and 100 seconds apart at an 8 s exposure. Keep
the inter-trial interval generous and do not comment on repetition between
blocks — if the participant notices the images are repeating they will start
comparing rather than viewing, which changes the task mid-experiment.

In the write-up this is one sentence, not a hedge: trials 7-18 were repeat
exposures within a single session, block 1 is the only naive block, and no
condition contrast is claimed.

## The six scenes

Chosen to span different mechanisms rather than to be the six prettiest.

| id | slug | mechanism |
|---|---|---|
| S01 | starbucks | sign above dog eye height |
| S02 | narrow-passage | 1.2 m occluder, blind behind it |
| S03 | poster-louvre | wall poster at 1.2 m |
| S04 | steps-handrail | handrail overhead from dog height |
| S05 | sofa-back | indoor, sofa back as occluder |
| S07 | hay-bale | installation hiding people behind it |

## No detection task

Free viewing, no target, no keypress. H1 is still addressable; H2 as written
("detected less often and later") is not, because there is no detection
response. RQ3 is unaffected — the visibility maps come from the segmentation
pipeline.

Instruction, identical in all three blocks, mentioning neither height nor dogs
nor colour:

> Imagine you are walking along this route. Look around the scene as you
> naturally would.

## Exposure

5–10 s. Report whole-trial dwell per height band and the first five fixations
separately; a viewpoint effect should be larger in the early window.

18 trials at 8 s is about 2.5 minutes of stimulus.

## What one participant can tell you

1. **Does the pipeline work end to end** — Neon capture, gaze export, AOI
   coding, heat maps on the stimulus.
2. **Do the heat maps for one scene move between its three conditions?** Look
   before computing. If they are indistinguishable, the manipulation is weaker
   than assumed and the dual-height pole matters more than assumed.
3. **Does the colour condition move gaze, or only change how the image looks?**
   A transform that does not move gaze is a presentation device, not a
   condition — worth knowing before the main study spends a third of its trials
   on it.
4. **Is 8 s right** — does viewing saturate early?
5. **Does anything break** — calibration drift, monitor too dark, wording.

Nothing here is a result. It is a check that the instrument works and that the
manipulation is visible at all.

## Limitations

The re-viewing problem above, plus those carried from the material: heading is
not locked between the two cameras; the human camera is probably not at 1.6 m,
so the height difference is smaller than the design calls for; within-pair
brightness differs and was left in the pixels, carried in `trial_lists.csv` as
`mean_grey` and `pair_grey_delta`; S05 is soft. Fuller notes in
`../eyetracking_stimuli/README.md`.

`dog_dichromat` is a human deuteranope simulation standing in for canine
dichromacy — a discriminability model, not an appearance model. The proposal
lists the colour transform as an extension not applied at this stage, so running
it as a third condition is a scope change and should be declared.
