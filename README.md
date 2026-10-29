# pet-vision

What a walk looks like from 0.45 m instead of 1.6 m.

Two cameras recorded the same 35-minute walk at the same instants — one at human
eye height, one on a harness on the dog's back — and a separate eye-tracking
session put one person in front of photographs taken from both heights. This
repository holds the analysis of both, and a site that plays the results back.

**→ [The visualisations](https://emmaruyiyang.github.io/pet-vision/)**

| | |
| --- | --- |
| **Gaze by scene** | 19 trials, three conditions each (human height, dog height, dog height through a dichromat transform). Every trial as an aggregate heat map and as a live scanpath, with dwell measured against how much of the image each class covers. |
| **Human / dog live AOI** | Five minutes of the walk, both viewpoints side by side, with composition, openness, obstacle cover, divergence and IMU motion advancing with the picture. |

## What is here

| Path | What it holds |
| --- | --- |
| [`docs/`](docs/) | the published site — both pages, their data and the video. [`docs/README.md`](docs/README.md) explains how to run it locally and how to rebuild it |
| [`scripts/`](scripts/) | every script that produces a figure or a CSV, in run order. [`scripts/README.md`](scripts/README.md) |
| [`aoi/`](aoi/) | the CSVs everything is drawn from, and the figures. [`aoi/README.md`](aoi/README.md) says what one row is, and which columns carry a warning |
| [`imu/`](imu/) | 20 Hz body motion, and the walking stops found in it |
| [`eyetracking_n1/`](eyetracking_n1/) | the stimuli as shown, and the trial table |
| [`viewer/`](viewer/) | the tool used to mark trial boundaries, and gaze mapped into stimulus pixels |

## Read it with these in mind

**n = 1.** One participant, one dog, one walk. This is a pipeline running end to
end and a look at whether the manipulation did anything — not a result about
people or about dogs.

**Gaze is placed to tens of stimulus pixels, not single ones.** The homography
comes from the monitor's own outline rather than from AprilTags. In this session
the rectified screen measures aspect 1.43 against the stimulus's 1.78, so
vertical positions are compressed by about 20%; horizontal is unaffected.

**`ground` excludes grass**, which ADE20K files under vegetation, and distances
ending `_cam` are in camera heights rather than metres. Far-field depth in this
data is not trustworthy; the near field is.

## What is not here

The raw capture is too large for a repository and is not in it: the Neon scene
and sensor video, the extracted frame pairs, and the full-length demo renders.
`.gitignore` lists them with the script that rebuilds each. Everything the
figures and the site are drawn from — the CSVs — is in the repo.

---

SCI 6384 · Towards a New Science of Design · Harvard GSD
