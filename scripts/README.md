# `scripts/`

Everything that reads the footage or draws a figure. Run them from the project
root, not from in here — they locate the data folders relative to their own
parent directory, so `python3 scripts/aoi_plots.py` works and `cd scripts &&
python3 aoi_plots.py` also works, but a copy of a script somewhere else will
not find `aoi/`.

Two kinds of file, and the difference decides how long you wait:

- **An `_frames` or `_extract` script reads the footage.** It runs a model or
  decodes video over all 2092 paired seconds, takes 10–35 minutes, and writes a
  CSV. Run it once.
- **A `_plots` or figure script reads that CSV.** It takes seconds. Run it as
  often as you like; nothing re-reads the footage.

That split is the whole point: no figure here re-runs a model, so a change to
a chart never costs half an hour.

## Order

```
                     A/export/pet-vision-combined.mp4
                                  │
            extract_paired_frames.py   (→ frames_paired/, 1 fps pairs)
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        │                         │                          │
  aoi_frames.py          elevation_frames.py          depth_frames.py
  (→ aoi_shares)        (→ elevation_shares,          (→ depth_metrics)
        │                 elev_azim_shares)                  │
        │                         │                          │
  aoi_plots.py           elevation_plots.py           depth_plots.py
  walk_contrasts.py ────── reads both aoi_shares and depth_metrics
  binocular_field.py ───── reads elev_azim_shares
  divergence_sheet.py                                 isovist.py
  stop_context.py ──────── also reads imu/imu.csv
  aoi_video.py


                     A/dog/VID_…_013.insv + _014.insv
                                  │
                        imu_extract.py   (→ imu/imu.csv, 20 Hz)
                                  │
                   imu_plots.py        imu_attitude.py
```

The eye-tracking side, separate from both:

```
      pupil_eye_tracking/<session>/        (Neon native recording)
                    │
        viewer/export_data.py   (→ viewer/data/<session>.json)
        screen_trace.py         (→ viewer/data/<session>.trace.json, optional)
        screen_homography.py    (→ viewer/data/<session>.gaze_px.csv)
                    │                        │
             viewer/serve.py → the viewer,   │  gaze in STIMULUS pixels
             where trial boundaries are      │
             marked by hand                  │
             (→ eyetracking_n1/P01_trials.csv)
                    └───────────┬────────────┘
                                │
   gaze_on_stimulus.py      gaze_aoi.py          gaze_headmotion.py
   (→ 12_gaze_heatmaps,     (→ gaze_aoi_shares.csv,  (→ head_motion.csv,
      13_scanpaths)            14_gaze_aoi_bias,        18_head_motion)
                               15_gaze_aoi_stack)
```

Both halves are needed: the homography gives the SPACE (which stimulus pixel),
the marking gives the TIME (which trial). Neither alone produces a figure.

Standing on their own, needing none of the above:

- `dog_vision.py` — the dichromat transform, run on any image or folder
- `dog_vision_figure.py` — what that transform does to urban target colours
- `build_stimuli.py` — cuts a stimulus set out of `stimulus_pairs/`
- `sphere_geometry.py` / `segment_frames.py` — the 360 machinery, for
  equirectangular sources; the combined export is rectilinear, so the current
  pipeline does not use them

## What each one is for

| | |
| --- | --- |
| `extract_frames.py` | one frame per real second from a video, with a sharpness probe |
| `extract_paired_frames.py` | the same, both panels at once, so a pair is the same instant by construction |
| `aoi_frames.py` | segment each panel, collapse ADE20K's 150 classes to AOI groups, write the per-frame shares |
| `aoi_plots.py` | mean composition, composition over time, the difference with its CI |
| `aoi_video.py` | the demo video: both panels side by side with a live AOI bar chart |
| `elevation_frames.py` | the same segmentation, but binned by elevation and azimuth first |
| `elevation_plots.py` | what fills each elevation band, and where each class moves between the heights |
| `depth_frames.py` | depth beside the segmentation: openness, enclosure, near-field distances. `--audit` prints the cross-view check that says why there are no height bands |
| `depth_plots.py` | openness over the walk, obstacle cover, how far the ground runs |
| `isovist.py` | the forward wedge as a plan-view fan, beside the frames it came from |
| `divergence_sheet.py` | the seconds where the two panels disagree most, filtered for black frames and self-occlusion |
| `stop_context.py` | where the dog stopped, and what was in front of it while it did |
| `imu_extract.py` | pull the IMU out of the `.insv` trailer, resample to 20 Hz |
| `imu_plots.py` | motion, stops, body pitch |
| `imu_attitude.py` | the tilt rose and the motion profile |
| `dog_vision_figure.py` | target–background separation under the dichromat transform |
| `colour_samples.py` | the same eight pairs as pictures rather than numbers, plus one real stimulus before and after — what "loses 86% of its separation" looks like |
| `screen_trace.py` | screen brightness and change traces for one Pupil session, as a hint layer for marking trial boundaries in the viewer |
| `screen_homography.py` | find the monitor's four corners per frame and map gaze into stimulus pixels. `--montage` rectifies a few frames so the geometry can be checked by eye |
| `binocular_field.py` | item 4: what sits in the dog's monocular wings (30-45 deg) against its binocular wedge, beside a schematic of both species' fields. The field angles are literature values, not measured here |
| `walk_contrasts.py` | items 7 and 10: the paired dog-minus-human difference second by second with its block-bootstrap CI, and the species x indoor/outdoor violins (whose indoor half is one scene, and says so) |
| `gaze_headmotion.py` | item 22: head angular speed per trial from the Neon's own gyroscope — a null result by design, there to rule head motion out |
| `gaze_aoi.py` | items 19 and the quantitative half of 18: segments the stimuli, scores each gaze sample by semantic class, and reports dwell share minus area share so the two heights can be compared although they frame different amounts of ground |
| `gaze_on_stimulus.py` | items 18 and 21: heat maps and scanpaths on the stimulus, one row per scene and one column per condition, so a scene's human / dog / dichromat versions sit side by side |
|  `export_gaze_web.py` | the data half of `docs/gaze-by-scene/`: gaze per trial in stimulus pixels, decimated to ~60 Hz, a downscaled copy of each stimulus, and the link to each trial’s heat map |
| `export_aoi_video_web.py` | the data half of `docs/live-aoi/`: composition, depth, divergence and IMU for one excerpt of the walk, keyed on source time |
| `bundle_viz_html.py` | fold a viz page into one double-clickable `.html`: the JSON inlined as `window.__DATA__`, every image as a `data:` URI. Writes `docs/standalone-build/` |

## Before you run the heavy ones

`depth_frames.py` and the two `_frames` scripts need `torch` and
`transformers` (already in `.venv`), and will download a model on first use.
`imu_extract.py` needs `telemetry-parser`. Everything else is numpy,
matplotlib and Pillow.

The two `export_*_web.py` scripts write into `docs/`, not `aoi/`; `docs/README.md`
says how to serve those pages and how to re-cut the video excerpt.

Figures land in `aoi/` and `imu/`; what each CSV holds, and which columns carry
a warning, is in `aoi/README.md`.
