# What is in `aoi/`

Every figure in this project is drawn from one of the CSVs below, and nothing
is drawn from numbers that live only inside a script. If you want a different
chart later, read the CSV; you should not have to re-run a model.

Two things are true of all of them. **Rows are one second of real time**, not
one video frame — the paired frames were cut at 1 fps by real timestamp, so
`t_s` is seconds into the combined export and `pair_index` is that same second
as an integer. And **the indoor block, t = 1140–1716 s, is still in the files**.
It is 9.4 minutes of one fixed scene and it dominates any mean taken over the
whole recording, so every figure drops it — but it is dropped at plotting time,
not here, so you can always put it back.

| File | Rows | What one row is | Written by |
| --- | --- | --- | --- |
| `aoi_shares.csv` | 4185 | one panel of one second: the share of the frame each AOI group covers | `scripts/aoi_frames.py` |
| `elevation_shares.csv` | 25k | one panel, one second, one **elevation band**: the same shares, inside that band only | `scripts/elevation_frames.py` |
| `elev_azim_shares.csv` | 72 | one panel, one elevation × azimuth cell, averaged over the walk | `scripts/elevation_frames.py` |
| `depth_metrics.csv` | 4184 | one panel of one second: openness, enclosure, obstacle cover, near-field distances | `scripts/depth_frames.py` |
| `divergence_tv.csv` | 2092 | one second: how far apart the two panels' compositions are, plus why a second was excluded | `scripts/divergence_sheet.py` |
| `stops_context.csv` | 23 | one walking stop: when, how long, and the mean composition of both panels while it lasted | `scripts/stop_context.py` |
| `../imu/imu.csv` | 46581 | one IMU sample at 20 Hz: gyro, accelerometer, relative pitch/roll, turn rate, motion magnitude | `scripts/imu_extract.py` |
| `../imu/stops.csv` | 26 | one pause detected in the IMU | `scripts/imu_plots.py` |

## The columns that need a warning

**`animal` in the share files is the dog's own fur**, not other animals you
might care about. The camera rides on a harness on the dog's back, so its fur
fills a corner of most dog frames. Every figure renormalises both panels
without it; if you use the raw shares, do the same, or the dog panel's other
classes are all squashed by whatever fraction the fur happened to cover.

**`ground` does not include grass.** ADE20K files grass under vegetation. So
"ground" here means paving and bare earth. Changing that is a one-line edit to
the AOI dictionary in `scripts/segment_frames.py`.

**Distances in `depth_metrics.csv` whose name ends `_cam` are in camera
heights, not metres**, and that is deliberate. The viewing geometry scales with
the camera: the nearest visible ground sits at `cam / tan(29.36°)` whatever the
camera height is, which is 2.84 m for the human panel and 0.80 m for the dog.
Plotting metres mostly plots that ratio back at you.

**Nothing here is a height band.** `scripts/depth_frames.py --audit` has the numbers:
the depth model returns ~3.1 m for the camera height of *both* panels, where
the truth is 1.60 and 0.45, and the two panels disagree about the height of the
same object at the same instant by a factor of 2.0–2.8. The near field survives
that (measured 2.68 and 0.77 m against 2.84 and 0.80 from pure geometry), the
far field does not. B0–B3 needs metric depth or a tape measure on site.

Every script that writes or reads these files lives in `scripts/`; `scripts/README.md` has the run order.

## Figures, and which file each comes from

| Figure | From |
| --- | --- |
| `01_mean_composition.png`, `02_composition_over_time.png`, `03_difference_over_time.png` | `aoi_shares.csv` |
| `04_divergence_sheet.png` | `aoi_shares.csv` + the frames themselves |
| `05_elevation_profile.png`, `06_disagreement_map.png` | `elevation_shares.csv`, `elev_azim_shares.csv` |
| `07_openness.png`, `08_obstacles.png` | `depth_metrics.csv` |
| `09_stops.png` | `stops_context.csv` |
| `10_colour_collapse.png` | no data file; computed from named colours in `scripts/dog_vision_figure.py` |
| `11_isovist.png` | `depth_metrics.csv`'s inputs, recomputed per frame by `scripts/isovist.py` |
| `../imu/01`–`04` | `../imu/imu.csv` |
| `../imu/05_attitude_rose.png`, `06_motion_profile.png` | `../imu/imu.csv` |
