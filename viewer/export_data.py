#!/usr/bin/env python3
"""Turn a Neon native recording into one JSON the browser viewer can read.

The Neon app writes its streams as flat binary: `<name> ps1.raw` holds the
records, `<name> ps1.time` holds one int64 nanosecond timestamp per record, and
`<name>.dtype` spells the record layout. Nothing here guesses those layouts --
each .dtype file is parsed and used, so a recording written by a different app
version still reads correctly as long as it ships its own .dtype.

TIME. Every stream carries absolute nanoseconds since the epoch. The browser
needs seconds since the start of the scene VIDEO, so everything is shifted by
the first scene-camera frame timestamp. The video is then assumed to play at a
constant rate from that instant, which is what an HTML5 <video> does; Neon
drops frames occasionally, so treat alignment as good to a frame or two rather
than exact.

GAZE IS IN SCENE-CAMERA PIXELS (1600x1200 here), not screen coordinates. The
viewer scales them to however large the video is drawn. Mapping them onto a
stimulus shown on a monitor is a different problem -- that is the surface
tracking this recording lacks, and why these sessions cannot be cut into
trials. See the work log.

Usage
-----
    python3 export_data.py                 # every session found
    python3 export_data.py 2026-10-27-04-12-39
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "pupil_eye_tracking"
OUT = Path(__file__).resolve().parent / "data"

#  keep every Nth gaze sample: 200 Hz is far more than a 60 Hz screen can show,
#  and the file size is what decides whether this loads at all
GAZE_STRIDE = 3


def read_dtype(path):
    """A .dtype file is a Python list of (name, type) pairs. Parse, don't eval."""
    pairs = ast.literal_eval(path.read_text())
    return np.dtype([(n, "<" + np.dtype(t).str[1:]) for n, t in pairs])


def stream(session, name):
    """-> (records, timestamps_ns) or (None, None) if the stream is absent."""
    raw = session / f"{name} ps1.raw"
    tim = session / f"{name} ps1.time"
    dt = session / f"{name}.dtype"
    if not raw.exists() or not dt.exists():
        return None, None
    rec = np.fromfile(raw, dtype=read_dtype(dt))
    ts = np.fromfile(tim, dtype="<i8") if tim.exists() else None
    return rec, ts


def export(session):
    info = json.loads((session / "info.json").read_text())
    scene_t = np.fromfile(session / "Neon Scene Camera v1 ps1.time", dtype="<i8")
    t0 = int(scene_t[0])

    gaze, gaze_t = stream(session, "gaze")
    if gaze is None:
        print(f"  {session.name}: no gaze stream, skipped")
        return None
    sec = (gaze_t - t0) / 1e9
    keep = slice(None, None, GAZE_STRIDE)
    samples = [[round(float(s), 3), round(float(x), 1), round(float(y), 1)]
               for s, x, y in zip(sec[keep], gaze["x"][keep], gaze["y"][keep])]

    fix, _ = stream(session, "fixations")
    fixations = []
    if fix is not None:
        for r in fix:
            #  event_type 1 = fixation, 0 = saccade; both are in this stream
            if int(r["event_type"]) != 1:
                continue
            a = (int(r["start_timestamp_ns"]) - t0) / 1e9
            b = (int(r["end_timestamp_ns"]) - t0) / 1e9
            fixations.append({
                "t0": round(a, 3), "t1": round(b, 3),
                "ms": round((b - a) * 1000),
                "x": round(float(r["mean_gaze_x"]), 1),
                "y": round(float(r["mean_gaze_y"]), 1),
                "amp": round(float(r["amplitude_angle_deg"]), 2),
            })

    blink, _ = stream(session, "blinks")
    blinks = []
    if blink is not None:
        for r in blink:
            blinks.append([round((int(r["start_timestamp_ns"]) - t0) / 1e9, 3),
                           round((int(r["end_timestamp_ns"]) - t0) / 1e9, 3)])

    events = []
    etxt, etime = session / "event.txt", session / "event.time"
    if etxt.exists() and etime.exists():
        names = etxt.read_text().split("\n")
        times = np.fromfile(etime, dtype="<i8")
        for n, t in zip(names, times):
            if n.strip():
                events.append({"t": round((int(t) - t0) / 1e9, 3), "name": n.strip()})

    worn, worn_t = stream(session, "worn")
    off = []
    if worn is not None and worn_t is not None:
        w = worn["worn"].astype(bool)
        ws = (worn_t - t0) / 1e9
        edges = np.flatnonzero(np.diff(w.astype(int)) != 0) + 1
        start = 0
        for e in list(edges) + [len(w)]:
            if not w[start]:
                off.append([round(float(ws[start]), 2), round(float(ws[e - 1]), 2)])
            start = e

    out = {
        "id": session.name,
        "label": info.get("template_data", {}).get("recording_name", session.name),
        "video": f"../pupil_eye_tracking/{session.name}/Neon Scene Camera v1 ps1.mp4",
        "width": 1600, "height": 1200,
        "duration": round(float(sec[-1]), 2),
        "nominal_hz": info.get("gaze_frequency"),
        "actual_hz": round(len(gaze_t) / float(sec[-1] - sec[0]), 1),
        "device": info.get("android_device_model"),
        "gaze": samples,
        "fixations": fixations,
        "blinks": blinks,
        "events": events,
        "not_worn": off,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{session.name}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  {session.name}: {len(samples)} gaze, {len(fixations)} fixations, "
          f"{len(blinks)} blinks, {out['duration']:.0f} s, "
          f"{out['actual_hz']:.0f} Hz actual")
    return {"id": out["id"], "label": out["label"], "duration": out["duration"],
            "fixations": len(fixations), "actual_hz": out["actual_hz"]}


def main():
    wanted = sys.argv[1:]
    dirs = sorted(d for d in SESSIONS.iterdir()
                  if d.is_dir() and (d / "info.json").exists())
    if wanted:
        dirs = [d for d in dirs if d.name in wanted]
    if not dirs:
        print(f"no sessions under {SESSIONS}")
        return
    print(f"{len(dirs)} session(s)")
    index = [r for r in (export(d) for d in dirs) if r]
    OUT.mkdir(exist_ok=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
