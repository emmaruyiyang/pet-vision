#!/usr/bin/env python3
"""Fold a viz page into ONE .html file you can double-click.

The pages are already plain HTML; the only reason they need a server is
`fetch("data.json")`, which the file:// origin blocks, and the files beside
them. This inlines both: the JSON goes in as `window.__DATA__`, and every image
path inside it becomes a data: URI, so the result has no siblings to lose.

THE VIDEO IS THE EXCEPTION. The clip is 74 MiB; base64 would make it ~100 MB of
text the browser has to parse before the first frame, and some browsers refuse
a media data: URI that large. It is copied next to the HTML instead, and the
page's `../media/...` src is rewritten to point at that copy. Still no server:
over file:// the browser reads the file directly, so seeking works, which is
exactly what it cannot do under `python3 -m http.server`.

Output goes to docs/standalone-build/, which is generated and git-ignored --
publishing the site does not need it, since fetch() works fine over http.

Usage
-----
    python3 scripts/bundle_viz_html.py                 # both pages
    python3 scripts/bundle_viz_html.py gaze-by-scene
"""

import base64, json, mimetypes, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "docs"
OUT = VIZ / "standalone-build"

#  page dir -> output name, and the sibling files it cannot inline: each is
#  (path relative to docs/, the src spelling in the page) and gets copied next
#  to the HTML with the page rewritten to its bare name
PAGES = {
    "gaze-by-scene": ("gaze-by-scene.html", []),
    "live-aoi": ("human-dog-live-aoi.html",
                 [("media/walk-05-10.mp4", "../media/walk-05-10.mp4")]),
}
#  which JSON fields hold a path to an image that should be inlined
IMAGE_FIELDS = ("img", "heat")


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def inline_images(obj, base: Path, seen: dict):
    """Walk the loaded JSON and swap image paths for data: URIs, once each."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in IMAGE_FIELDS and isinstance(v, str) and not v.startswith("data:"):
                p = base / v
                if p.exists():
                    if v not in seen:
                        seen[v] = data_uri(p)
                    obj[k] = seen[v]
            else:
                inline_images(v, base, seen)
    elif isinstance(obj, list):
        for v in obj:
            inline_images(v, base, seen)


def build(name: str) -> Path:
    out_name, siblings = PAGES[name]
    src = VIZ / name
    html = (src / "index.html").read_text()
    data = json.loads((src / "data.json").read_text())

    seen: dict = {}
    inline_images(data, src, seen)

    #  "</script>" inside the payload would end the tag early; escaping "<" is
    #  enough and keeps it valid JSON
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    tag = f'<script>window.__DATA__={payload};</script>\n'

    if "</head>" not in html:
        raise SystemExit(f"{name}: no </head> to inject before")
    html = html.replace("</head>", tag + "</head>", 1)

    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel, src_attr in siblings:
        f = VIZ / rel
        if not f.exists():
            raise SystemExit(f"{name}: missing {f}")
        if src_attr not in html:
            raise SystemExit(f"{name}: expected {src_attr!r} in the page, not found")
        html = html.replace(src_attr, f.name)
        shutil.copy2(f, OUT / f.name)
        copied.append(f.name)

    dst = OUT / out_name
    dst.write_text(html)

    mb = dst.stat().st_size / 1e6
    note = f"+ {', '.join(copied)} beside it" if copied else "self-contained"
    print(f"{dst.relative_to(ROOT)}  {mb:.1f} MB, {len(seen)} images inlined ({note})")
    return dst


if __name__ == "__main__":
    names = sys.argv[1:] or list(PAGES)
    for n in names:
        if n not in PAGES:
            raise SystemExit(f"unknown page {n!r}; pick from {', '.join(PAGES)}")
        build(n)
