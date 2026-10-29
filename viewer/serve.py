#!/usr/bin/env python3
"""Serve the project so the viewer can reach both itself and the recordings.

The viewer lives in viewer/ but its video sources point at
../pupil_eye_tracking/<session>/, so the server root has to be the PROJECT, not
the viewer folder. Opening viewer/index.html by double-clicking does not work:
fetch() of the JSON is blocked on file:// and the browser will not seek inside
a video it cannot range-request.

Range requests matter here. The scene videos are 200-500 MB and the viewer
seeks constantly; Python's stock SimpleHTTPRequestHandler ignores Range and
re-sends the whole file every time, which makes scrubbing unusable. This
handler implements single-range GET, which is all a <video> needs.

Usage
-----
    python3 viewer/serve.py            # http://localhost:8731/viewer/
    python3 viewer/serve.py 9000
"""

import http.server
import os
import re
import socketserver
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
RANGE = re.compile(r"bytes=(\d*)-(\d*)")


#  where the viewer's trial-marking panel writes; one file per session set, and
#  it is overwritten rather than appended, because a marking pass is a whole
#  answer and half of an old one mixed into a new one is worse than nothing
TRIALS_OUT = ROOT / "eyetracking_n1" / "P01_trials.csv"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("save-trials"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8")
        TRIALS_OUT.parent.mkdir(parents=True, exist_ok=True)
        TRIALS_OUT.write_text(body + "\n")
        rows = max(0, len(body.strip().split("\n")) - 1)
        msg = f"{rows} 行写入 {TRIALS_OUT.relative_to(ROOT)}"
        out = msg.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
        print(f"  saved {rows} trials -> {TRIALS_OUT}")

    def log_message(self, fmt, *args):
        if "206" not in (args[1] if len(args) > 1 else ""):
            super().log_message(fmt, *args)

    def send_head(self):
        header = self.headers.get("Range")
        if not header:
            return super().send_head()
        m = RANGE.match(header)
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()

        size = os.path.getsize(path)
        first = int(m.group(1)) if m.group(1) else 0
        last = int(m.group(2)) if m.group(2) else size - 1
        last = min(last, size - 1)
        if first > last:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        f = open(path, "rb")
        f.seek(first)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(last - first + 1))
        self.end_headers()
        self._remaining = last - first + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        self._remaining = None
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return          # the browser moved on; normal while scrubbing
            remaining -= len(chunk)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    url = f"http://localhost:{PORT}/viewer/"
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        print(f"serving {ROOT}\n  {url}\nCtrl-C to stop")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
