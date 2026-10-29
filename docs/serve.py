#!/usr/bin/env python3
"""Static server for viz/, with byte-range support.

python3 -m http.server answers every request with the whole file, so a <video>
can play from the start but cannot seek: the browser asks for a range, gets
200 + the whole thing, and gives up. The 5-minute clip is ~80 MB, so that
matters. This adds the one missing piece.

    python3 viz/serve.py            # http://localhost:8777
"""
import os, re, sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        m = RANGE.match(rng)
        if not m:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        start, end = m.group(1), m.group(2)
        start = int(start) if start else 0
        end = int(end) if end else size - 1
        end = min(end, size - 1)
        if start > end:
            f.close()
            self.send_error(416)
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        f.seek(start)
        return _Capped(f, end - start + 1)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass


class _Capped:
    """A file object that stops after n bytes, for copyfile()."""
    def __init__(self, f, n):
        self.f, self.left = f, n

    def read(self, k=-1):
        if self.left <= 0:
            return b""
        if k < 0 or k > self.left:
            k = self.left
        b = self.f.read(k)
        self.left -= len(b)
        return b

    def close(self):
        self.f.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    root = os.path.dirname(os.path.abspath(__file__))
    h = partial(RangeHandler, directory=root)
    print(f"serving {root} on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), h).serve_forever()
