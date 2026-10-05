"""Part bounding boxes on master_front.png: automatic pre-boxing + a local browser editor.

Boxes are stored in config/mechs/<mech_id>.boxes.yaml as {part_id: [x, y, w, h]} in master pixels.
Pre-boxing assumes the standardized master: a front A-pose on a plain light background, so the
silhouette's bounding box is split with fixed body proportions. The user then adjusts the boxes.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageFilter

from . import REPO_ROOT

# (x0, y0, x1, y1) as fractions of the silhouette bbox, for the parts on the VIEWER's left
# (= the mech's right side) plus the centered ones. Mech-left parts are mirrored from these.
TEMPLATE = {
    "HEAD_NECK": (0.36, 0.00, 0.64, 0.17),
    "CHEST_WAIST": (0.27, 0.12, 0.73, 0.42),
    "WAIST_HIP": (0.30, 0.36, 0.70, 0.56),
    "TORSO_FULL": (0.26, 0.12, 0.74, 0.56),
    "SHOULDER_UPPERARM_R": (0.00, 0.08, 0.38, 0.40),
    "FOREARM_HAND_R": (0.00, 0.32, 0.32, 0.62),
    "ARM_FULL_R": (0.00, 0.08, 0.38, 0.62),
    "THIGH_KNEE_R": (0.22, 0.50, 0.52, 0.76),
    "KNEE_SHIN_R": (0.20, 0.66, 0.52, 0.92),
    "ANKLE_FOOT_R": (0.14, 0.85, 0.52, 1.00),
    "LEG_FULL_R": (0.14, 0.50, 0.52, 1.00),
}
MIRROR = {k: k[:-2] + "_L" for k in TEMPLATE if k.endswith("_R")}


def boxes_path(mech_id: str) -> Path:
    return REPO_ROOT / "config" / "mechs" / f"{mech_id}.boxes.yaml"


def load(mech_id: str) -> dict[str, list[int]]:
    p = boxes_path(mech_id)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("parts", {}) if p.exists() else {}


def save(mech_id: str, master: Path, parts: dict[str, list[int]]) -> Path:
    p = boxes_path(mech_id)
    data = {"master": master.relative_to(REPO_ROOT).as_posix(), "image_size": list(Image.open(master).size),
            "parts": {k: [int(round(v)) for v in parts[k]] for k in sorted(parts)}}
    p.write_text("# Part boxes [x, y, w, h] on the master, edited with `mechpipe boxes`.\n"
                 + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=None), encoding="utf-8")
    return p


def silhouette_bbox(master: Path | Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of everything that differs from the background color (sampled at the corners)."""
    im = (master if isinstance(master, Image.Image) else Image.open(master)).convert("RGB")
    w, h = im.size
    corners = [im.getpixel(p) for p in [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]]
    bg = tuple(sorted(c[i] for c in corners)[len(corners) // 2] for i in range(3))
    diff = ImageChops.difference(im, Image.new("RGB", im.size, bg)).convert("L")
    mask = diff.point(lambda v: 255 if v > 28 else 0).filter(ImageFilter.MedianFilter(5))
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"no foreground found in {master}")
    return bbox


def prebox(master: Path | Image.Image) -> dict[str, list[int]]:
    x0, y0, x1, y1 = silhouette_bbox(master)
    sw, sh = x1 - x0, y1 - y0
    out = {}
    for pid, (a, b, c, d) in TEMPLATE.items():
        out[pid] = [x0 + a * sw, y0 + b * sh, (c - a) * sw, (d - b) * sh]
        if pid in MIRROR:
            out[MIRROR[pid]] = [x0 + (1 - c) * sw, y0 + b * sh, (c - a) * sw, (d - b) * sh]
    return {k: [int(round(v)) for v in box] for k, box in out.items()}


EDITOR = (Path(__file__).parent / "boxes_editor.html")


def serve(mech_id: str, master: Path, part_descs: dict[str, str], port: int = 8199):
    """Serve the box editor on http://127.0.0.1:<port>/ until Ctrl+C. Saving writes the boxes yaml."""
    state = {"boxes": load(mech_id) or prebox(master)}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(EDITOR.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/image":
                self._send(master.read_bytes(), "image/png")
            elif self.path == "/data":
                w, h = Image.open(master).size
                self._send(json.dumps({"mech_id": mech_id, "width": w, "height": h, "boxes": state["boxes"],
                                       "prebox": prebox(master), "descs": part_descs}, ensure_ascii=False).encode(),
                           "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/save":
                return self._send(b"not found", "text/plain", 404)
            boxes = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["boxes"] = boxes
            path = save(mech_id, master, boxes)
            print(f"saved {len(boxes)} boxes -> {path.relative_to(REPO_ROOT)}", flush=True)
            self._send(json.dumps({"saved": path.relative_to(REPO_ROOT).as_posix()}).encode(), "application/json")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"框選工具：http://127.0.0.1:{port}/  （按 Ctrl+C 結束）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
