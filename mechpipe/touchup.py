"""Touch-up node after S2: erase unwanted areas of a part image by brush, optionally let 2511 repaint
the erased area (e.g. a hole left by the eraser becomes a proper cross-section).

Every touch-up is a new version in runs/<mech_id>/s2_touch/<PART>/ (png + metadata, chain_seed kept),
and deliver.current_images() then uses the latest version. Outside the brushed area the result is
always the previous version, pixel for pixel: only the brushed area (grown and feathered) is taken
from the model output.
"""
import base64
import io
import json
import random
import shutil
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageFilter

from . import REPO_ROOT
from .crop import background
from .deliver import TOUCH_STAGE, current_images
from .jobs import Job, client, rel, settings
from .stages import _prompt, part_descs

EDITOR = Path(__file__).parent / "touchup_editor.html"


def _decode_mask(data_url: str, size: tuple[int, int]) -> Image.Image:
    """Brush layer from the editor (RGBA PNG, painted where alpha > 0) -> L mask 0/255 at image size."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    alpha = Image.open(io.BytesIO(raw)).convert("RGBA").getchannel("A")
    if alpha.size != size:
        alpha = alpha.resize(size, Image.NEAREST)
    return alpha.point(lambda v: 255 if v > 0 else 0)


def _fill(im: Image.Image, mask: Image.Image, rgb) -> Image.Image:
    out = im.copy()
    out.paste(Image.new("RGB", im.size, tuple(rgb)), (0, 0), mask)
    return out


def _marker_left(im: Image.Image, mask: Image.Image, rgb) -> float:
    """Share of the brushed area still showing the marker color (the model did not repaint it)."""
    px = list(im.getdata())
    m = list(mask.getdata())
    area = sum(1 for v in m if v)
    hits = sum(1 for p, v in zip(px, m) if v and all(abs(p[i] - rgb[i]) < 60 for i in range(3)))
    return hits / max(1, area)


class Session:
    def __init__(self, mech_id: str):
        self.mech_id = mech_id
        self.run = REPO_ROOT / settings()["paths"]["runs"] / mech_id
        self.cfg = settings()["touchup"]
        self.descs = part_descs()

    def entries(self) -> list[dict]:
        return [{"part": e["part"], "chain": e["chain_seed"], "png": rel(e["png"]), "touches": e["touches"],
                 "desc": self.descs[e["part"]]} for e in current_images(self.mech_id)]

    def current(self, part: str, chain: int) -> dict:
        for e in current_images(self.mech_id):
            if e["part"] == part and e["chain_seed"] == chain:
                return e
        raise KeyError(f"{part} chain {chain} not found")

    def _save(self, part: str, chain: int, image: Image.Image, meta: dict) -> Path:
        out_dir = self.run / TOUCH_STAGE / part
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y%m%d-%H%M%S}_{meta['seed']}"
        png = out_dir / f"{stem}.png"
        image.save(png)
        png.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return png

    def apply(self, part: str, chain: int, mask_url: str, repaint: bool, hint: str) -> dict:
        src = self.current(part, chain)["png"]
        im = Image.open(src).convert("RGB")
        mask = _decode_mask(mask_url, im.size)
        if not mask.getbbox():
            raise ValueError("沒有塗到任何地方")
        work = self.run / "touch_work"
        work.mkdir(parents=True, exist_ok=True)
        stamp = f"{part}_{chain}_{datetime.now():%Y%m%d-%H%M%S}"
        mask_path = work / f"{stamp}_mask.png"
        mask.save(mask_path)
        erased = _fill(im, mask, background(im))
        meta = {"mech_id": self.mech_id, "stage": TOUCH_STAGE, "part": part, "seed": chain,
                "touch_of": rel(src), "mask": rel(mask_path), "repaint": repaint, "hint": hint,
                "qc": {"ok": True, "problems": [], "chain_seed": chain},
                "timestamp": datetime.now().isoformat(timespec="seconds")}
        if not repaint:
            return {"png": rel(self._save(part, chain, erased, meta)), "problems": []}

        marked = work / f"{stamp}_marked.png"
        _fill(im, mask, self.cfg["fill_rgb"]).save(marked)
        prompt = _prompt("s2_fill.txt", fill_name=self.cfg["fill_name"],
                         hint_clause=f"補畫要求：{hint}。" if hint.strip() else "")
        grown = mask.filter(ImageFilter.MaxFilter(2 * self.cfg["grow"] + 1))
        soft = grown.filter(ImageFilter.GaussianBlur(self.cfg["feather"]))
        comfy = client()
        result, problems, job_meta = None, [], {}
        for attempt in range(self.cfg["tries"]):
            seed = random.randrange(2**32)
            job = Job(mech_id=self.mech_id, stage="s2_touch_raw", part=part, workflow="edit_keep.api.json", seed=seed,
                      mode="draft", prompt=prompt, input_images={"IN_IMAGE_1": rel(marked)},
                      unet=settings()["models"]["edit"]["unet"])
            raw = job.run(comfy)
            out = Image.open(raw).convert("RGB").resize(im.size, Image.LANCZOS)
            # outside the brushed area the previous version stays untouched
            result = Image.composite(out, erased, soft)
            left = _marker_left(result, mask, self.cfg["fill_rgb"])
            problems = [f"marker left {left:.0%}"] if left > 0.02 else []
            job_meta = {"raw": rel(raw), "raw_seed": seed, "attempt": attempt + 1, "prompt": prompt}
            if not problems:
                break
        meta.update(job_meta)
        meta["qc"] = {"ok": not problems, "problems": problems, "chain_seed": chain}
        return {"png": rel(self._save(part, chain, result, meta)), "problems": problems}

    def undo(self, part: str, chain: int) -> dict:
        """Move the latest touch-up of this part/chain to _rejected; the previous version is current again."""
        e = self.current(part, chain)
        if not e["touches"]:
            raise ValueError("沒有可以復原的修改")
        png = e["png"]
        dest = self.run / "_rejected" / TOUCH_STAGE / part
        dest.mkdir(parents=True, exist_ok=True)
        for p in (png, png.with_suffix(".json")):
            shutil.move(p, dest / p.name)
        return {"png": rel(self.current(part, chain)["png"])}


def serve(mech_id: str, port: int = 8198):
    session = Session(mech_id)
    runs = (REPO_ROOT / settings()["paths"]["runs"]).resolve()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json", code)

        def do_GET(self):
            if self.path == "/":
                self._send(EDITOR.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/list":
                self._json({"mech_id": mech_id, "items": session.entries()})
            elif self.path.startswith("/img/"):
                path = (REPO_ROOT / self.path[5:].split("?")[0]).resolve()
                if not path.is_relative_to(runs) or path.suffix != ".png" or not path.exists():
                    return self._send(b"not found", "text/plain", 404)
                self._send(path.read_bytes(), "image/png")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            try:
                if self.path == "/apply":
                    res = session.apply(body["part"], int(body["chain"]), body["mask"], bool(body["repaint"]),
                                        body.get("hint", ""))
                elif self.path == "/undo":
                    res = session.undo(body["part"], int(body["chain"]))
                else:
                    return self._send(b"not found", "text/plain", 404)
            except (KeyError, ValueError) as e:
                return self._json({"error": str(e)}, 400)
            print(f"{self.path} {body.get('part')} chain {body.get('chain')} -> {res}", flush=True)
            self._json(res)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"修圖工具：http://127.0.0.1:{port}/  （按 Ctrl+C 結束）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
