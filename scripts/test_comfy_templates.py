"""Run every API-format template once against the running ComfyUI (smoke test).
LoadImage inputs get real RC01 images; the touch-up template gets an image whose alpha is cleared in
a rectangle, which is how ComfyUI's mask editor marks the painted area."""
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

from mechpipe import REPO_ROOT
from mechpipe.jobs import client

API_DIR = REPO_ROOT / "comfy_nodes" / "0_MechPipeline" / "templates_api"
RC01 = REPO_ROOT / "runs" / "RC01"
INPUTS = {
    "原始機甲圖": REPO_ROOT / "assets" / "sources" / "RC01_source.jpg",
    "改好的機甲圖": RC01 / "master" / "master_edited.png",
    "正面 A-pose 全身圖": RC01 / "master" / "master_front.png",
    "正面部件圖": next((RC01 / "deliver").glob("RC01_FOREARM_HAND_R_*.png")),
}


def touchup_image() -> Path:
    im = Image.open(next((RC01 / "deliver").glob("RC01_CHEST_WAIST_*.png"))).convert("RGBA")
    w, h = im.size
    im.paste((0, 0, 0, 0), (int(w * 0.65), int(h * 0.3), int(w * 0.85), int(h * 0.55)))
    path = Path(tempfile.gettempdir()) / "touchup_test.png"
    im.save(path)
    return path


comfy = client()
only = sys.argv[1:]
for api in sorted(API_DIR.glob("*.api.json")):
    if only and not any(api.name.startswith(o) for o in only):
        continue
    wf = json.loads(api.read_text(encoding="utf-8"))
    for node in wf.values():
        if node["class_type"] != "LoadImage":
            continue
        title = node["_meta"]["title"]
        src = touchup_image() if title.startswith("要修的圖") else next(p for k, p in INPUTS.items() if title.startswith(k))
        node["inputs"]["image"] = comfy.upload(src)
    entry = comfy.wait(comfy.queue(wf))
    outs = [i["filename"] for o in entry["outputs"].values() for i in o.get("images", []) if i["type"] == "output"]
    print(api.name, "OK", outs, flush=True)
