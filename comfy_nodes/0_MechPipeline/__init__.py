"""ComfyUI nodes for the mech part pipeline (rubicon_image_gen).

The geometric and prompt steps of `mechpipe` as nodes, so the step-by-step templates in
example_workflows/ can run inside ComfyUI. Prompts, part list, boxes and masters are read from the
repo (prompts/, config/, runs/), the same files the mechpipe CLI uses. Model passes are the stock
Qwen-Image-Edit nodes in the templates; automatic QC and retries exist only in the CLI.

Installed into ComfyUI/custom_nodes/ as a symlink by scripts/setup_comfyui.sh.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechpipe import boxes as mech_boxes  # noqa: E402
from mechpipe.crop import background, content_bbox  # noqa: E402
from mechpipe.jobs import load_yaml, settings  # noqa: E402
from mechpipe.stages import SEGMENT_OF, TORSO_ERASE, TORSO_KEEP, part_canvas, part_descs  # noqa: E402

CATEGORY = "機甲轉圖"
PARTS = list(part_descs())
SEGMENTS = list(SEGMENT_OF)
PROMPTS = sorted(p.stem for p in (REPO_ROOT / "prompts").glob("*.txt")) + ["cam_lora"]
VIEWS = ["front", "45", "keep"]


def to_pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray((image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")


def to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im.convert("RGB")).astype(np.float32) / 255.0)[None]


def mask_to_pil(mask: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    m = mask if mask.dim() == 2 else mask[0]
    im = Image.fromarray((m.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "L")
    return im.resize(size, Image.NEAREST) if im.size != size else im


def pad(seg: Image.Image, bg, margin: float) -> Image.Image:
    m = int(max(seg.size) * margin)
    out = Image.new("RGB", (seg.width + 2 * m, seg.height + 2 * m), bg)
    out.paste(seg, (m, m))
    return out


def load_boxes(mech_id: str) -> dict:
    b = mech_boxes.load(mech_id)
    if not b:
        raise ValueError(f"{mech_id} 沒有框選資料，請先執行 mechpipe boxes {mech_id}")
    return b


class MechLoadMaster:
    """runs/<mech_id>/master/master_<view>.png (the images picked with `mechpipe pick`)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mech_id": ("STRING", {"default": "RC01"}),
                             "view": (["front", "45", "edited"], {"default": "45"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, mech_id, view):
        path = REPO_ROOT / settings()["paths"]["runs"] / mech_id / "master" / f"master_{view}.png"
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path.relative_to(REPO_ROOT)}，請先用 mechpipe pick 選定")
        return (to_tensor(Image.open(path)),)

    @classmethod
    def IS_CHANGED(cls, mech_id, view):
        path = REPO_ROOT / settings()["paths"]["runs"] / mech_id / "master" / f"master_{view}.png"
        return path.stat().st_mtime if path.exists() else ""


class MechPrompt:
    """Renders a prompt template from prompts/ for a part; `text` fills {change_text} / the repaint hint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"template": (PROMPTS,), "part": (["（無）"] + PARTS,), "view": (VIEWS, {"default": "keep"}),
                             "text": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, template, part, view, text):
        if template == "cam_lora":
            return (load_yaml("config/loras.yaml")["loras"]["cam_object"]["prompt"],)
        tu = settings()["touchup"]
        body = (REPO_ROOT / "prompts" / f"{template}.txt").read_text(encoding="utf-8").strip()
        return (body.format(
            part_desc=part_descs().get(part, ""), notes_clause="", view_clause=settings()["s2_views"][view],
            change_text=text, color_clause="", markings_clause="", fill_name=tu["fill_name"],
            hint_clause=f"補畫要求：{text}。" if text.strip() else ""),)


class MechPartCanvas:
    """Output size for a part with its box's aspect ratio (box from config/mechs/<mech_id>.boxes.yaml)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mech_id": ("STRING", {"default": "RC01"}), "part": (PARTS,),
                             "headroom": ("FLOAT", {"default": 1.35, "min": 1.0, "max": 2.0, "step": 0.05}),
                             "pixels": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 64})}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, mech_id, part, headroom, pixels):
        x, y, w, h = load_boxes(mech_id)[part]
        return part_canvas([0, 0, w, int(h * headroom)], pixels)


class MechImageCanvas:
    """Output size with the aspect ratio of the given image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "pixels": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 64})}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, pixels):
        return part_canvas([0, 0, image.shape[2], image.shape[1]], pixels)


class MechCutSegment:
    """Cut a segment (e.g. 右大腿含膝蓋) out of its clean whole part (整隻右腳) by the vertical ratios of
    their boxes on the front master, padded with background for the completion pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"whole": ("IMAGE",), "mech_id": ("STRING", {"default": "RC01"}), "segment": (SEGMENTS,),
                             "margin": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, whole, mech_id, segment, margin):
        b = load_boxes(mech_id)
        pb, sb = b[SEGMENT_OF[segment]], b[segment]
        top, bottom = max(0.0, (sb[1] - pb[1]) / pb[3]), min(1.0, (sb[1] + sb[3] - pb[1]) / pb[3])
        im = to_pil(whole)
        x0, y0, x1, y1 = content_bbox(im)
        seg = im.crop((x0, int(y0 + top * (y1 - y0)), x1, int(y0 + bottom * (y1 - y0))))
        return (to_tensor(pad(seg, background(im), margin)),)


class MechTorsoCut:
    """Front master with the arms removed -> head, legs and leftover shoulders painted out by their
    boxes -> cut by the torso box (padded for the completion pass)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"no_arms": ("IMAGE",), "mech_id": ("STRING", {"default": "RC01"}),
                             "margin": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, no_arms, mech_id, margin):
        b = load_boxes(mech_id)
        im = to_pil(no_arms)
        master = REPO_ROOT / settings()["paths"]["runs"] / mech_id / "master" / "master_front.png"
        mw, mh = Image.open(master).size
        sx, sy = im.width / mw, im.height / mh   # boxes are in master pixels; the edit may rescale

        def scaled(box):
            x, y, w, h = box
            return (int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy))

        mask = Image.new("L", im.size, 0)
        for k in TORSO_ERASE:
            mask.paste(255, scaled(b[k]))
        for k in TORSO_KEEP:
            mask.paste(0, scaled(b[k]))
        bg = background(im)
        im.paste(Image.new("RGB", im.size, bg), (0, 0), mask)
        return (to_tensor(pad(im.crop(scaled(b["TORSO_FULL"])), bg, margin)),)


class MechMarkMask:
    """Touch-up: the painted mask area as the marker color (for the repaint pass) and as white (erase only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("marked", "erased")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, mask):
        im = to_pil(image)
        m = mask_to_pil(mask, im.size).point(lambda v: 255 if v > 127 else 0)
        if not m.getbbox():
            raise ValueError("遮罩是空的：請在載入圖像節點上按右鍵 →「在遮罩編輯器中開啟」塗出要擦掉的範圍")
        marked, erased = im.copy(), im.copy()
        marked.paste(Image.new("RGB", im.size, tuple(settings()["touchup"]["fill_rgb"])), (0, 0), m)
        erased.paste(Image.new("RGB", im.size, (255, 255, 255)), (0, 0), m)
        return (to_tensor(marked), to_tensor(erased))


class MechMaskComposite:
    """Touch-up: take the repainted image only inside the mask (grown and feathered); outside it the
    erased original stays pixel for pixel."""

    @classmethod
    def INPUT_TYPES(cls):
        tu = settings()["touchup"]
        return {"required": {"erased": ("IMAGE",), "repainted": ("IMAGE",), "mask": ("MASK",),
                             "grow": ("INT", {"default": tu["grow"], "min": 0, "max": 64}),
                             "feather": ("INT", {"default": tu["feather"], "min": 0, "max": 64})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, erased, repainted, mask, grow, feather):
        base = to_pil(erased)
        new = to_pil(repainted).resize(base.size, Image.LANCZOS)
        m = mask_to_pil(mask, base.size).point(lambda v: 255 if v > 127 else 0)
        if grow:
            m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
        if feather:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        return (to_tensor(Image.composite(new, base, m)),)


NODE_CLASS_MAPPINGS = {
    "MechLoadMaster": MechLoadMaster,
    "MechPrompt": MechPrompt,
    "MechPartCanvas": MechPartCanvas,
    "MechImageCanvas": MechImageCanvas,
    "MechCutSegment": MechCutSegment,
    "MechTorsoCut": MechTorsoCut,
    "MechMarkMask": MechMarkMask,
    "MechMaskComposite": MechMaskComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MechLoadMaster": "機甲：載入選定的全身圖",
    "MechPrompt": "機甲：提示詞",
    "MechPartCanvas": "機甲：部位畫布尺寸（依框）",
    "MechImageCanvas": "機甲：畫布尺寸（依圖片）",
    "MechCutSegment": "機甲：從整件切出細分部位",
    "MechTorsoCut": "機甲：塗除頭腿並裁出軀幹",
    "MechMarkMask": "機甲：修圖遮罩標記",
    "MechMaskComposite": "機甲：只貼回遮罩範圍",
}
