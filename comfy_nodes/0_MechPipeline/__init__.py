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
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechpipe import boxes as mech_boxes  # noqa: E402
from mechpipe.crop import background, content_bbox  # noqa: E402
from mechpipe.jobs import load_yaml, settings  # noqa: E402
from mechpipe.pose import apose_skeleton  # noqa: E402
from mechpipe.qc import APOSE_COPY_IOU, same_view  # noqa: E402
from mechpipe.stages import SEGMENT_OF, TORSO_ERASE, TORSO_KEEP, part_canvas, part_descs  # noqa: E402

CATEGORY = "機甲轉圖"
PARTS = list(part_descs())
SEGMENTS = list(SEGMENT_OF)
PROMPTS = sorted(p.stem for p in (REPO_ROOT / "prompts").glob("*.txt")) + ["cam_lora"]
VIEWS = ["front", "45", "keep"]
# Shown on every prompt-producing node: filled in by web/mech_prompt.js with the prompt the node will use
# (re-rendered when the node's other fields change); edited text there is what the node outputs.
FULL_PROMPT = {"optional": {"full_prompt": ("STRING", {
    "multiline": True, "default": "",
    "tooltip": "目前實際使用的提示詞：自動產生，可直接修改；改上面的欄位會重新產生。留空則依上面的欄位產生。"})}}


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


def boxes_for(mech_id: str, boxes: dict | None) -> tuple[dict, tuple[int, int]]:
    """Part boxes and the size of the front image they refer to: the MECH_BOXES input when connected,
    else config/mechs/<mech_id>.boxes.yaml on runs/<mech_id>/master/master_front.png."""
    if boxes is not None:
        return boxes["parts"], tuple(boxes["size"])
    master = REPO_ROOT / settings()["paths"]["runs"] / mech_id / "master" / "master_front.png"
    return load_boxes(mech_id), Image.open(master).size


BOXES_OPT = {"optional": {"boxes": ("MECH_BOXES",)}}


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


def render_prompt(template: str, part: str = "（無）", view: str = "keep", text: str = "") -> str:
    if template == "cam_lora":
        return load_yaml("config/loras.yaml")["loras"]["cam_object"]["prompt"]
    tu = settings()["touchup"]
    body = (REPO_ROOT / "prompts" / f"{template}.txt").read_text(encoding="utf-8").strip()
    return body.format(
        part_desc=part_descs().get(part, ""), notes_clause="", view_clause=settings()["s2_views"][view],
        change_text=text, color_clause="", markings_clause="", fill_name=tu["fill_name"],
        hint_clause=f"補畫要求：{text}。" if text.strip() else "")


class MechPrompt:
    """Renders a prompt template from prompts/ for a part; `text` fills {change_text} / the repaint hint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"template": (PROMPTS,), "part": (["（無）"] + PARTS,), "view": (VIEWS, {"default": "keep"}),
                             "text": ("STRING", {"multiline": True, "default": ""})}, **FULL_PROMPT}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, template, part, view, text, full_prompt=""):
        return (full_prompt.strip() or render_prompt(template, part, view, text),)


class MechAposeSkeleton:
    """A-pose pose reference (OpenPose-style skeleton with mech proportions) for image 2 of the A-pose edit."""

    @classmethod
    def INPUT_TYPES(cls):
        w, h = settings()["master_size"]
        return {"required": {"width": ("INT", {"default": w, "min": 256, "max": 2048, "step": 16}),
                             "height": ("INT", {"default": h, "min": 256, "max": 2048, "step": 16})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, width, height):
        return (to_tensor(apose_skeleton((width, height))),)


class MechPickAPose:
    """Chooses between the two A-pose passes: the text-only result keeps the mech's proportions best,
    but for 3/4 or crouched mechs it just copies the input; then the pose-skeleton result is used."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"edited": ("IMAGE",), "text_result": ("IMAGE",), "skeleton_result": ("IMAGE",),
                             "copy_iou": ("FLOAT", {"default": APOSE_COPY_IOU, "min": 0.5, "max": 1.0, "step": 0.01,
                                                    "tooltip": "文字版和原圖輪廓重疊度達到這個值，就當作照抄原圖、改用骨架版"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "choice")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, edited, text_result, skeleton_result, copy_iou):
        iou = same_view(to_pil(edited), to_pil(text_result))
        if iou >= copy_iou:
            choice, image = f"骨架版（文字版照抄原圖，IoU {iou:.2f}）", skeleton_result
        else:
            choice, image = f"文字版（IoU {iou:.2f}）", text_result
        print(f"[MechPickAPose] {choice}")
        return (image, choice)


class MechPartCanvas:
    """Output size for a part with its box's aspect ratio (box from config/mechs/<mech_id>.boxes.yaml)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mech_id": ("STRING", {"default": "RC01"}), "part": (PARTS,),
                             "headroom": ("FLOAT", {"default": 1.35, "min": 1.0, "max": 2.0, "step": 0.05}),
                             "pixels": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 64})}, **BOXES_OPT}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, mech_id, part, headroom, pixels, boxes=None):
        x, y, w, h = boxes_for(mech_id, boxes)[0][part]
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
                             "margin": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05})}, **BOXES_OPT}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, whole, mech_id, segment, margin, boxes=None):
        b = boxes_for(mech_id, boxes)[0]
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
                             "margin": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05})}, **BOXES_OPT}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, no_arms, mech_id, margin, boxes=None):
        b, (mw, mh) = boxes_for(mech_id, boxes)
        im = to_pil(no_arms)
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


class MechBoxes:
    """The 18 part boxes for a front A-pose image: the boxes of `mech_id` adjusted in the box editor
    (scaled to this image), or, when mech_id is empty or has no boxes, automatic pre-boxing by body
    proportions (rough; check the preview)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"front": ("IMAGE",), "mech_id": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("MECH_BOXES",)
    RETURN_NAMES = ("boxes",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, front, mech_id):
        im = to_pil(front)
        saved = mech_boxes.load(mech_id.strip()) if mech_id.strip() else {}
        if saved:
            master = REPO_ROOT / settings()["paths"]["runs"] / mech_id.strip() / "master" / "master_front.png"
            mw, mh = Image.open(master).size
            sx, sy = im.width / mw, im.height / mh
            parts = {k: [int(x * sx), int(y * sy), int(w * sx), int(h * sy)] for k, (x, y, w, h) in saved.items()}
        else:
            parts = mech_boxes.prebox(im)
        return ({"size": list(im.size), "parts": parts},)


class MechDrawBoxes:
    """Preview of the part boxes drawn on the front image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"front": ("IMAGE",), "boxes": ("MECH_BOXES",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, front, boxes):
        im = to_pil(front)
        d = ImageDraw.Draw(im)
        for i, (k, (x, y, w, h)) in enumerate(sorted(boxes["parts"].items())):
            color = "hsl(%d, 90%%, 45%%)" % (i * 137 % 360)
            d.rectangle([x, y, x + w, y + h], outline=color, width=2)
            d.text((x + 3, y + 3), k, fill=color)
        return (to_tensor(im),)


TOUCH_MODES = {  # mode -> (prompt template, what the model sees in the brushed area)
    "刪除（補背景）": ("s2_remove", "erased"),
    "補畫結構": ("s2_fill", "marked"),
}
BG_SNAP = 24  # repainted pixels this close to the background become exactly the background


class MechMarkMask:
    """Touch-up: prepares the repaint pass for a brushed mask.

    刪除（補背景）: the area is filled with the image's background color and the model only tidies the
    cut edges (a marker-colored blob would make it draw a new part in that shape).
    補畫結構: the area is filled with the marker color and the model repaints a structure there.
    Both modes also output the erase-only image and the grown hard mask for MechMaskComposite."""

    @classmethod
    def INPUT_TYPES(cls):
        tu = settings()["touchup"]
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "mode": (list(TOUCH_MODES),),
                             "text": ("STRING", {"multiline": True, "default": ""}),
                             "grow": ("INT", {"default": tu["grow"], "min": 0, "max": 64})}, **FULL_PROMPT}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("model_input", "erased", "prompt", "mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, mask, mode, text, grow, full_prompt=""):
        im = to_pil(image)
        # any brushed pixel counts: the mask editor's soft brush leaves partial values at the stroke edge
        m = mask_to_pil(mask, im.size).point(lambda v: 255 if v > 0 else 0)
        if not m.getbbox():
            raise ValueError("遮罩是空的：請在載入圖像節點上按右鍵 →「在遮罩編輯器中開啟」塗出要擦掉的範圍")
        if grow:
            m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
        marked, erased = im.copy(), im.copy()
        marked.paste(Image.new("RGB", im.size, tuple(settings()["touchup"]["fill_rgb"])), (0, 0), m)
        erased.paste(Image.new("RGB", im.size, background(im)), (0, 0), m)
        template, seen = TOUCH_MODES[mode]
        model_input = erased if seen == "erased" else marked
        hard = torch.from_numpy(np.asarray(m).astype(np.float32) / 255.0)[None]
        return (to_tensor(model_input), to_tensor(erased), full_prompt.strip() or render_prompt(template, text=text), hard)


class MechMaskComposite:
    """Touch-up: take the repainted image only inside the mask (grown and feathered); outside it the
    erased original stays pixel for pixel. Repainted pixels close to the background color are snapped
    to it, so the model's slightly different background tone leaves no patches."""

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
        bg = background(base)
        new = to_pil(repainted).resize(base.size, Image.LANCZOS)
        near_bg = np.abs(np.asarray(new, dtype=np.int16) - np.array(bg, dtype=np.int16)).max(axis=2) <= BG_SNAP
        snapped = np.asarray(new).copy()
        snapped[near_bg] = bg
        new = Image.fromarray(snapped)
        m = mask_to_pil(mask, base.size).point(lambda v: 255 if v > 127 else 0)
        if grow:
            m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
        if feather:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        return (to_tensor(Image.composite(new, base, m)),)


def preview_prompt(query) -> str:
    """The prompt a MechPrompt / MechMarkMask node with these field values renders (for web/mech_prompt.js)."""
    if query.get("type") == "MechMarkMask":
        return render_prompt(TOUCH_MODES[query["mode"]][0], text=query.get("text", ""))
    return render_prompt(query["template"], query.get("part", "（無）"), query.get("view", "keep"), query.get("text", ""))


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/mech/prompt")
    async def _prompt_preview(request):
        try:
            return web.Response(text=preview_prompt(request.rel_url.query))
        except (KeyError, ValueError, FileNotFoundError) as e:
            return web.Response(status=400, text=str(e))
except ImportError:  # imported outside ComfyUI
    pass

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "MechLoadMaster": MechLoadMaster,
    "MechPrompt": MechPrompt,
    "MechAposeSkeleton": MechAposeSkeleton,
    "MechPickAPose": MechPickAPose,
    "MechPartCanvas": MechPartCanvas,
    "MechImageCanvas": MechImageCanvas,
    "MechCutSegment": MechCutSegment,
    "MechTorsoCut": MechTorsoCut,
    "MechBoxes": MechBoxes,
    "MechDrawBoxes": MechDrawBoxes,
    "MechMarkMask": MechMarkMask,
    "MechMaskComposite": MechMaskComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MechLoadMaster": "機甲：載入選定的全身圖",
    "MechPrompt": "機甲：提示詞",
    "MechAposeSkeleton": "機甲：A-pose 骨架圖（姿勢參考）",
    "MechPickAPose": "機甲：A-pose 自動挑選（文字版或骨架版）",
    "MechPartCanvas": "機甲：部位畫布尺寸（依框）",
    "MechImageCanvas": "機甲：畫布尺寸（依圖片）",
    "MechCutSegment": "機甲：從整件切出細分部位",
    "MechTorsoCut": "機甲：塗除頭腿並裁出軀幹",
    "MechBoxes": "機甲：部位框（自動預框或已框好的）",
    "MechDrawBoxes": "機甲：部位框預覽",
    "MechMarkMask": "機甲：修圖遮罩標記",
    "MechMaskComposite": "機甲：只貼回遮罩範圍",
}
