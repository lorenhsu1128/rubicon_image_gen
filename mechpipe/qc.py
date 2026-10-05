"""Automatic checks for S2 part images (what a script can judge; part identity, view and design
fidelity still need a human or visual review)."""
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

from .crop import background, content_bbox


def content_aspect(png: Path) -> float:
    x0, y0, x1, y1 = content_bbox(Image.open(png).convert("RGB"))
    return (x1 - x0) / (y1 - y0)


def _silhouette(png: Path | Image.Image) -> Image.Image:
    im = (png if isinstance(png, Image.Image) else Image.open(png)).convert("RGB")
    mask = ImageChops.difference(im, Image.new("RGB", im.size, background(im))).convert("L")
    mask = mask.point(lambda v: 255 if v > 40 else 0).filter(ImageFilter.MedianFilter(5))
    return mask.crop(mask.getbbox()).resize((64, 64))


def same_view(a: Path | Image.Image, b: Path | Image.Image) -> float:
    """Silhouette IoU of two part images, also against the mirror image. A turned part scores
    below ~0.8; an unturned (or merely mirrored) one ~0.9 (calibrated on RC01, 2026-10-04)."""
    sa, sb = _silhouette(a), _silhouette(b)

    def iou(x, y):
        px, py = list(x.getdata()), list(y.getdata())
        inter = sum(1 for u, v in zip(px, py) if u and v)
        union = sum(1 for u, v in zip(px, py) if u or v)
        return inter / max(1, union)

    return max(iou(sa, sb), iou(ImageOps.mirror(sa), sb))


# A-pose pass that only copied its input: silhouette IoU with the input >= this. Copies scored
# 0.92-1.00, real re-poses 0.59-0.66 (red crouched mech, yellow mech, RC01; 2026-10-05).
APOSE_COPY_IOU = 0.85


def apose_copied(src: Path | Image.Image, out: Path | Image.Image) -> bool:
    """True when the A-pose result kept the input's pose (the text-only pass does this for 3/4 or
    crouched mechs; then the pose-skeleton pass is used instead)."""
    return same_view(src, out) >= APOSE_COPY_IOU


def content_diff(a: Path, b: Path) -> float:
    """Mean grey-level difference (0-255) of two part images scaled to the same box, also against the
    mirror image. Unturned/mirrored parts score ~21-25, turned ones ~25-70 (RC01/RC02, 2026-10-04)."""
    def norm(png: Path) -> Image.Image:
        im = Image.open(png).convert("RGB")
        return im.crop(content_bbox(im)).convert("L").resize((48, 48))

    na, nb = norm(a), norm(b)
    return min(sum(ImageChops.difference(x, nb).getdata()) / (48 * 48) for x in (na, ImageOps.mirror(na)))


def new_colors(src: Path, out: Path) -> float:
    """Share of the output's part pixels whose color is far from every color of the source part.
    Catches redesigns (e.g. gold armor appearing on a red/grey mech)."""
    def part_pixels(png: Path) -> list[tuple[int, int, int]]:
        im = Image.open(png).convert("RGB")
        im.thumbnail((160, 160))
        bg = background(im)
        return [p for p in im.getdata() if max(abs(p[i] - bg[i]) for i in range(3)) > 40]

    src_px = part_pixels(src)
    palette = Image.new("RGB", (len(src_px), 1))
    palette.putdata(src_px)
    colors = [c for _, c in palette.quantize(32).convert("RGB").getcolors(4096)]
    out_px = part_pixels(out)
    far = sum(1 for p in out_px if min(sum((p[i] - c[i]) ** 2 for i in range(3)) for c in colors) > 70 ** 2)
    return far / max(1, len(out_px))


def check(png: Path) -> dict:
    im = Image.open(png).convert("RGB")
    w, h = im.size
    gray = im.convert("L")

    # 1. white background: a 2% border band should be near-white
    band = max(4, int(min(w, h) * 0.02))
    border = [gray.crop(b) for b in [(0, 0, w, band), (0, h - band, w, h), (0, 0, band, h), (w - band, 0, w, h)]]
    border_min = min(min(b.getdata()) for b in border)
    border_mean = sum(sum(b.getdata()) / (b.width * b.height) for b in border) / 4

    # 2. not cut by the frame: content keeps a margin on every side
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).convert("L")
    mask = diff.point(lambda v: 255 if v > 40 else 0).filter(ImageFilter.MedianFilter(5))
    bbox = mask.getbbox() or (0, 0, w, h)
    margin = min(bbox[0] / w, bbox[1] / h, (w - bbox[2]) / w, (h - bbox[3]) / h)

    # 3. ground shadow: light-grey, unsaturated pixels in the band just around the content's bottom
    x0, _, x1, y1 = bbox
    sy0, sy1 = max(0, y1 - int(0.06 * h)), min(h, y1 + int(0.02 * h))
    region = im.crop((max(0, x0 - int(0.1 * w)), sy0, min(w, x1 + int(0.1 * w)), sy1))
    px = list(region.getdata())
    shadow = sum(1 for r, g, b in px if 150 < (r + g + b) / 3 < 238 and max(r, g, b) - min(r, g, b) < 14) / max(1, len(px))

    result = {
        "white_bg": border_mean > 248 and border_min > 225,
        "not_cut": margin > 0.015,
        "no_shadow": shadow < 0.04,
        "border_mean": round(border_mean, 1), "margin": round(margin, 3), "shadow": round(shadow, 3),
    }
    result["pass"] = result["white_bg"] and result["not_cut"] and result["no_shadow"]
    return result
