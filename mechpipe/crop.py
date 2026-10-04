"""Geometric helpers for S2: find the drawn content, cut segments, pad them for completion."""
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter


def background(im: Image.Image) -> tuple[int, int, int]:
    """Median of the four corner pixels."""
    w, h = im.size
    corners = [im.getpixel(p) for p in [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]]
    return tuple(sorted(c[i] for c in corners)[2] for i in range(3))


def content_bbox(im: Image.Image, threshold: int = 30) -> tuple[int, int, int, int]:
    """Bounding box of everything that differs from the background color."""
    diff = ImageChops.difference(im, Image.new("RGB", im.size, background(im))).convert("L")
    bbox = diff.point(lambda v: 255 if v > threshold else 0).filter(ImageFilter.MedianFilter(5)).getbbox()
    if bbox is None:
        raise ValueError("image has no foreground")
    return bbox


def erase_boxes(src: Path, erase: list[list[int]], keep: list[list[int]], dest: Path) -> Path:
    """Paint every pixel that lies in an `erase` box but in no `keep` box with the background color
    (deterministic removal of neighbouring parts; the model later completes the cut edges)."""
    im = Image.open(src).convert("RGB")
    mask = Image.new("L", im.size, 0)
    for x, y, w, h in erase:
        mask.paste(255, (x, y, x + w, y + h))
    for x, y, w, h in keep:
        mask.paste(0, (x, y, x + w, y + h))
    im.paste(Image.new("RGB", im.size, background(im)), (0, 0), mask)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest)
    return dest


def _pad(seg: Image.Image, bg: tuple[int, int, int], margin: float, dest: Path) -> Path:
    # A wide margin keeps the cut edges away from the frame, so the model has room to complete them.
    m = int(max(seg.size) * margin)
    out = Image.new("RGB", (seg.width + 2 * m, seg.height + 2 * m), bg)
    out.paste(seg, (m, m))
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    return dest


def cut_band(src: Path, top: float, bottom: float, dest: Path, margin: float = 0.35) -> Path:
    """Cut rows [top, bottom] (fractions of the content height) out of an isolated part."""
    im = Image.open(src).convert("RGB")
    x0, y0, x1, y1 = content_bbox(im)
    seg = im.crop((x0, int(y0 + top * (y1 - y0)), x1, int(y0 + bottom * (y1 - y0))))
    return _pad(seg, background(im), margin, dest)


def cut_box(src: Path, bbox: list[int], dest: Path, margin: float = 0.35) -> Path:
    """Cut a [x, y, w, h] box out of an image (e.g. the master)."""
    im = Image.open(src).convert("RGB")
    x, y, w, h = bbox
    return _pad(im.crop((x, y, x + w, y + h)), background(im), margin, dest)
