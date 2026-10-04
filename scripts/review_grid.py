"""Review grid for S2: one row per part, [front | 45] of the accepted attempt of a chain seed, QC flags.
Usage: python scripts/review_grid.py <mech_id> <chain_seed> [part ...] -> runs/<mech_id>/_review_<seed>.png"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from mechpipe import REPO_ROOT
from mechpipe.stages import part_descs

mech, chain = sys.argv[1], int(sys.argv[2])
run = REPO_ROOT / "runs" / mech


def accepted(stage: str, part: str) -> tuple[Path, dict] | None:
    attempts = []
    for j in sorted((run / stage / part).glob("*.json")):
        meta = json.loads(j.read_text(encoding="utf-8"))
        if (meta.get("qc") or {}).get("chain_seed", meta["seed"]) == chain:
            attempts.append((j, meta))
    if not attempts:
        return None
    passed = [a for a in attempts if (a[1].get("qc") or {}).get("ok")]
    j, meta = (passed or attempts)[-1]
    return j.with_suffix(".png"), meta


parts = sys.argv[3:] or list(part_descs())
S = 230
g = Image.new("RGB", (170 + S * 2, S * len(parts)), "white")
d = ImageDraw.Draw(g)
for r, part in enumerate(parts):
    d.text((4, r * S + 4), part, fill="black")
    for c, stage in enumerate(["s2_front", "s2_45"]):
        hit = accepted(stage, part)
        if not hit:
            continue
        png, meta = hit
        im = Image.open(png).convert("RGB")
        im.thumbnail((S - 8, S - 22))
        g.paste(im, (170 + c * S + 4, r * S + 18))
        q = meta.get("qc") or {}
        label = f"{stage[3:]} " + ("OK" if q.get("ok") else "QC!" + ",".join(q.get("problems", [])))
        d.text((170 + c * S + 4, r * S + 4), label, fill="black" if q.get("ok") else "red")
out = run / f"_review_{chain}.png"
g.save(out)
print(out)
