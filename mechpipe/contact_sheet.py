"""Static HTML contact sheet for review: one row per part, columns = inputs | results | metadata.
Images are referenced by relative path, so the HTML must stay inside the repo (runs/...)."""
import html
import json
import os
from pathlib import Path

from . import REPO_ROOT

CSS = """
body{font-family:system-ui,sans-serif;margin:16px;background:#f4f4f4;color:#222}
h1{font-size:20px} h2{font-size:16px;margin:0 0 8px}
.row{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin-bottom:16px}
.cols{display:grid;grid-template-columns:auto 1fr 320px;gap:16px;align-items:start}
.imgs{display:flex;flex-wrap:wrap;gap:8px}
figure{margin:0;text-align:center;font-size:12px}
figure img{width:240px;height:240px;object-fit:contain;background:#e8e8e8;border:1px solid #ccc;display:block}
.refs figure img{width:140px;height:140px}
figcaption{margin-top:4px;word-break:break-all;max-width:240px}
pre{white-space:pre-wrap;font-size:12px;background:#fafafa;border:1px solid #eee;padding:8px;margin:0}
a{color:inherit}
"""


def _fig(path: Path, sheet_dir: Path, caption: str) -> str:
    src = html.escape(Path(os.path.relpath(path, sheet_dir)).as_posix())
    return f'<figure><a href="{src}"><img src="{src}" loading="lazy"></a><figcaption>{html.escape(caption)}</figcaption></figure>'


def _qc_label(q: dict | None) -> str:
    if not q:
        return ""
    return " | QC 通過" if q["ok"] else " | QC 未過：" + "、".join(q["problems"])


def build(stage_dir: Path, title: str | None = None) -> Path:
    stage_dir = stage_dir.resolve()
    rows = []
    for part_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
        metas = [(j, json.loads(j.read_text(encoding="utf-8"))) for j in sorted(part_dir.glob("*.json"))]
        if not metas:
            continue
        inputs = {}
        for _, m in metas:
            for t, p in m["input_images"].items():
                inputs.setdefault(p, t)
        refs = "".join(_fig(REPO_ROOT / p, stage_dir, f"{t}: {Path(p).name}") for p, t in inputs.items())
        results = "".join(
            _fig(j.with_suffix(".png"), stage_dir,
                 f"{j.stem} | {m['mode']} {m['steps']}步 CFG {m['cfg']} | {m['seconds']}s"
                 + "".join(f" | {l['name']} {l['strength']}" for l in m.get("loras", []) if l["strength"])
                 + _qc_label(m.get("qc")))
            for j, m in metas
        )
        m0 = metas[-1][1]
        summary = (f"workflow: {m0['workflow']}\nunet: {m0['unet']}\n"
                   f"loras: {', '.join(l['name'] for l in m0.get('loras', [])) or '-'}\n\nprompt:\n{m0['prompt']}")
        rows.append(f'<div class="row"><h2>{html.escape(part_dir.name)}（{len(metas)} 張）</h2><div class="cols">'
                    f'<div class="imgs refs">{refs}</div><div class="imgs">{results}</div>'
                    f'<pre>{html.escape(summary)}</pre></div></div>')
    title = title or f"{stage_dir.parent.name} / {stage_dir.name}"
    out = stage_dir / "contact_sheet.html"
    out.write_text(f'<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>{html.escape(title)}</title>'
                   f'<style>{CSS}</style><h1>{html.escape(title)}</h1>{"".join(rows)}</html>', encoding="utf-8")
    return out
