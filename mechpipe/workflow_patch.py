"""Patch ComfyUI API-format workflows by node title (`_meta.title`) and input name, never by node id.

Patch keys are "TITLE.input", e.g. {"SAMPLER.seed": 42, "IN_IMAGE_1.image": "abc.png"}.
"""
import copy
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def node_by_title(workflow: dict, title: str) -> dict:
    hits = [n for n in workflow.values() if n.get("_meta", {}).get("title") == title]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one node titled {title!r}, found {len(hits)}")
    return hits[0]


def patch(workflow: dict, patches: dict) -> dict:
    out = copy.deepcopy(workflow)
    for key, value in patches.items():
        title, field = key.split(".", 1)
        node = node_by_title(out, title)
        if field not in node["inputs"]:
            raise KeyError(f"node {title!r} ({node['class_type']}) has no input {field!r}")
        if isinstance(node["inputs"][field], list):
            raise KeyError(f"{key} is a link to another node, not a value")
        node["inputs"][field] = value
    return out
