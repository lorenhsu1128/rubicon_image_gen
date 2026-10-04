"""A Job is one ComfyUI run. Running it writes <timestamp>_<seed>.png plus a same-named .json
holding everything needed to run it again (`mechpipe rerun <json>`)."""
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from . import REPO_ROOT, workflow_patch
from .comfy_client import ComfyClient


def load_yaml(rel: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel).read_text(encoding="utf-8"))


def settings() -> dict:
    return load_yaml("config/settings.yaml")


def client() -> ComfyClient:
    s = settings()["comfy"]
    return ComfyClient(s["url"], s["timeout_s"])


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


@dataclass
class Job:
    mech_id: str
    stage: str
    part: str
    workflow: str                      # file name under workflows/
    seed: int
    mode: str                          # key of settings.modes
    prompt: str
    input_images: dict[str, str]       # node title -> repo-relative path
    unet: str
    loras: list[dict] = field(default_factory=list)   # [{"name", "file", "strength", "title"}]
    extra_patches: dict = field(default_factory=dict)  # other "TITLE.input" values
    family: str = "edit"                               # key of settings.models

    def mode_params(self) -> dict:
        return settings()["models"][self.family]["modes"][self.mode]

    def patches(self, uploaded: dict[str, str]) -> dict:
        mode = self.mode_params()
        p = {
            "UNET.unet_name": self.unet,
            "SAMPLER.seed": self.seed,
            "SAMPLER.steps": mode["steps"],
            "SAMPLER.cfg": mode["cfg"],
            "LORA_LIGHTNING.strength_model": mode["lightning"],
            settings()["models"][self.family]["prompt_input"]: self.prompt,
        }
        for lora in self.loras:
            p[f"{lora['title']}.lora_name"] = lora["file"]
            p[f"{lora['title']}.strength_model"] = lora["strength"]
        for title, name in uploaded.items():
            p[f"{title}.image"] = name
        p.update(self.extra_patches)
        return p

    def out_dir(self) -> Path:
        return REPO_ROOT / settings()["paths"]["runs"] / self.mech_id / self.stage / self.part

    def run(self, comfy: ComfyClient | None = None) -> Path:
        comfy = comfy or client()
        uploaded = {title: comfy.upload(REPO_ROOT / path) for title, path in self.input_images.items()}
        patches = self.patches(uploaded)
        wf = workflow_patch.patch(workflow_patch.load(REPO_ROOT / "workflows" / self.workflow), patches)

        started = time.time()
        prompt_id = comfy.queue(wf)
        entry = comfy.wait(prompt_id)
        images = comfy.output_images(entry)
        if len(images) != 1:
            raise RuntimeError(f"expected 1 output image, got {len(images)}")

        out_dir = self.out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y%m%d-%H%M%S}_{self.seed}"
        png = out_dir / f"{stem}.png"
        png.write_bytes(images[0])

        mode = self.mode_params()
        meta = {
            **asdict(self),
            "steps": mode["steps"],
            "cfg": mode["cfg"],
            "lightning_strength": mode["lightning"],
            "input_sha256": {t: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for t, p in self.input_images.items()},
            "patches": patches,
            "output": rel(png),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1),
            "comfy_prompt_id": prompt_id,
            "comfyui_version": comfy.system_stats()["system"]["comfyui_version"],
        }
        png.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return png


def from_metadata(path: Path) -> Job:
    meta = json.loads(path.read_text(encoding="utf-8"))
    return Job(**{k: meta[k] for k in Job.__dataclass_fields__ if k in meta})
