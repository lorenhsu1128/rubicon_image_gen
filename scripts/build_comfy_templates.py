"""Build the step-by-step ComfyUI templates (API format) of the mech pipeline.

Writes comfy_nodes/0_MechPipeline/templates_api/<name>.api.json. The UI-format copies in
example_workflows/ (what the template library shows) are produced from these by loading them in the
ComfyUI frontend and exporting the graph (see docs/comfy-templates.md).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "comfy_nodes" / "0_MechPipeline" / "templates_api"
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
CAM_LORA = "qwen_edit_2511_camera_angle_lora.safetensors"


class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add(self, class_type: str, title: str, **inputs) -> str:
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": class_type, "_meta": {"title": title}, "inputs": inputs}
        return nid

    def models(self, task_lora: str | None = None) -> tuple[list, list, list]:
        """GGUF UNet + Lightning (+ task LoRA), text encoder, VAE; same settings as the CLI (draft mode)."""
        unet = self.add("UnetLoaderGGUF", "UNET", unet_name="qwen-image-edit-2511-Q4_K_M.gguf")
        ms = self.add("ModelSamplingAuraFlow", "Model Sampling AuraFlow", model=[unet, 0], shift=3.1)
        norm = self.add("CFGNorm", "CFG Norm", model=[ms, 0], strength=1.0, pre_cfg=False)
        model = self.add("LoraLoaderModelOnly", "LORA_LIGHTNING", model=[norm, 0], lora_name=LIGHTNING, strength_model=1.0)
        if task_lora:
            model = self.add("LoraLoaderModelOnly", "LORA_TASK", model=[model, 0], lora_name=task_lora, strength_model=1.0)
        clip = self.add("CLIPLoader", "TEXT_ENCODER", clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
                        type="qwen_image", device="default")
        vae = self.add("VAELoader", "VAE", vae_name="qwen_image_vae.safetensors")
        return [model, 0], [clip, 0], [vae, 0]

    def edit(self, label: str, models, images: list, prompt, size=None) -> list:
        """One Qwen-Image-Edit-2511 pass. size=None edits in place on image 1's canvas (scaled to ~1MP);
        size=(w, h) draws on a new canvas of that size."""
        model, clip, vae = models
        first = images[0]
        if size is None:
            first = [self.add("FluxKontextImageScale", f"{label}：縮放", image=images[0]), 0]
        imgs = {f"image{i + 1}": img for i, img in enumerate([first] + images[1:])}
        pos = self.add("TextEncodeQwenImageEditPlus", f"{label}：提示詞", clip=clip, vae=vae, prompt=prompt, **imgs)
        neg = self.add("TextEncodeQwenImageEditPlus", f"{label}：負面", clip=clip, vae=vae, prompt="", **imgs)
        pos = self.add("FluxKontextMultiReferenceLatentMethod", f"{label}：參考方式", conditioning=[pos, 0],
                       reference_latents_method="index_timestep_zero")
        neg = self.add("FluxKontextMultiReferenceLatentMethod", f"{label}：參考方式（負）", conditioning=[neg, 0],
                       reference_latents_method="index_timestep_zero")
        if size is None:
            latent = self.add("VAEEncode", f"{label}：編碼", pixels=first, vae=vae)
        else:
            latent = self.add("EmptySD3LatentImage", f"{label}：畫布", width=size[0], height=size[1], batch_size=1)
        k = self.add("KSampler", f"{label}：取樣", model=model, positive=[pos, 0], negative=[neg, 0],
                     latent_image=[latent, 0], seed=0, steps=4, cfg=1.0, sampler_name="euler",
                     scheduler="simple", denoise=1.0)
        return [self.add("VAEDecode", f"{label}：解碼", samples=[k, 0], vae=vae), 0]

    def prompt(self, title: str, template: str, part: str = "（無）", view: str = "keep", text: str = "") -> list:
        return [self.add("MechPrompt", title, template=template, part=part, view=view, text=text), 0]

    def save(self, image, prefix: str, title: str = "輸出"):
        self.add("SaveImage", title, images=image, filename_prefix=prefix)

    def preview(self, image, title: str):
        self.add("PreviewImage", title, images=image)


def t01_edit():
    g = Graph()
    m = g.models()
    src = [g.add("LoadImage", "原始機甲圖", image="example.png"), 0]
    p = g.prompt("修改內容（在 text 填要改的地方）", "s1_edit", text="裝甲改成紅色")
    g.save(g.edit("改色", m, [src], p), "mech/01_edit")
    return g


def t02_apose():
    g = Graph()
    m = g.models()
    edited = [g.add("LoadImage", "改好的機甲圖（01 的結果）", image="example.png"), 0]
    src = [g.add("LoadImage", "原始機甲圖（補看不到的細節）", image="example.png"), 0]
    p = g.prompt("提示詞", "s1_apose")
    g.save(g.edit("轉 A-pose", m, [edited, src], p), "mech/02_apose")
    return g


def t03_turn45():
    g = Graph()
    m = g.models(task_lora=CAM_LORA)
    front = [g.add("LoadImage", "正面 A-pose 全身圖（02 的結果）", image="example.png"), 0]
    p = g.prompt("提示詞（camera-angle LoRA 原文）", "cam_lora")
    g.save(g.edit("轉 45°", m, [front], p), "mech/03_turn45")
    return g


def t04_whole():
    g = Graph()
    m = g.models()
    master = [g.add("MechLoadMaster", "45° 全身圖", mech_id="RC01", view="45"), 0]
    canvas = g.add("MechPartCanvas", "畫布尺寸（依框）", mech_id="RC01", part="ARM_FULL_R", headroom=1.35, pixels=1024)
    p = g.prompt("提示詞（選部位）", "s2_part", part="ARM_FULL_R", view="keep")
    g.save(g.edit("抽取部位", m, [master], p, size=([canvas, 0], [canvas, 1])), "mech/04_whole")
    return g


def t05_segment():
    g = Graph()
    m = g.models()
    master = [g.add("MechLoadMaster", "45° 全身圖", mech_id="RC01", view="45"), 0]
    canvas = g.add("MechPartCanvas", "整件畫布（依框）", mech_id="RC01", part="LEG_FULL_R", headroom=1.6, pixels=1024)
    pw = g.prompt("整件提示詞（選整件部位）", "s2_part", part="LEG_FULL_R", view="keep")
    whole = g.edit("抽取整件", m, [master], pw, size=([canvas, 0], [canvas, 1]))
    g.preview(whole, "整件預覽")
    cut = [g.add("MechCutSegment", "依框切出細分部位", whole=whole, mech_id="RC01", segment="THIGH_KNEE_R", margin=0.5), 0]
    g.preview(cut, "切段預覽")
    cc = g.add("MechImageCanvas", "畫布尺寸（依切段）", image=cut, pixels=1024)
    ps = g.prompt("補完提示詞（選細分部位）", "s2_complete", part="THIGH_KNEE_R", view="keep")
    g.save(g.edit("補完切口", m, [cut], ps, size=([cc, 0], [cc, 1])), "mech/05_segment")
    return g


def t06_torso():
    g = Graph()
    m = g.models()
    master = [g.add("MechLoadMaster", "正面 A-pose 全身圖", mech_id="RC01", view="front"), 0]
    no_arms = g.edit("刪除雙臂", m, [master], g.prompt("刪除雙臂提示詞", "s2_remove_arms"))
    g.preview(no_arms, "刪除雙臂預覽")
    cut = [g.add("MechTorsoCut", "塗除頭腿並裁出軀幹", no_arms=no_arms, mech_id="RC01", margin=0.35), 0]
    g.preview(cut, "裁切預覽")
    cc = g.add("MechImageCanvas", "畫布尺寸（依裁切）", image=cut, pixels=1024)
    torso = g.edit("補完軀幹", m, [cut], g.prompt("補完提示詞", "s2_complete", part="TORSO_FULL", view="front"),
                   size=([cc, 0], [cc, 1]))
    g.save(torso, "mech/06_torso_front", "正面軀幹")
    tc = g.add("MechImageCanvas", "畫布尺寸（軀幹）", image=torso, pixels=1024)
    g.save(g.edit("轉 45°", m, [torso], g.prompt("轉角度提示詞", "s2_rotate", view="45"), size=([tc, 0], [tc, 1])),
           "mech/06_torso_45", "45° 軀幹")
    return g


def t07_rotate():
    g = Graph()
    text_models = g.models()
    part = [g.add("LoadImage", "正面部件圖", image="example.png"), 0]
    cc = g.add("MechImageCanvas", "畫布尺寸", image=part, pixels=1024)
    size = ([cc, 0], [cc, 1])
    g.save(g.edit("文字轉 45°", text_models, [part], g.prompt("文字提示詞", "s2_rotate", view="45"), size=size),
           "mech/07_rotate_text", "45°（文字）")
    cam_models = g.models(task_lora=CAM_LORA)
    g.save(g.edit("LoRA 轉 45°", cam_models, [part], g.prompt("LoRA 提示詞", "cam_lora"), size=size),
           "mech/07_rotate_lora", "45°（camera LoRA）")
    return g


def t08_touchup():
    g = Graph()
    m = g.models()
    load = g.add("LoadImage", "要修的圖（右鍵 → 遮罩編輯器塗出範圍）", image="example.png")
    marks = g.add("MechMarkMask", "遮罩標記", image=[load, 0], mask=[load, 1])
    g.save([marks, 1], "mech/08_erase", "只擦除")
    p = g.prompt("補畫提示（text 可寫要求，例如：補成肩膀的圓形關節截面）", "s2_fill", text="")
    repainted = g.edit("補畫", m, [[marks, 0]], p)
    out = g.add("MechMaskComposite", "只貼回遮罩範圍", erased=[marks, 1], repainted=repainted, mask=[load, 1],
                grow=6, feather=4)
    g.save([out, 0], "mech/08_repaint", "擦除並補畫")
    return g


TEMPLATES = {
    "01_改色或修改（姿勢不動）": t01_edit,
    "02_轉成正面A-pose": t02_apose,
    "03_全身轉45度": t03_turn45,
    "04_抽取頭或四肢（45度）": t04_whole,
    "05_細分部位（整件切段補完）": t05_segment,
    "06_軀幹（刪臂裁切補完轉45度）": t06_torso,
    "07_單一部件轉45度": t07_rotate,
    "08_修圖（擦除與補畫）": t08_touchup,
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in TEMPLATES.items():
        (OUT / f"{name}.api.json").write_text(json.dumps(build().nodes, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name)
