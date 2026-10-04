"""Pipeline stages. S1 builds Jobs for the CLI to run; S2 chains Jobs through S2Run."""
import json
import shutil
from pathlib import Path

from PIL import Image

from . import REPO_ROOT, boxes, qc
from .crop import cut_band, cut_box, erase_boxes
from .jobs import Job, load_yaml, rel, settings

S1_VIEWS = {
    # view -> (model family, workflow, prompt template)
    "edit": ("edit", "edit_keep.api.json", "s1_edit.txt"),         # source + change text, same pose/canvas
    "apose": ("edit", "s1_master.api.json", "s1_apose.txt"),       # picked edit (+ source for detail) -> A-pose front
    "variant": ("edit", "edit_1img.api.json", "s1_variant.txt"),  # one step: change + A-pose (loses fidelity)
    "design": ("t2i", "s1_design.api.json", "s1_design.txt"),     # new design from text (Qwen-Image-2512)
    "restyle": ("edit", "s1_master.api.json", "s1_front.txt"),    # picked design in the source's style, A-pose
    "45": ("edit", "edit_keep_lora.api.json", None),            # front master turned 45 deg (camera LoRA)
}
# which master each S1 view's pick becomes
PICK_TARGET = {"edit": "edited", "apose": "front", "variant": "front", "design": "design", "restyle": "front", "45": "45"}


def load_mech(mech_id: str) -> dict:
    return load_yaml(f"config/mechs/{mech_id}.yaml")


def render_prompt(template: str, mech: dict) -> str:
    text = (REPO_ROOT / settings()["paths"]["prompts"] / template).read_text(encoding="utf-8").strip()
    color = mech.get("color_scheme") or ""
    markings = mech.get("markings") or ""
    return text.format(
        change_text=mech.get("change_text") or "",
        color_clause=f"，配色：{color}" if color else "",
        markings_clause=f"，標記：{markings}" if markings else "",
    )


def master_path(mech_id: str, view: str) -> Path:
    return REPO_ROOT / settings()["paths"]["runs"] / mech_id / "master" / f"master_{view}.png"


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{rel(path) if path.is_relative_to(REPO_ROOT) else path} missing; {hint}")
    return path


def s1_jobs(mech_id: str, view: str, seeds: list[int], mode: str) -> list[Job]:
    mech = load_mech(mech_id)
    family, workflow, template = S1_VIEWS[view]
    extra, loras = {}, []
    if view == "edit":
        inputs = {"IN_IMAGE_1": mech["source_image"]}
    elif view == "apose":
        edited = _require(master_path(mech_id, "edited"), "pick an edit result first (mechpipe pick --view edit)")
        inputs = {"IN_IMAGE_1": rel(edited), "IN_IMAGE_2": mech["source_image"]}
    elif view == "variant":
        inputs = {"IN_IMAGE_1": mech["source_image"]}
        w, h = settings()["master_size"]
        extra = {"OUT_SIZE.width": w, "OUT_SIZE.height": h}
        loras = [_lora_off()]
    elif view == "design":
        inputs = {}
        w, h = settings()["design_size"]
        extra = {"OUT_SIZE.width": w, "OUT_SIZE.height": h}
    elif view == "restyle":
        design = _require(master_path(mech_id, "design"), "pick a design first (mechpipe pick --view design)")
        inputs = {"IN_IMAGE_1": rel(design), "IN_IMAGE_2": mech["source_image"]}
    else:  # "45": the camera-angle LoRA turns the whole A-pose master to the TRELLIS.2 view
        front = _require(master_path(mech_id, "front"), "pick a front master first (mechpipe pick)")
        inputs = {"IN_IMAGE_1": rel(front)}
        loras = [_cam_lora()]
    return [
        Job(mech_id=mech_id, stage="s1", part=view, workflow=workflow, seed=seed, mode=mode,
            prompt=render_prompt(template, mech) if template else _cam_lora()["prompt"], input_images=inputs, loras=loras,
            unet=settings()["models"][family]["unet"], extra_patches=extra, family=family)
        for seed in seeds
    ]


def _lora_off() -> dict:
    """edit_1img.api.json always has a LORA_TASK node; strength 0 leaves the model untouched."""
    return {"name": "none", "title": "LORA_TASK", "file": load_yaml("config/loras.yaml")["loras"]["extract"]["file"], "strength": 0.0}


def _cam_lora() -> dict:
    cam = load_yaml("config/loras.yaml")["loras"]["cam_object"]
    return {"name": "cam_object", "title": "LORA_TASK", "file": cam["file"], "strength": cam["strength"] or 1.0,
            "prompt": cam["prompt"]}


def part_descs() -> dict[str, str]:
    return {p["id"]: p["desc"] for p in load_yaml("config/parts.yaml")["parts"]}


# S2 part hierarchy: A-pose master -> head, torso, four limbs -> segments of each.
# Head and limbs are isolated from the master by text alone. Asking for "the torso" by text also
# draws the arms, so the torso is asked for on a copy of the master with the arms removed (one removal
# edit; chaining more removals degrades the image) and head/legs painted out. Segments are cut out of
# their clean whole and completed without a reference image; redesigns and re-drawn wholes are
# caught by the QC checks and retried.
WHOLES = ("HEAD_NECK", "ARM_FULL_L", "ARM_FULL_R", "LEG_FULL_L", "LEG_FULL_R")  # isolated by text
SEGMENT_OF = {
    "SHOULDER_UPPERARM_L": "ARM_FULL_L", "FOREARM_HAND_L": "ARM_FULL_L",
    "SHOULDER_UPPERARM_R": "ARM_FULL_R", "FOREARM_HAND_R": "ARM_FULL_R",
    "THIGH_KNEE_L": "LEG_FULL_L", "KNEE_SHIN_L": "LEG_FULL_L", "ANKLE_FOOT_L": "LEG_FULL_L",
    "THIGH_KNEE_R": "LEG_FULL_R", "KNEE_SHIN_R": "LEG_FULL_R", "ANKLE_FOOT_R": "LEG_FULL_R",
    "CHEST_WAIST": "TORSO_FULL", "WAIST_HIP": "TORSO_FULL",
}
# The torso source: the master with the arms removed (model edit), then head, legs and leftover
# shoulder armour painted out geometrically (their boxes minus the torso boxes). The torso is then cut
# from it and completed like a segment; asking for "the torso" by text redraws head and legs.
TORSO_ERASE = ("HEAD_NECK", "LEG_FULL_L", "LEG_FULL_R", "ARM_FULL_L", "ARM_FULL_R")
TORSO_KEEP = ("CHEST_WAIST", "WAIST_HIP")
REMOVE_ARMS = ("刪除這台機甲的兩隻手臂，包括肩甲、上臂、前臂與手掌。肩膀處畫成乾淨的關節接座。"
               "其他部分完全不要改動：位置、大小、形狀、線條、配色與標記都和原圖相同。"
               "刪除後空出來的地方補成和原圖相同的純淺灰色背景。")


def part_canvas(bbox: list[int] | None, pixels: int) -> tuple[int, int]:
    """Output size with the part box's aspect ratio (~pixels**2 total, multiples of 16), so tall parts
    like whole legs are not cropped by a square canvas. Clamped to between 1:2 and 2:1."""
    if not bbox:
        return pixels, pixels
    aspect = min(max(bbox[2] / bbox[3], 0.5), 2.0)
    w = pixels * aspect ** 0.5
    h = pixels / aspect ** 0.5
    return int(round(w / 16)) * 16, int(round(h / 16)) * 16


def _prompt(template: str, **kw) -> str:
    return (REPO_ROOT / settings()["paths"]["prompts"] / template).read_text(encoding="utf-8").strip().format(**kw)


class S2Run:
    """Runs the S2 chain for one mech and one seed; every step is an ordinary Job with metadata.
    Output: every part in the TRELLIS.2 view (front-left 45 deg) under s2_45/.

    head, limbs (WHOLES):  45-deg master + text "draw only this part, keep the view"     [s2_45]
    their segments:        band of the clean whole, cut by the front-master box ratios (a
                           turn about the vertical axis keeps heights) -> cut completed  [s2_45]
    torso group:           master with arms removed [s2_prep], head/legs/shoulders painted
                           out by box, cut by the torso box -> completed in front view   [s2_front]
                           chest/waist: bands of that torso -> completed               [s2_front]
                           -> turned 45 deg: text prompt, else camera-angle LoRA         [s2_45]
    Turning single parts tips long ones over or leaves flat ones unturned, hence limbs come from
    the turned whole body. The torso turns reliably and cannot be cut out of the 45-deg body.
    """

    def __init__(self, mech_id: str, seed: int, mode: str, comfy, log=print):
        self.mech_id, self.seed, self.mode, self.comfy, self.log = mech_id, seed, mode, comfy, log
        self.master = _require(master_path(mech_id, "front"), "pick a front master first")
        self.master45 = _require(master_path(mech_id, "45"), "pick a 45-degree master first (s1 --view 45)")
        self.boxes = boxes.load(mech_id)
        if not self.boxes:
            raise FileNotFoundError(f"no boxes for {mech_id}; run `mechpipe boxes {mech_id}` first")
        self.notes = {k: (v or {}).get("notes", "") for k, v in (load_mech(mech_id).get("parts") or {}).items()}
        self.descs = part_descs()
        self.size = settings()["out_size"]
        self.work = REPO_ROOT / settings()["paths"]["runs"] / mech_id / "s2_work"
        self.done: dict[tuple[str, str], Path] = {}
        self._no_arms: Path | None = None
        self.failures: list[str] = []
        self.last_ok = False

    def _edit(self, stage: str, part: str, image: Path, prompt: str, canvas: tuple[int, int], validate,
              lora: dict | None = None, tries: int = 0, flag: bool = True) -> Path:
        """Run one edit step; if validate(out) reports a problem, retry with the next seeds.
        The verdict is written into the output's metadata under "qc"."""
        out = None
        self.last_ok = False
        for attempt in range(tries or TRIES):
            seed = self.seed + attempt * 1000
            job = Job(mech_id=self.mech_id, stage=stage, part=part, seed=seed, mode=self.mode, prompt=prompt,
                      workflow="edit_1img.api.json", input_images={"IN_IMAGE_1": rel(image)},
                      unet=settings()["models"]["edit"]["unet"], loras=[lora or _lora_off()],
                      extra_patches={"OUT_SIZE.width": canvas[0], "OUT_SIZE.height": canvas[1]})
            out = job.run(self.comfy)
            problems = validate(out)
            drift = qc.new_colors(self.master, out)
            if drift > NEW_COLOR_MAX:
                problems.append(f"new colors {drift:.0%}")
            meta_path = out.with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["qc"] = {"ok": not problems, "problems": problems, "attempt": attempt + 1, "chain_seed": self.seed}
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            self.log(f"  {stage}/{part} seed={seed} -> {rel(out)}" + (f"  QC: {', '.join(problems)}" if problems else "  OK"))
            if not problems:
                self.last_ok = True
                return out
        if flag:
            self.failures.append(f"{stage}/{part} (chain seed {self.seed})")
        return out

    def no_arms(self) -> Path:
        if self._no_arms is None:
            job = Job(mech_id=self.mech_id, stage="s2_prep", part="no_arms", workflow="edit_keep.api.json",
                      seed=self.seed, mode=self.mode, prompt=REMOVE_ARMS, input_images={"IN_IMAGE_1": rel(self.master)},
                      unet=settings()["models"]["edit"]["unet"])
            self._no_arms = job.run(self.comfy)
            self.log(f"  s2_prep/no_arms -> {rel(self._no_arms)}")
        return self._no_arms

    def torso_source(self) -> Path:
        return erase_boxes(self.no_arms(), [self.boxes[k] for k in TORSO_ERASE], [self.boxes[k] for k in TORSO_KEEP],
                           self.work / f"torso_source_{self.seed}.png")

    def _notes(self, part: str) -> str:
        return f"，務必保留：{self.notes[part]}" if self.notes.get(part) else ""

    def _view_clause(self, view: str) -> str:
        return settings()["s2_views"]["keep" if view == "45" else "front"]

    def _whole(self, part: str, view: str) -> Path:
        """Head or limb isolated by text from the master of that view."""
        source = self.master45 if view == "45" else self.master
        prompt = _prompt("s2_part.txt", part_desc=self.descs[part], notes_clause=self._notes(part),
                         view_clause=self._view_clause(view))
        x, y, w, h = self.boxes[part]
        # extra headroom: wholes taken from the turned body were cut at the top on a box-shaped canvas
        canvas = part_canvas([0, 0, w, int(h * (1.6 if part.startswith("LEG") else 1.35))], self.size)

        def validate(o: Path) -> list[str]:
            # user boxes are loose, so the aspect check is wide; the whole-mech check does the real work
            problems = _check_part(o, w / h, tolerance=2.0)
            # round heads match the bulky body silhouette; head isolation never drew the whole mech
            if part != "HEAD_NECK" and qc.same_view(source, o) > WHOLE_MECH_IOU:
                problems.append("whole mech")
            return problems

        return self._edit(f"s2_{view}", part, source, prompt, canvas, validate)

    def _completed(self, part: str, view: str, cut: Path, whole: Path | None) -> Path:
        w, h = Image.open(cut).size
        prompt = _prompt("s2_complete.txt", part_desc=self.descs[part], notes_clause=self._notes(part),
                         view_clause=self._view_clause(view))
        cut_aspect = qc.content_aspect(cut)

        master = self.master45 if view == "45" else self.master

        torso = part in TORSO_GROUP

        def validate(o: Path) -> list[str]:
            # torso cuts tend to be "completed" into a whole robot (arms make it wider): tight aspect
            problems = _check_part(o, cut_aspect, tolerance=TORSO_ASPECT_TOL if torso else 1.6)
            if qc.same_view(master, o) > WHOLE_MECH_IOU:
                problems.append("whole mech")
            if whole is not None and qc.same_view(whole, o) > COPIED_WHOLE_IOU:
                problems.append("redrew the whole")
            return problems

        # No reference image here: given the whole as image 2, the model redraws the whole.
        return self._edit(f"s2_{view}", part, cut, prompt, part_canvas([0, 0, w, h], self.size), validate,
                          tries=TORSO_TRIES if torso else 0)

    def part(self, part: str, view: str) -> Path:
        """`part` drawn alone in `view` ("front" for the torso group, "45" for head and limbs)."""
        key = (part, view)
        if key in self.done:
            return self.done[key]
        if part in WHOLES:
            out = self._whole(part, view)
            if view == "45" and not self.last_ok:
                # fallback (RC02 legs kept coming out cut at the top): isolate in the front view and turn
                self.failures.pop()
                out = self._turn(part, self._whole(part, "front"))
        elif part == "TORSO_FULL":
            cut = cut_box(self.torso_source(), self.boxes[part], self.work / f"{part}_{self.seed}.png")
            out = self._completed(part, view, cut, None)
        else:
            parent = SEGMENT_OF[part]
            whole = self.part(parent, view)
            pb, sb = self.boxes[parent], self.boxes[part]
            top = max(0.0, (sb[1] - pb[1]) / pb[3])
            bottom = min(1.0, (sb[1] + sb[3] - pb[1]) / pb[3])
            cut = cut_band(whole, top, bottom, self.work / f"{part}_{view}_{self.seed}.png", margin=SEGMENT_MARGIN)
            out = self._completed(part, view, cut, whole)
        self.done[key] = out
        return out

    def final(self, part: str) -> Path:
        """The TRELLIS.2 deliverable for `part` (s2_45)."""
        if part not in TORSO_GROUP:
            return self.part(part, "45")
        return self._turn(part, self.part(part, "front"))

    def _turn(self, part: str, front: Path) -> Path:
        """Front-view part -> 45 deg: text prompt first, camera-angle LoRA when the text did not turn it."""
        w, h = Image.open(front).size
        cam = _cam_lora()

        def validate(out: Path) -> list[str]:
            # a turned part keeps its aspect within ~1.5x; parts tipped over by the LoRA reach 1.8x+
            problems = _check_part(out, qc.content_aspect(front), tolerance=1.6)
            # silhouettes of round parts barely change when turned, so also require similar content
            if qc.same_view(front, out) > NOT_TURNED_IOU and qc.content_diff(front, out) < NOT_TURNED_DIFF:
                problems.append("not turned")
            return problems

        canvas = part_canvas([0, 0, w, h], self.size)
        prompt = _prompt("s2_rotate.txt", view_clause=settings()["s2_views"]["45"], notes_clause=self._notes(part))
        out = self._edit("s2_45", part, front, prompt, canvas, validate, tries=2, flag=False)
        if self.last_ok:
            return out
        return self._edit("s2_45", part, front, cam["prompt"], canvas, validate, lora=cam)


TORSO_GROUP = ("TORSO_FULL", "CHEST_WAIST", "WAIST_HIP")
TORSO_ASPECT_TOL = 1.15   # good torso completions kept ~0.93x the cut's aspect, whole robots 1.18x+
TORSO_TRIES = 5
SEGMENT_MARGIN = 0.5      # white margin around a cut segment; 0.35 left feet extending to the frame
TRIES = 3                 # seeds tried per S2 step before keeping the last result and flagging it
COPIED_WHOLE_IOU = 0.85  # silhouette IoU with its whole above which a segment is really the whole again
WHOLE_MECH_IOU = 0.72    # silhouette IoU with the master above which a "part" is really the whole mech
NEW_COLOR_MAX = 0.08      # see qc.new_colors; redesigned parts scored ~0.2, faithful ones <0.01
NOT_TURNED_IOU = 0.90     # see qc.same_view; turned compact parts score up to ~0.89, mirrored ones 0.92+
NOT_TURNED_DIFF = 24.5    # see qc.content_diff


def _check_part(png: Path, expected_aspect: float, tolerance: float = 1.6) -> list[str]:
    """Problems a script can see: background, cropping, ground shadow, and a content aspect ratio far
    from the expected one (e.g. a whole mech or two arms drawn instead of one arm)."""
    r = qc.check(png)
    problems = [k for k in ("white_bg", "not_cut", "no_shadow") if not r[k]]
    ratio = qc.content_aspect(png) / expected_aspect
    if not 1 / tolerance <= ratio <= tolerance:
        problems.append(f"aspect x{ratio:.2f}")
    return problems


def pick(mech_id: str, view: str, png: Path) -> Path:
    """Promote a chosen S1 result to runs/<mech_id>/master/master_<target>.png (+ its metadata)."""
    dest = master_path(mech_id, PICK_TARGET[view])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png, dest)
    meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    meta["picked_from"] = rel(png)
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest
