"""Collect the current TRELLIS.2-view part images into runs/<mech_id>/deliver/ with a manifest."""
import json
import shutil
from pathlib import Path

from . import REPO_ROOT
from .jobs import rel, settings
from .stages import TORSO_GROUP, part_descs

TOUCH_STAGE = "s2_touch"


def _chain(meta: dict) -> int:
    return (meta.get("qc") or {}).get("chain_seed", meta["seed"])


def current_images(mech_id: str, view: str = "45") -> list[dict]:
    """One entry per part and chain seed: the latest touch-up if there is one, else the last attempt
    of the S2 step that passed QC (or the last attempt, flagged, when none did)."""
    run = REPO_ROOT / settings()["paths"]["runs"] / mech_id
    out = []
    for part in part_descs():
        by_chain: dict[int, list[tuple[Path, dict]]] = {}
        for j in sorted((run / f"s2_{view}" / part).glob("*.json")):
            meta = json.loads(j.read_text(encoding="utf-8"))
            by_chain.setdefault(_chain(meta), []).append((j, meta))
        touches: dict[int, list[tuple[Path, dict]]] = {}
        for j in sorted((run / TOUCH_STAGE / part).glob("*.json")):
            meta = json.loads(j.read_text(encoding="utf-8"))
            touches.setdefault(_chain(meta), []).append((j, meta))
        for chain, attempts in sorted(by_chain.items()):
            passed = [a for a in attempts if (a[1].get("qc") or {}).get("ok")]
            j, meta = (passed or attempts)[-1]
            entry = {"part": part, "chain_seed": chain, "png": j.with_suffix(".png"), "meta": meta,
                     "qc_ok": bool(passed), "base": j.with_suffix(".png"), "touches": len(touches.get(chain, []))}
            if chain in touches:
                tj, tmeta = touches[chain][-1]
                entry.update(png=tj.with_suffix(".png"), meta=tmeta)
            out.append(entry)
    return out


def collect(mech_id: str, view: str = "45") -> Path:
    """Copy current_images() to deliver/<mech_id>_<PART>_<chain seed>.png."""
    out_dir = REPO_ROOT / settings()["paths"]["runs"] / mech_id / "deliver"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    manifest = {"mech_id": mech_id, "view": view, "parts": []}
    for e in current_images(mech_id, view):
        dest = out_dir / f"{mech_id}_{e['part']}_{e['chain_seed']}.png"
        shutil.copyfile(e["png"], dest)
        # the torso group is only right about half the time (extra limbs QC cannot see): one
        # candidate per chain is delivered and a person picks
        manifest["parts"].append({"part": e["part"], "chain_seed": e["chain_seed"], "file": dest.name,
                                  "qc_ok": e["qc_ok"], "manual_pick": e["part"] in TORSO_GROUP,
                                  "touch_ups": e["touches"], "qc": e["meta"].get("qc"), "source": rel(e["png"])})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
