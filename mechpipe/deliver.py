"""Collect the accepted TRELLIS.2-view part images into runs/<mech_id>/deliver/ with a manifest."""
import json
import shutil
from pathlib import Path

from . import REPO_ROOT
from .jobs import rel, settings
from .stages import TORSO_GROUP, part_descs


def collect(mech_id: str, view: str = "45") -> Path:
    """For every part and chain seed, take the last attempt that passed QC (or the last attempt,
    flagged, when none did) and copy it to deliver/<mech_id>_<PART>_<chain seed>.png."""
    run = REPO_ROOT / settings()["paths"]["runs"] / mech_id
    out_dir = run / "deliver"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    manifest = {"mech_id": mech_id, "view": view, "parts": []}
    for part in part_descs():
        by_chain: dict[int, list[tuple[Path, dict]]] = {}
        for j in sorted((run / f"s2_{view}" / part).glob("*.json")):
            meta = json.loads(j.read_text(encoding="utf-8"))
            chain = (meta.get("qc") or {}).get("chain_seed", meta["seed"])
            by_chain.setdefault(chain, []).append((j, meta))
        for chain, attempts in sorted(by_chain.items()):
            passed = [a for a in attempts if (a[1].get("qc") or {}).get("ok")]
            j, meta = (passed or attempts)[-1]
            dest = out_dir / f"{mech_id}_{part}_{chain}.png"
            shutil.copyfile(j.with_suffix(".png"), dest)
            # the torso group is only right about half the time (extra limbs QC cannot see): one
            # candidate per chain is delivered and a person picks
            manifest["parts"].append({"part": part, "chain_seed": chain, "file": dest.name, "qc_ok": bool(passed),
                                      "manual_pick": part in TORSO_GROUP,
                                      "qc": meta.get("qc"), "source": rel(j.with_suffix(".png"))})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
