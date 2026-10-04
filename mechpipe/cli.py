import argparse
import random
from pathlib import Path

from . import boxes, contact_sheet, deliver, stages
from .jobs import client, from_metadata, rel

MODES = ["draft", "final"]   # same names in every model family of config/settings.yaml


def _seeds(args) -> list[int]:
    if args.seed:
        return args.seed
    return [random.randrange(2**32) for _ in range(args.count)]


def cmd_s1(args):
    jobs = stages.s1_jobs(args.mech_id, args.view, _seeds(args), args.mode)
    comfy = client()
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] s1 {args.view} seed={job.seed} mode={job.mode} ...", flush=True)
        print("  ->", rel(job.run(comfy)), flush=True)
    sheet = contact_sheet.build(jobs[0].out_dir().parent)
    print("contact sheet:", rel(sheet))


def cmd_s2(args):
    parts = args.part or list(stages.part_descs())
    comfy = client()
    run_dir, failures = None, []
    for seed in _seeds(args):
        run = stages.S2Run(args.mech_id, seed, args.mode, comfy)
        for i, part in enumerate(parts, 1):
            print(f"[seed {seed}] {i}/{len(parts)} {part}", flush=True)
            out = run.final(part)
            run_dir = out.parent.parent.parent
        failures += run.failures
    print("QC failures after retries:", ", ".join(failures) if failures else "none")
    for stage in ("s2_front", "s2_45"):
        if (run_dir / stage).exists():
            print("contact sheet:", rel(contact_sheet.build(run_dir / stage)))


def cmd_boxes(args):
    master = stages.master_path(args.mech_id, "front")
    if not master.exists():
        raise SystemExit(f"{rel(master)} missing; pick a front master first")
    boxes.serve(args.mech_id, master, stages.part_descs(), args.port)


def cmd_deliver(args):
    out = deliver.collect(args.mech_id, args.view)
    print("deliver:", rel(out))


def cmd_pick(args):
    print("master:", rel(stages.pick(args.mech_id, args.view, Path(args.png))))


def cmd_rerun(args):
    job = from_metadata(Path(args.json))
    if args.mode:
        job.mode = args.mode
    print(f"rerun {job.stage}/{job.part} seed={job.seed} mode={job.mode} ...", flush=True)
    print("  ->", rel(job.run()))


def cmd_sheet(args):
    print("contact sheet:", rel(contact_sheet.build(Path(args.stage_dir))))


def main():
    p = argparse.ArgumentParser(prog="mechpipe")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("s1", help="S1: variant (2511 edit) | design (2512) -> restyle (2511), then 34")
    s.add_argument("mech_id")
    s.add_argument("--view", choices=list(stages.S1_VIEWS), default="variant")
    s.add_argument("--count", type=int, default=4, help="number of random seeds")
    s.add_argument("--seed", type=int, action="append", help="explicit seed (repeatable)")
    s.add_argument("--mode", choices=MODES, default="draft")
    s.set_defaults(func=cmd_s1)

    s = sub.add_parser("s2", help="S2: every part alone in the TRELLIS.2 view (needs master_front, master_45 and boxes)")
    s.add_argument("mech_id")
    s.add_argument("--part", action="append", help="part id from config/parts.yaml (repeatable; default all)")
    s.add_argument("--count", type=int, default=2, help="number of random seeds; each seed runs its own chain")
    s.add_argument("--seed", type=int, action="append", help="explicit seed (repeatable)")
    s.add_argument("--mode", choices=MODES, default="draft")
    s.set_defaults(func=cmd_s2)

    s = sub.add_parser("boxes", help="pre-box the parts on master_front and adjust them in the browser")
    s.add_argument("mech_id")
    s.add_argument("--port", type=int, default=8199)
    s.set_defaults(func=cmd_boxes)

    s = sub.add_parser("deliver", help="copy the accepted part images to runs/<mech_id>/deliver/ + manifest.json")
    s.add_argument("mech_id")
    s.add_argument("--view", choices=["45", "front"], default="45")
    s.set_defaults(func=cmd_deliver)

    s = sub.add_parser("pick",help="promote an S1 result to runs/<mech_id>/master/master_<view>.png")
    s.add_argument("mech_id")
    s.add_argument("png")
    s.add_argument("--view", choices=list(stages.S1_VIEWS), required=True, help="S1 view the png came from")
    s.set_defaults(func=cmd_pick)

    s = sub.add_parser("rerun", help="run a job again from its metadata .json")
    s.add_argument("json")
    s.add_argument("--mode", choices=MODES, help="override mode, e.g. final")
    s.set_defaults(func=cmd_rerun)

    s = sub.add_parser("sheet", help="(re)build the contact sheet of a stage dir")
    s.add_argument("stage_dir")
    s.set_defaults(func=cmd_sheet)

    args = p.parse_args()
    args.func(args)
