"""
Safe-GS end-to-end entry point (PLAN_REBUILD_BASE_FRAMEWORK.md, Phase 4). Relative paths only.

Local CPU steps run here; GPU steps are launched on Modal (printed, or executed with --modal):
    0-2      scripts/modal_come_train.py    COLMAP, CoMe 3DGS, renders           (Volume safe-gs-come)
    4-12     scripts/modal_come_pipeline.py Inpaint360GS + CoMe rasterizer       (Volume inpaint360gs)
    13       scripts/modal_come_mesh.py     CoMe extract_mesh_tets.py            (Volume inpaint360gs)
    6id      local: src/tools_safe/select_target_id.py   target_id from a reference mask / click (after 4-7 download)
    13p      local: src/tools_mesh/estimate_table_plane.py  <run>/table_plane.json (support plane under the object)
    13b      local: src/tools_mesh/crop_table.py         full mesh -> table prism (table_plane.json)
    14.1     local: src/tools_instantmesh/extract_object.py
    14 / 14.2-14.5  scripts/modal_instantmesh.py InstantMesh                       (Volume instantmesh)
    15       local: src/tools_instantmesh/place_on_table.py

  python scripts/run_pipeline.py --check-env
  python scripts/run_pipeline.py --list
  python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 14.1 --object-id 44
  python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 15
  python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 4-12 --dry-run   # commands of each part
  python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 4-12 --modal     # launch on Modal
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SRC), str(SCRIPTS)]

from common.paths import CONFIGS_DIR, DATA_DIR, OUTPUTS_DIR, get_run_dir, step_dir  # noqa: E402

LOCAL_STEPS = {"6id", "11c", "13p", "13b", "14.1", "15"}
ORDER = ["0", "1", "2", "4", "5", "6", "6id", "7", "8", "9", "10", "11", "12", "11c", "13", "13p", "13b", "14.1", "14", "15"]


def _utf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------------- check-env
def check_env() -> int:
    ok = True
    print(f"python   {sys.version.split()[0]} ({sys.executable})  {platform.system()} {platform.release()}")
    print(f"root     {ROOT}")
    for name, mod, need in [("numpy", "numpy", True), ("scipy", "scipy", True), ("open3d", "open3d", True),
                            ("plyfile", "plyfile", True), ("scikit-learn", "sklearn", True), ("opencv", "cv2", True),
                            ("pillow", "PIL", True), ("trimesh", "trimesh", False), ("pyyaml", "yaml", True),
                            ("modal", "modal", True), ("torch", "torch", True)]:
        try:
            m = importlib.import_module(mod)
            print(f"  [ok]   {name:13s} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok &= not need
            print(f"  [{'MISS' if need else 'opt '}] {name:13s} {type(e).__name__}: {e}"[:120])
    try:
        import torch
        print(f"  torch cuda available: {torch.cuda.is_available()} (local steps 13b/14.1/15 run on CPU)")
    except Exception:  # noqa: BLE001
        print("  torch missing: step 14.1 (extract_object.py) needs it (CPU build is enough)")
    if shutil.which("modal") or importlib.util.find_spec("modal"):
        r = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True)
        prof = r.stdout.strip() or r.stderr.strip()[-200:]
        want = os.environ.get("MODAL_PROFILE")
        print(f"modal profile: {prof or '?'}" + (f" (MODAL_PROFILE={want})" if want else ""))
    print(f"PYTHONIOENCODING={os.environ.get('PYTHONIOENCODING', '')!r} (set utf-8 on Windows PowerShell)")
    print("structure:")
    for rel in ["src/come/train.py", "src/come/extract_mesh_tets.py", "src/come/configs/hierarchical.json",
                "src/tools_instantmesh/extract_object.py", "src/tools_mesh/crop_table.py", "src/common/paths.py",
                "configs/paths.yaml", "configs/datasets/lerf.yaml", "configs/datasets/replica.yaml",
                "configs/datasets/inpaint360gs.yaml", "scripts/modal_come_train.py", "scripts/modal_come_pipeline.py",
                "scripts/modal_come_mesh.py", "scripts/modal_instantmesh.py"]:
        e = (ROOT / rel).exists()
        ok &= e
        print(f"  [{'ok' if e else 'MISS'}]   {rel}")
    fix = "_appearance_capture" in (SRC / "come" / "train.py").read_text(encoding="utf-8", errors="replace")
    ok &= fix
    print(f"  [{'ok' if fix else 'MISS'}]   src/come/train.py checkpoint-resume fix")
    print(f"data dir    {DATA_DIR} ({'exists' if DATA_DIR.exists() else 'missing'}; override SAFE_GS_DATA_DIR)")
    runs = sorted(p.relative_to(OUTPUTS_DIR).as_posix() for d in OUTPUTS_DIR.glob("*") if d.is_dir()
                  and not d.name.startswith("_") for p in d.iterdir() if p.is_dir()) if OUTPUTS_DIR.exists() else []
    print(f"outputs dir {OUTPUTS_DIR}: {runs or 'empty (normal after git clone: outputs/ is not in git; download '}"
          + ("" if runs else "with the ::download entrypoints or modal_sync.py)"))
    print("RESULT:", "OK" if ok else "missing items above")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------- steps
def expand(spec: str) -> list[str]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in ("all", ""):
            return list(ORDER)
        if "-" in tok:
            a, b = tok.split("-")
            i, j = ORDER.index(a), ORDER.index(b)
            out += ORDER[i:j + 1]
        elif tok in ORDER or tok in ("14.2", "14.3", "14.4", "14.5"):
            out.append(tok)
        else:
            raise SystemExit(f"unknown step '{tok}' ({', '.join(ORDER)})")
    return out


def modal_cmds(step: str, cfg: dict, args) -> list[tuple[list[str], bool]] | None:
    """[(command, is_download)] of a Modal step: uploads / prep before, download of the results after (the local
    steps 6id, 13p, 13b, 14.1, 15 read them from outputs/)."""
    ds, run, scene = cfg["dataset"], cfg["run"], cfg["scene"]
    m = [sys.executable, "-m", "modal", "run"]
    det = ["--detach"] if args.detach else []
    tr = f"{SCRIPTS / 'modal_come_train.py'}"
    sc = ["--dataset", ds, "--scene", scene] + (["--volume-key", args.volume_key] if args.volume_key else [])
    if step in ("0", "1", "2"):
        ent = {"0": "colmap", "1": "train", "2": "render"}[step]
        out = [(m + [f"{tr}::upload_images"] + sc, False)] if step == "0" else []
        out.append((m + det + [f"{tr}::{ent}"] + sc, False))
        if step == "2":
            out.append((m + [f"{tr}::download"] + sc + ["--run", run], True))
        return out
    if step.startswith("4-12:"):                      # all requested parts of 4-12 in one chain on Modal
        parts = step.split(":", 1)[1]
        if parts.split(",")[0] == "4":
            parts = "prep," + parts                   # scene data / CoMe model on Volume inpaint360gs (idempotent)
        pl = f"{SCRIPTS / 'modal_come_pipeline.py'}"
        out = []
        if args.source == "local":
            out.append((m + [f"{pl}::upload_local", "--dataset", ds, "--run", run], False))
        out.append((m + det + [f"{pl}::run", "--dataset", ds, "--run", run, "--steps", parts, "--source", args.source]
                    + (["--come-key", args.volume_key] if args.volume_key else []), False))
        out.append((m + [f"{pl}::download", "--dataset", ds, "--run", run], True))
        return out
    if step == "13":
        me = f"{SCRIPTS / 'modal_come_mesh.py'}"
        src11 = []                                    # step 11c cleaned the step-11 ply locally -> upload it, redo 13
        rd = Path(cfg["run_dir"])
        c11 = rd / "step11_inpaint_3dgs_come_loss" / "step11c_clear_object_volume.json"
        if c11.exists():
            d13 = rd / "step13_come_mesh" / "done.json"
            src11 = ["--source", "local"] + (["--force"] if args.force or not d13.exists()
                                             or d13.stat().st_mtime < c11.stat().st_mtime else [])
        return [(m + det + [f"{me}::run", "--dataset", ds, "--run", run] + src11, False),
                (m + [f"{me}::download", "--dataset", ds, "--run", run] + (["--overwrite"] if src11 else []), True)]
    if step in ("14", "14.2", "14.3", "14.4", "14.5"):
        parts = "14.2,14.3,14.4" if step in ("14", "14.5") else step
        im = f"{SCRIPTS / 'modal_instantmesh.py'}"
        return [(m + det + [f"{im}::start", "--dataset", ds, "--run-name", run, "--parts", parts], False),
                (m + [f"{im}::download", "--dataset", ds, "--run-name", run], True)]
    return None


def local_cmd(step: str, cfg: dict, args) -> list[str]:
    run_dir = cfg["run_dir"]
    py = [sys.executable]
    plane = ["--plane", args.plane] if args.plane else []
    if step == "6id":
        if args.ref_masks:
            extra = ["--ref_masks", args.ref_masks] + (["--ref_id", str(args.ref_id)] if args.ref_id is not None else [])
        elif args.click:
            extra = ["--click"] + args.click
        else:
            raise SystemExit("step 6id needs --ref-masks <dir> [--ref-id N] or --click <frame> <x> <y>")
        return py + [str(SRC / "tools_safe" / "select_target_id.py"), "--run", run_dir, "--object",
                     cfg.get("object") or "object"] + extra
    if step == "11c":
        oid = args.object_id or cfg.get("target_id")
        if oid is None:
            raise SystemExit("--object-id is required (no target_id in the config)")
        return py + [str(SRC / "tools_safe" / "step11c_clear_object_volume.py"), "--run", run_dir,
                     "--object_id", str(oid)] + plane
    if step == "13p":
        oid = args.object_id or cfg.get("target_id")
        return py + [str(SRC / "tools_mesh" / "estimate_table_plane.py"), "--run", run_dir] + (["--force"] if args.force else []) +             (["--object_id", str(oid)] if oid is not None else [])
    if step == "13b":
        return py + [str(SRC / "tools_mesh" / "crop_table.py"), "--run", run_dir] + plane +             (["--force"] if args.force else [])
    if step == "14.1":
        oid = args.object_id or cfg.get("target_id")
        if oid is None:
            raise SystemExit("--object-id is required (no target_id in the config)")
        return py + [str(SRC / "tools_instantmesh" / "extract_object.py"), "--run", run_dir, "--object_id", str(oid),
                     "--max_height_frac", str(cfg.get("object_max_height_frac", 3.0))] + plane +             (["--force"] if args.force else [])
    if step == "15":
        gap = ["--gap", str(args.gap if args.gap is not None else cfg.get("gap", 0.001))]
        return py + [str(SRC / "tools_instantmesh" / "place_on_table.py"), "--run", run_dir] + gap + plane + (["--force"] if args.force else [])
    raise ValueError(step)


def dry_run_4_12(cfg: dict, parts: list[str]):
    import pipeline_steps as ps
    spec = ps.make_spec(cfg)
    sel = ",".join(p for p in parts if p in [str(i) for i in range(4, 13)])
    if sel:
        print(ps.describe(spec, ps.expand_steps(sel)))


def main():
    _utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-env", action="store_true", help="check Python packages, Modal profile, folder structure")
    ap.add_argument("--list", action="store_true", help="runs declared in configs/datasets/*.yaml")
    ap.add_argument("--dataset", default="LERF", help="LERF | Replica | Inpaint360GS")
    ap.add_argument("--run", default="teatime_napkin", help="<scene>_<object>, e.g. teatime_napkin")
    ap.add_argument("--scene", default=None, help="scene of an undeclared run (default: run minus the last _part)")
    ap.add_argument("--step", default=None, help="one step: 0..15, 13b, 14.1, 14.2 ...")
    ap.add_argument("--steps", default=None, help="range / list: 4-12, 13b,14.1, all")
    ap.add_argument("--object-id", type=int, default=None, help="step 14.1 (default: target_id of the config)")
    ap.add_argument("--plane", default=None, help="table_plane.json for 13b / 14.1 / 15")
    ap.add_argument("--gap", type=float, default=None, help="step 15 clearance")
    ap.add_argument("--ref-masks", default=None, help="step 6id: reference masks of the object (per frame)")
    ap.add_argument("--ref-id", type=int, default=None, help="step 6id: id of the object in --ref-masks")
    ap.add_argument("--click", nargs=3, default=None, metavar=("FRAME", "X", "Y"), help="step 6id: pixel on a photo")
    ap.add_argument("--source", default="volume", help="step 4 prep source for modal_come_pipeline: volume | local")
    ap.add_argument("--volume-key", default="", help="key of the scene on Volume safe-gs-come (default <dataset>/<scene>; "
                    "the teatime CoMe model trained on come-setup is at key 'teatime')")
    ap.add_argument("--modal", action="store_true", help="execute the Modal commands (default: print them)")
    ap.add_argument("--detach", action="store_true", help="modal run --detach")
    ap.add_argument("--force", action="store_true", help="local steps: redo even if the result exists (14.1)")
    ap.add_argument("--dry-run", action="store_true", help="print commands only (also for local steps)")
    args = ap.parse_args()

    if args.check_env:
        raise SystemExit(check_env())
    from common.datasets import YAML_OF, list_runs, resolve_run
    if args.list:
        for ds in YAML_OF:
            for run, scene, obj in list_runs(ds):
                rd = get_run_dir(ds, run)
                print(f"{ds:13s} {run:22s} scene={scene:12s} object={obj:8s} outputs={'yes' if rd.exists() else 'no'}")
        return
    if not (args.step or args.steps):
        ap.error("give --step / --steps (or --check-env / --list)")
    cfg = resolve_run(args.dataset, args.run, args.scene)
    for k in ("ref_masks", "plane"):                  # local tools run with cwd=src: make user paths absolute
        if getattr(args, k):
            setattr(args, k, str(Path(getattr(args, k)).resolve()))
    steps = expand(args.step or args.steps)
    if "6id" in steps and len(steps) > 1 and not (args.ref_masks or args.click):
        steps.remove("6id")                           # optional: only with --ref-masks / --click
        if cfg.get("target_id") is None and any(s_ in steps for s_ in ("8", "9", "10", "11", "12")):
            print("[note] no target_id in the YAML: run 4-7, then --step 6id --ref-masks/--click, then 8-15")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print(f"[safe-gs] {cfg['dataset']}/{cfg['run']} scene={cfg['scene']} target_id={cfg.get('target_id')} "
          f"run_dir={cfg['run_dir']}")
    if args.dry_run:
        dry_run_4_12(cfg, steps)
    grouped, run412 = [], []                          # consecutive parts of 4-12 -> one Modal chain each
    for s_ in steps + [None]:
        if s_ in [str(i) for i in range(4, 13)]:
            run412.append(s_)
            continue
        if run412:
            grouped.append("4-12:" + ",".join(run412)); run412 = []
        if s_ is not None:
            grouped.append(s_)
    steps = grouped
    if "6id" in steps and any(x.startswith("4-12:") and any(int(k) >= 8 for k in x[5:].split(","))
                              for x in steps[steps.index("6id"):]):
        print("[note] steps 8-12 need the target_id chosen at 6id: download 4-7, run 6id, put target_id in the YAML, "
              "then run --steps 8-15")
    for st in steps:
        if st in LOCAL_STEPS:
            cmd = local_cmd(st, cfg, args)
            print(f"\n[step {st}] local: {' '.join(cmd)}")
            if not args.dry_run:
                Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
                rc = subprocess.run(cmd, cwd=SRC, env=env).returncode
                if rc:
                    raise SystemExit(f"step {st} failed ({rc})")
            continue
        cmds = modal_cmds(st, cfg, args)
        if cmds is None:
            print(f"\n[step {st}] nothing to run (step 3 = reference mask, made by --step 6id --ref-masks)")
            continue
        for cmd, is_dl in cmds:
            later = is_dl and args.detach
            print(f"\n[step {st}] modal{' (download - run it when the detached job has finished)' if later else ''}: "
                  f"{' '.join(cmd)}")
            if args.modal and not args.dry_run and not later:
                rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
                if rc:
                    raise SystemExit(f"step {st} failed ({rc})")
        if args.modal and args.detach and not args.dry_run:
            print("[note] --detach: stopping after the first detached Modal step (later steps need its results)")
            return
    if not args.modal and not args.dry_run and any(s not in LOCAL_STEPS for s in steps):
        print("\n(Modal steps were only printed: add --modal [--detach] to launch them)")


if __name__ == "__main__":
    main()
