"""
Relative path helpers for Safe-GS (PLAN_REBUILD_BASE_FRAMEWORK.md, Phase 2).

Everything is derived from the location of this file, so the repo works on any drive / OS after `git pull`.
Standard library only (importable from scripts/, from src/ with cwd=src, and from src/come).

    import sys; sys.path.insert(0, str(<PROJECT_ROOT>/"src"))
    from common.paths import PROJECT_ROOT, get_run_dir

Environment overrides:
    SAFE_GS_DATA_DIR   dataset root (default <PROJECT_ROOT>/Data)
    SAFE_GS_OUTPUTS_DIR outputs root (default <PROJECT_ROOT>/outputs)
    MODAL_PROFILE      Modal profile expected by the runners (default: whatever profile is active)
A `.env` file at PROJECT_ROOT with KEY=VALUE lines is read for these keys (real env vars win).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # src/common/paths.py -> src/common -> src -> root


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _resolve(value: str | os.PathLike) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


SRC_DIR = PROJECT_ROOT / "src"
COME_DIR = SRC_DIR / "come"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_CONFIGS_DIR = CONFIGS_DIR / "datasets"
BASED_MODEL_DIR = PROJECT_ROOT / "Based_Model"
DATA_DIR = _resolve(os.environ.get("SAFE_GS_DATA_DIR", PROJECT_ROOT / "Data"))
OUTPUTS_DIR = _resolve(os.environ.get("SAFE_GS_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
MODAL_PROFILE = os.environ.get("MODAL_PROFILE") or None

DATASETS = ("LERF", "Replica", "Inpaint360GS")

# Folder names of the 16 steps, as in the reference run outputs/LERF/ours/teatime_napkin
STEP_DIRS = {
    "0": "step0_reuse_colmap",
    "1": "step1_reuse_come_3dgs",
    "2": "step2_reuse_come_renders",
    "3": "step3_reuse_napkin_reference",
    "4": "step4_renderer_check",
    "5": "step5_mask2d_hqsam",
    "6": "step6_sync_mask",
    "7": "step7_id_distill",
    "8": "step8_removal_virtual_cam_v2",
    "9": "step9_inpaint_masks_samt",
    "10": "step10_inpaint2d_lama",
    "11": "step11_inpaint_3dgs_come_loss",
    "12": "step12_render",
    "13": "step13_come_mesh",
    "14": "step14_instantmesh",
    "15": "step15_object_mesh_merge",
}
# Names used for scenes produced by the new runners (no "reuse")
NEW_STEP_DIRS = {**STEP_DIRS, "0": "step0_colmap", "1": "step1_come_3dgs", "2": "step2_come_renders",
                 "3": "step3_object_reference"}


OURS = "ours"                  # outputs/<dataset>/<method>/...: ours | CoMe | MILo | Inpaint360GS | NeRFiller


def get_method_dir(dataset: str, method: str = OURS) -> Path:
    """outputs/<dataset>/<method> (e.g. outputs/Replica/ours, outputs/Replica/CoMe)."""
    return OUTPUTS_DIR / dataset / method


def get_run_dir(dataset: str, run_name: str) -> Path:
    """outputs/<dataset>/ours/<run_name> (e.g. outputs/LERF/ours/teatime_napkin)."""
    return get_method_dir(dataset) / run_name


def step_dir(run_dir: Path, step: str | int) -> Path:
    """Existing folder of a step inside a run (reference names first, then new-runner names)."""
    key = str(step).split(".")[0]
    for table in (STEP_DIRS, NEW_STEP_DIRS):
        cand = Path(run_dir) / table[key]
        if cand.exists():
            return cand
    return Path(run_dir) / NEW_STEP_DIRS[key]


def find_plane(run_dir: Path | None = None) -> Path:
    """table_plane.json of a run: <run>/table_plane.json -> <run>/step13_come_mesh/table_plane.json
    -> src/tools_mesh/table_plane.json (teatime default)."""
    if run_dir is not None:
        for cand in (Path(run_dir) / "table_plane.json", Path(run_dir) / "step13_come_mesh" / "table_plane.json"):
            if cand.is_file():
                return cand
    return SRC_DIR / "tools_mesh" / "table_plane.json"


if __name__ == "__main__":
    for k in ("PROJECT_ROOT", "SRC_DIR", "COME_DIR", "CONFIGS_DIR", "DATA_DIR", "OUTPUTS_DIR", "MODAL_PROFILE"):
        print(f"{k:14s} {globals()[k]}")
