"""Doc file yaml dataset trong thu muc "Dataset Configs" (nam ngoai repo, ngang
hang voi repo tren cung o dia) va tra ve duong dan da resolve.

Dung chung 1 quy uoc voi "Dataset Configs/load_dataset_config.py":
  - vars: cac bien nguoi dung sua tay (root, scene, scan, category, ...)
  - paths: duong dan mau chua {ten_bien} de the vao tu vars

Doi may/o dia: set bien moi truong DATASET_ROOT (vd DATASET_ROOT=/mnt/data) de
doi "root" ma khong can sua file yaml. Neu thu muc "Dataset Configs" khong nam
ngang hang voi repo, set them DATASET_CONFIGS_DIR tro thang toi no.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _default_configs_dir() -> Path:
    # src/common/dataset_config.py -> repo root la parents[2]; "Dataset Configs"
    # nam ngang hang voi repo (cung thu muc cha).
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "Dataset Configs"


def _derive_numeric_vars(vars_dict: dict) -> dict:
    derived = {}
    for key, value in vars_dict.items():
        if isinstance(value, str):
            match = re.search(r"(\d+)$", value)
            if match:
                derived[f"{key}_num"] = int(match.group(1))
    return derived


def _resolve(obj: Any, fmt_vars: dict) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve(v, fmt_vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, fmt_vars) for v in obj]
    if isinstance(obj, str):
        return obj.format(**fmt_vars)
    return obj


def load_dataset_paths(name: str, **overrides: str) -> dict:
    """Doc `<name>.yaml` trong Dataset Configs va tra ve dict duong dan da resolve.

    `name`: ten dataset, co hoac khong co duoi ".yaml" (vd "replica" hoac
    "replica.yaml").
    `overrides`: ghi de bien, vd scene="room0".

    Vi du: load_dataset_paths("replica", scene="office0")["images_dir"]
    """
    configs_dir = Path(os.environ.get("DATASET_CONFIGS_DIR", _default_configs_dir()))
    yaml_path = configs_dir / (name if name.endswith(".yaml") else f"{name}.yaml")

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset config: {yaml_path}\n"
            "Kiem tra lai thu muc 'Dataset Configs' co nam ngang hang voi repo "
            "khong, hoac set bien moi truong DATASET_CONFIGS_DIR."
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_vars = dict(cfg.get("vars", {}))
    cfg_vars.update(overrides)

    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        cfg_vars["root"] = env_root
    elif not Path(cfg_vars.get("root", "")).exists():
        repo_root = Path(__file__).resolve().parents[2]
        if (repo_root.parent / "Dataset Configs").exists():
            cfg_vars["root"] = str(repo_root.parent)

    fmt_vars = {**cfg_vars, **_derive_numeric_vars(cfg_vars)}
    return _resolve(cfg.get("paths", {}), fmt_vars)

