"""
Dataset / run configuration (configs/datasets/*.yaml) -> flat parameters for Step 0-15.

    from common.datasets import resolve_run
    cfg = resolve_run("LERF", "teatime_napkin")
    cfg["target_id"], cfg["scene"], cfg["data_dir"], cfg["run_dir"], ...

A run name is <scene>_<object> (declared under scenes.<scene>.objects.<object>.run in the YAML). Precedence:
defaults < scene < object (< `others` block of inpaint360gs.yaml for the "others" split).
Requires pyyaml (local only - Modal containers receive the resolved dict, not the YAML).
"""
from __future__ import annotations

from pathlib import Path

from .paths import DATA_DIR, DATASET_CONFIGS_DIR, get_run_dir

YAML_OF = {"LERF": "lerf.yaml", "Replica": "replica.yaml", "Inpaint360GS": "inpaint360gs.yaml"}
_ALIASES = {k.lower(): k for k in YAML_OF}


def canonical_dataset(name: str) -> str:
    try:
        return _ALIASES[name.lower()]
    except KeyError:
        raise SystemExit(f"unknown dataset '{name}' (choose {', '.join(YAML_OF)})") from None


def load_dataset_yaml(dataset: str) -> dict:
    import yaml
    dataset = canonical_dataset(dataset)
    with open(DATASET_CONFIGS_DIR / YAML_OF[dataset], encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def list_runs(dataset: str) -> list[tuple[str, str, str]]:
    """[(run_name, scene, object)] declared in the YAML."""
    y = load_dataset_yaml(dataset)
    out = []
    for block in ("scenes", "others"):
        for scene, sc in (y.get(block) or {}).items():
            for obj, oc in ((sc or {}).get("objects") or {}).items():
                out.append(((oc or {}).get("run", f"{scene}_{obj}"), scene, obj))
    return out


def resolve_run(dataset: str, run: str, scene: str | None = None) -> dict:
    """Flat config of one run. If `run` is not declared, `scene` (or the part of `run` before the last '_')
    gives the scene and only scene-level values are used (target_id stays None)."""
    dataset = canonical_dataset(dataset)
    y = load_dataset_yaml(dataset)
    found = None
    for block in ("scenes", "others"):
        for sc_name, sc in (y.get(block) or {}).items():
            for obj, oc in ((sc or {}).get("objects") or {}).items():
                if (oc or {}).get("run", f"{sc_name}_{obj}") == run:
                    found = (block, sc_name, obj)
    if found:
        block, scene, obj = found
    else:
        block, obj = "scenes", None
        scene = scene or (run.rsplit("_", 1)[0] if "_" in run else run)
        if scene not in (y.get("scenes") or {}) and scene in (y.get("others") or {}):
            block = "others"
    sc = dict((y.get(block) or {}).get(scene) or {})
    objects = sc.pop("objects", {}) or {}
    cfg = _merge(y.get("defaults", {}), sc)
    if obj is not None:
        cfg = _merge(cfg, objects.get(obj) or {})
    cfg.update(dataset=dataset, scene=scene, object=obj, run=run, split=block)
    cfg["data_dir"] = str(DATA_DIR / cfg.get("data_subdir", dataset) / scene)
    cfg["run_dir"] = str(get_run_dir(dataset, run))
    for k in ("target_surrounding_id",):
        if cfg.get(k) in (None, "", "null"):
            cfg[k] = "None"
    return cfg


if __name__ == "__main__":
    import json
    import sys
    # cd src && python -m common.datasets LERF teatime_napkin
    a = sys.argv[1:]
    ds, rn = (a[0] if a else "LERF"), (a[1] if len(a) > 1 else "teatime_napkin")
    print(json.dumps(resolve_run(ds, rn), indent=2, ensure_ascii=False))
