"""
Safe-GS src package root.
Dynamically exposes submodules relocated into domain directories (Module0-4) to
maintain compatibility with existing import paths (e.g. src.module2_scene_gs, etc.).
"""
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent

# Extend __path__ so that subpackages relocated into Module directories remain importable
_DOMAIN_DIRS = [
    _SRC_DIR / "Module0-Preprocess",
    _SRC_DIR / "Module1-Perception",
    _SRC_DIR / "Module2-3DGS",
    _SRC_DIR / "Module2-3DGS" / "optimization",
    _SRC_DIR / "Module3-Mesh",
    _SRC_DIR / "Module3-Mesh" / "optimization",
    _SRC_DIR / "Module4-Physic_Engine",
    _SRC_DIR / "Module4-Physic_Engine" / "simulation",
    _SRC_DIR / "common",
]

for _d in _DOMAIN_DIRS:
    if _d.exists():
        _d_str = str(_d)
        if _d_str not in __path__:
            __path__.append(_d_str)
