"""Script 06: Run Module 4 simulation (formerly Module 6) - MuJoCo Physics Simulation / Interactive Viewer / Benchmark.

Usage:
    python scripts/06_run_simulation.py [--input=office0|room0|...] [--rebuild] [--no-bg-collision] [--check] [--no-view]
                                        [--variant=unified|high|adaptive|coarse] [--benchmark]
                                        [--start=x,y,z] [--target=ROI_x] [--manual]

Example:
    python scripts/06_run_simulation.py                       # agent -> nearest ROI -> collision -> report
    python scripts/06_run_simulation.py --no-view             # same task headless (task_report.json)
    python scripts/06_run_simulation.py --manual              # old viewer, drag objects by hand
    python scripts/06_run_simulation.py --rebuild --check     # build, stability check, viewer
    python scripts/06_run_simulation.py --benchmark           # M_High / M_Adaptive / M_Coarse
    python scripts/06_run_simulation.py --input=room0         # scene of another dataset (run 05 for it first)
"""

import os
import sys
import argparse

# preprocessing/ (Module 5) and simulation/ (Module 6) live in src/Module4-Physic_Engine/; the
# folder name has a "-" so it cannot be imported as a package -> put it on sys.path directly.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(ROOT_DIR, "src", "Module4-Physic_Engine")
for _path in (ROOT_DIR, ENGINE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--input")
_args, _ = _parser.parse_known_args()
_dataset = (_args.input or os.environ.get("MODULE5_INPUT_FOLDER", "office0")).strip()
if not _dataset or any(c in _dataset for c in "/\\:") or _dataset in {".", ".."}:
    _parser.error(f"Invalid dataset folder: {_dataset!r}")
os.environ["MODULE5_INPUT_FOLDER"] = _dataset

from simulation.env_mujoco import main

if __name__ == "__main__":
    main()
