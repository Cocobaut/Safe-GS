"""Script 05: Run Module 4 preprocessing (formerly Module 5) Adaptive Mesh Decimation Pipeline.

Steps:
  1-2: Mesh cleaning and connected component filtering
  3:   Topology guard & boundary check
  4-6: Dataset ROI mapping and ROI-aware adaptive QEM decimation
  7:   Lightweight task-driven candidate selection
  8:   Structural & Chamfer quality validation
  9:   Collision mesh export + per-segment collision hulls

Usage:
    python scripts/05_run_postprocessing.py [--input=office0|room0|...]

Outputs go to src/Module4-Physic_Engine/preprocessing/Output/<input>/ (default input: office0).
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

from preprocessing import (
    adaptive_mesh,
    collision_mesh,
    mesh_preprocessing,
    mesh_selection,
    mesh_validation,
    topology_repair,
)


def main():
    from preprocessing.roi_config import INPUT_FOLDER_NAME, OUTPUT_DIR
    print(f"MODULE 4 PREPROCESSING PIPELINE - dataset: {INPUT_FOLDER_NAME} -> {OUTPUT_DIR}")
    mesh_preprocessing.main()
    topology_repair.main()
    adaptive_mesh.main()
    mesh_selection.main()
    mesh_validation.main()
    collision_mesh.main()
    print("\nMODULE 4 PREPROCESSING PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
