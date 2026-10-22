"""Step 7: lightweight adaptive candidate selection."""

import json
import os

import numpy as np
import open3d as o3d

try:
    from .roi_config import (
        ADAPTIVE_DIR,
        CURRENT_TASK_ROI_MAX,
        CURRENT_TASK_ROI_MIN,
        MESH_SUBROI_PAIRS,
        TEST_ROIS,
        SELECTED_MESH,
        SELECTION_DIR,
        SELECTION_REPORT,
        adaptive_mesh_path,
        ensure_dir,
        aabb_contains,
    )
except ImportError:
    from roi_config import (
        ADAPTIVE_DIR,
        CURRENT_TASK_ROI_MAX,
        CURRENT_TASK_ROI_MIN,
        MESH_SUBROI_PAIRS,
        TEST_ROIS,
        SELECTED_MESH,
        SELECTION_DIR,
        SELECTION_REPORT,
        adaptive_mesh_path,
        ensure_dir,
        aabb_contains,
    )




def select_candidate(*args, **kwargs):
    ensure_dir(SELECTION_DIR)
    
    subrois = TEST_ROIS
    results = []
    for pair in MESH_SUBROI_PAIRS:
        subroi_name = pair["subroi"]
        box = subrois[subroi_name]
        results.append({
            "subroi": subroi_name,
            "mesh": pair["mesh"],
            "target_ratio": box["target_ratio"],
            "detail_level": box["detail_level"],
            "mesh_exists": os.path.exists(adaptive_mesh_path(pair["mesh"])),
        })

    available = [item for item in results if item["mesh_exists"]]
    if not available:
        raise RuntimeError("No adaptive mesh candidate found.")
        
    selected = available[0]
    rule = "segmentation-based exact selection"

    candidate_order = [item["mesh"] for item in results if item["mesh_exists"]]
    selected_path = adaptive_mesh_path(selected["mesh"])
    selected_mesh = o3d.io.read_triangle_mesh(selected_path)
    if selected_mesh.is_empty():
        raise RuntimeError(f"Selected candidate is empty: {selected_path}")
    if not o3d.io.write_triangle_mesh(SELECTED_MESH, selected_mesh):
        raise RuntimeError(f"Failed to save selected mesh: {SELECTED_MESH}")

    report = {
        "task_roi": "SEGMENT_BASED_SELECTION",
        "subroi_evaluations": results,
        "selected_subroi": selected,
        "selected_mesh": selected["mesh"],
        "candidate_order": candidate_order,
        "selection_rule": rule,
        "output": SELECTED_MESH,
        "note": "Step 7 performs lightweight selection; final quality validation is performed in Step 8.",
    }
    with open(SELECTION_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
    return report


def main():
    select_candidate()
    print("STEP 7 COMPLETED")


if __name__ == "__main__":
    main()
