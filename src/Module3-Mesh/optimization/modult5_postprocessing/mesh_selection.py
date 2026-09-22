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
        MESH_LEVELS,
        MESH_SUBROI_PAIRS,
        SELECTED_MESH,
        SELECTION_DIR,
        SELECTION_REPORT,
        create_subrois,
        ensure_dir,
        aabb_contains,
    )
except ImportError:
    from roi_config import (
        ADAPTIVE_DIR,
        CURRENT_TASK_ROI_MAX,
        CURRENT_TASK_ROI_MIN,
        MESH_LEVELS,
        MESH_SUBROI_PAIRS,
        SELECTED_MESH,
        SELECTION_DIR,
        SELECTION_REPORT,
        create_subrois,
        ensure_dir,
        aabb_contains,
    )


def box_volume(box_min, box_max):
    return float(np.prod(np.maximum(np.asarray(box_max) - np.asarray(box_min), 0.0)))


def intersection_volume(min1, max1, min2, max2):
    return box_volume(np.maximum(min1, min2), np.minimum(max1, max2))


def _candidate_order(mesh_name):
    start = MESH_LEVELS.index(mesh_name)
    return list(reversed(MESH_LEVELS[: start + 1]))


def select_candidate():
    ensure_dir(SELECTION_DIR)
    task_min = np.asarray(CURRENT_TASK_ROI_MIN, dtype=np.float64)
    task_max = np.asarray(CURRENT_TASK_ROI_MAX, dtype=np.float64)
    task_volume = box_volume(task_min, task_max)
    subrois = create_subrois()
    results = []
    for pair in MESH_SUBROI_PAIRS:
        subroi_name = pair["subroi"]
        box = subrois[subroi_name]
        overlap = intersection_volume(task_min, task_max, box["min"], box["max"])
        results.append({
            "subroi": subroi_name,
            "mesh": pair["mesh"],
            "ratio": box["ratio"],
            "detail_level": box["detail_level"],
            "contains_task_roi": bool(aabb_contains(box["min"], box["max"], task_min, task_max)),
            "intersection_volume": overlap,
            "overlap_ratio": overlap / task_volume if task_volume > 0 else 0.0,
            "mesh_exists": os.path.exists(os.path.join(ADAPTIVE_DIR, f"{pair['mesh']}.ply")),
        })

    containing = [item for item in results if item["contains_task_roi"] and item["mesh_exists"]]
    if containing:
        selected = min(containing, key=lambda item: item["ratio"])
        rule = "finest available mesh whose corresponding Sub-ROI contains the task ROI"
    else:
        available = [item for item in results if item["mesh_exists"]]
        if not available:
            raise RuntimeError("No adaptive mesh candidate found.")
        selected = max(available, key=lambda item: (item["overlap_ratio"], -item["ratio"]))
        rule = "highest valid task-ROI/Sub-ROI overlap ratio, finer resolution as tie-breaker"

    candidate_order = [
        name for name in _candidate_order(selected["mesh"])
        if os.path.exists(os.path.join(ADAPTIVE_DIR, f"{name}.ply"))
    ]
    if not candidate_order:
        raise RuntimeError("Selected candidate has no available mesh file.")

    selected_path = os.path.join(ADAPTIVE_DIR, f"{selected['mesh']}.ply")
    selected_mesh = o3d.io.read_triangle_mesh(selected_path)
    if selected_mesh.is_empty():
        raise RuntimeError(f"Selected candidate is empty: {selected_path}")
    if not o3d.io.write_triangle_mesh(SELECTED_MESH, selected_mesh):
        raise RuntimeError(f"Failed to save selected mesh: {SELECTED_MESH}")

    report = {
        "task_roi": {"min": task_min.tolist(), "max": task_max.tolist(), "volume": task_volume},
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