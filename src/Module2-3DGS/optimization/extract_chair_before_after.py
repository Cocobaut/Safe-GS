"""
Tach rieng Gaussian cua cai ghe (instance_id=4) tu 2 checkpoint:
  - "before": scene-gs goc (chua refine)
  - "after" : sau khi refine ROI-GS (office0_4_chair_roigs.ply)
De xem 2 ban truoc/sau ngay trong 1 viewer, so sanh chan ghe.

Chay:
    /home/ml4u/conda_envs/safe-gs/bin/python tmp/extract_chair_before_after.py
"""
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

INSTANCE_ID = 4
CATEGORY = "chair"
BEFORE_PLY = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/3dgs/office0/point_cloud/iteration_30000/point_cloud.ply"
AFTER_PLY = "/media/ml4u/Extreme SSD/Safe-GS/tmp/roi_gs_objects/office0_4_chair_roigs.ply"
BOXES_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json"
OUTPUT_DIR = Path("/media/ml4u/Extreme SSD/Safe-GS/tmp")
# Nới rộng box 1 chút để không cắt mất phần chân ghế sát biên OBB gốc
MARGIN = 1.15


def load_gaussian_ply(path):
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    return vertex, xyz


def points_in_obb(xyz, obb, margin=1.0):
    center = np.array(obb["center"])
    sizes = np.array(obb["sizes"]) * margin
    R = np.array(obb["rotation_matrix"])
    local = (xyz - center) @ R.T
    half = sizes / 2.0
    return np.all(np.abs(local) <= half[None, :], axis=1)


def save_subset_ply(vertex, mask, out_path):
    subset = vertex.data[mask]
    el = PlyElement.describe(subset, "vertex")
    PlyData([el]).write(str(out_path))
    return len(subset)


def main():
    boxes = json.loads(Path(BOXES_JSON).read_text())
    obb = next(b["arbitrary_3d_obb"] for b in boxes if b["instance_id"] == INSTANCE_ID)

    for tag, ply_path in [("before", BEFORE_PLY), ("after", AFTER_PLY)]:
        vertex, xyz = load_gaussian_ply(ply_path)
        mask = points_in_obb(xyz, obb, margin=MARGIN)
        out_path = OUTPUT_DIR / f"chair4_{tag}.ply"
        n = save_subset_ply(vertex, mask, out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"[{tag}] {n} gaussians -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
