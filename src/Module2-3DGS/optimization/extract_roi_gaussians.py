"""
Tach Gaussian nam trong tung ROI (3D bounding box) ra file .ply rieng, de xem thu chat
luong 3DGS cua tung object da bbox.

Dung "arbitrary_3d_obb" (oriented box, xoay theo vat the - chinh xac hon AABB) de loc
diem: chuyen xyz ve he toa do cuc bo cua box (qua rotation_matrix + center) roi kiem
tra co nam trong nua-kich-thuoc (size/2) khong.

Giu nguyen TOAN BO field goc cua checkpoint GOF (bao gom filter_3D) khi ghi file con,
de file .ply sinh ra van la 1 checkpoint Gaussian hop le, load lai duoc binh thuong.

Chay:
    /home/ml4u/conda_envs/safe-gs/bin/python tmp/extract_roi_gaussians.py
"""
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

PLY_PATH = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/3dgs/office0/point_cloud/iteration_30000/point_cloud.ply"
BOXES_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json"
OUTPUT_DIR = Path("/media/ml4u/Extreme SSD/Safe-GS/tmp")
SCENE_LABEL = "Replica_office0"


def load_gaussian_ply(path):
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    return vertex, xyz


def points_in_obb(xyz, obb):
    center = np.array(obb["center"])
    sizes = np.array(obb["sizes"])
    R = np.array(obb["rotation_matrix"])  # (3,3), hang la truc cuc bo cua box trong he world

    local = (xyz - center) @ R.T  # chieu ve he toa do cuc bo cua box
    half = sizes / 2.0
    inside = np.all(np.abs(local) <= half[None, :], axis=1)
    return inside


def save_subset_ply(vertex, mask, out_path):
    subset = vertex.data[mask]
    el = PlyElement.describe(subset, "vertex")
    PlyData([el]).write(str(out_path))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Loading] Đọc checkpoint Gaussian: {PLY_PATH}")
    vertex, xyz = load_gaussian_ply(PLY_PATH)
    print(f"[Loading] Tổng số Gaussian trong scene: {len(xyz)}")

    boxes = json.loads(Path(BOXES_JSON).read_text())
    print(f"[Loading] Số object (ROI) cần tách: {len(boxes)}")

    for box in boxes:
        instance_id = box["instance_id"]
        category = box["category"]
        obb = box["arbitrary_3d_obb"]

        mask = points_in_obb(xyz, obb)
        n_found = int(mask.sum())

        out_name = f"{SCENE_LABEL}_3dgs_object_{instance_id}_{category}.ply"
        out_path = OUTPUT_DIR / out_name
        save_subset_ply(vertex, mask, out_path)

        print(f"  [{instance_id:>3}] {category:<15} - Gaussian trong ROI: {n_found:>7} -> {out_name}")

    print(f"\n[Done] Đã tách {len(boxes)} file vào: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
