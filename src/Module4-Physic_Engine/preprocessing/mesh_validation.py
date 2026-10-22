"""Step 8: structural and ROI-Chamfer mesh validation."""

import json
import os

import numpy as np
import open3d as o3d

try:
    from .roi_config import (
        ADAPTIVE_DIR,
        adaptive_mesh_path,
        REPAIRED_MESH,
        CHAMFER_SAMPLES,
        CURRENT_TASK_ROI_MAX,
        CURRENT_TASK_ROI_MIN,
        DEGENERATE_AREA_EPS,
        SELECTION_REPORT,
        SELECTED_MESH,
        TAU_ROI,
        VALIDATED_MESH,
        VALIDATION_DIR,
        VALIDATION_REPORT,
        ensure_dir,
        point_inside_aabb,
        TEST_ROIS,
        triangle_centroids,
    )
except ImportError:
    from roi_config import (
        ADAPTIVE_DIR,
        adaptive_mesh_path,
        REPAIRED_MESH,
        CHAMFER_SAMPLES,
        CURRENT_TASK_ROI_MAX,
        CURRENT_TASK_ROI_MIN,
        DEGENERATE_AREA_EPS,
        SELECTION_REPORT,
        SELECTED_MESH,
        TAU_ROI,
        VALIDATED_MESH,
        VALIDATION_DIR,
        VALIDATION_REPORT,
        ensure_dir,
        point_inside_aabb,
        TEST_ROIS,
        triangle_centroids,
    )


def check_mesh_structure(mesh):
    if mesh.is_empty() or len(mesh.triangles) == 0:
        return False, "EMPTY_MESH"
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not np.all(np.isfinite(vertices)):
        return False, "NON_FINITE_VERTICES"
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        return False, "INVALID_TRIANGLE_INDEX"
    tri_vertices = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]),
        axis=1,
    )
    degenerate_ratio = float(np.mean(areas <= DEGENERATE_AREA_EPS))
    if degenerate_ratio > 0.05:
        return False, "TOO_MANY_DEGENERATE_TRIANGLES"
    return True, "VALID"


def crop_to_roi(mesh, roi):
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return o3d.geometry.TriangleMesh()
        
    centroids = triangle_centroids(mesh)
    
    if "segment_vertices" in roi and "segment_faces" in roi and len(roi["segment_faces"]) > 0:
        from scipy.spatial import cKDTree
        mesh_verts_all = np.asarray(mesh.vertices)
        dist, _ = cKDTree(roi["segment_vertices"]).query(mesh_verts_all, k=1)
        inside = (dist < 1e-3)[triangles].all(axis=1)
    else:
        inside = np.array([
            point_inside_aabb(centroid, roi["min"], roi["max"])
            for centroid in centroids
        ], dtype=bool)
        
    if not np.any(inside):
        return o3d.geometry.TriangleMesh()
        
    roi_mesh = o3d.geometry.TriangleMesh()
    roi_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    roi_mesh.triangles = o3d.utility.Vector3iVector(triangles[inside])
    roi_mesh.remove_unreferenced_vertices()
    return roi_mesh


def calculate_chamfer(mesh1, mesh2):
    pcd1 = mesh1.sample_points_uniformly(number_of_points=CHAMFER_SAMPLES)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=CHAMFER_SAMPLES)
    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        return None
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    return float((np.mean(d1) + np.mean(d2)) / 2.0)


def validate_candidates():
    ensure_dir(VALIDATION_DIR)
    with open(SELECTION_REPORT, "r", encoding="utf-8") as handle:
        selection = json.load(handle)
    candidates = selection["candidate_order"]
    reference_path = REPAIRED_MESH
    reference_mesh = o3d.io.read_triangle_mesh(reference_path)
    reference_valid, reference_status = check_mesh_structure(reference_mesh)
    if not reference_valid:
        raise RuntimeError(f"M0 reference mesh is invalid: {reference_status}")
    results = []
    final_mesh = None
    final_result = None
    for mesh_name in candidates:
        print(f"Testing: {mesh_name}")
        mesh = o3d.io.read_triangle_mesh(adaptive_mesh_path(mesh_name))
        valid, status = check_mesh_structure(mesh)
        print(f"Structure: {status}")
        result = {"mesh": mesh_name, "structurally_valid": valid, "validation_status": status, "passed": False}
        if valid:
            roi = TEST_ROIS[mesh_name]
            reference_roi = crop_to_roi(reference_mesh, roi)
            candidate_roi = crop_to_roi(mesh, roi)
            if candidate_roi.is_empty() or reference_roi.is_empty():
                result["validation_status"] = "INVALID / FAIL: Empty crop generated. No geometric data for validation."
                result["chamfer"] = None
                result["threshold"] = float(TAU_ROI)
                result["roi"] = {"min": roi["min"].tolist(), "max": roi["max"].tolist()}
                result["passed"] = False
                print(f"ROI Chamfer: FAIL (Empty crop)")
            else:
                chamfer = calculate_chamfer(candidate_roi, reference_roi)
                result["chamfer"] = None if chamfer is None else float(chamfer)
                result["threshold"] = float(TAU_ROI)
                result["roi"] = {"min": roi["min"].tolist(), "max": roi["max"].tolist()}
                result["passed"] = chamfer is not None and chamfer <= TAU_ROI
                print(f"ROI Chamfer: {chamfer}")
        results.append(result)
        print("PASS" if result["passed"] else "FAIL")
        if result["passed"]:
            final_mesh = mesh
            final_result = result
            break
    if final_mesh is None:
        raise RuntimeError("No candidate mesh passed validation.")
    final_mesh.remove_unreferenced_vertices()
    final_mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(VALIDATED_MESH, final_mesh):
        raise RuntimeError(f"Failed to save validated mesh: {VALIDATED_MESH}")
    report = {
        "candidate_order": candidates,
        "selected_mesh": final_result["mesh"],
        "validation_results": results,
        "reference_mesh": REPAIRED_MESH,
        "chamfer_threshold": float(TAU_ROI),
        "output": VALIDATED_MESH,
    }
    with open(VALIDATION_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
    return report


def main():
    validate_candidates()
    print("STEP 8 COMPLETED")


if __name__ == "__main__":
    main()
