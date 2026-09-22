"""Steps 4-6: ROI mapping, global QEM candidates, and linking."""

import json
import os

import numpy as np
import open3d as o3d

try:
    from .roi_config import (
        ADAPTIVE_DIR,
        LINKING_DIR,
        LINKING_MAPPING,
        LINKING_REPORT,
        MESH_SUBROI_PAIRS,
        OUTPUT_DIR,
        QEM_REPORT,
        REPAIRED_MESH,
        ROI_LABELS,
        ROI_MAPPING_REPORT,
        ROI_MAX,
        ROI_MIN,
        SPATIAL_INFLUENCE_SIGMA,
        SUBROIS,
        TARGET_TRIANGLES,
        TRIANGLE_CENTROIDS,
        TRIANGLE_ROI_DISTANCES,
        TRIANGLE_SPATIAL_INFLUENCE,
        create_subrois,
        distance_to_aabb,
        ensure_dir,
        point_inside_aabb,
        roi_center_size,
        triangle_centroids,
    )
except ImportError:
    from roi_config import (
        ADAPTIVE_DIR,
        LINKING_DIR,
        LINKING_MAPPING,
        LINKING_REPORT,
        MESH_SUBROI_PAIRS,
        OUTPUT_DIR,
        QEM_REPORT,
        REPAIRED_MESH,
        ROI_LABELS,
        ROI_MAPPING_REPORT,
        ROI_MAX,
        ROI_MIN,
        SPATIAL_INFLUENCE_SIGMA,
        SUBROIS,
        TARGET_TRIANGLES,
        TRIANGLE_CENTROIDS,
        TRIANGLE_ROI_DISTANCES,
        TRIANGLE_SPATIAL_INFLUENCE,
        create_subrois,
        distance_to_aabb,
        ensure_dir,
        point_inside_aabb,
        roi_center_size,
        triangle_centroids,
    )


def classify_centroid(centroid, subrois):
    for name in SUBROIS:
        box = subrois[name]
        if point_inside_aabb(centroid, box["min"], box["max"]):
            return name
    return "NON_ROI"


def map_roi_subrois():
    mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)
    if mesh.is_empty():
        raise RuntimeError(f"Mesh could not be loaded: {REPAIRED_MESH}")
    subrois = create_subrois()
    centroids = triangle_centroids(mesh)
    labels = np.array([classify_centroid(c, subrois) for c in centroids], dtype="<U10")
    distances = np.array(
        [distance_to_aabb(c, ROI_MIN, ROI_MAX) for c in centroids],
        dtype=np.float64,
    )
    influence = np.exp(-distances / SPATIAL_INFLUENCE_SIGMA)
    statistics = {}
    for name in ["R0", "R1", "R2", "R3", "NON_ROI"]:
        count = int(np.sum(labels == name))
        statistics[name] = {
            "triangles": count,
            "percentage": count / len(labels) * 100.0 if len(labels) else 0.0,
        }

    ensure_dir(OUTPUT_DIR)
    np.save(ROI_LABELS, labels)
    np.save(TRIANGLE_CENTROIDS, centroids)
    np.save(TRIANGLE_ROI_DISTANCES, distances)
    np.save(TRIANGLE_SPATIAL_INFLUENCE, influence)
    center, size = roi_center_size()
    report = {
        "input_mesh": REPAIRED_MESH,
        "roi_source": "manual_aabb_standin_for_module2",
        "mesh_statistics": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)},
        "roi": {"min": ROI_MIN.tolist(), "max": ROI_MAX.tolist(), "center": center.tolist(), "size": size.tolist()},
        "subroi_definition": {
            name: {"ratio": box["ratio"], "detail_level": box["detail_level"], "min": box["min"].tolist(), "max": box["max"].tolist()}
            for name, box in subrois.items()
        },
        "classification_rule": "Triangle centroid priority: R0 > R1 > R2 > R3 > NON_ROI.",
        "spatial_influence": {"formula": "I = exp(-d / sigma)", "sigma": SPATIAL_INFLUENCE_SIGMA},
        "statistics": statistics,
        "outputs": {
            "triangle_labels": ROI_LABELS,
            "triangle_centroids": TRIANGLE_CENTROIDS,
            "triangle_roi_distances": TRIANGLE_ROI_DISTANCES,
            "triangle_spatial_influence": TRIANGLE_SPATIAL_INFLUENCE,
        },
    }
    with open(ROI_MAPPING_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
    return mesh, subrois


def mesh_statistics(mesh):
    return {"vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.triangles))}


def _clean_candidate(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def simplify_mesh(mesh, target_triangles):
    if target_triangles is None or target_triangles <= 0 or len(mesh.triangles) <= target_triangles:
        return o3d.geometry.TriangleMesh(mesh)
    return _clean_candidate(mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_triangles)))


def generate_global_qem():
    ensure_dir(ADAPTIVE_DIR)
    mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)
    if mesh.is_empty():
        raise RuntimeError(f"Input mesh is empty or cannot be loaded: {REPAIRED_MESH}")
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    original_stats = mesh_statistics(mesh)
    report = {"input": REPAIRED_MESH, "original_mesh": original_stats, "method": "Global QEM simplification", "meshes": {}}
    for mesh_name, target in TARGET_TRIANGLES.items():
        candidate = simplify_mesh(mesh, target)
        stats = mesh_statistics(candidate)
        reduction = ((original_stats["triangles"] - stats["triangles"]) / original_stats["triangles"] * 100.0) if original_stats["triangles"] else 0.0
        output_path = os.path.join(ADAPTIVE_DIR, f"{mesh_name}.ply")
        if not o3d.io.write_triangle_mesh(output_path, candidate):
            raise RuntimeError(f"Failed to save {mesh_name}")
        report["meshes"][mesh_name] = {
            "target_triangles": None if target is None else int(target),
            "actual_vertices": stats["vertices"],
            "actual_triangles": stats["triangles"],
            "triangle_reduction_percent": float(reduction),
            "output": output_path,
        }
    with open(QEM_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
    return report


def process_pair(mesh_name, subroi_name, subrois):
    mesh_path = os.path.join(ADAPTIVE_DIR, f"{mesh_name}.ply")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"Cannot load mesh: {mesh_path}")
    box = subrois[subroi_name]
    inside = np.array([point_inside_aabb(c, box["min"], box["max"]) for c in triangle_centroids(mesh)], dtype=bool)
    labels_path = os.path.join(LINKING_DIR, f"{mesh_name}_{subroi_name}_InsideSubROI.npy")
    report_path = os.path.join(LINKING_DIR, f"{mesh_name}_{subroi_name}_Report.json")
    np.save(labels_path, inside)
    report = {
        "mesh": mesh_name,
        "subroi": subroi_name,
        "pair": [mesh_name, subroi_name],
        "mesh_path": mesh_path,
        "mesh_statistics": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)},
        "subroi_definition": {"ratio": box["ratio"], "detail_level": box["detail_level"], "min": box["min"].tolist(), "max": box["max"].tolist()},
        "linking_statistics": {
            "triangles_inside_subroi": int(np.sum(inside)),
            "total_triangles": len(mesh.triangles),
            "triangle_percentage": float(np.mean(inside) * 100.0) if len(inside) else 0.0,
        },
        "label_file": labels_path,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
    return report


def link_meshes_to_subrois(subrois=None):
    ensure_dir(LINKING_DIR)
    subrois = create_subrois() if subrois is None else subrois
    reports = {}
    for pair in MESH_SUBROI_PAIRS:
        reports[f"({pair['mesh']}, {pair['subroi']})"] = process_pair(pair["mesh"], pair["subroi"], subrois)
    mapping = {
        "mesh_set": [pair["mesh"] for pair in MESH_SUBROI_PAIRS],
        "subroi_set": [pair["subroi"] for pair in MESH_SUBROI_PAIRS],
        "pairs": MESH_SUBROI_PAIRS,
        "mapping": {pair["subroi"]: pair["mesh"] for pair in MESH_SUBROI_PAIRS},
        "purpose": "Link each pre-generated adaptive mesh to its corresponding Sub-ROI. Mesh selection is performed in Step 7.",
    }
    with open(LINKING_MAPPING, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=4)
    with open(LINKING_REPORT, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=4)
    return reports


def main():
    _, subrois = map_roi_subrois()
    generate_global_qem()
    link_meshes_to_subrois(subrois)
    print("STEPS 4-6 COMPLETED")


if __name__ == "__main__":
    main()