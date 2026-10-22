"""Steps 4-6: dataset ROI mapping and ROI-aware adaptive decimation."""

import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

try:
    from .roi_config import (
        ADAPTIVE_DIR,
        adaptive_mesh_path,
        adaptive_mesh_report_path,
        LINKING_DIR,
        LINKING_MAPPING,
        LINKING_REPORT,
        MESH_SUBROI_PAIRS,
        OUTPUT_DIR,
        QEM_REPORT,
        REPAIRED_MESH,
        ROI_BOX_FILE,
        ROI_LABELS,
        ROI_MAPPING_REPORT,
        ROI_MAX,
        ROI_MIN,
        TEST_ROIS,
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
        optimized_roi_dir,
        optimized_visual_path,
        optimized_background_path,
        segment_color,
    )
except ImportError:
    from roi_config import (
        ADAPTIVE_DIR,
        adaptive_mesh_path,
        adaptive_mesh_report_path,
        LINKING_DIR,
        LINKING_MAPPING,
        LINKING_REPORT,
        MESH_SUBROI_PAIRS,
        OUTPUT_DIR,
        QEM_REPORT,
        REPAIRED_MESH,
        ROI_BOX_FILE,
        ROI_LABELS,
        ROI_MAPPING_REPORT,
        ROI_MAX,
        ROI_MIN,
        TEST_ROIS,
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
        optimized_roi_dir,
        optimized_visual_path,
        optimized_background_path,
        segment_color,
    )


SEGMENT_MATCH_TOL = 1e-3  # m, segment vertices are copies of scene-mesh vertices


def classify_centroids_with_segments(centroids, rois):
    labels = np.full(len(centroids), "NON_ROI", dtype="<U10")
    if not len(centroids):
        return labels
    
    # Check if we have segments
    has_segments = any("segment_vertices" in b for b in rois.values())
    
    if has_segments:
        # Avoid Open3D's native KDTree query crash in the current Windows env.
        kdtree = cKDTree(centroids)
        
        for name, box in rois.items():
            if "segment_vertices" in box and "segment_faces" in box and len(box["segment_faces"]) > 0:
                seg_mesh = o3d.geometry.TriangleMesh()
                seg_mesh.vertices = o3d.utility.Vector3dVector(box["segment_vertices"])
                seg_mesh.triangles = o3d.utility.Vector3iVector(box["segment_faces"])
                seg_centroids = triangle_centroids(seg_mesh)
                
                if len(seg_centroids):
                    _, indices = kdtree.query(seg_centroids, k=1)
                    labels[indices] = name
            else:
                # Fallback for this ROI if it doesn't have segments
                for i, c in enumerate(centroids):
                    if labels[i] == "NON_ROI" and point_inside_aabb(c, box["min"], box["max"]):
                        labels[i] = name
    else:
        for i, c in enumerate(centroids):
            for name, box in rois.items():
                if point_inside_aabb(c, box["min"], box["max"]):
                    labels[i] = name
                    break
    
    return labels


def map_roi_subrois():
    mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)
    if mesh.is_empty():
        raise RuntimeError(f"Mesh could not be loaded: {REPAIRED_MESH}")
    subrois = TEST_ROIS
    centroids = triangle_centroids(mesh)
    labels = classify_centroids_with_segments(centroids, subrois)
    distances = np.array(
        [distance_to_aabb(c, ROI_MIN, ROI_MAX) for c in centroids],
        dtype=np.float64,
    )
    influence = np.exp(-distances / SPATIAL_INFLUENCE_SIGMA)
    statistics = {}
    for name in [*subrois, "NON_ROI"]:
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
        "roi_source": ROI_BOX_FILE,
        "mesh_statistics": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)},
        "roi": {"min": ROI_MIN.tolist(), "max": ROI_MAX.tolist(), "center": center.tolist(), "size": size.tolist()},
        "subroi_definition": {
            name: {"target_ratio": box["target_ratio"], "detail_level": box["detail_level"], "min": box["min"].tolist(), "max": box["max"].tolist()}
            for name, box in subrois.items()
        },
        "classification_rule": "Triangle centroid is assigned to the first matching dataset AABB; otherwise NON_ROI.",
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


def topology_statistics(mesh):
    edge_counts = {}
    for triangle in np.asarray(mesh.triangles):
        edges = ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
        for v0, v1 in edges:
            edge = tuple(sorted((int(v0), int(v1))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    counts = list(edge_counts.values())
    boundary = sum(count == 1 for count in counts)
    non_manifold = sum(count > 2 for count in counts)
    return {
        "edges": len(edge_counts),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "watertight": boundary == 0 and non_manifold == 0,
        "valid": len(mesh.vertices) > 0 and len(mesh.triangles) > 0 and non_manifold == 0,
    }


def _clean_candidate(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _mesh_from_triangle_mask(mesh, mask):
    triangles = np.asarray(mesh.triangles)[mask]
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    result.triangles = o3d.utility.Vector3iVector(triangles)
    result.remove_unreferenced_vertices()
    return _clean_candidate(result)


def simplify_mesh(mesh, target_triangles):
    if target_triangles is None or target_triangles <= 0 or len(mesh.triangles) <= target_triangles:
        return o3d.geometry.TriangleMesh(mesh)
    return _clean_candidate(
        mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(target_triangles),
            boundary_weight=1000.0,
        )
    )


def _merge_preserved_and_simplified(preserved, simplified):
    vertices = np.vstack([np.asarray(preserved.vertices), np.asarray(simplified.vertices)])
    preserved_triangles = np.asarray(preserved.triangles)
    simplified_triangles = np.asarray(simplified.triangles) + len(preserved.vertices)
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(vertices)
    result.triangles = o3d.utility.Vector3iVector(
        np.vstack([preserved_triangles, simplified_triangles])
    )
    result.merge_close_vertices(1e-8)
    result.remove_non_manifold_edges()
    return _clean_candidate(result)


def generate_adaptive_mesh(mesh, roi_id, roi, target_ratio, color_kdtree=None):
    centroids = triangle_centroids(mesh)
    mesh_verts_all = np.asarray(mesh.vertices)
    
    if "segment_vertices" in roi and "segment_faces" in roi and len(roi["segment_faces"]) > 0:
        from scipy.spatial import cKDTree

        # The mask JSON keeps only one triangle per quad of the source mesh, so its faces leave a
        # checkerboard of holes. Take the object's triangles from the full scene mesh instead:
        # every triangle whose 3 vertices are segment vertices (all of them, referenced or not).
        mesh_tris = np.asarray(mesh.triangles)
        dist, _ = cKDTree(roi["segment_vertices"]).query(mesh_verts_all, k=1)
        inside = (dist < SEGMENT_MATCH_TOL)[mesh_tris].all(axis=1)
        preserved = _mesh_from_triangle_mask(mesh, inside)

        print(f"[{roi_id}] segment faces: {len(roi['segment_faces'])}, matched background hole triangles: {np.sum(inside)}")
    else:
        inside = np.array(
            [point_inside_aabb(c, roi["min"], roi["max"]) for c in centroids],
            dtype=bool,
        )
        preserved = _mesh_from_triangle_mask(mesh, inside)
        
    outside = ~inside
    
    # Extract Background (Non-ROI)
    # Extract Background (Non-ROI)
    non_roi = _mesh_from_triangle_mask(mesh, outside)
    
    # Decimate Background using QEM
    target = max(1, int(len(non_roi.triangles) * target_ratio))
    simplified = simplify_mesh(non_roi, target)
    
    # Save unified Adaptive Mesh (for testing/static scenes)
    adaptive = _merge_preserved_and_simplified(preserved, simplified)
    output_path = adaptive_mesh_path(roi_id)
    if not o3d.io.write_triangle_mesh(output_path, adaptive):
        raise RuntimeError(f"Failed to save adaptive mesh: {output_path}")
        
    # [NEW] Save Separated Assets for Dynamic Physics Engine Pipeline
    ensure_dir(optimized_roi_dir(roi_id))
    
    # Calculate object centroid and lowest vertex in world coordinates
    obj_verts = np.asarray(preserved.vertices)
    if len(obj_verts) > 0:
        centroid = obj_verts.mean(axis=0)
        min_z = float(obj_verts[:, 2].min())
    else:
        centroid = np.zeros(3)
        min_z = 0.0
        
    # Save High-Res Object Visual Mesh centered at local (0, 0, 0)
    local_visual = o3d.geometry.TriangleMesh(preserved)
    local_visual.vertices = o3d.utility.Vector3dVector(obj_verts - centroid)
    local_visual.compute_vertex_normals()
    visual_path = optimized_visual_path(roi_id)
    o3d.io.write_triangle_mesh(visual_path, local_visual, write_ascii=True)
    
    # Save QEM-Simplified Background Mesh in static world space
    bg_path = optimized_background_path(roi_id)
    o3d.io.write_triangle_mesh(bg_path, simplified, write_ascii=True)
    
    # Save asset info for collision/physics pipeline
    color = None
    if mesh.has_vertex_colors():
        color = segment_color(mesh_verts_all, np.asarray(mesh.vertex_colors), obj_verts, color_kdtree)
    info_path = os.path.join(optimized_roi_dir(roi_id), f"Object_{roi_id}_Info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "roi_id": roi_id,
            "centroid": centroid.tolist(),
            "min_z": min_z,
            "color": color,
            "visual_mesh": visual_path,
            "background_mesh": bg_path,
        }, f, indent=4)
    
    return {
        "roi_id": roi_id,
        "roi": {"min": roi["min"].tolist(), "max": roi["max"].tolist()},
        "input_triangles": int(len(mesh.triangles)),
        "roi_triangles_before": int(np.sum(inside)),
        "non_roi_triangles_before": int(np.sum(outside)),
        "non_roi_triangles_after": int(len(simplified.triangles)),
        "output_triangles": int(len(adaptive.triangles)),
        "non_roi_reduction_ratio": float(
            1.0 - len(simplified.triangles) / np.sum(outside)
        ) if np.sum(outside) else 0.0,
        "roi_preserved": True,
        "topology": topology_statistics(adaptive),
        "boundary_protection": "QEM applied only to non-ROI mesh with boundary_weight=100.0",
        "output": output_path,
        "visual_mesh": visual_path,
        "background_mesh": bg_path,
    }


def generate_adaptive_candidates():
    ensure_dir(ADAPTIVE_DIR)
    mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)
    if mesh.is_empty():
        raise RuntimeError(f"Input mesh is empty or cannot be loaded: {REPAIRED_MESH}")
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    color_kdtree = None
    if mesh.has_vertex_colors():
        color_kdtree = cKDTree(np.asarray(mesh.vertices))
    report = {
        "input": REPAIRED_MESH,
        "method": "ROI-preserving mesh plus non-ROI QEM simplification",
        "candidates": {},
    }
    
    # Only test ROI 75 and ROI 41-49 to match user's physics tests previously
    # Actually, iterate over ALL TEST_ROIS, but limit them in the config if needed
    for roi_id, roi in TEST_ROIS.items():
        print(f"[{roi_id}] Running QEM Background Decimation...")
        candidate_report = generate_adaptive_mesh(
            mesh,
            roi_id,
            roi,
            roi["target_ratio"],
            color_kdtree,
        )
        report["candidates"][roi_id] = candidate_report
        with open(adaptive_mesh_report_path(roi_id), "w", encoding="utf-8") as handle:
            json.dump(candidate_report, handle, indent=4)
            
    with open(QEM_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
        
    return report


def process_pair(mesh_name, subroi_name, subrois):
    mesh_path = adaptive_mesh_path(mesh_name)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"Cannot load mesh: {mesh_path}")
    box = subrois[subroi_name]
    
    if "segment_vertices" in box and "segment_faces" in box and len(box["segment_faces"]) > 0:
        seg_mesh = o3d.geometry.TriangleMesh()
        seg_mesh.vertices = o3d.utility.Vector3dVector(box["segment_vertices"])
        seg_mesh.triangles = o3d.utility.Vector3iVector(box["segment_faces"])
        seg_centroids = triangle_centroids(seg_mesh)
        
        mesh_cents = triangle_centroids(mesh)
        inside = np.zeros(len(mesh_cents), dtype=bool)
        if len(mesh_cents) and len(seg_centroids):
            _, indices = cKDTree(mesh_cents).query(seg_centroids, k=1)
            inside[indices] = True
    else:
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
        "subroi_definition": {"target_ratio": box["target_ratio"], "detail_level": box["detail_level"], "min": box["min"].tolist(), "max": box["max"].tolist()},
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
    subrois = TEST_ROIS if subrois is None else subrois
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
    print("STEP 4 - ROI MAPPING", flush=True)
    _, subrois = map_roi_subrois()
    print("STEP 5 - ADAPTIVE DECIMATION", flush=True)
    generate_adaptive_candidates()
    print("STEP 6 - MESH-ROI LINKING", flush=True)
    link_meshes_to_subrois(subrois)
    print("STEPS 4-6 COMPLETED")


if __name__ == "__main__":
    main()
