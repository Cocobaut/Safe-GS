"""Step 9: export the validated mesh as collision geometry."""

import json
import os

import numpy as np
import open3d as o3d

try:
    from .roi_config import COLLISION_DIR, COLLISION_MESH, COLLISION_REPORT, VALIDATED_MESH, ensure_dir
except ImportError:
    from roi_config import COLLISION_DIR, COLLISION_MESH, COLLISION_REPORT, VALIDATED_MESH, ensure_dir


def topology_statistics(mesh):
    edge_map = {}
    for tri_id, tri in enumerate(np.asarray(mesh.triangles)):
        for edge in (tuple(sorted((int(tri[0]), int(tri[1])))), tuple(sorted((int(tri[1]), int(tri[2])))), tuple(sorted((int(tri[2]), int(tri[0]))))):
            edge_map.setdefault(edge, []).append(tri_id)
    counts = [len(incident) for incident in edge_map.values()]
    boundary = sum(count == 1 for count in counts)
    non_manifold = sum(count > 2 for count in counts)
    return {
        "edges": len(edge_map),
        "boundary_edges": boundary,
        "manifold_edges": sum(count == 2 for count in counts),
        "non_manifold_edges": non_manifold,
        "watertight": boundary == 0 and non_manifold == 0,
    }


def generate_collision_mesh():
    ensure_dir(COLLISION_DIR)
    if not os.path.exists(VALIDATED_MESH):
        raise FileNotFoundError(f"Input mesh not found: {VALIDATED_MESH}")
    mesh = o3d.io.read_triangle_mesh(VALIDATED_MESH)
    if mesh.is_empty():
        raise RuntimeError("Validated mesh is empty.")
    original_stats = {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles), **topology_statistics(mesh)}
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output_stats = {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles), **topology_statistics(mesh)}
    if not o3d.io.write_triangle_mesh(COLLISION_MESH, mesh):
        raise RuntimeError(f"Failed to save collision mesh: {COLLISION_MESH}")
    report = {
        "step": 9,
        "input_mesh": VALIDATED_MESH,
        "output_mesh": COLLISION_MESH,
        "method": "Export validated adaptive mesh as collision geometry (no additional uniform QEM).",
        "input": original_stats,
        "output": {**output_stats, "polygon_reduction_percent": 0.0},
        "valid": len(mesh.vertices) > 0 and len(mesh.triangles) > 0,
    }
    with open(COLLISION_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
    return report


def main():
    generate_collision_mesh()
    print("STEP 9 COMPLETED")


if __name__ == "__main__":
    main()