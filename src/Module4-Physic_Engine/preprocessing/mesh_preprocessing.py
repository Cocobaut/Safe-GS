"""Steps 1-2: mesh cleaning and connected-component filtering."""

import json
import os

import numpy as np
import open3d as o3d

try:
    from .roi_config import (
        CLEANED_MESH,
        COMPONENT_LABELS,
        COMPONENT_MESH,
        COMPONENT_REPORT,
        DEGENERATE_AREA_EPS,
        OBJECT_MESH,
        OUTPUT_DIR,
        RAW_SCENE_MESH,
        ROI_MAX,
        ROI_MIN,
        SMALL_COMPONENT_AREA,
        SMALL_COMPONENT_TRIANGLES,
        VERTEX_MERGE_EPS,
        ensure_dir,
        maybe_visualize,
        point_inside_aabb,
        triangle_centroids,
    )
except ImportError:
    from roi_config import (
        CLEANED_MESH,
        COMPONENT_LABELS,
        COMPONENT_MESH,
        COMPONENT_REPORT,
        DEGENERATE_AREA_EPS,
        OBJECT_MESH,
        OUTPUT_DIR,
        RAW_SCENE_MESH,
        ROI_MAX,
        ROI_MIN,
        SMALL_COMPONENT_AREA,
        SMALL_COMPONENT_TRIANGLES,
        VERTEX_MERGE_EPS,
        ensure_dir,
        maybe_visualize,
        point_inside_aabb,
        triangle_centroids,
    )


def clean_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    finite_mask = np.isfinite(vertices).all(axis=1)
    invalid_vertices = int(np.sum(~finite_mask))
    print("Invalid vertices:", invalid_vertices)

    if invalid_vertices > 0:
        vertex_map = -np.ones(len(vertices), dtype=np.int64)
        valid_indices = np.where(finite_mask)[0]
        vertex_map[valid_indices] = np.arange(len(valid_indices))
        triangle_valid_mask = finite_mask[triangles].all(axis=1)
        triangles = vertex_map[triangles[triangle_valid_mask]]
        vertices = vertices[finite_mask]
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

    print("Merging vertices within epsilon:", VERTEX_MERGE_EPS)
    before = len(mesh.vertices)
    mesh.merge_close_vertices(VERTEX_MERGE_EPS)
    print("Duplicated / near-duplicate vertices removed:", before - len(mesh.vertices))

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) > 0:
        sorted_triangles = np.sort(triangles, axis=1)
        _, unique_indices = np.unique(sorted_triangles, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles[unique_indices])

        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        tri_vertices = vertices[triangles]
        area = 0.5 * np.linalg.norm(
            np.cross(tri_vertices[:, 1] - tri_vertices[:, 0],
                     tri_vertices[:, 2] - tri_vertices[:, 0]),
            axis=1,
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            triangles[area > DEGENERATE_AREA_EPS]
        )

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh

def load_mesh_safely(filepath):
    """Read with Open3D, explicitly triangulating the Replica quad PLY layout."""
    counts = {}
    if os.path.splitext(filepath)[1].lower() == ".ply":
        with open(filepath, "rb") as handle:
            header = []
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Incomplete PLY header: {filepath}")
                line = line.decode("ascii").strip()
                header.append(line)
                if line == "end_header":
                    break
            for line in header:
                if line.startswith("element "):
                    _, name, count = line.split()
                    counts[name] = int(count)
            # Replica exports fixed-size quads with RGB vertices and object IDs.
            # Open3D's polygon triangulator can abort on degenerate scan faces.
            layout = [line for line in header if not line.startswith(("comment ", "obj_info "))]
            expected = [
                "ply", "format binary_little_endian 1.0",
                f"element vertex {counts.get('vertex')}",
                "property float x", "property float y", "property float z",
                "property uchar red", "property uchar green", "property uchar blue",
                f"element face {counts.get('face')}",
                "property list uchar uint vertex_indices", "property ushort object_id",
                "end_header",
            ]
            if layout == expected:
                handle.seek(counts["vertex"] * 15, os.SEEK_CUR)
                quad_type = np.dtype([("count", "u1"), ("indices", "<u4", (4,)), ("object", "<u2")])
                faces = np.fromfile(handle, dtype=quad_type, count=counts["face"])
                if (len(faces) != counts["face"] or not len(faces)
                        or np.any(faces["count"] != 4) or handle.read(1)):
                    raise ValueError(f"Expected complete Replica quad PLY data: {filepath}")
                quads = faces["indices"]
                if np.any(quads >= counts["vertex"]):
                    raise ValueError(f"PLY contains out-of-range vertex indices: {filepath}")
                cloud = o3d.io.read_point_cloud(filepath)
                if len(cloud.points) != counts["vertex"] or len(cloud.colors) != counts["vertex"]:
                    raise RuntimeError(f"Open3D did not read all PLY vertices/colors: {filepath}")
                # Same quad diagonal as the previous loader; cleaning removes degenerate faces.
                triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [2, 3, 0]]))
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = cloud.points
                mesh.vertex_colors = cloud.colors
                mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(triangles, dtype=np.int32))
                print(f"Open3D: {len(mesh.vertices)} vertices, {len(quads)} quads -> {len(mesh.triangles)} triangles")
                return mesh
    mesh = o3d.io.read_triangle_mesh(filepath)
    if mesh.is_empty() or not len(mesh.triangles):
        raise RuntimeError(f"Open3D could not read mesh: {filepath}")
    if counts and (len(mesh.vertices) != counts.get("vertex")
                   or len(mesh.triangles) < counts.get("face", 0)):
        raise RuntimeError(f"Open3D returned an incomplete PLY mesh: {filepath}")
    return mesh


def clean_object_mesh():
    print("STEP 1 - MESH CLEANING")
    if not os.path.exists(RAW_SCENE_MESH):
        raise FileNotFoundError(f"No reference mesh found: {RAW_SCENE_MESH}")
    print("Loading detailed reference mesh:", RAW_SCENE_MESH)
    mesh = load_mesh_safely(RAW_SCENE_MESH)
    if mesh.is_empty():
        raise RuntimeError(f"Mesh is empty or cannot be loaded: {RAW_SCENE_MESH}")
    mesh = clean_mesh(mesh)
    if mesh.is_empty():
        raise RuntimeError("Mesh became empty after cleaning.")
    ensure_dir(OUTPUT_DIR)
    if not o3d.io.write_triangle_mesh(CLEANED_MESH, mesh):
        raise RuntimeError("Failed to save cleaned mesh.")
    maybe_visualize(mesh, "Cleaned Object Mesh")
    return mesh


def _component_keep_mask(mesh, triangle_clusters, cluster_n_triangles):
    centroids = triangle_centroids(mesh)
    keep_mask = cluster_n_triangles >= SMALL_COMPONENT_TRIANGLES
    for component_id, triangle_count in enumerate(cluster_n_triangles):
        if keep_mask[component_id]:
            continue
        component_centroids = centroids[triangle_clusters == component_id]
        keep_mask[component_id] = any(
            point_inside_aabb(centroid, ROI_MIN, ROI_MAX)
            for centroid in component_centroids
        )
    return keep_mask


def filter_connected_components():
    print("STEP 2 - CONNECTED COMPONENT FILTERING")
    mesh = o3d.io.read_triangle_mesh(CLEANED_MESH)
    if mesh.is_empty():
        raise RuntimeError(f"Cannot load mesh: {CLEANED_MESH}")

    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    keep_mask = _component_keep_mask(mesh, triangle_clusters, cluster_n_triangles)
    triangle_keep = keep_mask[triangle_clusters]

    filtered_mesh = o3d.geometry.TriangleMesh(mesh)
    filtered_mesh.remove_triangles_by_mask(~triangle_keep)
    filtered_mesh.remove_unreferenced_vertices()
    filtered_mesh.compute_vertex_normals()
    ensure_dir(OUTPUT_DIR)
    if not o3d.io.write_triangle_mesh(COMPONENT_MESH, filtered_mesh):
        raise RuntimeError("Failed to save component-filtered mesh.")
    np.save(COMPONENT_LABELS, triangle_clusters[triangle_keep])

    report = {
        "input_mesh": CLEANED_MESH,
        "output_mesh": COMPONENT_MESH,
        "triangle_threshold": SMALL_COMPONENT_TRIANGLES,
        "area_threshold_report_only": SMALL_COMPONENT_AREA,
        "small_component_rule": "Keep small components when any triangle centroid is inside the ROI.",
        "num_components": int(len(cluster_n_triangles)),
        "triangles_before": int(len(mesh.triangles)),
        "triangles_after": int(len(filtered_mesh.triangles)),
        "triangles_removed": int(len(mesh.triangles) - len(filtered_mesh.triangles)),
        "components": [
            {
                "component": int(index),
                "triangles": int(cluster_n_triangles[index]),
                "area": float(cluster_area[index]),
                "kept": bool(keep_mask[index]),
            }
            for index in range(len(cluster_n_triangles))
        ],
    }
    with open(COMPONENT_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)
    return filtered_mesh


def main():
    clean_object_mesh()
    filter_connected_components()
    print("STEPS 1-2 COMPLETED")


if __name__ == "__main__":
    main()
