import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roi_config import (
    OBJECT_MESH,
    OUTPUT_DIR,
    RAW_SCENE_MESH,
    ROI_MAX,
    ROI_MIN,
    SCENE_MESH,
    ensure_dir,
    maybe_visualize,
)


def main():
    print("=" * 60)
    print("CROP OBJECT / SCENE BY ROI AABB")
    print("=" * 60)
    print("Loading:", RAW_SCENE_MESH)

    mesh = o3d.io.read_triangle_mesh(RAW_SCENE_MESH)
    if mesh.is_empty():
        raise RuntimeError("Cannot load mesh.")
    mesh.compute_vertex_normals()

    roi_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=ROI_MIN,
        max_bound=ROI_MAX,
    )
    object_mesh = mesh.crop(roi_box)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_centers = (
        vertices[triangles[:, 0]]
        + vertices[triangles[:, 1]]
        + vertices[triangles[:, 2]]
    ) / 3.0
    inside = np.all(
        (triangle_centers >= ROI_MIN) & (triangle_centers <= ROI_MAX),
        axis=1,
    )
    scene_triangles = triangles[~inside]
    used_vertices = np.unique(scene_triangles)
    old_to_new = -np.ones(len(vertices), dtype=np.int64)
    old_to_new[used_vertices] = np.arange(len(used_vertices))

    scene_mesh = o3d.geometry.TriangleMesh()
    scene_mesh.vertices = o3d.utility.Vector3dVector(vertices[used_vertices])
    scene_mesh.triangles = o3d.utility.Vector3iVector(old_to_new[scene_triangles])
    scene_mesh.compute_vertex_normals()

    ensure_dir(OUTPUT_DIR)
    o3d.io.write_triangle_mesh(OBJECT_MESH, object_mesh)
    o3d.io.write_triangle_mesh(SCENE_MESH, scene_mesh)
    print("Object Mesh:", OBJECT_MESH)
    print("Scene Mesh :", SCENE_MESH)

    maybe_visualize(object_mesh, "Object ROI")
    print("DONE")


if __name__ == "__main__":
    main()
