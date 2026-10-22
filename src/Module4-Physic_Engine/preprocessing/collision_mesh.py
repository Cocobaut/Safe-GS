"""Step 9: export the validated mesh as collision geometry."""

import json
import os

import numpy as np
import open3d as o3d

try:
    from .roi_config import (
        COLLISION_DIR,
        COLLISION_MESH,
        COLLISION_REPORT,
        OPTIMIZED_DIR,
        TEST_ROIS,
        VALIDATED_MESH,
        ensure_dir,
        is_planar_roi,
        is_static_roi,
        optimized_roi_dir,
    )
except ImportError:
    from roi_config import (
        COLLISION_DIR,
        COLLISION_MESH,
        COLLISION_REPORT,
        OPTIMIZED_DIR,
        TEST_ROIS,
        VALIDATED_MESH,
        ensure_dir,
        is_planar_roi,
        is_static_roi,
        optimized_roi_dir,
    )


def topology_statistics(mesh):
    edge_map = {}
    for tri_id, tri in enumerate(np.asarray(mesh.triangles)):
        for edge in (
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0])))),
        ):
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
    """Export the validated adaptive mesh as collision geometry (Step 9)."""
    ensure_dir(COLLISION_DIR)
    if not os.path.exists(VALIDATED_MESH):
        raise FileNotFoundError(f"Input mesh not found: {VALIDATED_MESH}")
    mesh = o3d.io.read_triangle_mesh(VALIDATED_MESH)
    if mesh.is_empty():
        raise RuntimeError("Validated mesh is empty.")
    original_stats = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        **topology_statistics(mesh),
    }
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output_stats = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        **topology_statistics(mesh),
    }
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
    print(f"Collision mesh exported to: {COLLISION_MESH}")
    return report


# "patch": thin convex plates that hug the scanned surface (default).
# "coacd": CoACD decomposition; it voxelises open surfaces and inflates them by several cm.
COLLISION_METHOD = os.environ.get("MODULE5_COLLISION_METHOD", "patch").strip().lower()
PATCH_THICKNESS = 0.01          # m, plates are extruded behind the surface to this thickness
STATIC_PATCH_CELL = 0.35        # m, patch size for static segments
MIN_DYNAMIC_PATCH_CELL = 0.04   # m
MAX_HULLS_STATIC = 128
MAX_HULLS_DYNAMIC = 64


def surface_patch_hulls(vertices, faces, cell_size, thickness=PATCH_THICKNESS, max_hulls=64):
    """Split an open surface into voxel-cell patches and return one thin convex hull per patch.

    Segments are single-sided scans, so a convex decomposition of their "volume" is ill-posed:
    CoACD voxelises them and the hulls bulge several cm into neighbouring objects. Plates that
    follow the surface keep neighbours from starting interpenetrated.
    """
    tris = vertices[faces]
    centers = tris.mean(axis=1)
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # area weighted

    origin = centers.min(axis=0)
    while True:
        keys = np.floor((centers - origin) / cell_size).astype(np.int64)
        _, cell_of_face = np.unique(keys, axis=0, return_inverse=True)
        cell_of_face = cell_of_face.reshape(-1)
        if cell_of_face.max() + 1 <= max_hulls:
            break
        cell_size *= 1.5

    parts = []
    for cell in range(cell_of_face.max() + 1):
        selected = cell_of_face == cell
        points = np.unique(tris[selected].reshape(-1, 3), axis=0)
        if len(points) < 3:
            continue
        centered = points - points.mean(axis=0)
        thin_axis = np.linalg.svd(centered, full_matrices=False)[2][-1]
        projection = centered @ thin_axis
        if projection.max() - projection.min() < thickness:
            normal = face_normals[selected].sum(axis=0)
            norm = np.linalg.norm(normal)
            direction = normal / norm if norm > 1e-12 else thin_axis
            # Extrude behind the visible side so the plate does not stick out of the surface.
            points = np.vstack([points, points - direction * thickness])
        try:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64)))
            hull, _ = cloud.compute_convex_hull()
            hull.orient_triangles()
            volume = hull.get_volume()
        except Exception:
            continue
        if len(hull.triangles) < 4 or abs(volume) < 1e-9:
            continue
        parts.append((np.asarray(hull.vertices).copy(), np.asarray(hull.triangles).copy()))
    return parts


def coacd_parts(vertices, faces, planar):
    import coacd

    return coacd.run_coacd(
        coacd.Mesh(vertices, faces),
        threshold=0.05 if planar else 0.08,
        max_convex_hull=30 if planar else 15,
        preprocess_mode="auto",
        preprocess_resolution=50,
        resolution=1000,
        mcts_nodes=15,
        mcts_iterations=80,
        mcts_max_depth=3,
        pca=False,
        merge=True,
        seed=0,
    )


def build_hull_parts(centered_vertices, triangles, static, planar):
    """Convex collision parts [(vertices, faces), ...] for one object mesh in local coordinates."""
    if COLLISION_METHOD == "coacd":
        return coacd_parts(centered_vertices, triangles, planar)
    extent = float((centered_vertices.max(axis=0) - centered_vertices.min(axis=0)).max())
    return surface_patch_hulls(
        centered_vertices,
        triangles,
        STATIC_PATCH_CELL if static else max(MIN_DYNAMIC_PATCH_CELL, extent / 8.0),
        max_hulls=MAX_HULLS_STATIC if static else MAX_HULLS_DYNAMIC,
    )


def generate_coacd_hulls():
    """Generate convex collision hulls per ROI (surface patches or CoACD) and a per-ROI MuJoCo XML."""
    print("\n" + "=" * 60)
    print(f"CONVEX HULL GENERATION (PHYSICS READY) - method: {COLLISION_METHOD}")
    print("=" * 60)

    reports = {}
    for roi_id in TEST_ROIS.keys():
        roi_dir = optimized_roi_dir(roi_id)
        if not os.path.exists(roi_dir):
            continue

        visual_path = os.path.join(roi_dir, f"Object_{roi_id}_Visual.obj")
        if not os.path.exists(visual_path):
            continue

        print(f"[{roi_id}] Building collision hulls for {visual_path}...")
        mesh = o3d.io.read_triangle_mesh(visual_path)
        if len(mesh.triangles) == 0:
            print(f"[{roi_id}] Visual mesh has no faces, skipping.")
            continue

        vertices = np.asarray(mesh.vertices)
        info_path = os.path.join(roi_dir, f"Object_{roi_id}_Info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            centroid = np.asarray(info["centroid"], dtype=np.float64)
            min_z = float(info["min_z"])
            centered_vertices = np.asarray(mesh.vertices)
        else:
            centroid = vertices.mean(axis=0)
            min_z = float(np.min(vertices[:, 2]))
            centered_vertices = vertices - centroid

        roi = TEST_ROIS[roi_id]
        static = is_static_roi(roi_id, roi)
        planar = is_planar_roi(roi)

        parts = build_hull_parts(centered_vertices, np.asarray(mesh.triangles), static, planar)
        print(f"[{roi_id}] {'static' if static else 'dynamic'}: {len(parts)} hulls")

        # Remove hulls from previous runs so stale parts are not picked up by the scene builder.
        for old in os.listdir(roi_dir):
            if old.startswith(f"Object_{roi_id}_CoACD_") and old.endswith(".obj"):
                os.remove(os.path.join(roi_dir, old))

        hulls_saved = []
        for i, (vs, fs) in enumerate(parts):
            vs = np.asarray(vs, dtype=np.float64)
            part_mesh = o3d.geometry.TriangleMesh()
            part_mesh.vertices = o3d.utility.Vector3dVector(vs)
            part_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(fs))
            part_mesh.compute_vertex_normals()

            hull_filename = f"Object_{roi_id}_CoACD_{i}.obj"
            hull_path = os.path.join(roi_dir, hull_filename)
            o3d.io.write_triangle_mesh(hull_path, part_mesh, write_ascii=True)
            hulls_saved.append(hull_path)

        xml_path = os.path.join(roi_dir, f"Object_{roi_id}_MuJoCo.xml")
        cx, cy, cz = centroid
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(f'<mujoco model="{roi_id}">\n')
            f.write('  <compiler angle="radian"/>\n')
            f.write('  <size memory="50M"/>\n')
            f.write('  <option gravity="0 0 -9.81" timestep="0.005"/>\n')
            f.write('  <asset>\n')
            f.write(f'    <mesh name="visual" file="Object_{roi_id}_Visual.obj"/>\n')
            f.write(f'    <mesh name="background" file="Background_{roi_id}_QEM.obj"/>\n')
            for i in range(len(parts)):
                f.write(f'    <mesh name="col_{i}" file="Object_{roi_id}_CoACD_{i}.obj"/>\n')
            f.write('  </asset>\n')
            f.write('  <worldbody>\n')
            f.write(f'    <light diffuse=".7 .7 .7" pos="{cx:.3f} {cy:.3f} {cz + 3.0:.3f}" dir="0 0 -1"/>\n')
            f.write(f'    <geom name="floor_plane" type="plane" pos="0 0 {min_z:.4f}" size="20 20 0.1" rgba="0.3 0.3 0.3 0.0" condim="3" friction="0.01 0.001 0.0001"/>\n')
            f.write('    <geom name="background_mesh" type="mesh" mesh="background" contype="0" conaffinity="0" pos="0 0 0" rgba="0.40 0.44 0.50 0.92"/>\n')
            f.write(f'    <body name="{roi_id}" pos="{cx:.4f} {cy:.4f} {cz:.4f}">\n')
            f.write('      <inertial mass="5.0" pos="0 0 0" diaginertia="0.2 0.2 0.2"/>\n')
            f.write('      <camera name="track_cam" mode="trackcom" pos="-1.8 -2.0 1.2" xyaxes="0.707 -0.707 0 0.408 0.408 0.816"/>\n')
            f.write(f'      <joint name="{roi_id}_slide_x" type="slide" axis="1 0 0" damping="1.0" limited="true" range="-3.0 3.0"/>\n')
            f.write(f'      <joint name="{roi_id}_slide_y" type="slide" axis="0 1 0" damping="1.0" limited="true" range="-3.0 3.0"/>\n')
            f.write(f'      <joint name="{roi_id}_slide_z" type="slide" axis="0 0 1" damping="1.0" limited="true" range="-0.05 1.0"/>\n')
            f.write(f'      <joint name="{roi_id}_rotate_z" type="hinge" axis="0 0 1" damping="0.2"/>\n')
            f.write('      <geom type="mesh" mesh="visual" contype="0" conaffinity="0" rgba="0.10 0.85 1.00 1.0"/>\n')
            for i in range(len(parts)):
                f.write(f'      <geom type="mesh" mesh="col_{i}" group="3" rgba="1 0.2 0.2 0.0" condim="3" friction="0.01 0.001 0.0001"/>\n')
            f.write('    </body>\n')
            f.write('  </worldbody>\n')
            f.write('</mujoco>\n')

        reports[roi_id] = {"num_hulls": len(parts), "hulls": hulls_saved}
    return reports


def main():
    generate_collision_mesh()
    generate_coacd_hulls()
    print("STEP 9 COMPLETED")


if __name__ == "__main__":
    main()
