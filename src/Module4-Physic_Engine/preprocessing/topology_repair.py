import json
import os
import sys
from collections import defaultdict

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .roi_config import (
        COMPONENT_MESH,
        MAX_HOLE_LOOP_EDGES,
        OUTPUT_DIR,
        REPAIRED_MESH,
        TOPOLOGY_REPORT,
        ensure_dir,
        maybe_visualize,
    )
except ImportError:
    from roi_config import (
        COMPONENT_MESH,
        MAX_HOLE_LOOP_EDGES,
        OUTPUT_DIR,
        REPAIRED_MESH,
        TOPOLOGY_REPORT,
        ensure_dir,
        maybe_visualize,
    )


def analyze_topology(mesh):
    triangles = np.asarray(mesh.triangles)
    edge_to_triangles = defaultdict(list)

    for triangle_id, tri in enumerate(triangles):
        v0, v1, v2 = map(int, tri)
        for edge in (
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0))),
        ):
            edge_to_triangles[edge].append(triangle_id)

    boundary_edges = []
    manifold_edges = []
    non_manifold_edges = []

    for edge, triangle_ids in edge_to_triangles.items():
        count = len(triangle_ids)
        if count == 1:
            boundary_edges.append(edge)
        elif count == 2:
            manifold_edges.append(edge)
        else:
            non_manifold_edges.append((edge, triangle_ids))

    boundary_graph = defaultdict(set)
    for v0, v1 in boundary_edges:
        boundary_graph[v0].add(v1)
        boundary_graph[v1].add(v0)

    visited_edges = set()
    boundary_loops = []

    for start_vertex in boundary_graph:
        for next_vertex in boundary_graph[start_vertex]:
            edge_key = tuple(sorted((start_vertex, next_vertex)))
            if edge_key in visited_edges:
                continue

            loop = []
            current = start_vertex
            previous = None

            while True:
                loop.append(current)
                neighbors = boundary_graph[current]
                candidates = [
                    n for n in neighbors
                    if tuple(sorted((current, n))) not in visited_edges
                ]
                if not candidates:
                    break

                if previous is not None:
                    non_back = [n for n in candidates if n != previous]
                    next_v = non_back[0] if non_back else candidates[0]
                else:
                    next_v = candidates[0]

                visited_edges.add(tuple(sorted((current, next_v))))
                previous = current
                current = next_v
                if current == start_vertex:
                    break

            if loop and loop[0] == loop[-1]:
                loop = loop[:-1]
            if len(loop) >= 3 and current == start_vertex:
                boundary_loops.append(loop)

    return {
        "vertices": len(mesh.vertices),
        "triangles": len(triangles),
        "edges": len(edge_to_triangles),
        "boundary_edges": boundary_edges,
        "manifold_edges": manifold_edges,
        "non_manifold_edges": non_manifold_edges,
        "boundary_loops": boundary_loops,
        "edge_to_triangles": edge_to_triangles,
    }


def _existing_triangle_for_edge(triangles, edge_to_triangles, v0, v1):
    key = tuple(sorted((v0, v1)))
    ids = edge_to_triangles.get(key, [])
    if not ids:
        return None
    return np.asarray(triangles[ids[0]], dtype=np.int64)


def _edge_winding_in_triangle(tri, v0, v1):
    tri = [int(x) for x in tri]
    for i in range(3):
        a = tri[i]
        b = tri[(i + 1) % 3]
        if a == v0 and b == v1:
            return 1
        if a == v1 and b == v0:
            return -1
    return 0


def fill_small_boundary_loops(mesh, max_loop_edges=MAX_HOLE_LOOP_EDGES):
    info = analyze_topology(mesh)
    triangles = np.asarray(mesh.triangles).tolist()
    filled = 0
    skipped = 0

    for loop in info["boundary_loops"]:
        if len(loop) < 3 or len(loop) > max_loop_edges:
            skipped += 1
            continue

        sample = _existing_triangle_for_edge(
            np.asarray(mesh.triangles),
            info["edge_to_triangles"],
            loop[0],
            loop[1],
        )
        reverse = False
        if sample is not None:
            winding = _edge_winding_in_triangle(sample, loop[0], loop[1])
            reverse = winding > 0

        ordered = list(reversed(loop)) if reverse else list(loop)
        origin = ordered[0]
        for i in range(1, len(ordered) - 1):
            triangles.append([origin, ordered[i], ordered[i + 1]])
        filled += 1

    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return mesh, filled, skipped


def main():
    print("=" * 70)
    print("STEP 3 - TOPOLOGY DETECTION AND REPAIR")
    print("=" * 70)
    print("\n[1] Loading mesh...")

    mesh = o3d.io.read_triangle_mesh(COMPONENT_MESH)
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty or cannot be loaded.")

    print("Vertices :", len(mesh.vertices))
    print("Triangles:", len(mesh.triangles))

    initial = analyze_topology(mesh)
    print("\nInitial boundary edges:", len(initial["boundary_edges"]))
    print("Initial boundary loops:", len(initial["boundary_loops"]))
    print("Initial non-manifold edges:", len(initial["non_manifold_edges"]))

    triangles_before = len(mesh.triangles)
    mesh.remove_non_manifold_edges()
    removed_non_manifold = triangles_before - len(mesh.triangles)
    print("\nNon-manifold triangles removed:", removed_non_manifold)

    mesh, holes_filled, holes_skipped = fill_small_boundary_loops(mesh)
    print("Small boundary loops filled:", holes_filled)
    print("Boundary loops skipped (too large):", holes_skipped)

    before = len(mesh.triangles)
    mesh.remove_degenerate_triangles()
    degenerate_removed = before - len(mesh.triangles)
    before = len(mesh.triangles)
    mesh.remove_duplicated_triangles()
    duplicate_removed = before - len(mesh.triangles)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    final = analyze_topology(mesh)
    watertight = (
        len(final["boundary_edges"]) == 0
        and len(final["non_manifold_edges"]) == 0
    )

    print("\nFinal boundary edges:", len(final["boundary_edges"]))
    print("Final boundary loops:", len(final["boundary_loops"]))
    print("Watertight:", watertight)

    ensure_dir(OUTPUT_DIR)
    if not o3d.io.write_triangle_mesh(REPAIRED_MESH, mesh):
        raise RuntimeError("Failed to save repaired mesh.")

    report = {
        "input": COMPONENT_MESH,
        "output": REPAIRED_MESH,
        "initial": {
            "vertices": int(initial["vertices"]),
            "triangles": int(initial["triangles"]),
            "boundary_edges": len(initial["boundary_edges"]),
            "non_manifold_edges": len(initial["non_manifold_edges"]),
            "boundary_loops": len(initial["boundary_loops"]),
        },
        "repair": {
            "non_manifold_triangles_removed": int(removed_non_manifold),
            "small_holes_filled": int(holes_filled),
            "large_holes_skipped": int(holes_skipped),
            "degenerate_triangles_removed": int(degenerate_removed),
            "duplicated_triangles_removed": int(duplicate_removed),
        },
        "final": {
            "vertices": int(final["vertices"]),
            "triangles": int(final["triangles"]),
            "boundary_edges": len(final["boundary_edges"]),
            "non_manifold_edges": len(final["non_manifold_edges"]),
            "boundary_loops": len(final["boundary_loops"]),
            "watertight": bool(watertight),
        },
    }

    with open(TOPOLOGY_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("Saved:", REPAIRED_MESH)
    print("Report:", TOPOLOGY_REPORT)
    maybe_visualize(mesh, "Topology Repaired Mesh")
    print("\nSTEP 3 COMPLETED")


if __name__ == "__main__":
    main()
