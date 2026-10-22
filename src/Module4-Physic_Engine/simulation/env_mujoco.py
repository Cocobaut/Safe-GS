"""Module 4 simulation - MuJoCo environment: Unified Physics Scene built from the Module 4 preprocessing outputs.

Static structure (walls/floor/...) + free-falling segmented objects, each with a visual mesh
(rendering only) and separate convex collision hulls (Module 4 preprocessing, collision_mesh.py), plus the
task agent (kinematic probe).

By default the scene runs the autonomous simulation task (auto_task.py): double-click an ROI
object (the actor) and then another one (the target) in the viewer; the actor moves to the
target, collides with it and the predicted outcome is reported (headless / --auto: the agent
probe moves into the nearest object). --manual gives the old
viewer where objects are dragged by hand.

Usage (from the repository root):
    python scripts/06_run_simulation.py [--rebuild] [--no-bg-collision] [--check] [--no-view]
                                        [--variant=unified|high|adaptive|coarse] [--benchmark]
                                        [--start=x,y,z] [--actor=ROI_x] [--target=ROI_y] [--auto] [--manual]

    --rebuild           regenerate background + XML even if the XML already exists
    --no-bg-collision   leave the leftover (non-segment) background visual-only
    --check             run a headless stability test after building
    --no-view           no viewer: run the task headless and only write the report
    --variant=NAME      scene resolution variant (default: unified, see VARIANTS)
    --benchmark         run the M_High / M_Adaptive / M_Coarse comparison (benchmarks.py)
    --start=x,y,z       agent start point in metres (default: middle of the room, 1.2 m up)
    --actor=ROI_x       this object (not the agent) moves into the target
    --target=ROI_y      object to hit, instead of the one nearest to the actor
    --auto              viewer: agent -> nearest object right away instead of waiting for clicks
    --speed=K           viewer: play the simulation K times faster than real time (default 1)
    --manual            interactive viewer without the task (drag objects with the mouse)
    --no-sleep          physics on for every object from t=0, at the resolver-adjusted pose
                        (default: objects stay exactly at their scanned pose, asleep, until
                        dragged or touched - see topology_guard.py)

Scene variants (README Module 4 simulation, step 10):
    unified   every segment at full resolution, leftover background capped at 20k triangles
              (the interactive scene)
    high      M_High     - nothing decimated
    adaptive  M_Adaptive - target ROI at full resolution, everything else QEM-simplified
    coarse    M_Coarse   - everything QEM-simplified with the same ratio (blind global QEM)
"""

import sys
import os
import time
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import mujoco
import mujoco.viewer

# src/Module4-Physic_Engine/ (holds preprocessing/ and simulation/)
ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)


from preprocessing.roi_config import (
    ALL_ROIS,
    INPUT_FOLDER_NAME,
    REPAIRED_MESH,
    OPTIMIZED_DIR,
    ensure_dir,
    is_planar_roi,
    is_static_roi,
    is_thin_roi,
    THIN_CATEGORIES,
    segment_color,
)
from preprocessing.collision_mesh import build_hull_parts, surface_patch_hulls
from simulation.topology_guard import TopologyGuard

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# One output folder per dataset (same name as the Module 4 preprocessing Input/ folder).
OUTPUT_DIR = os.path.join(MODULE_DIR, "Output", INPUT_FOLDER_NAME)
SCENES_DIR = os.path.join(OUTPUT_DIR, "scenes")

VARIANTS = ("unified", "high", "adaptive", "coarse")
DEFAULT_VARIANT = "unified"
UNIFIED_BG_TRIANGLES = 20000    # background budget of the interactive scene
DEFAULT_QEM_RATIO = 0.30        # Module 4 preprocessing target_ratio, used when an ROI does not define one

# Physics parameters
OBJECT_DENSITY = 200.0          # kg/m^3 of the segment AABB ...
OBJECT_FILL = 0.3               # ... times the fraction of that box that is actually solid
MASS_RANGE = (0.05, 30.0)       # kg
AREAL_DENSITY = 1.0             # kg/m^2 of scanned surface for thin / soft segments (blanket, towel, ...)
# Collision strength of soft / thin segments (flat splats, cloth): 1 = as hard as any object,
# 0.1 = a soft, giving contact (10% of the stiffness), 0 = no collision at all (pass through).
SOFT_COLLISION = 0.1
BASE_SOLREF_TIME = 0.01         # s, contact time constant of the "collision" default class
SCENE_VERSION = 3               # bump when the scene XML changes meaning: older scenes are rebuilt
DEFAULT_RGB = [0.8, 0.8, 0.8]
PENETRATION_TOL = 0.002         # m, contacts deeper than this at t=0 are resolved
PENETRATION_GAP = 0.001         # m, clearance left after pushing a body out
MAX_RESOLVE_SHIFT = 0.05        # m, max total displacement of a body by the resolver
MAX_RESOLVE_ITERS = 20
COLLISION_METHOD = os.environ.get("MODULE5_COLLISION_METHOD", "patch").strip().lower()
BG_PATCH_CELL = 0.15            # m, background patches (connected component by component)
BG_MIN_COMPONENT_TRIANGLES = 10 # smaller scraps get no collision
BG_MAX_HULLS_PER_COMPONENT = 64
SUPPORT_TEST_TIME = 0.3         # s, drop test used to find wall/ceiling-mounted objects
SUPPORT_MAX_DROP = 0.10         # m, free fall covers ~0.44 m in 0.3 s; resting objects barely move
FLOOR_CATEGORIES = {"floor", "rug"}


def scene_dir(variant):
    return ensure_dir(os.path.join(SCENES_DIR, variant))


def scene_xml(variant):
    return os.path.join(SCENES_DIR, variant, "Unified_MuJoCo.xml")


def _fmt(vec):
    return " ".join(f"{v:.5f}" for v in vec)


def _path(p):
    return os.path.abspath(p).replace("\\", "/")


def _qem_ratio(roi):
    return float(roi.get("target_ratio", DEFAULT_QEM_RATIO))


def object_ratio(variant, roi_id, target_roi):
    """Fraction of an object's triangles kept in a variant."""
    if variant in ("unified", "high"):
        return 1.0
    if variant == "adaptive" and roi_id == target_roi:
        return 1.0
    return _qem_ratio(ALL_ROIS[roi_id])


def generate_unified_background(out_dir, variant=DEFAULT_VARIANT):
    print("Loading repaired mesh...")
    mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)
    if mesh.is_empty():
        raise RuntimeError(f"Cannot load {REPAIRED_MESH}")

    print("Gathering all object vertices from ROI config...")
    all_seg_verts = []
    seg_labels = []
    for label, (roi_id, roi) in enumerate(ALL_ROIS.items()):
        if "segment_vertices" in roi and len(roi["segment_vertices"]) > 0:
            all_seg_verts.append(roi["segment_vertices"])
            seg_labels.append(np.full(len(roi["segment_vertices"]), label))

    if not all_seg_verts:
        raise ValueError("No segment vertices found in ROIs. Are you using AABB only?")

    giant_verts = np.vstack(all_seg_verts)
    giant_labels = np.concatenate(seg_labels)

    print(f"Building KDTree for {len(giant_verts)} object vertices...")
    mesh_verts = np.asarray(mesh.vertices)
    mesh_tris = np.asarray(mesh.triangles)
    # A triangle belongs to an object only when all 3 vertices are vertices of the SAME segment
    # (same rule as adaptive_mesh, so the hole matches the object visual meshes exactly).
    # Triangles spanning two segments belong to neither object mesh, so they stay in the
    # background; otherwise every segment border would show a thin open gap.
    dist, idx = cKDTree(giant_verts).query(mesh_verts, k=1)
    vert_label = np.where(dist < 1e-3, giant_labels[idx], -1)
    tri_labels = vert_label[mesh_tris]
    is_object = (tri_labels[:, 0] >= 0) & (tri_labels[:, 0] == tri_labels[:, 1]) & (tri_labels[:, 0] == tri_labels[:, 2])
    # Seam = triangle whose vertices all lie on segments but not on a single one. Seams sit
    # right against objects, so they are rendered (full resolution) but get no collision.
    is_seam = ~is_object & (tri_labels >= 0).all(axis=1)

    bg_tris = mesh_tris[~is_object & ~is_seam]
    seam_tris = mesh_tris[is_seam]
    print(f"Original triangles: {len(mesh_tris)}, Background triangles: {len(bg_tris)}, "
          f"segment seam triangles: {len(seam_tris)}")

    if variant == "unified":
        target_total = UNIFIED_BG_TRIANGLES
    elif variant == "high":
        target_total = len(bg_tris)
    else:
        target_total = int(len(bg_tris) * DEFAULT_QEM_RATIO)

    for old in os.listdir(out_dir):
        if old.startswith("Unified_Background") and old.endswith(".obj"):
            os.remove(os.path.join(out_dir, old))

    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    def write_colour_parts(tris, prefix, target):
        """MuJoCo has no vertex colours, so split by the .ply colour of each triangle and
        give every part its own rgba. Returns (parts, merged mesh)."""
        if colors is not None:
            tri_colors = np.round(colors[tris[:, 0]] * 255).astype(np.int64)
        else:
            tri_colors = np.full((len(tris), 3), 204, dtype=np.int64)
        palette, color_of_tri = np.unique(tri_colors, axis=0, return_inverse=True)
        color_of_tri = color_of_tri.reshape(-1)
        parts, merged = [], o3d.geometry.TriangleMesh()
        for k, rgb in enumerate(palette):
            part = o3d.geometry.TriangleMesh()
            part.vertices = o3d.utility.Vector3dVector(mesh_verts)
            part.triangles = o3d.utility.Vector3iVector(tris[color_of_tri == k])
            part.remove_unreferenced_vertices()
            if target is not None:
                part_target = max(4, int(target * len(part.triangles) / max(len(tris), 1)))
                if len(part.triangles) > part_target:
                    part = part.simplify_quadric_decimation(part_target, boundary_weight=100.0)
            if len(part.triangles) == 0 or len(part.vertices) < 4:  # MuJoCo needs >= 4 vertices
                continue
            part.compute_vertex_normals()
            name = f"{prefix}{k}.obj"
            o3d.io.write_triangle_mesh(os.path.join(out_dir, name), part, write_ascii=True)
            parts.append({"file": name, "color": (rgb / 255.0).tolist()})
            merged += part
        return parts, merged

    parts, merged = write_colour_parts(bg_tris, "Unified_Background_c", target_total)
    seam_parts, _ = write_colour_parts(seam_tris, "Unified_Background_seam_c", None) if len(seam_tris) else ([], None)
    print(f"Background split into {len(parts)} colour parts ({len(merged.triangles)} triangles) "
          f"+ {len(seam_parts)} seam parts")
    # `merged` (collision source) excludes the seams; the visual parts include them.
    return merged, parts + seam_parts


def _background_patch_parts(bg_mesh, centered_vertices):
    """Surface patch hulls per connected component, so scraps that merely share a voxel
    cell (floor edge + wall edge) are never bridged by one convex block."""
    triangles = np.asarray(bg_mesh.triangles)
    labels, counts, _ = bg_mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    parts = []
    for comp, count in enumerate(np.asarray(counts)):
        if count < BG_MIN_COMPONENT_TRIANGLES:
            continue
        parts += surface_patch_hulls(
            centered_vertices, triangles[labels == comp], BG_PATCH_CELL, max_hulls=BG_MAX_HULLS_PER_COMPONENT
        )
    return parts


def run_coacd_on_background(bg_mesh, out_dir):
    print(f"Building collision hulls for the unsegmented background (method: {COLLISION_METHOD})...")
    vertices = np.asarray(bg_mesh.vertices)
    centroid = vertices.mean(axis=0)
    centered_vertices = vertices - centroid

    if COLLISION_METHOD == "coacd":
        import coacd

        parts = coacd.run_coacd(
            coacd.Mesh(centered_vertices, np.asarray(bg_mesh.triangles)),
            threshold=0.05,
            max_convex_hull=60,
            preprocess_mode="auto",
            preprocess_resolution=30,
            resolution=1000,
            mcts_nodes=15,
            mcts_iterations=80,
            mcts_max_depth=3,
            pca=False,
            merge=True,
            seed=0,
        )
    else:
        parts = _background_patch_parts(bg_mesh, centered_vertices)
    print(f"Background collision hulls: {len(parts)}")

    hulls = []
    for i, (vs, fs) in enumerate(parts):
        part_mesh = o3d.geometry.TriangleMesh()
        part_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vs))
        part_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(fs))
        part_mesh.compute_vertex_normals()

        hull_name = f"Unified_Background_CoACD_{i}.obj"
        o3d.io.write_triangle_mesh(os.path.join(out_dir, hull_name), part_mesh, write_ascii=True)
        hulls.append(hull_name)

    return hulls


_COLOR_CACHE = {}


def _color_lookup(seg_vertices):
    if "tree" not in _COLOR_CACHE:
        mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)
        verts = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        _COLOR_CACHE.update(tree=cKDTree(verts) if len(verts) else None, verts=verts, colors=colors)
    if _COLOR_CACHE["tree"] is None:
        return None
    return segment_color(_COLOR_CACHE["verts"], _COLOR_CACHE["colors"], seg_vertices, _COLOR_CACHE["tree"])


def _box_inertial(extent, area=None):
    """Mass/Inertia is an approximation derived from AABB volume bounding, not the true physical mass.
    Hulls are thin plates, so their volume says nothing about the real mass.
    A thin / soft segment (`area` given) weighs its surface area times AREAL_DENSITY instead:
    a blanket draped over a sofa fills a big box but is light.
    """
    ex, ey, ez = np.maximum(extent, 0.01)
    raw = AREAL_DENSITY * area if area is not None else OBJECT_DENSITY * ex * ey * ez * OBJECT_FILL
    mass = float(np.clip(raw, *MASS_RANGE))
    inertia = mass / 12.0 * np.array([ey**2 + ez**2, ex**2 + ez**2, ex**2 + ey**2])
    return mass, inertia


def _soft_attrs(obj):
    """Extra collision-geom attributes of a soft segment. Contact stiffness scales with
    1/timeconst^2, so SOFT_COLLISION = k uses timeconst / sqrt(k); priority makes these soft
    parameters win over the other geom's in a contact (default mixing would average them)."""
    if not obj.get("soft"):
        return ""
    if SOFT_COLLISION <= 0.0:
        return ' contype="0" conaffinity="0"'
    if SOFT_COLLISION >= 1.0:
        return ""
    return f' solref="{BASE_SOLREF_TIME / np.sqrt(SOFT_COLLISION):.4f} 1" priority="1"'


def _rgba(rgb):
    return f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} 1"


def _decimated_object(roi_id, roi, visual_path, ratio, out_dir):
    """QEM-simplified copy of an object's visual mesh and hulls for a benchmark variant.
    Returns (visual_path, hull_dir, hull_files); cached until the Module 4 preprocessing visual changes."""
    obj_dir = ensure_dir(os.path.join(out_dir, "objects", roi_id))
    dec_visual = os.path.join(obj_dir, f"Object_{roi_id}_Visual.obj")
    prefix = f"Object_{roi_id}_CoACD_"

    cached = (
        os.path.exists(dec_visual)
        and os.path.getmtime(dec_visual) >= os.path.getmtime(visual_path)
        and any(f.startswith(prefix) for f in os.listdir(obj_dir))
    )
    if not cached:
        mesh = o3d.io.read_triangle_mesh(visual_path)
        target = max(4, int(len(mesh.triangles) * ratio))
        if len(mesh.triangles) > target:
            mesh = mesh.simplify_quadric_decimation(target, boundary_weight=100.0)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(dec_visual, mesh, write_ascii=True)

        for old in os.listdir(obj_dir):
            if old.startswith(prefix):
                os.remove(os.path.join(obj_dir, old))
        parts = build_hull_parts(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles), is_static_roi(roi_id, roi), is_planar_roi(roi)
        )
        for i, (vs, fs) in enumerate(parts):
            part = o3d.geometry.TriangleMesh()
            part.vertices = o3d.utility.Vector3dVector(np.asarray(vs))
            part.triangles = o3d.utility.Vector3iVector(np.asarray(fs))
            o3d.io.write_triangle_mesh(os.path.join(obj_dir, f"{prefix}{i}.obj"), part, write_ascii=True)

    hull_files = [f for f in os.listdir(obj_dir) if f.startswith(prefix) and f.endswith(".obj")]
    hull_files.sort(key=lambda f: int(f[len(prefix):-4]))
    return dec_visual, obj_dir, hull_files


def load_scene_objects(variant=DEFAULT_VARIANT, out_dir=None, target_roi=None, forced_static=()):
    """Collect every ROI that has a visual mesh + collision hulls, tagged static or dynamic."""
    objects = []
    for roi_id, roi in ALL_ROIS.items():
        roi_dir = os.path.join(OPTIMIZED_DIR, roi_id)
        info_path = os.path.join(roi_dir, f"Object_{roi_id}_Info.json")
        if not os.path.exists(info_path):
            continue

        prefix = f"Object_{roi_id}_CoACD_"
        hull_files = [f for f in os.listdir(roi_dir) if f.startswith(prefix) and f.endswith(".obj")]
        if not hull_files:
            continue
        hull_files.sort(key=lambda f: int(f[len(prefix):-4]))
        visual_path = os.path.join(roi_dir, f"Object_{roi_id}_Visual.obj")
        hull_dir = roi_dir

        ratio = object_ratio(variant, roi_id, target_roi)
        if ratio < 1.0:
            visual_path, hull_dir, hull_files = _decimated_object(roi_id, roi, visual_path, ratio, out_dir)
            if not hull_files:
                continue

        with open(info_path, "r") as f:
            info = json.load(f)

        color = info.get("color")
        if color is None and "segment_vertices" in roi:
            # Info.json from an older run: look the colour up in the coloured scene mesh.
            color = _color_lookup(roi["segment_vertices"])

        objects.append({
            "id": roi_id,
            "category": str(roi.get("category", "unknown")),
            "static": is_static_roi(roi_id, roi) or roi_id in forced_static,
            "centroid": np.asarray(info["centroid"], dtype=np.float64),
            "min_z": float(info["min_z"]),
            "extent": np.asarray(roi["max"]) - np.asarray(roi["min"]),
            "color": color or DEFAULT_RGB,
            "visual_path": visual_path,
            "hulls": hull_files,
            "dir": hull_dir,
            "ratio": ratio,
            "thin": is_thin_roi(roi),
            # Soft = movable thin segment, or a cloth category even when fixed (curtain).
            "soft": (is_thin_roi(roi) and not (is_static_roi(roi_id, roi) or roi_id in forced_static))
                    or str(roi.get("category", "")).lower() in THIN_CATEGORIES,
        })
    return objects


def floor_height(objects):
    floors = [o["min_z"] for o in objects if o["category"].lower() in FLOOR_CATEGORIES]
    if floors:
        return min(floors)
    return min(o["min_z"] for o in objects)


def build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid, positions=None, excludes=(), probe=None,
                      pose_mode="resolved"):
    """Write the scene XML. `positions` overrides body positions, `excludes` are body-name
    pairs and `probe` ({"pos", "half_size"}) adds the kinematic probe of the benchmark task."""
    positions = positions or {}
    out_dir = os.path.dirname(xml_path)
    floor_z = floor_height(objects)
    cx, cy, cz = bg_centroid
    lines = [
        '<mujoco model="unified_scene">',
        f'  <!-- start_pose: {pose_mode} -->',
        f'  <!-- scene_version: {SCENE_VERSION} -->',
        '  <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="3 3" balanceinertia="true"/>',
        '  <size memory="500M"/>',
        '  <visual>',
        '    <headlight ambient="0.45 0.45 0.45" diffuse="0.45 0.45 0.45" specular="0 0 0"/>',
        '  </visual>',
        '  <option gravity="0 0 -9.81" timestep="0.002" integrator="Euler" solver="Newton" iterations="100" tolerance="1e-8" cone="elliptic" noslip_iterations="3"/>',
        '  <default>',
        '    <joint damping="0.05"/>',
        '    <default class="visual">',
        '      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" material="matte"/>',
        '    </default>',
        '    <default class="collision">',
        '      <geom type="mesh" group="3" condim="3" density="0" margin="0.001"'
        ' solref="0.01 1" solimp="0.9 0.95 0.001" friction="0.8 0.005 0.0001" rgba="1 0.2 0.2 0.3"/>',
        '    </default>',
        '  </default>',
        '  <asset>',
        # Flat, non-shiny material so the .ply segment colours read like in the mesh viewer.
        '    <material name="matte" specular="0.05" shininess="0.1"/>',
    ]
    for k, part in enumerate(bg_parts):
        lines.append(f'    <mesh name="bg_visual_{k}" file="{_path(os.path.join(out_dir, part["file"]))}" inertia="shell"/>')
    for i, hull in enumerate(bg_hulls):
        lines.append(f'    <mesh name="bg_col_{i}" file="{_path(os.path.join(out_dir, hull))}"/>')
    for obj in objects:
        lines.append(f'    <mesh name="{obj["id"]}_visual" file="{_path(obj["visual_path"])}" inertia="shell"/>')
        for hull in obj["hulls"]:
            lines.append(f'    <mesh name="{obj["id"]}_{hull}" file="{_path(os.path.join(obj["dir"], hull))}"/>')
    lines += [
        '  </asset>',
        '  <worldbody>',
        # No shadows: shadow maps on the ~600k-triangle scan add black/white acne.
        f'    <light diffuse=".5 .5 .5" specular="0 0 0" castshadow="false" pos="{cx:.3f} {cy:.3f} {cz + 4.0:.3f}" dir="0 0 -1"/>',
        f'    <light diffuse=".3 .3 .3" specular="0 0 0" castshadow="false" pos="{cx + 2:.3f} {cy + 2:.3f} {cz + 3.0:.3f}" dir="-0.5 -0.5 -1"/>',
        # Safety net at the real floor height so nothing falls out of the room.
        f'    <geom name="floor_plane" type="plane" pos="0 0 {floor_z:.4f}" size="20 20 0.1"'
        ' rgba="0.3 0.3 0.3 0.0" condim="3" friction="1.0 0.005 0.0001"/>',
        # Background meshes are in world coordinates; background hulls are centred on bg_centroid.
        '    <body name="environment">',
    ]
    for k, part in enumerate(bg_parts):
        lines.append(f'      <geom class="visual" mesh="bg_visual_{k}" rgba="{_rgba(part["color"])}"/>')
    for i in range(len(bg_hulls)):
        lines.append(f'      <geom class="collision" mesh="bg_col_{i}" pos="{cx:.4f} {cy:.4f} {cz:.4f}" rgba="0.5 0.5 0.5 0.2"/>')
    lines.append('    </body>')

    for obj in objects:
        pos = positions.get(obj["id"], obj["centroid"])
        lines.append(f'    <body name="{obj["id"]}" pos="{_fmt(pos)}">')
        if obj["static"]:
            # Structural segment: welded to the world, only collides with dynamic bodies.
            lines.append(f'      <geom class="visual" mesh="{obj["id"]}_visual" rgba="{_rgba(obj["color"])}"/>')
        else:
            area = None
            if obj["thin"]:
                area = o3d.io.read_triangle_mesh(obj["visual_path"]).get_surface_area()
            mass, inertia = _box_inertial(obj["extent"], area)
            lines.append(f'      <freejoint name="{obj["id"]}_free"/>')
            # Full precision: rounding a flat box's inertia can break A + B >= C.
            inertia_str = " ".join(f"{v:.10g}" for v in inertia)
            lines.append(f'      <inertial pos="0 0 0" mass="{mass:.6g}" diaginertia="{inertia_str}"/>')
            lines.append(f'      <geom class="visual" mesh="{obj["id"]}_visual" rgba="{_rgba(obj["color"])}"/>')
        for hull in obj["hulls"]:
            lines.append(f'      <geom class="collision" mesh="{obj["id"]}_{hull}"{_soft_attrs(obj)}/>')
        lines.append('    </body>')

    if probe is not None:
        # Kinematic task probe (robot end-effector stand-in), driven through data.mocap_pos.
        lines.append(f'    <body name="probe" mocap="true" pos="{_fmt(probe["pos"])}">')
        lines.append(f'      <geom name="probe_geom" type="box" size="{_fmt(probe["half_size"])}" condim="3"'
                     ' friction="0.8 0.005 0.0001" rgba="1 0.1 0.1 0.9"/>')
        lines.append('    </body>')
    lines.append('  </worldbody>')

    if excludes:
        lines.append('  <contact>')
        for a, b in excludes:
            lines.append(f'    <exclude body1="{a}" body2="{b}"/>')
        lines.append('  </contact>')
    lines.append('</mujoco>')

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Unified XML saved to {xml_path}")


def _penetrations(model, data, tol):
    """Return [(body1, body2, depth, normal_from_1_to_2)] for contacts deeper than tol."""
    mujoco.mj_forward(model, data)
    result = []
    for i in range(data.ncon):
        con = data.contact[i]
        if con.dist >= -tol:
            continue
        b1 = int(model.geom_bodyid[con.geom[0]])
        b2 = int(model.geom_bodyid[con.geom[1]])
        result.append((b1, b2, -float(con.dist), np.array(con.frame[:3])))
    return result


def resolve_initial_penetration(xml_path, objects):
    """Push dynamic bodies out of whatever they start interpenetrated with.

    Segments come from one fused mesh, so neighbours (chair on rug, plant on stand) touch and
    their convex hulls overlap at t=0. MuJoCo would resolve that with huge contact forces and
    the scene explodes. Returns adjusted positions and the body pairs that must be excluded.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dynamic = {}  # body id -> qpos address of its freejoint
    for obj in objects:
        if obj["static"]:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj["id"])
        dynamic[bid] = int(model.jnt_qposadr[model.body_jntadr[bid]])
    start = {bid: data.qpos[adr:adr + 3].copy() for bid, adr in dynamic.items()}

    initial = _penetrations(model, data, PENETRATION_TOL)
    max_before = max((p[2] for p in initial), default=0.0)
    print(f"Initial penetrations > {PENETRATION_TOL * 1000:.0f} mm: {len(initial)} contacts, "
          f"max depth {max_before * 1000:.1f} mm")

    stuck = set()
    remaining = initial
    for _ in range(MAX_RESOLVE_ITERS):
        if not remaining:
            break
        moves = {}
        for b1, b2, depth, normal in remaining:
            if b1 not in dynamic and b2 not in dynamic:
                continue
            if b1 in dynamic and b2 in dynamic:
                # Move the body resting on top (higher centre) away from the other.
                upper_is_2 = data.xpos[b2][2] >= data.xpos[b1][2]
                mover, direction = (b2, normal) if upper_is_2 else (b1, -normal)
            elif b2 in dynamic:
                mover, direction = b2, normal
            else:
                mover, direction = b1, -normal
            if (min(b1, b2), max(b1, b2)) in stuck:
                continue
            if direction[2] > 0.5:
                # Resting contact: lift straight up instead of sliding the body sideways.
                shift = np.array([0.0, 0.0, (depth + PENETRATION_GAP) / direction[2]])
            else:
                shift = direction * (depth + PENETRATION_GAP)
            if mover not in moves or np.linalg.norm(shift) > np.linalg.norm(moves[mover][0]):
                moves[mover] = (shift, (min(b1, b2), max(b1, b2)))
        if not moves:
            break
        for bid, (shift, pair) in moves.items():
            adr = dynamic[bid]
            new_pos = data.qpos[adr:adr + 3] + shift
            if np.linalg.norm(new_pos - start[bid]) > MAX_RESOLVE_SHIFT:
                stuck.add(pair)
            else:
                data.qpos[adr:adr + 3] = new_pos
        remaining = _penetrations(model, data, PENETRATION_TOL)

    remaining = _penetrations(model, data, PENETRATION_TOL)
    excludes = set(stuck)
    for b1, b2, _, _ in remaining:
        if b1 != b2 and (b1 in dynamic or b2 in dynamic) and 0 not in (b1, b2):
            excludes.add((min(b1, b2), max(b1, b2)))
    # A thin / soft segment (blanket on a cushion) overlaps what it lies on in the scan. Against
    # another movable object keep the collision, so one can push the other (the sleeping start
    # and the wake speed clamp separate the overlap gently). Against a fixed structure it is
    # draped over (sofa, scan background) it stays excluded: a rigid shell hooked over a sofa
    # arm cannot slide off it, only a real cloth could.
    thin = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o["id"])
            for o in objects if o.get("thin") and not o["static"]}
    kept = {pair for pair in excludes if thin & set(pair) and pair[0] in dynamic and pair[1] in dynamic}
    excludes -= kept

    def name(bid):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)

    positions = {name(bid): data.qpos[adr:adr + 3].copy() for bid, adr in dynamic.items()}
    exclude_names = sorted((name(a), name(b)) for a, b in excludes)
    moved = {n: float(np.linalg.norm(positions[n] - start[b])) for b, n in ((b, name(b)) for b in dynamic)}
    moved = {n: d for n, d in moved.items() if d > 1e-6}
    world_left = [(name(b1), name(b2), d) for b1, b2, d, _ in remaining if 0 in (b1, b2)]

    print(f"Resolver moved {len(moved)} bodies, excluded {len(exclude_names)} pairs, "
          f"kept {len(kept)} overlapping pairs with thin objects")
    for a, b in exclude_names:
        print(f"  exclude {a} <-> {b}")
    for a, b, d in world_left:
        print(f"  still touching floor plane: {a} <-> {b} ({d * 1000:.1f} mm)")

    report = {
        "tolerance_m": PENETRATION_TOL,
        "thin_objects": sorted(name(b) for b in thin),
        "kept_thin_pairs": sorted((name(a), name(b)) for a, b in kept),
        "before": {"contacts": len(initial), "max_depth_m": max_before},
        "moved_bodies_m": moved,
        "excluded_pairs": exclude_names,
        "after_floor_plane_contacts": [[a, b, d] for a, b, d in world_left],
    }
    return positions, exclude_names, report


def find_unsupported_bodies(xml_path, duration=SUPPORT_TEST_TIME, max_drop=SUPPORT_MAX_DROP):
    """Short drop test: dynamic bodies that fall freely have nothing under them in the scan
    (ceiling lights, wall fixtures), so they are really mounted and should be static."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    steps = int(duration / model.opt.timestep)

    def fallen(frozen):
        """Joints that drop more than max_drop while the `frozen` joints are held in place."""
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        qpos0 = data.qpos.copy()
        for _ in range(steps):
            mujoco.mj_step(model, data)
            for j in frozen:
                qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
                data.qpos[qa:qa + 7] = qpos0[qa:qa + 7]
                data.qvel[da:da + 6] = 0.0
        return {
            j for j in free
            if j not in frozen and qpos0[model.jnt_qposadr[j] + 2] - data.qpos[model.jnt_qposadr[j] + 2] > max_drop
        }

    candidates = fallen(set())
    # An object may only fall because what it rests on fell (tablet on a table top), so
    # re-test each candidate alone with the other candidates held in place.
    mounted = [j for j in candidates if j in fallen(candidates - {j})]
    return sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])) for j in mounted)


def build_scene(variant=DEFAULT_VARIANT, target_roi=None, bg_collision=True, forced_static=None, probe=None,
                layout=None, keep_scanned_pose=False):
    """Build the scene XML of a variant (README Module 4 simulation steps 1-3).

    `forced_static` reuses the static/dynamic split of another variant (so every benchmark
    variant simulates the same set of dynamic bodies); when None, the drop test decides.
    `layout` = (positions, excludes) of another variant: the resolver is skipped so that
    variants differ only in their meshes, not in start poses or excluded pairs.
    `keep_scanned_pose` writes every body at its scanned pose (the resolver still decides the
    excluded pairs); use it with the sleeping Topology Guard so nothing moves until touched.
    Returns a dict with the XML path, the build report and the layout used.
    """
    out_dir = scene_dir(variant)
    xml_path = scene_xml(variant)
    t0 = time.time()
    print(f"\n[1] Building '{variant}' scene...")
    bg_mesh, bg_parts = generate_unified_background(out_dir, variant)
    bg_centroid = np.asarray(bg_mesh.vertices).mean(axis=0)
    bg_hulls = run_coacd_on_background(bg_mesh, out_dir) if bg_collision else []

    objects = load_scene_objects(variant, out_dir, target_roi, forced_static or ())
    n_static = sum(o["static"] for o in objects)
    print(f"Scene objects: {n_static} static, {len(objects) - n_static} dynamic")
    for o in objects:
        qem = "" if o["ratio"] >= 1.0 else f"  (QEM x{o['ratio']:.2f})"
        print(f"  {o['id']:<8} {o['category']:<16} {'STATIC' if o['static'] else 'dynamic'}"
              f"{'  soft' if o['soft'] else ''}{qem}")

    mounted = []
    if layout is not None:
        positions, excludes = layout
        build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid, positions, excludes)
        model = mujoco.MjModel.from_xml_path(xml_path)
        residual = _penetrations(model, mujoco.MjData(model), PENETRATION_TOL)
        print(f"\n[1b] Reusing reference layout: {len(residual)} contacts deeper than "
              f"{PENETRATION_TOL * 1000:.0f} mm, max {max((p[2] for p in residual), default=0.0) * 1000:.1f} mm")
        report = {
            "tolerance_m": PENETRATION_TOL,
            "layout": "reused from reference variant",
            "before": {"contacts": len(residual), "max_depth_m": max((p[2] for p in residual), default=0.0)},
            "excluded_pairs": [list(pair) for pair in excludes],
        }
        build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid, positions, excludes, probe)
        report.update({
            "variant": variant,
            "target_roi": target_roi,
            "auto_static": [],
            "static_ids": sorted(o["id"] for o in objects if o["static"]),
            "dynamic_ids": sorted(o["id"] for o in objects if not o["static"]),
            "build_time_s": time.time() - t0,
        })
        with open(os.path.join(out_dir, "penetration_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        return {"xml": xml_path, "objects": objects, "report": report, "layout": layout}

    build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid)
    print("\n[1b] Resolving initial interpenetration...")
    positions, excludes, report = resolve_initial_penetration(xml_path, objects)
    build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid, positions, excludes)

    if forced_static is None:
        print("\n[1c] Detecting unsupported (wall/ceiling-mounted) objects...")
        mounted = find_unsupported_bodies(xml_path)
        if mounted:
            print(f"Making {len(mounted)} unsupported objects static: {', '.join(mounted)}")
            for obj in objects:
                if obj["id"] in mounted:
                    obj["static"] = True
            build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid)
            positions, excludes, report = resolve_initial_penetration(xml_path, objects)
    if keep_scanned_pose:
        build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid, {}, excludes, probe, "scanned")
    else:
        build_unified_xml(xml_path, objects, bg_parts, bg_hulls, bg_centroid, positions, excludes, probe)

    report.update({
        "variant": variant,
        "target_roi": target_roi,
        "start_pose": "scanned" if keep_scanned_pose else "resolved",
        "auto_static": mounted,
        "static_ids": sorted(o["id"] for o in objects if o["static"]),
        "dynamic_ids": sorted(o["id"] for o in objects if not o["static"]),
        "build_time_s": time.time() - t0,
    })
    with open(os.path.join(out_dir, "penetration_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    return {"xml": xml_path, "objects": objects, "report": report, "layout": (positions, excludes)}


def park_probe(model, data):
    """Keep the task agent far above the scene when no task drives it."""
    probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")
    if probe >= 0:
        data.mocap_pos[model.body_mocapid[probe]] = [0.0, 0.0, 100.0]


def check_stability(xml_path, steps=2000, sleep=False):
    """Headless run through the Topology Guard: velocities, drift and penetration of dynamic bodies."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    guard = TopologyGuard(model, data, sleep=sleep)
    park_probe(model, data)
    guard.start()
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    start = {j: data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3].copy() for j in free}

    t0 = time.time()
    peak_speed, max_pen = 0.0, 0.0
    for _ in range(steps):
        park_probe(model, data)
        guard.step()
        speeds = [np.linalg.norm(data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 3]) for j in free]
        peak_speed = max(peak_speed, max(speeds, default=0.0))
        if data.ncon:
            max_pen = max(max_pen, max(-data.contact[i].dist for i in range(data.ncon)))
    wall = time.time() - t0

    final_speed = max((np.linalg.norm(data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 3]) for j in free), default=0.0)
    drift = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j):
            float(np.linalg.norm(data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3] - start[j]))
        for j in free
    }
    worst = sorted(drift.items(), key=lambda kv: -kv[1])[:5]
    print("\n[Stability check]")
    print(f"  simulated {steps * model.opt.timestep:.1f}s in {wall:.1f}s wall time")
    print(f"  peak linear speed {peak_speed:.3f} m/s, final {final_speed:.3f} m/s")
    print(f"  max penetration during run {max_pen * 1000:.1f} mm")
    print(f"  topology guard faults: {len(guard.faults)}"
          + (f", sleeping mode (woken: {len(guard.woken)})" if sleep else ""))
    print("  largest drift: " + ", ".join(f"{n} {d:.3f} m" for n, d in worst))
    return {"peak_speed": peak_speed, "final_speed": final_speed, "max_penetration": max_pen,
            "drift": drift, "guard": guard.report()}


def _console_commands():
    """Read "ROI_a ROI_b" lines typed in the terminal while the viewer runs (fallback when
    double-click picking is awkward). Returns a queue of name lists."""
    import queue
    import threading

    commands = queue.Queue()

    def roi_name(word):
        digits = word.upper().replace("ROI", "").strip("_")
        return f"ROI_{digits}" if digits.isdigit() else word

    def reader():
        for line in sys.stdin:
            names = [roi_name(w) for w in line.replace(",", " ").split()]
            if names:
                commands.put(names)

    threading.Thread(target=reader, daemon=True).start()
    return commands


def launch_viewer(xml_path, sleep=False, task_args=None, speed=1.0):
    """Interactive viewer. With `task_args` ({"start", "target", "actor", "wait_for_click"}) the
    autonomous task drives the actor and its report is saved when a run finishes; without,
    objects are dragged by hand."""
    print(f"\nLoading {xml_path}...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"MuJoCo failed to parse XML: {e}")
        sys.exit(1)

    data = mujoco.MjData(model)
    guard = TopologyGuard(model, data, sleep=sleep)
    task = None
    if task_args is not None:
        from simulation.auto_task import AutonomousTask, print_summary, save_report
        task = AutonomousTask(model, data, guard=guard, **task_args)
        probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")
    else:
        park_probe(model, data)
    guard.start()

    print("\n" + "="*60)
    print("                 MUJOCO UNIFIED SIMULATION")
    print("="*60)
    print("Huong dan thao tac:")
    print("  • Chuot Trai/Phai/Giua: Xoay, cuon, di chuyen camera")
    if task is not None and task.wait_for_click:
        print("  • DOUBLE-click (chuot trai, 2 lan nhanh) vat 1, roi DOUBLE-click vat 2")
        print("    -> vat 1 di theo duong thang toi va cham vao vat 2, roi bao cao ket qua")
        print("  • Hoac go vao terminal nay: ROI_39 ROI_41  (Enter)")
        print("  • Vat 1 phai la vat dong (dynamic); vat 2 co the la vat bat ky, ke ca tuong")
    elif task is not None:
        print("  • Vat / agent (hop do) tu di toi muc tieu, va cham, roi bao cao ket qua")
    else:
        print("  • Double-click chon vat the -> Ctrl + Chuot Phai de keo tha")
    print("  • Space: Tam dung/Tiep tuc mo phong")
    print("  • Backspace: Reset mo phong" + (" (chon lai tu dau)" if task is not None else ""))
    print("  • Phim 3: Bat/tat hien thi collision hull")
    if sleep:
        print("  • Vat dung yen dung vi tri scan cho toi khi bi keo/cham vao (--no-sleep de tat)")
    print("="*60 + "\n")

    last_report = None
    last_select = 0
    commands = _console_commands() if task is not None else None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if task is not None:
                # Double-click selects a body (viewer.perturb.select): 1st = actor, 2nd = target.
                select = int(viewer.perturb.select)
                if select != last_select:
                    last_select = select
                    if select > 0 and select != probe_id:
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, select) or ""
                        print(f"[Viewer] selected {name or f'body {select}'}")
                        try:
                            task.select(name)
                        except RuntimeError as e:
                            print(f"[Task] {e}")
                while not commands.empty():
                    names = commands.get()
                    try:
                        if len(names) >= 2:
                            if task._check_actor(names[0]):
                                task.start_run(names[0], names[1], selected_by="console")
                        else:
                            task.select(names[0])
                    except RuntimeError as e:
                        print(f"[Task] {e}")
                task.before_step()
            else:
                park_probe(model, data)
            if not guard.step():
                print(f"[TopologyGuard] {guard.faults[-1]['reason']} at t={guard.faults[-1]['time']:.3f}s -> restored")
            if task is not None:
                task.after_step()
                if task.report is not None and task.report is not last_report:
                    last_report = task.report
                    print_summary(task.report)
                    print(f"\nTask report saved to {save_report(task.report, OUTPUT_DIR)}")
            viewer.sync()
            # Keep the viewer close to real time.
            # `speed` > 1 plays the simulation faster than real time (same physics).
            remaining = model.opt.timestep / speed - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def scene_is_stale(xml_path, pose_mode=None, need_probe=False):
    """Return a reason when the scene must be rebuilt: no XML yet, built for another start pose,
    without the task agent, or a mesh it references is missing / newer than the XML (Module 4 preprocessing re-run)."""
    if not os.path.exists(xml_path):
        return "no scene XML yet"
    import re

    xml_time = os.path.getmtime(xml_path)
    with open(xml_path, encoding="utf-8") as f:
        text = f.read()
    if any(name in text for name in ("module5_postprocessing", "module6_physics")):
        return "scene references paths from before the Module 4 migration"
    if pose_mode is not None and f"start_pose: {pose_mode}" not in text:
        return f"start pose changed to '{pose_mode}'"
    if f"scene_version: {SCENE_VERSION}" not in text:
        return "scene format changed (thin objects)"
    if need_probe and '<body name="probe"' not in text:
        return "scene has no task agent yet"
    for mesh_file in re.findall(r'<mesh name="[^"]*" file="([^"]+)"', text):
        if not os.path.isabs(mesh_file):
            mesh_file = os.path.join(os.path.dirname(xml_path), mesh_file)
        if not os.path.exists(mesh_file):
            return f"missing mesh {mesh_file}"
        if os.path.getmtime(mesh_file) > xml_time:
            return f"mesh newer than scene: {mesh_file}"
    return None


def _arg_value(prefix, default):
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def main():
    print("=" * 70)
    print(f"MODULE 4 SIMULATION - UNIFIED INTERACTIVE MESH SIMULATION  (dataset: {INPUT_FOLDER_NAME})")
    print("=" * 70)
    if not os.path.exists(REPAIRED_MESH):
        raise SystemExit(
            f"No Module 4 preprocessing output for dataset '{INPUT_FOLDER_NAME}' ({REPAIRED_MESH}).\n"
            f"Run: python scripts/05_run_postprocessing.py --input={INPUT_FOLDER_NAME}"
        )

    if "--benchmark" in sys.argv:
        from simulation import benchmarks
        benchmarks.main()
        return

    variant = _arg_value("--variant=", DEFAULT_VARIANT)
    if variant not in VARIANTS:
        raise SystemExit(f"Unknown variant '{variant}', choose one of {VARIANTS}")
    xml_path = scene_xml(variant)

    from simulation import auto_task

    start = _arg_value("--start=", None)
    task_args = None if "--manual" in sys.argv else {
        "start": auto_task.parse_start(start) if start else None,
        "target": _arg_value("--target=", None),
        "actor": _arg_value("--actor=", None),
    }

    sleep = "--no-sleep" not in sys.argv
    stale = scene_is_stale(xml_path, "scanned" if sleep else "resolved", need_probe=True)
    if stale or "--rebuild" in sys.argv:
        if stale:
            print(f"\n[1] Rebuilding scene: {stale}")
        # The agent is part of every interactive scene; its pose is set at run time.
        build_scene(variant, bg_collision="--no-bg-collision" not in sys.argv, keep_scanned_pose=sleep,
                    probe=auto_task.agent_spec())
    else:
        print(f"\n[1] Found existing Unified Scene XML: {xml_path}")
        print("    (Use --rebuild to force regeneration of the scene)")

    if "--check" in sys.argv:
        check_stability(xml_path, sleep=sleep)

    if "--no-view" not in sys.argv:
        print("\n[2] Launching Viewer...")
        if task_args is not None:
            # Viewer: wait for double-clicked actor + target unless --auto / --target / --actor say what to do.
            task_args["wait_for_click"] = "--auto" not in sys.argv
        launch_viewer(xml_path, sleep=sleep, task_args=task_args, speed=float(_arg_value("--speed=", 1.0)))
    elif task_args is not None:
        print("\n[2] Running the autonomous task (headless)...")
        auto_task.run_headless(xml_path, OUTPUT_DIR, sleep=sleep, **task_args)


if __name__ == "__main__":
    main()
