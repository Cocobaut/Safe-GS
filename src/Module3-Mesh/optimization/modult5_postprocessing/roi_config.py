"""Shared ROI, paths, and thresholds for Module 5.

Module 2 AABB is not wired yet, so ROI_MIN / ROI_MAX are the
current hand-picked object box. CURRENT_TASK_ROI uses the same
box until a task-specific ROI is provided.
"""

import os

import numpy as np
import open3d as o3d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
INPUT_DIR = os.path.join(BASE_DIR, "Input")

# Temporary Module-2 stand-in (manual AABB).
ROI_MIN = np.array([1.31638556, -1.97638444, -1.2299122], dtype=np.float64)
ROI_MAX = np.array([2.24038578, -0.92432044, -0.49570042], dtype=np.float64)

# Task ROI: same as object ROI until Module 2 / task config exists.
CURRENT_TASK_ROI_MIN = ROI_MIN.copy()
CURRENT_TASK_ROI_MAX = ROI_MAX.copy()

SUBROIS = {
    "R0": {"ratio": 0.40, "detail_level": "HIGH"},
    "R1": {"ratio": 0.60, "detail_level": "MEDIUM_HIGH"},
    "R2": {"ratio": 0.80, "detail_level": "MEDIUM"},
    "R3": {"ratio": 1.00, "detail_level": "COARSE"},
}

MESH_LEVELS = [
    "M0_HighRes",
    "M1_MediumHigh",
    "M2_Medium",
    "M3_Coarse",
]

MESH_SUBROI_PAIRS = [
    {"mesh": "M0_HighRes", "subroi": "R0"},
    {"mesh": "M1_MediumHigh", "subroi": "R1"},
    {"mesh": "M2_Medium", "subroi": "R2"},
    {"mesh": "M3_Coarse", "subroi": "R3"},
]

MESH_MAPPING = {
    "R0": "M0_HighRes.ply",
    "R1": "M1_MediumHigh.ply",
    "R2": "M2_Medium.ply",
    "R3": "M3_Coarse.ply",
}

# Keep the high-detail core while QEM-simplifying the rest.
PROTECT_SUBROI_FOR_MESH = {
    "M0_HighRes": None,
    "M1_MediumHigh": "R0",
    "M2_Medium": "R0",
    "M3_Coarse": "R0",
}

TARGET_TRIANGLES = {
    "M0_HighRes": None,
    "M1_MediumHigh": 18000,
    "M2_Medium": 12000,
    "M3_Coarse": 6000,
}

VERTEX_MERGE_EPS = 1e-6
DEGENERATE_AREA_EPS = 1e-12
SMALL_COMPONENT_TRIANGLES = 2900
SMALL_COMPONENT_AREA = 0.005
MAX_HOLE_LOOP_EDGES = 16
SPATIAL_INFLUENCE_SIGMA = 0.10
CHAMFER_SAMPLES = 30000
TAU_ROI = 0.0045
CONTAINMENT_TOL = 1e-8
VISUALIZE = False

RAW_SCENE_MESH = os.path.join(INPUT_DIR, "office0_mesh.ply")
OBJECT_MESH = os.path.join(OUTPUT_DIR, "Object_ROI1.ply")
SCENE_MESH = os.path.join(OUTPUT_DIR, "Scene_ROI1.ply")
CLEANED_MESH = os.path.join(OUTPUT_DIR, "Cleaned_Object_ROI1.ply")
COMPONENT_MESH = os.path.join(OUTPUT_DIR, "ComponentFiltered_Object_ROI1.ply")
COMPONENT_LABELS = os.path.join(OUTPUT_DIR, "ConnectedComponents_ROI1.npy")
COMPONENT_REPORT = os.path.join(OUTPUT_DIR, "ConnectedComponentReport_ROI1.json")
REPAIRED_MESH = os.path.join(OUTPUT_DIR, "TopologyRepaired_Object_ROI1.ply")
TOPOLOGY_REPORT = os.path.join(OUTPUT_DIR, "TopologyRepairReport_ROI1.json")
ROI_MAPPING_REPORT = os.path.join(OUTPUT_DIR, "ROISubROIMappingReport_ROI1.json")
ROI_LABELS = os.path.join(OUTPUT_DIR, "TriangleSubROILabels_ROI1.npy")
TRIANGLE_CENTROIDS = os.path.join(OUTPUT_DIR, "TriangleCentroids_ROI1.npy")
TRIANGLE_ROI_DISTANCES = os.path.join(OUTPUT_DIR, "TriangleROIDistances_ROI1.npy")
TRIANGLE_SPATIAL_INFLUENCE = os.path.join(
    OUTPUT_DIR,
    "TriangleSpatialInfluence_ROI1.npy",
)
ADAPTIVE_DIR = os.path.join(OUTPUT_DIR, "AdaptiveMeshes_ROI1")
QEM_REPORT = os.path.join(ADAPTIVE_DIR, "QEMAdaptiveMeshReport_ROI1.json")
LINKING_DIR = os.path.join(OUTPUT_DIR, "MeshSubROILinking_ROI1")
LINKING_REPORT = os.path.join(LINKING_DIR, "MeshSubROILinkingSummary_ROI1.json")
LINKING_MAPPING = os.path.join(LINKING_DIR, "MeshSubROIMapping_ROI1.json")
SELECTION_DIR = os.path.join(OUTPUT_DIR, "AdaptiveMeshSelection_ROI1")
SELECTED_MESH = os.path.join(SELECTION_DIR, "Selected_Mesh_ROI1.ply")
SELECTION_REPORT = os.path.join(SELECTION_DIR, "AdaptiveMeshSelectionReport_ROI1.json")
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "MeshQualityValidation_ROI1")
VALIDATED_MESH = os.path.join(VALIDATION_DIR, "Validated_Mesh_ROI1.ply")
VALIDATION_REPORT = os.path.join(VALIDATION_DIR, "MeshQualityValidationReport_ROI1.json")
COLLISION_DIR = os.path.join(OUTPUT_DIR, "CollisionMesh_ROI1")
COLLISION_MESH = os.path.join(COLLISION_DIR, "Collision_Mesh_ROI1.ply")
COLLISION_REPORT = os.path.join(COLLISION_DIR, "CollisionMeshReport_ROI1.json")


def roi_center_size(roi_min=None, roi_max=None):
    roi_min = ROI_MIN if roi_min is None else np.asarray(roi_min, dtype=np.float64)
    roi_max = ROI_MAX if roi_max is None else np.asarray(roi_max, dtype=np.float64)
    return (roi_min + roi_max) / 2.0, roi_max - roi_min


def create_subrois(roi_min=None, roi_max=None):
    roi_min = ROI_MIN if roi_min is None else np.asarray(roi_min, dtype=np.float64)
    roi_max = ROI_MAX if roi_max is None else np.asarray(roi_max, dtype=np.float64)
    center, size = roi_center_size(roi_min, roi_max)
    half = size / 2.0

    subrois = {}
    for name, config in SUBROIS.items():
        ratio = config["ratio"]
        subrois[name] = {
            "ratio": ratio,
            "detail_level": config["detail_level"],
            "min": center - half * ratio,
            "max": center + half * ratio,
        }
    return subrois


def point_inside_aabb(point, box_min, box_max, tol=0.0):
    point = np.asarray(point)
    return np.all((point >= box_min - tol) & (point <= box_max + tol))


def aabb_contains(outer_min, outer_max, inner_min, inner_max, tol=CONTAINMENT_TOL):
    return np.all(inner_min >= outer_min - tol) and np.all(inner_max <= outer_max + tol)


def distance_to_aabb(point, box_min, box_max):
    point = np.asarray(point, dtype=np.float64)
    delta = np.maximum(np.maximum(box_min - point, 0.0), point - box_max)
    return float(np.linalg.norm(delta))


def triangle_centroids(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.mean(vertices[triangles], axis=1)


def maybe_visualize(mesh, window_name):
    if not VISUALIZE:
        return
    view = o3d.geometry.TriangleMesh(mesh)
    view.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries(
        [view],
        window_name=window_name,
        width=1400,
        height=900,
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
