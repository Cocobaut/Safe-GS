"""Shared paths and ROI configuration for Module 4 preprocessing."""

import json
import os
import glob

import numpy as np
import open3d as o3d

def _validate_dataset(value):
    value = value.strip()
    if not value or value in {".", ".."} or any(c in value for c in "/\\:"):
        raise ValueError(f"Invalid dataset folder: {value!r}")
    return value


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "Input")

# Dataset = one folder under Input/ (office0, room0, ...). Set with MODULE5_INPUT_FOLDER or
# the --input=NAME argument of scripts/05_run_postprocessing.py / 06_run_simulation.py.
INPUT_FOLDER_NAME = _validate_dataset(os.environ.get("MODULE5_INPUT_FOLDER", "office0"))
INPUT_FOLDER = os.path.join(INPUT_DIR, INPUT_FOLDER_NAME)
# Outputs are kept per dataset so runs on different scenes never mix.
OUTPUT_DIR = os.path.join(BASE_DIR, "Output", INPUT_FOLDER_NAME)

if not os.path.isdir(INPUT_FOLDER):
    raise FileNotFoundError(f"Dataset input folder does not exist: {INPUT_FOLDER}")
_mask_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*_3d_masks.json")))
if not _mask_files:
    raise FileNotFoundError(f"No *_3d_masks.json found in {INPUT_FOLDER}")
if len(_mask_files) > 1:
    raise ValueError(f"Multiple ROI mask files found in {INPUT_FOLDER}: {_mask_files}")
ROI_BOX_FILE = _mask_files[0]



def mesh_for_roi_file(path):
    """Resolve the detailed reference mesh paired with a mask JSON file."""
    folder = os.path.dirname(path)
    stem = os.path.basename(folder)
    candidates = [
        os.path.join(folder, f"visualization_3d_mask_origin_{stem}.ply"),
        os.path.join(folder, f"visualization_3d_mask_{stem}.ply"),
        os.path.join(INPUT_DIR, f"visualization_roi_combined_{stem}.ply"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.environ.get("MODULE5_MESH_FILE", candidates[0] if candidates else "")


def load_roi_boxes(path=ROI_BOX_FILE):
    """Load Segments exported by the ROI detection stage."""
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    rois = {}
    for index, record in enumerate(records, start=1):
        roi_id = f"ROI_{record.get('instance_id', index)}"
        
        verts_dict = record.get("vertices", {})
        if not verts_dict:
            # Fallback for old bbox format if needed
            aabb = record.get("aabb", {})
            if "min" in aabb and "max" in aabb:
                rois[roi_id] = {
                    "min": np.asarray(aabb["min"], dtype=np.float64),
                    "max": np.asarray(aabb["max"], dtype=np.float64),
                    "target_ratio": 0.30,
                    "detail_level": "HIGH",
                    "instance_id": record.get("instance_id", index),
                    "name": record.get("name", "unknown"),
                    "category": record.get("category", "unknown"),
                    "source": path,
                }
            continue
            
        verts_list = [v for k, v in verts_dict.items()]
        verts_np = np.asarray(verts_list, dtype=np.float64)
        if len(verts_np) == 0:
            continue
            
        faces_dict = record.get("faces", {})
        faces_list = [f for k, f in faces_dict.items()]
        faces_np = np.asarray(faces_list, dtype=np.int32)
        
        roi_min = verts_np.min(axis=0)
        roi_max = verts_np.max(axis=0)
        
        rois[roi_id] = {
            "min": roi_min,
            "max": roi_max,
            "target_ratio": 0.30,
            "detail_level": "HIGH",
            "instance_id": record.get("instance_id", index),
            "name": record.get("name", "unknown"),
            "category": record.get("category", "unknown"),
            "source": path,
            "segment_vertices": verts_np,
            "segment_faces": faces_np,
        }

    if not rois:
        raise ValueError(f"No valid segments found in ROI file: {path}")
    return rois


ALL_ROIS = load_roi_boxes()
_ROI_FILTER = os.environ.get("MODULE5_ROI_IDS", "").strip()
TEST_ROIS = ALL_ROIS
if _ROI_FILTER:
    requested_roi_ids = {item.strip() for item in _ROI_FILTER.split(",") if item.strip()}
    TEST_ROIS = {
        roi_id: roi for roi_id, roi in TEST_ROIS.items()
        if roi_id in requested_roi_ids
    }
    if not TEST_ROIS:
        raise ValueError(f"No requested ROI IDs found in {ROI_BOX_FILE}: {sorted(requested_roi_ids)}")
_ALL_ROI_MINS = np.asarray([roi["min"] for roi in ALL_ROIS.values()])
_ALL_ROI_MAXS = np.asarray([roi["max"] for roi in ALL_ROIS.values()])
ROI_MIN = _ALL_ROI_MINS.min(axis=0)
ROI_MAX = _ALL_ROI_MAXS.max(axis=0)

# The first supplied ROI is the default task ROI until a task-specific one is set.
CURRENT_TASK_ROI_MIN = next(iter(TEST_ROIS.values()))["min"].copy()
CURRENT_TASK_ROI_MAX = next(iter(TEST_ROIS.values()))["max"].copy()

SUBROIS = {
    "R0": {"ratio": 0.40, "detail_level": "HIGH"},
    "R1": {"ratio": 0.60, "detail_level": "MEDIUM_HIGH"},
    "R2": {"ratio": 0.80, "detail_level": "MEDIUM"},
    "R3": {"ratio": 1.00, "detail_level": "COARSE"},
}

MESH_LEVELS = list(TEST_ROIS)

MESH_SUBROI_PAIRS = [
    {"mesh": name, "subroi": name}
    for name in MESH_LEVELS
]

MESH_MAPPING = {
    name: f"AdaptiveMesh_{name.replace('_', '')}.ply"
    for name in MESH_LEVELS
}

# Keep the high-detail core while QEM-simplifying the rest.
PROTECT_SUBROI_FOR_MESH = {name: name for name in MESH_LEVELS}

TARGET_TRIANGLES = {
    name: None
    for name in MESH_LEVELS
}

VERTEX_MERGE_EPS = 1e-6
DEGENERATE_AREA_EPS = 1e-12
SMALL_COMPONENT_TRIANGLES = 0
SMALL_COMPONENT_AREA = 0.005
MAX_HOLE_LOOP_EDGES = 16
SPATIAL_INFLUENCE_SIGMA = 0.10
CHAMFER_SAMPLES = 30000
TAU_ROI = 0.0045
CONTAINMENT_TOL = 1e-8
VISUALIZE = False

RAW_SCENE_MESH = os.environ.get("MODULE5_MESH_FILE", mesh_for_roi_file(ROI_BOX_FILE))
if not os.path.isfile(RAW_SCENE_MESH):
    raise FileNotFoundError(f"Reference mesh does not exist: {RAW_SCENE_MESH}")
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

# Unified Mesh Optimization Outputs
OPTIMIZED_DIR = os.path.join(OUTPUT_DIR, "OptimizedMeshes")

# Physics role of each segment: structural / fixed segments are welded to the world,
# everything else gets a freejoint in the unified MuJoCo scene.
STATIC_CATEGORIES = {
    "wall", "floor", "ceiling", "rug", "door", "blinds", "window", "vent", "pillar",
    "wall-plug", "switch", "panel", "clock", "tv-screen", "lamp", "sofa", "shelf",
    "cabinet", "picture", "curtain",
}
# Thin sheet-like static segments: CoACD needs more, tighter hulls for these.
PLANAR_CATEGORIES = {"wall", "floor", "ceiling", "rug", "door", "blinds", "window"}
# Any segment larger than this (max AABB edge, metres) is treated as static.
STATIC_MAX_EXTENT = 1.5


def _id_set_from_env(name):
    return {item.strip() for item in os.environ.get(name, "").split(",") if item.strip()}


_FORCED_STATIC_IDS = _id_set_from_env("MODULE5_STATIC_IDS")
_FORCED_DYNAMIC_IDS = _id_set_from_env("MODULE5_DYNAMIC_IDS")


def is_static_roi(roi_id, roi):
    """Return True when a segment should be fixed to the world instead of free-falling."""
    if roi_id in _FORCED_STATIC_IDS:
        return True
    if roi_id in _FORCED_DYNAMIC_IDS:
        return False
    if str(roi.get("category", "")).lower() in STATIC_CATEGORIES:
        return True
    extent = np.asarray(roi["max"]) - np.asarray(roi["min"])
    return float(extent.max()) > STATIC_MAX_EXTENT


def segment_color(mesh_vertices, mesh_colors, seg_vertices, kdtree=None):
    """Dominant vertex colour (RGB in [0, 1]) of a segment, looked up in the coloured scene mesh."""
    if mesh_colors is None or len(mesh_colors) == 0 or len(seg_vertices) == 0:
        return None
    if kdtree is None:
        from scipy.spatial import cKDTree
        kdtree = cKDTree(mesh_vertices)
    _, idx = kdtree.query(np.asarray(seg_vertices, dtype=np.float64), k=1)
    quantized = np.round(np.asarray(mesh_colors)[idx] * 255).astype(np.int64)
    values, counts = np.unique(quantized, axis=0, return_counts=True)
    return (values[np.argmax(counts)] / 255.0).tolist()


def is_planar_roi(roi):
    return str(roi.get("category", "")).lower() in PLANAR_CATEGORIES


# Thin / soft segments (cloth, paper, ...). The physics scene can only simulate rigid bodies,
# so they stay rigid shells, but light ones: mass from the surface area, not from the box.
THIN_CATEGORIES = {"blanket", "towel", "cloth", "clothes", "sheet", "curtain", "paper", "bag", "mat"}
THIN_FLATNESS = 0.05    # smallest / largest PCA spread of the segment below this = flat splat


def is_thin_roi(roi):
    """Cloth-like category, or a segment flat enough to be a sheet (paper, flat splat)."""
    if str(roi.get("category", "")).lower() in THIN_CATEGORIES:
        return True
    verts = roi.get("segment_vertices")
    if verts is None or len(verts) < 4:
        return False
    spread = np.linalg.svd(np.asarray(verts) - np.mean(verts, axis=0), compute_uv=False)
    return bool(spread[-1] < THIN_FLATNESS * spread[0])


def optimized_roi_dir(roi_id):
    return os.path.join(OPTIMIZED_DIR, roi_id)

def optimized_visual_path(roi_id):
    return os.path.join(optimized_roi_dir(roi_id), f"Object_{roi_id}_Visual.obj")

def optimized_background_path(roi_id):
    return os.path.join(optimized_roi_dir(roi_id), f"Background_{roi_id}_QEM.obj")



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


def adaptive_mesh_filename(roi_id):
    return f"AdaptiveMesh_{roi_id.replace('_', '')}.ply"


def adaptive_mesh_path(roi_id):
    return os.path.join(ADAPTIVE_DIR, adaptive_mesh_filename(roi_id))


def adaptive_mesh_report_path(roi_id):
    return os.path.join(
        ADAPTIVE_DIR,
        f"AdaptiveMesh_{roi_id.replace('_', '')}_Report.json",
    )


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
