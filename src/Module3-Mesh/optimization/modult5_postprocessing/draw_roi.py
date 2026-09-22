import open3d as o3d
import numpy as np


# ============================================================
# CONFIG
# ============================================================

MESH_PATH = r"src\modult5_postprocessing\Input\office0_mesh.ply"

# Tên các ROI cần tạo
ROI_NAMES = [
    "Animal",
    "Plant",
    "AirConditioner",
    "TV"
]


# ============================================================
# LOAD MESH
# ============================================================

print("=" * 60)
print("Loading mesh...")
print("=" * 60)

mesh = o3d.io.read_triangle_mesh(MESH_PATH)

if mesh.is_empty():
    raise RuntimeError("Mesh is empty or could not be loaded.")

mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print(f"Vertices : {len(vertices):,}")
print(f"Triangles: {len(triangles):,}")

mesh_min = vertices.min(axis=0)
mesh_max = vertices.max(axis=0)

mesh_center = (mesh_min + mesh_max) / 2.0
mesh_extent = mesh_max - mesh_min

print("\nScene bounding box:")
print("Min:", mesh_min)
print("Max:", mesh_max)
print("Center:", mesh_center)
print("Extent:", mesh_extent)


# ============================================================
# ROI INITIALIZATION
# ============================================================

# ROI bắt đầu nhỏ, khoảng 15% kích thước scene
initial_size = mesh_extent * 0.15

# Đảm bảo ROI không có kích thước quá nhỏ
initial_size = np.maximum(initial_size, 0.01)

roi_center = mesh_center.copy()
roi_size = initial_size.copy()

# Bước di chuyển = 2% kích thước scene
move_step = mesh_extent * 0.02
move_step = np.maximum(move_step, 0.001)

# Bước resize = 5% kích thước ROI ban đầu
resize_step = initial_size * 0.05
resize_step = np.maximum(resize_step, 0.001)


# ============================================================
# CREATE ROI BOX
# ============================================================

def create_roi_box(center, size):
    """
    Tạo wireframe box từ center + size.
    """

    half = size / 2.0

    x, y, z = center
    hx, hy, hz = half

    points = np.array([
        [x - hx, y - hy, z - hz],  # 0
        [x + hx, y - hy, z - hz],  # 1
        [x + hx, y + hy, z - hz],  # 2
        [x - hx, y + hy, z - hz],  # 3
        [x - hx, y - hy, z + hz],  # 4
        [x + hx, y - hy, z + hz],  # 5
        [x + hx, y + hy, z + hz],  # 6
        [x - hx, y + hy, z + hz],  # 7
    ], dtype=np.float64)

    lines = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],

        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],

        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ], dtype=np.int32)

    box = o3d.geometry.LineSet()

    box.points = o3d.utility.Vector3dVector(points)
    box.lines = o3d.utility.Vector2iVector(lines)

    # Màu ROI
    colors = np.tile(
        np.array([[1.0, 0.0, 0.0]]),
        (len(lines), 1)
    )

    box.colors = o3d.utility.Vector3dVector(colors)

    return box


roi_box = create_roi_box(
    roi_center,
    roi_size
)


# ============================================================
# STATE
# ============================================================

state = {
    "center": roi_center.copy(),
    "size": roi_size.copy(),
    "done": False,
    "cancel": False
}


# ============================================================
# UPDATE ROI
# ============================================================

def update_roi(vis):
    """
    Update vị trí/kích thước ROI box.
    """

    new_box = create_roi_box(
        state["center"],
        state["size"]
    )

    # Xóa box cũ
    vis.clear_geometries()

    # Thêm lại mesh
    vis.add_geometry(mesh)

    # Thêm ROI
    vis.add_geometry(new_box)

    return False


# ============================================================
# MOVE FUNCTIONS
# ============================================================

def move_x_plus(vis):
    state["center"][0] += move_step[0]
    update_roi(vis)
    print("ROI center:", state["center"])
    return False


def move_x_minus(vis):
    state["center"][0] -= move_step[0]
    update_roi(vis)
    print("ROI center:", state["center"])
    return False


def move_y_plus(vis):
    state["center"][1] += move_step[1]
    update_roi(vis)
    print("ROI center:", state["center"])
    return False


def move_y_minus(vis):
    state["center"][1] -= move_step[1]
    update_roi(vis)
    print("ROI center:", state["center"])
    return False


def move_z_plus(vis):
    state["center"][2] += move_step[2]
    update_roi(vis)
    print("ROI center:", state["center"])
    return False


def move_z_minus(vis):
    state["center"][2] -= move_step[2]
    update_roi(vis)
    print("ROI center:", state["center"])
    return False


# ============================================================
# RESIZE FUNCTIONS
# ============================================================

def grow_roi(vis):
    state["size"] += resize_step

    # Không cho ROI vượt quá scene
    state["size"] = np.minimum(
        state["size"],
        mesh_extent
    )

    update_roi(vis)

    print("ROI size:", state["size"])

    return False


def shrink_roi(vis):
    min_size = np.maximum(
        mesh_extent * 0.01,
        0.001
    )

    state["size"] -= resize_step

    state["size"] = np.maximum(
        state["size"],
        min_size
    )

    update_roi(vis)

    print("ROI size:", state["size"])

    return False


# ============================================================
# SAVE ROI
# ============================================================

def save_roi(vis):
    state["done"] = True

    print("\n" + "=" * 60)
    print("ROI SELECTED")
    print("=" * 60)

    center = state["center"]
    size = state["size"]

    half = size / 2.0

    roi_min = center - half
    roi_max = center + half

    print("Center:")
    print(center)

    print("\nSize:")
    print(size)

    print("\nROI Min:")
    print(roi_min)

    print("\nROI Max:")
    print(roi_max)

    print("=" * 60)

    return False


# ============================================================
# CANCEL
# ============================================================

def cancel_roi(vis):
    state["cancel"] = True
    state["done"] = True

    print("\nROI selection cancelled.")

    return False


# ============================================================
# VISUALIZER
# ============================================================

vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window(
    window_name="Safe-GS - ROI Selection",
    width=1400,
    height=900
)

# Add mesh
vis.add_geometry(mesh)

# Add ROI
vis.add_geometry(roi_box)


# ============================================================
# KEYBOARD CALLBACKS
# ============================================================

# X axis
vis.register_key_callback(
    ord("D"),
    move_x_plus
)

vis.register_key_callback(
    ord("A"),
    move_x_minus
)


# Y axis
vis.register_key_callback(
    ord("W"),
    move_y_plus
)

vis.register_key_callback(
    ord("S"),
    move_y_minus
)


# Z axis
vis.register_key_callback(
    ord("E"),
    move_z_plus
)

vis.register_key_callback(
    ord("Q"),
    move_z_minus
)


# Resize
vis.register_key_callback(
    ord("R"),
    grow_roi
)

vis.register_key_callback(
    ord("F"),
    shrink_roi
)


# Save
vis.register_key_callback(
    257,       # ENTER
    save_roi
)


# Cancel
vis.register_key_callback(
    256,       # ESC
    cancel_roi
)


# ============================================================
# RENDER SETTINGS
# ============================================================

render_option = vis.get_render_option()

render_option.background_color = np.array([
    0.05,
    0.05,
    0.05
])


# ============================================================
# CAMERA
# ============================================================

view_control = vis.get_view_control()

view_control.set_lookat(mesh_center)

# Zoom scene
view_control.set_zoom(0.8)


# ============================================================
# HELP
# ============================================================

print("\n")
print("=" * 60)
print("SAFE-GS ROI SELECTOR")
print("=" * 60)

print("""
Mouse:
    Left drag       : Rotate scene
    Wheel           : Zoom
    Right drag      : Pan

Move ROI:
    A               : X -
    D               : X +

    S               : Y -
    W               : Y +

    Q               : Z -
    E               : Z +

Resize ROI:
    R               : Increase ROI size
    F               : Decrease ROI size

Finish:
    ENTER           : Save ROI
    ESC             : Cancel

Current ROI:
""")

print("Center:", state["center"])
print("Size  :", state["size"])

print("=" * 60)


# ============================================================
# MAIN LOOP
# ============================================================

while not state["done"]:

    vis.poll_events()
    vis.update_renderer()


# ============================================================
# CLOSE
# ============================================================

vis.destroy_window()


# ============================================================
# FINAL OUTPUT
# ============================================================

if not state["cancel"]:

    center = state["center"]
    size = state["size"]

    half = size / 2.0

    roi_min = center - half
    roi_max = center + half

    print("\n")
    print("=" * 60)
    print("FINAL ROI")
    print("=" * 60)

    print("ROI Min =", roi_min)
    print("ROI Max =", roi_max)

    print("\nBạn có thể copy 2 dòng này vào roi_config.py:")
    print(f"ROI_MIN = np.array({roi_min.tolist()})")
    print(f"ROI_MAX = np.array({roi_max.tolist()})")

    print("=" * 60)