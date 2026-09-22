import os
import time
import open3d as o3d
import mujoco


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XML_PATH = os.path.join(
    BASE_DIR,
    "scene.xml"
)

PLY_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Output",
    "CollisionMesh_ROI1",
    "Collision_Mesh_ROI1.ply"
)

OBJ_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Output",
    "CollisionMesh_ROI1",
    "Collision_Mesh_ROI1.obj"
)

PLY_PATH = os.path.abspath(PLY_PATH)
OBJ_PATH = os.path.abspath(OBJ_PATH)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("STEP 10 - MUJOCO SIMULATION")
print("=" * 60)


# ============================================================
# STEP 10.1 - PLY -> OBJ
# ============================================================

print("\n[1] CONVERT COLLISION MESH")

print(f"Input PLY : {PLY_PATH}")
print(f"Output OBJ: {OBJ_PATH}")

if not os.path.exists(PLY_PATH):
    raise FileNotFoundError(
        f"Collision mesh not found:\n{PLY_PATH}"
    )

mesh = o3d.io.read_triangle_mesh(PLY_PATH)

if mesh.is_empty():
    raise RuntimeError(
        "Failed to load collision mesh from PLY."
    )

print(f"Vertices  : {len(mesh.vertices)}")
print(f"Triangles : {len(mesh.triangles)}")

success = o3d.io.write_triangle_mesh(
    OBJ_PATH,
    mesh,
    write_ascii=True
)

if not success or not os.path.exists(OBJ_PATH):
    raise RuntimeError(
        "PLY -> OBJ conversion failed."
    )

print("PLY -> OBJ conversion: PASS")


# ============================================================
# STEP 10.2 - UPDATE XML
# ============================================================

print("\n[2] PREPARE MUJOCO MODEL")

# MuJoCo scene.xml nằm cùng thư mục với script.
# scene.xml sẽ trỏ tới OBJ.
print(f"XML: {XML_PATH}")

if not os.path.exists(XML_PATH):
    raise FileNotFoundError(
        f"scene.xml not found:\n{XML_PATH}"
    )


# ============================================================
# STEP 10.3 - LOAD MUJOCO
# ============================================================

print("\n[3] LOAD MUJOCO")

try:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("MuJoCo loading: PASS")

except Exception as e:
    print("MuJoCo loading: FAILED")
    print(f"Error: {e}")
    raise


# ============================================================
# STEP 10.4 - MODEL INFORMATION
# ============================================================

print("\n" + "-" * 60)
print("MODEL INFORMATION")
print("-" * 60)

print(f"Number of bodies : {model.nbody}")
print(f"Number of geoms  : {model.ngeom}")
print(f"Timestep          : {model.opt.timestep}")


# ============================================================
# STEP 10.5 - RUN SIMULATION
# ============================================================

print("\n" + "-" * 60)
print("RUNNING SIMULATION")
print("-" * 60)

SIMULATION_STEPS = 5000

print(f"\nSimulation steps : {SIMULATION_STEPS}")

start_time = time.perf_counter()

for _ in range(SIMULATION_STEPS):
    mujoco.mj_step(model, data)

end_time = time.perf_counter()

total_time = end_time - start_time
average_step = total_time / SIMULATION_STEPS
steps_per_second = SIMULATION_STEPS / total_time


# ============================================================
# RESULTS
# ============================================================

print(f"Total time       : {total_time:.6f} s")
print(f"Average step     : {average_step * 1000:.6f} ms")
print(f"Steps / second   : {steps_per_second:.2f}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("STEP 10 TEST COMPLETED")
print("=" * 60)