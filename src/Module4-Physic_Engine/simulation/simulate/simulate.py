import os
import time
import shutil

import mujoco
import mujoco.viewer
import open3d as o3d


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COLLISION_PLY = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "Output",
        "CollisionMesh_ROI1",
        "Collision_Mesh_ROI1.ply"
    )
)

OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

OBJ_PATH = os.path.join(
    OUTPUT_DIR,
    "Collision_Mesh_ROI1.obj"
)

XML_PATH = os.path.join(
    OUTPUT_DIR,
    "scene.xml"
)


# ============================================================
# STEP 1 - PLY -> OBJ
# ============================================================

def convert_ply_to_obj():

    print("=" * 70)
    print("STEP 11 - INTERACTIVE MESH SIMULATION")
    print("=" * 70)

    print("\n[1] Loading collision mesh:")
    print(COLLISION_PLY)

    if not os.path.exists(COLLISION_PLY):
        raise FileNotFoundError(
            f"Collision mesh not found:\n{COLLISION_PLY}"
        )

    mesh = o3d.io.read_triangle_mesh(COLLISION_PLY)

    if mesh.is_empty():
        raise RuntimeError(
            "Collision mesh is empty."
        )

    print("Vertices :", len(mesh.vertices))
    print("Triangles:", len(mesh.triangles))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = o3d.io.write_triangle_mesh(
        OBJ_PATH,
        mesh,
        write_vertex_normals=False,
        write_vertex_colors=False
    )

    if not success:
        raise RuntimeError(
            "Failed to convert PLY to OBJ."
        )

    print("\nPLY -> OBJ: PASS")
    print("OBJ:", OBJ_PATH)


# ============================================================
# STEP 2 - CREATE MUJOCO XML
# ============================================================

def create_scene_xml():

    xml = f"""
<mujoco model="Safe-GS Interactive Simulation">

    <compiler angle="degree" meshdir="{OUTPUT_DIR}" />

    <option timestep="0.002"
            gravity="0 0 -9.81"
            integrator="Euler" />

    <visual>
        <headlight diffuse="0.8 0.8 0.8"
                    ambient="0.3 0.3 0.3"
                    specular="0.1 0.1 0.1" />

        <global azimuth="140"
                elevation="-20" />
    </visual>

    <asset>

        <mesh
            name="collision_mesh"
            file="Collision_Mesh_ROI1.obj"
            scale="1 1 1"
        />

    </asset>

    <worldbody>

        <!-- Ground -->
        <geom
            name="ground"
            type="plane"
            pos="0 0 -1.2"
            size="5 5 0.1"
            friction="1 0.5 0.1"
            contype="1"
            conaffinity="1"
        />

        <!-- Reconstructed object -->
        <body
            name="reconstructed_object"
            pos="0 0 0.5"
        >

            <!-- Allow object to move freely -->
            <freejoint />

            <geom
                name="collision_mesh_geom"
                type="mesh"
                mesh="collision_mesh"
                density="1000"
                friction="1 0.5 0.1"
                contype="1"
                conaffinity="1"
            />

        </body>

    </worldbody>

</mujoco>
"""

    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)

    print("\n[2] MuJoCo XML created:")
    print(XML_PATH)


# ============================================================
# STEP 3 - LOAD MODEL
# ============================================================

def load_model():

    print("\n[3] Loading MuJoCo model...")

    model = mujoco.MjModel.from_xml_path(
        XML_PATH
    )

    data = mujoco.MjData(model)

    print("MuJoCo loading: PASS")
    print("Bodies:", model.nbody)
    print("Geoms :", model.ngeom)

    return model, data


# ============================================================
# STEP 4 - INTERACTIVE VIEWER
# ============================================================

def run_viewer(model, data):

    print("\n" + "=" * 70)
    print("INTERACTIVE MUJOCO VIEWER")
    print("=" * 70)

    print("""
Controls:

    Left mouse drag
        -> rotate camera

    Right mouse drag
        -> move camera

    Mouse wheel
        -> zoom

    Ctrl + mouse interaction on object
        -> select / manipulate object

    ESC
        -> close viewer

The reconstructed Collision Mesh is a FREE BODY,
so it can move, rotate and fall under gravity.
""")

    print("=" * 70)

    # Native MuJoCo viewer
    mujoco.viewer.launch(
        model,
        data
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Check MuJoCo
    print("MuJoCo version:", mujoco.__version__)

    # PLY -> OBJ
    convert_ply_to_obj()

    # XML
    create_scene_xml()

    # Load model
    model, data = load_model()

    # Run interactive viewer
    run_viewer(model, data)

    print("\nSimulation closed.")
    print("STEP 11 COMPLETED")


if __name__ == "__main__":
    main()