"""Run Module 5 steps 1-9 in order.

ROI AABB currently comes from roi_config.py (Module 2 stand-in).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from . import adaptive_mesh, collision_mesh, mesh_preprocessing, mesh_selection, mesh_validation, topology_repair
except ImportError:
    import adaptive_mesh
    import collision_mesh
    import mesh_preprocessing
    import mesh_selection
    import mesh_validation
    import topology_repair


def main():
    mesh_preprocessing.main()
    topology_repair.main()
    adaptive_mesh.main()
    mesh_selection.main()
    mesh_validation.main()
    collision_mesh.main()
    print("\nMODULE 5 PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
