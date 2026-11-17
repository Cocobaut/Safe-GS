"""Step 16 - physics engine (MuJoCo) on top of steps 8 / 11 / 13 / 15 (ported from teatime_napkin/step14_physics_engine).

    16.1 prepare_assets.py  Z-up sim frame, textured table + object meshes, colliders, GS binding, MJCF
    16.2 run_sim.py         serial robot arm (analytic IK) grasps / lifts / carries the object -> trajectory + report
    16.3 render_3dgs.py     same trajectory rendered with the Gaussians (step 11 background + deformed step 8 object)
"""
