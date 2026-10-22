"""Module 4 simulation - Benchmarks: the same interaction task on M_High, M_Adaptive and M_Coarse.

README Module 4 simulation, steps 4 and 9-11:
  * A kinematic probe (stand-in for the robot end-effector) follows one fixed trajectory into
    the target ROI object and back, identical for every mesh variant.
  * Per variant: polygon count, build/simulation/wall time and FPS, Topology Guard faults
    (explosion rate), interpenetration, contact stability while settling, probe/object
    trajectories and the contacts in the ROI.
  * Against M_High: error collisions (FP/FN), collision accuracy and Error_ROI (<= tau_ROI),
    plus Reduction = 1 - N_Adaptive / N_High.

Usage (from the repository root):
    python scripts/06_run_simulation.py --benchmark
Environment:
    MODULE6_TARGET_ROI=ROI_4   target object (default: first dynamic, non-"unknown" segment)
"""

import json
import os
import sys
import time

import numpy as np
import mujoco

# src/Module4-Physic_Engine/ (holds preprocessing/ and simulation/)
ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)


from preprocessing.roi_config import ALL_ROIS, OPTIMIZED_DIR, TAU_ROI, is_static_roi
from simulation.collision_verifier import ContactRecorder, compare_to_reference
from simulation.env_mujoco import OUTPUT_DIR, build_scene, ensure_dir
from simulation.topology_guard import TopologyGuard

BENCHMARK_REPORT = os.path.join(OUTPUT_DIR, "benchmark_report.json")
BENCHMARK_VARIANTS = ("high", "adaptive", "coarse")   # "high" first: it is the reference

# Task: probe trajectory (seconds of simulated time)
PROBE_HALF_SIZE = (0.06, 0.06, 0.06)  # m
PROBE_STANDOFF = 0.6                  # m between the probe start and the ROI bounding radius
T_APPROACH_START = 1.0                # probe waits while the scene settles
T_APPROACH_END = 3.0                  # probe reaches the ROI centroid
T_RETREAT_END = 4.0                   # probe back at its start point
T_END = 4.5
TRAJECTORY_SAMPLE_EVERY = 10          # steps between stored trajectory samples


def default_target_roi():
    env = os.environ.get("MODULE6_TARGET_ROI", "").strip()
    if env:
        return env
    for roi_id, roi in ALL_ROIS.items():
        if not is_static_roi(roi_id, roi) and roi.get("category", "unknown") != "unknown":
            return roi_id
    raise RuntimeError("No dynamic ROI available as benchmark target")


def _roi_centroid(roi_id):
    with open(os.path.join(OPTIMIZED_DIR, roi_id, f"Object_{roi_id}_Info.json"), encoding="utf-8") as f:
        return np.asarray(json.load(f)["centroid"], dtype=np.float64)


def make_probe_task(target_roi):
    """Straight-line trajectory: start beside the target (on the room-centre side), push to
    the target centroid, retreat."""
    target = _roi_centroid(target_roi)
    extent = np.asarray(ALL_ROIS[target_roi]["max"]) - np.asarray(ALL_ROIS[target_roi]["min"])
    room_center = np.mean([np.asarray(r["min"]) + np.asarray(r["max"]) for r in ALL_ROIS.values()], axis=0) / 2.0
    direction = room_center - target
    direction[2] = 0.0
    if np.linalg.norm(direction) < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    start = target + direction * (0.5 * float(extent[:2].max()) + PROBE_STANDOFF)
    return {"target_roi": target_roi, "start": start, "end": target.copy(),
            "pos": start, "half_size": np.asarray(PROBE_HALF_SIZE)}


def probe_position(task, t):
    if t <= T_APPROACH_START:
        return task["start"]
    if t <= T_APPROACH_END:
        a = (t - T_APPROACH_START) / (T_APPROACH_END - T_APPROACH_START)
        return task["start"] + a * (task["end"] - task["start"])
    if t <= T_RETREAT_END:
        a = (t - T_APPROACH_END) / (T_RETREAT_END - T_APPROACH_END)
        return task["end"] + a * (task["start"] - task["end"])
    return task["start"]


def polygon_counts(model, target_roi):
    """Triangles in visual meshes vs collision hulls of a compiled model."""
    visual = collision = target_roi_faces = other_roi_faces = bg_faces = 0
    for m in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m) or ""
        faces = int(model.mesh_facenum[m])
        if name.endswith("_visual") or name.startswith("bg_visual"):
            visual += faces
            if name.startswith("bg_visual"):
                bg_faces += faces
            elif name == f"{target_roi}_visual":
                target_roi_faces += faces
            else:
                other_roi_faces += faces
        else:
            collision += faces
    return {
        "visual": visual, "collision": collision, "total": visual + collision,
        "target_roi_faces": target_roi_faces,
        "other_roi_faces": other_roi_faces,
        "bg_faces": bg_faces
    }


def run_task(xml_path, task):
    """Simulate the probe task once; returns (metrics, ContactRecorder)."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    guard = TopologyGuard(model, data)
    guard.start()
    recorder = ContactRecorder(model, target_body=task["target_roi"])

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task["target_roi"])
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    qpos0 = {j: data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3].copy() for j in free}
    settle_drift = None
    settle_ncon = []
    probe_traj, target_traj = [], []

    n_steps = int(T_END / model.opt.timestep)
    t0 = time.time()
    for step in range(n_steps):
        data.mocap_pos[0] = probe_position(task, data.time)
        guard.step()
        recorder.record(data)
        if data.time <= T_APPROACH_START:
            settle_ncon.append(int(data.ncon))
        elif settle_drift is None:
            # Contact stability: how far resting objects moved before the probe acts.
            settle_drift = max(
                (float(np.linalg.norm(data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3] - qpos0[j]))
                 for j in free), default=0.0)
        if step % TRAJECTORY_SAMPLE_EVERY == 0:
            probe_traj.append([float(data.time), *map(float, data.mocap_pos[0])])
            target_traj.append([float(data.time), *map(float, data.xpos[target_id])])
    wall = time.time() - t0

    target_traj_np = np.asarray(target_traj)
    metrics = {
        "polygons": polygon_counts(model, task["target_roi"]),
        "simulated_time_s": float(n_steps * model.opt.timestep),
        "wall_time_s": wall,
        "fps": n_steps / wall if wall > 0 else 0.0,
        "realtime_factor": n_steps * model.opt.timestep / wall if wall > 0 else 0.0,
        "topology_guard": guard.report(),
        "contact_stability": {
            "settle_max_drift_m": settle_drift or 0.0,
            "settle_contacts_std": float(np.std(settle_ncon)) if settle_ncon else 0.0,
        },
        "contacts": recorder.summary(),
        "target_displacement_m": float(np.linalg.norm(target_traj_np[-1, 1:] - target_traj_np[0, 1:])),
        "probe_trajectory": probe_traj,
        "target_trajectory": target_traj,
    }
    return metrics, recorder, target_traj_np


def main():
    target_roi = default_target_roi()
    task = make_probe_task(target_roi)
    ensure_dir(OUTPUT_DIR)
    print("=" * 70)
    print(f"MODULE 4 SIMULATION BENCHMARK - target {target_roi} ({ALL_ROIS[target_roi].get('category')})")
    print("=" * 70)

    results, recorders, trajectories = {}, {}, {}
    forced_static = layout = None
    for variant in BENCHMARK_VARIANTS:
        scene = build_scene(variant, target_roi=target_roi, forced_static=forced_static, probe=task, layout=layout)
        if forced_static is None:
            # Same static/dynamic split, start poses and excluded pairs for every variant,
            # decided on the M_High scene, so only the meshes differ between runs.
            forced_static = scene["report"]["static_ids"]
            layout = scene["layout"]
        if target_roi in scene["report"]["static_ids"]:
            raise RuntimeError(f"Target {target_roi} became static (unsupported); pick another MODULE6_TARGET_ROI")
        print(f"\n[{variant}] running probe task...")
        metrics, recorder, traj = run_task(scene["xml"], task)
        metrics["build_time_s"] = scene["report"]["build_time_s"]
        metrics["excluded_pairs"] = scene["report"]["excluded_pairs"]
        results[variant], recorders[variant], trajectories[variant] = metrics, recorder, traj

    reference = recorders["high"]
    n_high = results["high"]["polygons"]
    for variant in BENCHMARK_VARIANTS:
        if variant == "high":
            continue
        cmp = compare_to_reference(recorders[variant], reference, TAU_ROI)
        steps = min(len(trajectories[variant]), len(trajectories["high"]))
        cmp["target_trajectory_error_m"] = float(np.mean(np.linalg.norm(
            trajectories[variant][:steps, 1:] - trajectories["high"][:steps, 1:], axis=1)))
        results[variant]["vs_high"] = cmp
        results[variant]["reduction"] = {
            key: 1.0 - results[variant]["polygons"][key] / n_high[key] if n_high[key] else 0.0
            for key in ("visual", "collision", "total")
        }

    report = {
        "target_roi": target_roi,
        "task": {
            "probe_start": task["start"].tolist(),
            "probe_end": task["end"].tolist(),
            "probe_half_size": list(PROBE_HALF_SIZE),
            "timeline_s": [T_APPROACH_START, T_APPROACH_END, T_RETREAT_END, T_END],
        },
        "tau_roi_m": TAU_ROI,
        "variants": results,
    }
    with open(BENCHMARK_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 125)
    print(f"{'variant':<10}{'polys(vis)':>12}{'target_roi':>12}{'other_roi':>12}{'bg_faces':>10}{'polys(col)':>12}{'reduction':>11}{'fps':>8}{'faults':>8}"
          f"{'pen max mm':>12}{'ROI steps':>11}{'FP':>6}{'FN':>6}{'Err_ROI mm':>12}{'<=tau':>7}")
    for variant in BENCHMARK_VARIANTS:
        r = results[variant]
        cmp = r.get("vs_high", {})
        red = r.get("reduction", {}).get("total")
        err = cmp.get("error_roi_m")
        print(f"{variant:<10}{r['polygons']['visual']:>12}{r['polygons']['target_roi_faces']:>12}{r['polygons']['other_roi_faces']:>12}{r['polygons']['bg_faces']:>10}{r['polygons']['collision']:>12}"
              f"{'-' if red is None else f'{red:.1%}':>11}{r['fps']:>8.0f}{r['topology_guard']['fault_count']:>8}"
              f"{r['contacts']['interpenetration_max_m'] * 1000:>12.1f}{r['contacts']['roi_contact_steps']:>11}"
              f"{cmp.get('false_positive_contacts', '-'):>6}{cmp.get('false_negative_contacts', '-'):>6}"
              f"{'-' if err is None else f'{err * 1000:.2f}':>12}{str(cmp.get('error_roi_within_tau', '-')):>7}")
    print(f"\nBenchmark report saved to {BENCHMARK_REPORT}")
    return report


if __name__ == "__main__":
    main()
