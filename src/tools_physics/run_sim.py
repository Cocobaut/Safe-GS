"""Step 16.2 (local CPU): MuJoCo pick-and-lift of the object by the serial robot arm (ported from teatime_napkin
run_real_teatime_sim.py + test_cloth_draping_sim.py; geometry / waypoints come from 16.1 robot.json).

Phases (time-based): SETTLE -> APPROACH -> DESCEND -> GRASP (fingers close, weld lock) -> LIFT -> CARRY -> HOLD.
The object is a free body (rigid), 3 hinged segments (articulated cloth) or a <flexcomp> (flex), see 16.1 --mode.

Outputs in <run>/step16_physics_engine/:
  trajectory.npz (qpos + object body poses / flex nodes per video frame)  trajectory.json (light, per frame)
  final_physics_report.json  videos/pick_place_mujoco.mp4  videos/pick_place_dualview.mp4  videos/keyframe_*.png

  python tools_physics/run_sim.py --run <run dir | run name> [--dataset LERF] [--view | --manual] [--no-video]
"""
from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import physics_io as pio


class Scene:
    def __init__(self, out: Path):
        import mujoco
        self.mj = mujoco
        self.out = out
        self.rob = json.loads((out / "robot.json").read_text(encoding="utf-8"))
        self.st = json.loads((out / "sim_transform.json").read_text(encoding="utf-8"))
        self.mode = self.st["object"]["mode"]
        self.model = mujoco.MjModel.from_xml_path(str(out / "mjcf" / "scene_simulation.xml"))
        self.data = mujoco.MjData(self.model)
        m = self.model
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        self.airbot = self.rob.get("type") == "airbot_play"
        self.grasp_mode = self.rob.get("grasp_mode", "weld")
        self.grip = nid(mujoco.mjtObj.mjOBJ_BODY, "link6" if self.airbot else "robot_gripper")
        if self.airbot:
            jn = [nid(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
            self.arm_qadr = np.array([m.jnt_qposadr[j] for j in jn])
            self.qwp = {k: np.array(v["q"]) for k, v in self.rob["ik"].items()}
            for f_ in ("endleft", "endright"):
                j = nid(mujoco.mjtObj.mjOBJ_JOINT, f_)
                setattr(self, f"{f_}_qadr", m.jnt_qposadr[j])
        self.welds = [i for i in range(m.neq) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or "").startswith("grasp_weld")]
        if self.mode == "articulated":
            self.bodies = [s["body"] for s in self.st["object"]["segments"]]
        elif self.mode == "rigid":
            self.bodies = ["object_body"]
        else:
            self.bodies = []
        self.body_ids = [nid(mujoco.mjtObj.mjOBJ_BODY, b) for b in self.bodies]
        self.table_geom = nid(mujoco.mjtObj.mjOBJ_GEOM, "table_collider")
        robot_root = nid(mujoco.mjtObj.mjOBJ_BODY, "robot_base")
        self.robot_bodies = {b for b in range(m.nbody) if self._is_desc(b, robot_root)}
        self.fold_qadr = [m.jnt_qposadr[j] for j in range(m.njnt)
                          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("fold")]
        ph = self.st["config"]["phases"]
        self.phases = list(ph.items())
        self.t_end = np.cumsum([d for _, d in self.phases])
        self.duration = float(self.t_end[-1])
        wp = {k: np.array(v) for k, v in self.rob["waypoints"].items()}
        R = self.rob["L1"] + self.rob["L2"]
        wp["HOME"] = wp["HOVER"] + np.array([0, -0.25 * R, 0.15 * R])
        self.wp = wp
        self.locked = False
        self.grip_ref = None

    def _is_desc(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.model.body_parentid[b]
        return False

    # ------------------------------------------------------------------ control
    def ik(self, target):
        sh = np.array(self.rob["shoulder"])
        L1, L2, Lee = self.rob["L1"], self.rob["L2"], self.rob["L_ee"]
        d = np.array([target[0], target[1], target[2] + Lee]) - sh
        yaw = np.arctan2(-d[0], d[1])
        r = np.hypot(d[0], d[1])
        D = np.clip(np.hypot(r, d[2]), abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3)
        elbow = np.arccos(np.clip((D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1))
        beta = np.arccos(np.clip((L1 ** 2 + D ** 2 - L2 ** 2) / (2 * L1 * D), -1, 1))
        sp = np.arctan2(d[2], r) + beta
        return yaw, sp, -elbow, -np.pi / 2 - (sp - elbow)

    def phase_at(self, t):
        i = int(np.searchsorted(self.t_end, t, side="right"))
        i = min(i, len(self.phases) - 1)
        t0 = self.t_end[i - 1] if i else 0.0
        return self.phases[i][0], np.clip((t - t0) / self.phases[i][1], 0, 1)

    def lock(self, on=True):
        d, m = self.data, self.model
        if on:
            p1, q1 = d.xpos[self.grip], d.xquat[self.grip]
            R1 = d.xmat[self.grip].reshape(3, 3)
            for e in self.welds:
                b2 = m.eq_obj2id[e]
                m.eq_data[e, 0:3] = 0
                m.eq_data[e, 3:6] = R1.T @ (d.xpos[b2] - p1)
                q = np.zeros(4)
                self.mj.mju_negQuat(q, q1)
                qr = np.zeros(4)
                self.mj.mju_mulQuat(qr, q, d.xquat[b2])
                m.eq_data[e, 6:10] = qr
        for e in self.welds:
            d.eq_active[e] = int(on)
        self.locked = on

    def reset(self):
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        if self.airbot:                                              # start at the HOME pose, gripper open
            d.qpos[self.arm_qadr] = self.qwp["HOME"]
            d.qpos[self.endleft_qadr], d.qpos[self.endright_qadr] = self.rob["jaw_open"], -self.rob["jaw_open"]
            d.ctrl[:6] = self.qwp["HOME"]
            d.ctrl[6] = self.rob["jaw_open"]
        self.lock(False)
        mj.mj_forward(m, d)

    def control_airbot(self, name, a, s):
        q, op, cl = self.qwp, self.rob["jaw_open"], self.rob["jaw_closed"]
        tgt, jaw = {"SETTLE": (q["HOME"], op),
                    "APPROACH": ((1 - s) * q["HOME"] + s * q["HOVER"], op),
                    "DESCEND": ((1 - s) * q["HOVER"] + s * q["GRASP"], op),
                    "GRASP": (q["GRASP"], op + (cl - op) * min(1.0, a / 0.6)),
                    "LIFT": ((1 - s) * q["GRASP"] + s * q["LIFT"], cl),
                    "CARRY": ((1 - s) * q["LIFT"] + s * q["CARRY"], cl),
                    "HOLD": (q["CARRY"], cl)}[name]
        if name == "GRASP" and a >= 0.7 and self.grasp_mode == "friction" and self.grip_ref is None:
            self.grip_ref = self._grip_obj_dist()                    # friction grasp: no weld, only measure slip
        if name == "GRASP" and a >= 0.7 and self.grasp_mode == "weld" and not self.locked:
            self.lock(True)
        self.data.ctrl[:6] = tgt
        self.data.ctrl[6] = jaw
        return name

    def _grip_obj_dist(self):
        p = self.data.xpos[self.body_ids[0]] if self.body_ids else self.data.flexvert_xpos.mean(0)
        return float(np.linalg.norm(p - self.data.xpos[self.grip]))

    def control(self, t):
        name, a = self.phase_at(t)
        s = 0.5 * (1 - np.cos(np.pi * a))
        if self.airbot:
            return self.control_airbot(name, a, s)
        w, jc = self.wp, self.rob["jaw_closed"]
        tgt, jaw = {"SETTLE": (w["HOME"], 0.0),
                    "APPROACH": ((1 - s) * w["HOME"] + s * w["HOVER"], 0.0),
                    "DESCEND": ((1 - s) * w["HOVER"] + s * w["GRASP"], 0.0),
                    "GRASP": (w["GRASP"], jc * min(1.0, a / 0.6)),
                    "LIFT": ((1 - s) * w["GRASP"] + s * w["LIFT"], jc),
                    "CARRY": ((1 - s) * w["LIFT"] + s * w["CARRY"], jc),
                    "HOLD": (w["CARRY"], jc)}[name]
        if name == "GRASP" and a >= 0.7 and not self.locked:
            self.lock(True)
        self.data.ctrl[:4] = self.ik(tgt)
        self.data.ctrl[4] = 0.0
        self.data.ctrl[5:7] = jaw
        return name

    # ------------------------------------------------------------------ measurements
    def object_com(self):
        if self.body_ids:
            return np.average(self.data.xipos[self.body_ids], axis=0, weights=self.model.body_mass[self.body_ids])
        return self.data.flexvert_xpos.mean(0)

    def table_penetration(self):
        """deepest contact between the table collider and the object (not the robot)."""
        d, m = self.data, self.model
        pen = 0.0
        for c in d.contact[:d.ncon]:
            g = (c.geom1, c.geom2)
            if self.table_geom not in g:
                continue
            other = g[1] if g[0] == self.table_geom else g[0]
            if other >= 0 and m.geom_bodyid[other] in self.robot_bodies:
                continue
            pen = max(pen, -float(c.dist))
        return pen


def draw_hud(img, lines):
    import cv2
    ov = img.copy()
    cv2.rectangle(ov, (12, 12), (min(img.shape[1] - 12, 900), 24 + 26 * len(lines)), (20, 24, 30), -1)
    cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (24, 38 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    return img


def friction_trial(sc: Scene):
    """Simulate up to the end of LIFT with the friction grasp only. -> object lift (com z rise after GRASP)."""
    mj, m, d = sc.mj, sc.model, sc.data
    sc.reset()
    t_lift_end = float(sc.t_end[[p for p, _ in sc.phases].index("LIFT")])
    z0 = None
    while d.time < t_lift_end:
        ph = sc.control(d.time)
        mj.mj_step(m, d)
        if ph == "GRASP":
            z0 = float(sc.object_com()[2])
        if not np.isfinite(d.qpos).all():
            return float("nan")
    return float(sc.object_com()[2] - z0) if z0 is not None else 0.0


def run_headless(sc: Scene, video=True):
    import cv2
    mj, m, d = sc.mj, sc.model, sc.data
    trial = None
    if sc.airbot and sc.grasp_mode == "friction":                 # geometry says it fits: check that it really holds
        trial = friction_trial(sc)
        need = 0.3 * sc.rob["lift_height"]
        ok = np.isfinite(trial) and trial >= need
        print(f"[16.2] friction grasp trial: object lifted {trial:.3f} (need {need:.3f}) -> "
              f"{'friction grasp kept' if ok else 'falls back to the weld lock'}")
        if not ok:
            sc.grasp_mode = "weld"
            sc.rob["grasp_reason"] = (f"friction trial failed: the closed gripper lifted the object {trial:.3f} "
                                      f"(< {need:.3f}); pads slip on the rigid strip colliders")
        sc.grip_ref = None
    sc.reset()
    fps = int(sc.st["config"].get("fps", 30))
    vids = sc.out / "videos"
    vids.mkdir(exist_ok=True)
    if video:
        rend = mj.Renderer(m, height=720, width=1280)
        cam_names = [c for c in ("cam_side", "cam_robot") if mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, c) >= 0]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w_main = cv2.VideoWriter(str(vids / "pick_place_mujoco.mp4"), fourcc, fps, (1280, 720))
        w_dual = cv2.VideoWriter(str(vids / "pick_place_dualview.mp4"), fourcc, fps, (1280, 720))
    rec = {"t": [], "qpos": [], "phase": [], "body_pos": [], "body_quat": [], "flex": [], "grip_pos": []}
    frames_json = []
    com0 = sc.object_com().copy()
    settle_com = com0.copy()
    max_pen, max_fold, nan = 0.0, 0.0, False
    keys = {}
    next_t, t_wall = 0.0, time.time()
    n_steps = int(np.ceil(sc.duration / m.opt.timestep))
    for _ in range(n_steps + 1):
        phase = sc.control(d.time)
        if d.time + 1e-9 >= next_t:
            com = sc.object_com()
            lift = float(com[2] - settle_com[2])
            rec["t"].append(d.time); rec["qpos"].append(d.qpos.copy()); rec["phase"].append(phase)
            rec["grip_pos"].append(d.xpos[sc.grip].copy())
            if sc.body_ids:
                rec["body_pos"].append(d.xpos[sc.body_ids].copy()); rec["body_quat"].append(d.xquat[sc.body_ids].copy())
            if m.nflexvert:
                rec["flex"].append(d.flexvert_xpos.copy())
            frames_json.append({"t": round(float(d.time), 4), "phase": phase, "object_com": np.round(com, 5).tolist(),
                                "grip_pos": np.round(d.xpos[sc.grip], 5).tolist(), "lift": round(lift, 5),
                                "contacts": int(d.ncon)})
            if video:
                txt = [(f"Safe-GS Step 16 | {sc.mode} object | phase {phase}", (90, 230, 140)),
                       (f"t={d.time:4.2f}s  lift={lift:+.3f}  contacts={d.ncon}  max table penetration={max_pen:.4f}",
                        (230, 230, 230))]
                rend.update_scene(d, camera=cam_names[0] if cam_names else -1)
                img = draw_hud(cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR), txt)
                w_main.write(img)
                halves = []
                for c in cam_names[:2]:
                    rend.update_scene(d, camera=c)
                    halves.append(cv2.resize(cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR), (640, 720)))
                if len(halves) == 2:
                    w_dual.write(draw_hud(np.hstack(halves), txt[:1]))
                if phase not in keys:
                    keys[phase] = True
                    cv2.imwrite(str(vids / f"keyframe_{len(keys)}_{phase.lower()}.png"), img)
            next_t += 1.0 / fps
        mj.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            nan = True
            print("[16.2] simulation diverged (NaN)")
            break
        name, _ = sc.phase_at(d.time)
        if name == "SETTLE":
            settle_com = sc.object_com().copy()
        max_pen = max(max_pen, sc.table_penetration())
        if sc.fold_qadr:
            max_fold = max(max_fold, float(np.abs(d.qpos[sc.fold_qadr]).max()))
    wall = time.time() - t_wall
    if video:
        w_main.release(); w_dual.release(); rend.close()

    np.savez_compressed(sc.out / "trajectory.npz", t=np.array(rec["t"]), qpos=np.array(rec["qpos"]),
                        phase=np.array(rec["phase"]), grip_pos=np.array(rec["grip_pos"]),
                        body_names=np.array(sc.bodies), body_pos=np.array(rec["body_pos"]),
                        body_quat=np.array(rec["body_quat"]), flex=np.array(rec["flex"]))
    (sc.out / "trajectory.json").write_text(json.dumps({"fps": fps, "mode": sc.mode, "frames": frames_json}),
                                            encoding="utf-8")
    com = sc.object_com()
    lift_h = sc.rob["lift_height"]
    final_lift = float(com[2] - settle_com[2])
    s = sc.st["config"]["sim_scale"]
    report = {
        "mode": sc.mode, "sim_scale": s, "units": "sim units (= scene units x sim_scale; scene units are not metres)",
        "duration_s": sc.duration, "timestep": m.opt.timestep, "frames": len(rec["t"]),
        "wall_time_s": round(wall, 2), "realtime_factor": round(sc.duration / max(wall, 1e-9), 3),
        "stable_no_nan": not nan,
        "settle_drift": round(float(np.linalg.norm((settle_com - com0)[:2])), 5),
        "settle_drop": round(float(com0[2] - settle_com[2]), 5),
        "max_table_penetration": round(max_pen, 6),
        "max_table_penetration_rel_object_width": round(max_pen / max(np.ptp(np.array([sc.st["object"]["bbox_min"],
                                                                                    sc.st["object"]["bbox_max"]]), 0)[:2].max(), 1e-9), 6),
        "commanded_lift": round(lift_h, 4), "final_object_lift": round(final_lift, 5),
        "grasp_success": bool(not nan and final_lift > 0.5 * lift_h),
        "max_fold_angle_deg": round(float(np.degrees(max_fold)), 2) if sc.fold_qadr else None,
        "robot_reachable": sc.rob["reachable"], "robot_scale_k": sc.rob["k"],
        "robot": sc.rob.get("type", "serial"), "grasp_mode": sc.grasp_mode,
        "grasp": ("friction only: the closed gripper holds the object through contacts" if sc.grasp_mode == "friction"
                  else "weld lock activated at 70% of the GRASP phase (fingers close, then the object is attached)"),
    }
    if sc.airbot:
        report["friction_trial_lift"] = None if trial is None else round(trial, 5)
        report["pinch_widths"] = sc.rob["pinch_widths"]
        report["gripper_max_pinch"] = sc.rob["max_pinch"]
        report["grasp_reason"] = sc.rob.get("grasp_reason")
        report["wall_angle_deg"] = sc.rob.get("wall_angle_deg")
        report["ik_errors_cm"] = {k: round(v["pos_err"] * 100, 2) for k, v in sc.rob["ik"].items()}
        if sc.grasp_mode == "friction" and sc.grip_ref is not None:
            report["grasp_slip"] = round(abs(sc._grip_obj_dist() - sc.grip_ref), 5)
    if m.nflexvert:
        rest = np.array(json.loads((sc.out / "object_gs_binding.json").read_text())["rest_nodes"])
        report["flex_node_order_max_err"] = round(float(np.abs(np.array(rec["flex"][0]) - rest).max()), 6)
    (sc.out / "final_physics_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def run_viewer(sc: Scene, manual=False):
    import mujoco.viewer
    mj, m, d = sc.mj, sc.model, sc.data
    sc.reset()
    print("viewer: mouse = rotate/pan/zoom, Space = pause, Backspace = reset"
          + (" | manual: drag the actuators in the Control panel (fingers > 60% closed = grasp)" if manual else ""))
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = sc.wp["GRASP"]
        v.cam.distance = 3.0 * sc.rob["L1"]
        v.cam.azimuth, v.cam.elevation = 125.0, -25.0
        while v.is_running():
            t0 = time.time()
            if not manual:
                if d.time > sc.duration + 1.0:
                    sc.reset()
                sc.control(d.time)
            else:
                closed = (d.ctrl[6] <= 0.01 if sc.airbot else min(d.ctrl[5], d.ctrl[6]) >= 0.6 * sc.rob["jaw_closed"])
                if sc.grasp_mode == "friction":
                    closed = sc.locked
                if closed and not sc.locked:
                    sc.lock(True)
                elif not closed and sc.locked:
                    sc.lock(False)
            mj.mj_step(m, d)
            v.sync()
            time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="LERF")
    ap.add_argument("--view", action="store_true", help="interactive viewer, automatic trajectory")
    ap.add_argument("--manual", action="store_true", help="interactive viewer, manual actuator sliders")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = pio.out_dir(pio.resolve_run_dir(args.run, args.dataset))
    if not (out / "mjcf" / "scene_simulation.xml").exists():
        raise SystemExit(f"run step 16.1 first ({out}/mjcf/scene_simulation.xml missing)")
    sc = Scene(out)
    if args.view or args.manual:
        run_viewer(sc, manual=args.manual)
    else:
        rep = run_headless(sc, video=not args.no_video)
        sys.exit(0 if rep["stable_no_nan"] else 1)


if __name__ == "__main__":
    main()
