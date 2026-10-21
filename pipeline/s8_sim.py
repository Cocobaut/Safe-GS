"""
S8: mô phỏng MuJoCo trên scene.xml của S7. Kịch bản (config S8_sim.scenarios):
  settle       : để yên t_total giây -> vật có đứng vững không (độ dời cuối, có NaN không).
  drop_all     : nâng mọi vật lên (vật thấp nâng ít, cao dần theo thứ tự đáy, cộng xoay nhẹ) rồi thả rơi.
  lift_<vật>   : nhấc vật lên height_m trong t_up giây (động học, theo đường mượt), giữ t_hold giây, thả rơi tự do;
                 các vật khác mô phỏng bình thường.
Mỗi kịch bản lưu pose (vị trí + quaternion w,x,y,z của từng body, hệ sim) ở fps khung hình.
Pose = phép biến đổi cứng áp lên toạ độ sim ban đầu của vật (qpos ban đầu = đơn vị).
Xuất: <kịch bản>/trajectory.npz (t, names, pos[T,N,3], quat[T,N,4]), <kịch bản>/report.json, <kịch bản>/height.png,
      summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def quat_from_axis_angle(axis, ang):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def run(model, names, setup, total, fps, kin=None):
    import mujoco
    data = mujoco.MjData(model)
    adr = {n: model.jnt_qposadr[model.joint(f"{n}_free").id] for n in names}
    vadr = {n: model.jnt_dofadr[model.joint(f"{n}_free").id] for n in names}
    setup(data, adr)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    every = max(1, int(round(1.0 / (fps * dt))))
    T, P, Q = [], [], []
    nsteps = int(round(total / dt))
    for s in range(nsteps + 1):
        if s % every == 0:
            T.append(data.time)
            P.append([data.qpos[adr[n]:adr[n] + 3].copy() for n in names])
            Q.append([data.qpos[adr[n] + 3:adr[n] + 7].copy() for n in names])
        if kin is not None:
            kin(data, adr, vadr)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            break
    return np.array(T), np.array(P), np.array(Q), bool(np.isfinite(data.qpos).all())


def plot(t, P, names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, n in enumerate(names):
        ax.plot(t, P[:, i, 2], label=n)
    ax.set_xlabel("t (s)"); ax.set_ylabel("độ cao dịch chuyển z (m)"); ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--s7", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text(encoding="utf-8"))["S8_sim"]
    s7, out = Path(a.s7), Path(a.out)
    model = mujoco.MjModel.from_xml_path(str(s7 / "scene.xml"))
    model.opt.timestep = cfg["timestep"]                        # S8 quyết định timestep (S7 chỉ ghi giá trị khởi tạo)
    names = [model.body(i).name for i in range(1, model.nbody)]
    phys = json.loads((s7 / "physics.json").read_text())["objects"]
    summary = {}

    attached = json.loads((s7 / "physics.json").read_text()).get("attached", {})

    def finish(key, t, P, Q, ok, extra=None):
        d = out / key; d.mkdir(parents=True, exist_ok=True)
        # vật gắn vào vật cha (S7 attach) đi theo pose của cha
        all_names, P2, Q2 = list(names), P, Q
        for c, par in attached.items():
            i = names.index(par)
            all_names.append(c)
            P2 = np.concatenate([P2, P[:, i:i + 1]], 1); Q2 = np.concatenate([Q2, Q[:, i:i + 1]], 1)
        np.savez_compressed(d / "trajectory.npz", t=t, names=np.array(all_names), pos=P2, quat=Q2)
        disp = {n: float(np.linalg.norm(P[-1, i])) for i, n in enumerate(names)}
        ang = {n: float(2 * np.degrees(np.arccos(np.clip(abs(Q[-1, i, 0]), 0, 1)))) for i, n in enumerate(names)}
        rep = {"frames": int(len(t)), "seconds": float(t[-1]), "finite": ok,
               "final_translation_m": disp, "final_rotation_deg": ang, **(extra or {})}
        (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        plot(t, P, names, d / "height.png", key)
        summary[key] = {"finite": ok, "max_final_translation_m": max(disp.values()) if disp else 0}
        print(key, json.dumps(summary[key]), flush=True)

    sc = cfg["scenarios"]
    if "settle" in sc:
        t, P, Q, ok = run(model, names, lambda d, adr: None, sc["settle"]["t_total"], cfg["fps"])
        finish("settle", t, P, Q, ok, {"note": "vật đứng vững thì độ dời cuối nhỏ (vài mm)"})

    if "drop_all" in sc:
        c = sc["drop_all"]
        rng = np.random.default_rng(0)
        # vật có đáy thấp nâng ít, các vật sau cao dần -> không chồng nhau lúc bắt đầu
        order = sorted(names, key=lambda n: phys.get(n, {}).get("collision_volume_m3", 0), reverse=True)
        off = {n: c["base_offset_m"] + c["step_m"] * k for k, n in enumerate(order)}

        def setup(d, adr):
            for n in names:
                # xoay quanh trọng tâm vật (gốc body = gốc hệ sim, có thể cách vật cả mét): p = cm - R cm + nâng
                q = quat_from_axis_angle(rng.normal(size=3), np.radians(c["tilt_deg"]))
                R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
                cm = model.body_ipos[model.body(n).id]
                d.qpos[adr[n]:adr[n] + 3] = cm - R @ cm + np.array([0, 0, off[n]])
                d.qpos[adr[n] + 3:adr[n] + 7] = q
        t, P, Q, ok = run(model, names, setup, c["t_total"], cfg["fps"])
        finish("drop_all", t, P, Q, ok, {"start_height_m": off})

    for obj in sc.get("lift", {}).get("objects", []):
        c = sc["lift"]
        if obj not in names:
            continue

        def kin(d, adr, vadr, obj=obj, c=c):
            tt = d.time
            if tt <= c["t_up"] + c["t_hold"]:
                s = min(tt / c["t_up"], 1.0)
                z = c["height_m"] * (3 * s ** 2 - 2 * s ** 3)            # smoothstep
                d.qpos[adr[obj]:adr[obj] + 3] = [0, 0, z]
                d.qpos[adr[obj] + 3:adr[obj] + 7] = [1, 0, 0, 0]
                d.qvel[vadr[obj]:vadr[obj] + 6] = 0
        t, P, Q, ok = run(model, names, lambda d, adr: None, c["t_total"], cfg["fps"], kin=kin)
        finish(f"lift_{obj}", t, P, Q, ok, {"lifted": obj, "height_m": c["height_m"],
                                             "release_at_s": c["t_up"] + c["t_hold"]})

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
