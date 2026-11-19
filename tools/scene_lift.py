"""
Thử nhấc từng vật trong CẢ cảnh (MuJoCo), đối chiếu với đồ thị chống đỡ. Cùng một quy tắc cho mọi vật.

Đầu vào: thư mục của tools/frame_mesh.py (<vật>/coll_<k>_<j>.obj = các mảnh lồi CoACD, đúng vị trí trong cảnh, hệ bàn, mét; report.json)
và graph.json của tools/support_graph.py.
  - Bàn = plane z = 0. Vật đặt trên bàn hoặc do vật khác đỡ: freejoint, 500 kg/m3. Vật không đặt trên bàn (gấu, cừu: ngồi ghế,
    ghế không có trong cảnh): cố định, chỉ làm vật cản.
  - Mỗi lần thử: reset, để yên --settle s; rồi hàn (weld) vật được nhấc vào một mocap, kéo lên --lift_cm trong 1 s, giữ 0.5 s.
  - Vật khác: "mang theo" nếu cao lên > 50 % độ nhấc; "bị xô" nếu dời > --disturb_mm mà không được mang theo.
  - Kỳ vọng từ đồ thị: mang theo = hậu duệ của vật được nhấc. Báo khớp / lệch.
Cũng báo: các cặp lún vào nhau > 2 mm ngay lúc đầu (mesh chồng lấn), vật dời trong lúc để yên.
Xuất: lift.json, lift.md, scene.xml
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--support_graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--lift_cm", type=float, default=10.0)
    ap.add_argument("--disturb_mm", type=float, default=5.0)
    ap.add_argument("--record", default="", help="vật cần ghi quỹ đạo (cách nhau dấu phẩy) để render Gaussian; mặc định: vật nào có hậu duệ + 1 vật bất kỳ trên bàn")
    ap.add_argument("--support", default="", help="support.json để ghi world_to_sim.json (hệ cảnh -> hệ bàn, mét)")
    ap.add_argument("--drop_test", default="", help="vật thử nhấc + thả (phẩy): nhấc --drop_cm trong 1.5 s, giữ 0.5 s, thả, chờ 2 s")
    ap.add_argument("--drop_cm", type=float, default=20.0)
    ap.add_argument("--com_frame", type=int, default=0,
                    help="1: khung thân vật động đặt tại tâm khối (hình học) thay vì gốc cảnh -> mối hàn nhấc neo ở vật, không ở gốc cảnh xa")
    a = ap.parse_args()
    md = Path(a.meshes)
    rep = json.loads((md / "report.json").read_text())
    g = json.loads(Path(a.support_graph).read_text())
    children = g.get("support_children", {})
    desc = g.get("descendants", {})
    names = [n for n, r in rep.items() if "parts" in r]
    dynamic = [n for n in names if rep[n].get("rests_on_table_mesh") or rep[n].get("support_parent") not in ("TABLE", "?")]
    assets, bodies, eqs, mocaps = [], [], [], []
    com = {}
    for n in names:
        geoms = []
        files = sorted((md / n).glob("coll_*.obj"))
        c = np.zeros(3)
        if a.com_frame and n in dynamic and files:
            import trimesh
            ms = [trimesh.load(f, force="mesh") for f in files]
            vols = np.array([max(abs(x.volume), 1e-12) for x in ms])
            c = (np.stack([x.center_mass if x.is_volume else x.centroid for x in ms]) * vols[:, None]).sum(0) / vols.sum()
            com[n] = c
        gp = f' pos="{-c[0]:.6f} {-c[1]:.6f} {-c[2]:.6f}"' if n in com else ""
        for f in files:
            mn = f"{n}__{f.stem}"
            assets.append(f'<mesh name="{mn}" file="{f.relative_to(md).as_posix()}"/>')
            geoms.append(f'<geom type="mesh" mesh="{mn}"{gp} density="500" friction="0.8 0.01 0.001" condim="4" solref="0.002 1"/>')
        if not geoms:
            continue
        if n in dynamic:
            bp = f' pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"' if n in com else ""
            bodies.append(f'<body name="{n}"{bp}><freejoint/>{"".join(geoms)}</body>')
            mocaps.append(f'<body name="mc_{n}" mocap="true"{bp}/>')        # cùng tư thế đầu với vật -> relpose của weld = đồng nhất
            eqs.append(f'<weld name="w_{n}" body1="mc_{n}" body2="{n}" active="false" solref="0.01 1"/>')
        else:
            bodies.append(f'<body name="{n}">{"".join(geoms)}</body>')
    xml = f"""<mujoco><compiler meshdir="{md.as_posix()}" boundmass="0.01" boundinertia="1e-6"/>
<option timestep="0.001" gravity="0 0 -9.81"/>
<asset>{''.join(assets)}</asset>
<worldbody><geom name="table" type="plane" size="3 3 0.1" friction="0.8 0.01 0.001" solref="0.002 1"/>
{''.join(bodies)}{''.join(mocaps)}</worldbody>
<equality>{''.join(eqs)}</equality></mujoco>"""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
    dyn = [n for n in dynamic if bid.get(n, -1) >= 0]

    # mesh chồng lấn lúc đầu
    mujoco.mj_forward(m, d)
    over = {}
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist < -0.002:
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            k = tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or "table",
                              mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or "table")))
            over[k] = max(over.get(k, 0.0), -1000 * float(con.dist))
    # gỡ chồng lấn ban đầu: đẩy vật nằm trên (con trong đồ thị đỡ; không có thì vật có đáy cao hơn) lên khỏi vật bên dưới
    parent = {c: p for p, cs in children.items() for c in cs}
    qadr = {n: m.jnt_qposadr[m.body_jntadr[bid[n]]] for n in dyn}
    lifted0 = {}
    for _ in range(50):
        mujoco.mj_forward(m, d)
        worst = {}
        for c in range(d.ncon):
            con = d.contact[c]
            if con.dist >= -0.0005:
                continue
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom1])
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[con.geom2])
            if n1 not in dyn or n2 not in dyn:
                continue
            up = n1 if parent.get(n1) == n2 else n2 if parent.get(n2) == n1 else (n1 if d.xipos[bid[n1]][2] > d.xipos[bid[n2]][2] else n2)
            worst[up] = max(worst.get(up, 0.0), -float(con.dist))
        if not worst:
            break
        for n, dz in worst.items():
            d.qpos[qadr[n] + 2] += dz + 0.0002
            lifted0[n] = lifted0.get(n, 0.0) + dz + 0.0002
    qpos0 = d.qpos.copy()
    steps_settle = int(a.settle / m.opt.timestep)

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos0
        mujoco.mj_forward(m, d)

    def nwarn():
        """Số cảnh báo phân kỳ (MuJoCo tự mj_resetData khi gặp, mô phỏng sau đó KHÔNG còn hợp lệ)."""
        return int(sum(d.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                                                     mujoco.mjtWarning.mjWARN_BADQVEL)))

    def pos():
        return {n: d.xipos[bid[n]].copy() for n in dyn}        # tâm khối lượng (gốc thân = gốc cảnh, xoay làm phóng đại dời)

    # để yên
    reset()
    p0 = pos()
    for _ in range(steps_settle):
        mujoco.mj_step(m, d)
    p1 = pos()
    settle = {n: 1000 * float(np.linalg.norm(p1[n] - p0[n])) for n in dyn}
    settle_warn = nwarn()

    res = {"dynamic": dyn, "static": [n for n in names if n not in dyn], "initial_overlap_mm": {f"{k[0]}|{k[1]}": v for k, v in over.items()},
           "settle_displacement_mm": settle, "settle_divergence_warnings": settle_warn, "overlap_resolved_lift_mm": {k: 1000 * v for k, v in lifted0.items()}, "lifts": {}}
    lift = a.lift_cm / 100
    for n in dyn:
        reset()
        for _ in range(steps_settle):
            mujoco.mj_step(m, d)
        base = pos()
        mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"mc_{n}")]
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"w_{n}")
        # mocap trùng tư thế hiện tại của vật -> hàn không giật
        d.mocap_pos[mid] = d.xpos[bid[n]].copy()
        d.mocap_quat[mid] = d.xquat[bid[n]].copy()
        # eq_data của weld để mặc định: relpose đồng nhất (mocap đặt đúng khung thân vật ở trên)
        d.eq_active[eid] = 1
        start = d.mocap_pos[mid].copy()
        T = int(1.0 / m.opt.timestep)
        for t in range(T):
            d.mocap_pos[mid] = start + np.array([0, 0, lift * (t + 1) / T])
            mujoco.mj_step(m, d)
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        end = pos()
        rise = {k: float(end[k][2] - base[k][2]) for k in dyn}
        shift = {k: 1000 * float(np.linalg.norm(end[k] - base[k])) for k in dyn}
        carried = sorted(k for k in dyn if k != n and rise[k] > 0.5 * lift)
        disturbed = sorted(k for k in dyn if k != n and k not in carried and shift[k] > a.disturb_mm)
        expect = sorted(x for x in desc.get(n, []) if x in dyn)
        d.eq_active[eid] = 0
        res["lifts"][n] = {"lifted_cm": 100 * rise[n], "carried": carried, "expected_carried": expect,
                           "carried_rise_cm": {k: round(100 * rise[k], 3) for k in carried},
                           "match": carried == expect, "disturbed": disturbed, "divergence_warnings": nwarn(),
                           "disturbed_mm": {k: round(shift[k], 1) for k in disturbed}, "nan": bool(np.isnan(d.qpos).any())}
        print(n, json.dumps(res["lifts"][n]))
    # ---------------------------------------------------------------- nhấc cao rồi thả (như lift_coc_su của s8_sim v1)
    res["drop_tests"] = {}
    for n in [x for x in a.drop_test.split(",") if x]:
        if n not in dyn:
            continue
        reset()
        for _ in range(steps_settle):
            mujoco.mj_step(m, d)
        base = pos()
        q_base = d.xquat[bid[n]].copy()
        mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"mc_{n}")]
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"w_{n}")
        d.mocap_pos[mid] = d.xpos[bid[n]].copy(); d.mocap_quat[mid] = d.xquat[bid[n]].copy()
        d.eq_active[eid] = 1
        start = d.mocap_pos[mid].copy()
        T = int(1.5 / m.opt.timestep)
        for t in range(T):
            d.mocap_pos[mid] = start + np.array([0, 0, a.drop_cm / 100 * (t + 1) / T])
            mujoco.mj_step(m, d)
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        top = float(d.xpos[bid[n]][2] - base[n][2])
        d.eq_active[eid] = 0
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        end = pos()
        dq = abs(float(np.dot(d.xquat[bid[n]], q_base)))
        others = {k: round(1000 * float(np.linalg.norm(end[k] - base[k])), 2) for k in dyn if k != n}
        res["drop_tests"][n] = {"drop_cm": a.drop_cm, "reached_cm": 100 * top,
                                "landing_offset_xy_mm": 1000 * float(np.linalg.norm(end[n][:2] - base[n][:2])),
                                "landing_offset_z_mm": 1000 * float(end[n][2] - base[n][2]),
                                "tilt_deg": float(2 * np.degrees(np.arccos(min(1.0, dq)))),
                                "final_speed": float(np.abs(d.qvel).max()), "nan": bool(np.isnan(d.qpos).any()),
                                "divergence_warnings": nwarn(),
                                "others_moved_mm": others}
        print("drop", n, json.dumps(res["drop_tests"][n]))
    (out / "lift.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------------- ghi quỹ đạo để render (cùng quy ước s9_render: names, t, pos, quat)
    rec = [x for x in a.record.split(",") if x] or ([n for n in dyn if desc.get(n)] + [n for n in dyn if not desc.get(n)][:1])
    for n in rec:
        if n not in dyn:
            continue
        reset()
        mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"mc_{n}")]
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"w_{n}")
        ts, P, Q = [], [], []

        def snap():
            ts.append(d.time)
            # tư thế của khung "gốc cảnh" (s9v3_render dùng): p = R p_cảnh + (xpos - R c)
            P.append(np.stack([d.xpos[bid[k]] - d.xmat[bid[k]].reshape(3, 3) @ com.get(k, np.zeros(3)) for k in dyn]))
            Q.append(np.stack([d.xquat[bid[k]] for k in dyn]))
        every = int(0.04 / m.opt.timestep)                         # 25 khung/s
        k = 0
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d); k += 1
            if k % every == 0: snap()
        d.mocap_pos[mid] = d.xpos[bid[n]].copy(); d.mocap_quat[mid] = d.xquat[bid[n]].copy()
        d.eq_active[eid] = 1
        start = d.mocap_pos[mid].copy()
        T = int(1.5 / m.opt.timestep)
        for t in range(T):
            d.mocap_pos[mid] = start + np.array([0, 0, 0.15 * (t + 1) / T])
            mujoco.mj_step(m, d); k += 1
            if k % every == 0: snap()
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d); k += 1
            if k % every == 0: snap()
        d.eq_active[eid] = 0                                       # thả
        for _ in range(int(1.0 / m.opt.timestep)):
            mujoco.mj_step(m, d); k += 1
            if k % every == 0: snap()
        td = out / f"traj_lift_{n}"
        td.mkdir(exist_ok=True)
        np.savez(td / "trajectory.npz", names=np.array(dyn), t=np.array(ts), pos=np.array(P), quat=np.array(Q))
        print(f"quy dao {n}: {len(ts)} khung -> {td}")
    if a.support:
        sup = json.loads(Path(a.support).read_text())
        n_t, d_t, mpu = np.array(sup["table"]["normal"]), float(sup["table"]["d"]), float(sup["meters_per_unit"])
        b0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n_t, b0); e1 /= np.linalg.norm(e1); e2 = np.cross(n_t, e1)
        Rm = np.stack([e1, e2, n_t], 0)
        p0 = -d_t * n_t
        T4 = np.eye(4); T4[:3, :3] = Rm * mpu; T4[:3, 3] = -(Rm @ p0) * mpu          # p_sim = mpu * R (p - p0)
        (out / "world_to_sim.json").write_text(json.dumps({"T": T4.tolist(), "note": "p_sim = T @ [p_scene; 1], hệ bàn, mét"}))
    L = ["# Thu nhac tung vat trong ca canh (MuJoCo)", "",
         f"Vat dong: {', '.join(dyn)}; vat co dinh (khong dat tren ban): {', '.join(res['static']) or '-'}.", "",
         "Chong lan luc dau (> 2 mm): " + (", ".join(f"{k} {v:.1f} mm" for k, v in res["initial_overlap_mm"].items()) or "khong"), "",
         "Da day len de go chong lan (mm): " + (", ".join(f"{k} {1000 * v:.1f}" for k, v in lifted0.items()) or "-"), "",
         "Dich chuyen khi de yen 1 s (mm): " + ", ".join(f"{k} {v:.2f}" for k, v in settle.items()) + f" (canh bao phan ky: {settle_warn})", "",
         "| nhac | len duoc (cm) | mang theo (MuJoCo) | mang theo (do thi) | khop | bi xo (mm) | canh bao phan ky |", "|---|---|---|---|---|---|---|"]
    for n, r in res["lifts"].items():
        L.append(f"| {n} | {r['lifted_cm']:.1f} | {', '.join(r['carried']) or '-'} | {', '.join(r['expected_carried']) or '-'} | "
                 f"{'co' if r['match'] else 'KHONG'} | {', '.join(f'{k} {v}' for k, v in r['disturbed_mm'].items()) or '-'} | {r['divergence_warnings']} |")
    for n, r in res["drop_tests"].items():
        L += ["", f"Nhac {n} {r['drop_cm']:.0f} cm roi tha (cho 2 s): len duoc {r['reached_cm']:.1f} cm; lech cho cu ngang "
              f"{r['landing_offset_xy_mm']:.1f} mm, doc {r['landing_offset_z_mm']:.1f} mm, nghieng {r['tilt_deg']:.2f} do; van toc cuoi "
              f"{r['final_speed']:.4f}; NaN {r['nan']}; canh bao phan ky {r['divergence_warnings']}; vat khac doi (mm): " + ", ".join(f"{k} {v}" for k, v in r["others_moved_mm"].items())]
    (out / "lift.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
