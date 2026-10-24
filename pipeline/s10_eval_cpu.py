"""
S10 (phần CPU): đo S6 (mesh) và S8 (mô phỏng).

S6, mỗi vật (hệ sim, mét):
  - kín (watertight), pháp tuyến nhất quán (winding), số mặt, thể tích, kích thước bao (cm);
  - Chamfer mesh <-> tâm Gaussian của vật (opacity >= 0.5, đổi sang hệ sim): trung bình 2 chiều + phân vị 95 (mm),
    mẫu 20k điểm trên mesh (như cách đo trong các bài ACD: 20k điểm / thành phần);
  - CoACD: số mảnh, tỉ số thể tích (tổng thể tích mảnh lồi / thể tích mesh, càng gần 1 càng sát), Hausdorff 1 chiều
    mesh -> hợp các mảnh lồi (mm).
S8, từ scene.xml của S7:
  - settle 2 s: độ dời cuối (mm), góc xoay cuối (độ), độ lún sâu nhất cuối cùng (mm, khoảng cách tiếp xúc âm),
    vận tốc lớn nhất cuối (m/s), có NaN không, tốc độ mô phỏng so với thời gian thực;
  - lift: sai lệch vị trí khi vật rơi về (cm) so với vị trí gốc.
Xuất: eval_cpu.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def read_ply_xyz_opacity(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        lines = header.decode("ascii").splitlines()
        n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
        props = [l.split()[-1] for l in lines if l.startswith("property")]
        d = np.fromfile(f, dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    op = 1 / (1 + np.exp(-d["opacity"].astype(np.float64)))
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64), op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    for k in ("s5", "s6", "s7", "s8", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    s5, s6, s7, s8, out = map(Path, (a.s5, a.s6, a.s7, a.s8, a.out))
    out.mkdir(parents=True, exist_ok=True)
    T = np.array(json.loads((s6 / "world_to_sim.json").read_text())["T"])
    res = {"s6_mesh": {}, "s8_sim": {}}

    for od in sorted((s6 / "objects").iterdir()):
        info = json.loads((od / "info.json").read_text())
        if info.get("skipped") or not (od / "visual.ply").exists():
            res["s6_mesh"][od.name] = {"skipped": info.get("skipped", "không có mesh")}
            continue
        m = trimesh.load(od / "visual.ply", force="mesh")
        hulls = [trimesh.load(p, force="mesh") for p in sorted(od.glob("collision_*.obj"))]
        xyz, op = read_ply_xyz_opacity(s5 / "objects" / od.name / "gaussians.ply")
        g = xyz[op >= 0.5] @ T[:3, :3].T + T[:3, 3]
        pts, _ = trimesh.sample.sample_surface(m, 20000, seed=0)
        d1 = cKDTree(g).query(pts)[0]          # mesh -> Gaussian
        d2 = cKDTree(pts).query(g)[0]          # Gaussian -> mesh
        hull_union = trimesh.util.concatenate(hulls) if hulls else None
        hv = float(sum(abs(h.volume) for h in hulls))
        mv = float(abs(m.volume)) if m.is_volume else float(abs(m.convex_hull.volume))
        haus = None
        if hull_union is not None:
            hp, _ = trimesh.sample.sample_surface(hull_union, 20000, seed=1)
            haus = float(cKDTree(hp).query(pts)[0].max() * 1000)
        res["s6_mesh"][od.name] = {
            "watertight": bool(m.is_watertight), "winding_consistent": bool(m.is_winding_consistent),
            "faces": int(len(m.faces)), "volume_cm3": mv * 1e6, "bbox_cm": (m.extents * 100).round(1).tolist(),
            "chamfer_mm": float((d1.mean() + d2.mean()) / 2 * 1000),
            "chamfer_p95_mm": float(max(np.percentile(d1, 95), np.percentile(d2, 95)) * 1000),
            "gaussians_used": int(len(g)),
            "coacd_parts": len(hulls), "hull_volume_ratio": hv / mv if mv > 0 else None,
            "hausdorff_mesh_to_hulls_mm": haus}
        print(od.name, json.dumps(res["s6_mesh"][od.name]), flush=True)

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(s7 / "scene.xml"))
    model.opt.timestep = json.loads(Path(a.cfg).read_text(encoding="utf-8"))["S8_sim"]["timestep"]
    data = mujoco.MjData(model)
    names = [model.body(i).name for i in range(1, model.nbody)]
    adr = {n: model.jnt_qposadr[model.joint(f"{n}_free").id] for n in names}
    t0 = time.time()
    while data.time < 2.0:
        mujoco.mj_step(model, data)
    wall = time.time() - t0
    pen = [data.contact[i].dist for i in range(data.ncon)]
    disp = {n: float(np.linalg.norm(data.qpos[adr[n]:adr[n] + 3]) * 1000) for n in names}
    rot = {n: float(2 * np.degrees(np.arccos(np.clip(abs(data.qpos[adr[n] + 3]), 0, 1)))) for n in names}
    res["s8_sim"]["settle"] = {"final_displacement_mm": disp, "final_rotation_deg": rot,
                               "max_displacement_mm": max(disp.values()), "max_rotation_deg": max(rot.values()),
                               "max_penetration_mm": float(-min(pen) * 1000) if pen and min(pen) < 0 else 0.0,
                               "contacts": int(data.ncon), "max_speed_end_m_s": float(np.abs(data.qvel).max()),
                               "finite": bool(np.isfinite(data.qpos).all()), "realtime_factor": 2.0 / wall}
    for sc in sorted(p.name for p in s8.iterdir() if p.name.startswith("lift_")):
        rep = json.loads((s8 / sc / "report.json").read_text())
        obj = rep["lifted"]
        res["s8_sim"][sc] = {"return_error_cm": rep["final_translation_m"][obj] * 100,
                             "others_max_displacement_cm": max(v for k, v in rep["final_translation_m"].items() if k != obj) * 100,
                             "finite": rep["finite"]}
    if (s8 / "drop_all" / "report.json").exists():
        rep = json.loads((s8 / "drop_all" / "report.json").read_text())
        res["s8_sim"]["drop_all"] = {"finite": rep["finite"], "final_translation_m": rep["final_translation_m"]}
    print("S8", json.dumps(res["s8_sim"]["settle"]), flush=True)
    (out / "eval_cpu.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
