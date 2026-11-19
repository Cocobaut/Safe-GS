"""
Bước 3: mesh kín cho từng vật từ khung chặt (pipeline/s3m_mold.py, frame/<vật>.npz gồm cả ruột), kiểm tra kín bằng bảng băm cạnh,
rồi thả vào MuJoCo lên mặt bàn phẳng để đo đứng yên. Cùng một quy tắc cho mọi vật.

  1. Lưới chiếm chỗ của khung (đệm 2 ô) -> làm mịn Gauss sigma = --sigma ô -> marching cubes ở mức 0.5 -> đổi sang mét theo hệ bàn
     (trục z = pháp tuyến bàn, z = 0 là mặt bàn).
  2. Bỏ mảnh rất nhỏ (< --min_part % thể tích); mỗi mảnh còn lại là một mesh.
  3. Kiểm tra kín (cạnh -> số mặt, bảng băm): cạnh biên (1 mặt) = 0, cạnh non-manifold (> 2 mặt) = 0, thể tích > 0;
     đặc trưng Euler V - E + F; số lỗ chui (genus) = 1 - chi / 2 cho mesh kín định hướng được.
  4. MuJoCo: mặt bàn = plane; vật = freejoint + geom mesh (MuJoCo va chạm bằng bao lồi của mesh), mật độ 500 kg/m3, khối lượng / quán tính
     tính từ chính mesh kín; đặt đáy cách bàn 1 mm, chạy 2 s: dời lớn nhất, xoay lớn nhất, lún sâu nhất, vận tốc cuối.
Xuất: <out>/<vật>/part_<k>.obj, report.json, report.md
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, binary_closing, binary_dilation, binary_erosion, label, distance_transform_edt
from skimage.measure import marching_cubes


def convex_parts(mesh, threshold):
    """Chia lồi CoACD (va chạm MuJoCo đúng hình lõm, vd lòng cốc); lỗi thì trả bao lồi."""
    try:
        import coacd
        coacd.set_log_level("error")
        res = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold)
        return [trimesh.Trimesh(v, f, process=True) for v, f in res]
    except Exception:
        return [mesh.convex_hull]


def edge_stats(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    key = e[:, 0].astype(np.int64) * (1 << 32) + e[:, 1]
    u, c = np.unique(key, return_counts=True)
    return len(u), int((c == 1).sum()), int((c > 2).sum())


def mujoco_rest(mesh_paths, out_xml, density=500.0, settle_s=2.0):
    import mujoco
    geoms, bodies = [], []
    assets = "\n".join(f'<mesh name="m{i}" file="{p.name}"/>' for i, p in enumerate(mesh_paths))
    for i, p in enumerate(mesh_paths):
        bodies.append(f'<body name="b{i}" pos="0 0 0.001"><freejoint/><geom type="mesh" mesh="m{i}" density="{density}" '
                      f'friction="0.8 0.01 0.001" condim="4" solref="0.002 1"/></body>')
    xml = f"""<mujoco><compiler meshdir="{mesh_paths[0].parent.as_posix()}" inertiafromgeom="true"/>
<option timestep="0.001" gravity="0 0 -9.81"/>
<asset>{assets}</asset>
<worldbody><geom name="table" type="plane" size="2 2 0.1" friction="0.8 0.01 0.001" solref="0.002 1"/>{''.join(bodies)}</worldbody></mujoco>"""
    out_xml.write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0 = d.qpos.copy()
    maxd, maxr, pen = 0.0, 0.0, 0.0
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
        for i in range(len(mesh_paths)):
            p, q = d.qpos[7 * i:7 * i + 3], d.qpos[7 * i + 3:7 * i + 7]
            maxd = max(maxd, float(np.linalg.norm(p[:2] - q0[7 * i:7 * i + 2])))
            maxr = max(maxr, float(2 * np.degrees(np.arccos(min(1.0, abs(q[0]))))))
        for c in range(d.ncon):
            pen = max(pen, float(-d.contact[c].dist))
    v = float(np.abs(d.qvel).max()) if len(d.qvel) else 0.0
    return {"mass_kg": [float(m.body_mass[i + 1]) for i in range(len(mesh_paths))], "max_slide_mm": 1000 * maxd,
            "max_rot_deg": maxr, "max_penetration_mm": 1000 * pen, "final_speed": v, "nan": bool(np.isnan(d.qpos).any())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--min_part", type=float, default=2.0)
    ap.add_argument("--close_vox", type=int, default=2, help="bán kính đóng hình thái (ô) để lấp đường hầm nhỏ")
    ap.add_argument("--split_vox", type=int, default=2, help="số ô co để tách mảnh nối bằng cầu mỏng")
    ap.add_argument("--coacd_threshold", type=float, default=0.05)
    ap.add_argument("--support_graph", default="", help="graph.json: chỉ thử MuJoCo riêng cho vật do bàn đỡ")
    ap.add_argument("--min_thick_vox", type=float, default=0.0,
                    help="chốt độ dày (cùng quy tắc mọi vật): nếu lõi dày nhất của khung (max khoảng cách EDT tới mặt, ô) < giá trị này "
                         "thì nở khung thêm ceil(thiếu) ô; 0 = tắt. Vật đủ dày không đổi.")
    ap.add_argument("--objects", default="", help="chỉ làm các vật này (phẩy); report.json cũ trong --out được giữ cho vật khác")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sup = json.loads(Path(a.support).read_text())
    n_t, d_t, mpu = np.array(sup["table"]["normal"]), float(sup["table"]["d"]), float(sup["meters_per_unit"])
    b0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n_t, b0); e1 /= np.linalg.norm(e1); e2 = np.cross(n_t, e1)
    p0 = -d_t * n_t
    R = np.stack([e1, e2, n_t], 0)                                  # hàng = trục hệ bàn
    parent = {}
    if a.support_graph:
        for p_, cs in json.loads(Path(a.support_graph).read_text()).get("support_children", {}).items():
            for c_ in cs:
                parent[c_] = p_
    rep = json.loads((out / "report.json").read_text()) if a.objects and (out / "report.json").exists() else {}
    only = set(x for x in a.objects.split(",") if x)
    for f in sorted(Path(a.frames).glob("*.npz")):
        n = f.stem
        if only and n not in only:
            continue
        z = np.load(f)
        X, vox = z["xyz"].astype(np.float64), float(z["voxel"])
        if len(X) < 30:
            rep[n] = {"skipped": "khung qua nho"}
            continue
        lo = X.min(0) - 2 * vox
        ijk = np.round((X - lo) / vox).astype(int)
        grid = np.zeros(ijk.max(0) + 3, np.float32)
        grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
        grid = grid > 0
        n_raw = int(grid.sum())
        # (a) lấp đường hầm nhỏ (genus giả): đóng hình thái với cầu bán kính --close_vox ô; lỗ thật lớn hơn (quai cốc) vẫn còn
        r = a.close_vox
        zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
        ball = (xx ** 2 + yy ** 2 + zz ** 2) <= r * r
        grid = np.pad(grid, r + 1)
        grid = binary_closing(grid, ball)[r + 1:-(r + 1), r + 1:-(r + 1), r + 1:-(r + 1)]
        guard = 0
        if a.min_thick_vox > 0:
            core = float(distance_transform_edt(np.pad(grid, 1)).max())
            if core < a.min_thick_vox:
                guard = int(np.ceil(a.min_thick_vox - core))
                g2 = np.pad(grid, guard + 1)
                zz, yy, xx = np.mgrid[-guard:guard + 1, -guard:guard + 1, -guard:guard + 1]
                grid = binary_dilation(g2, (xx ** 2 + yy ** 2 + zz ** 2) <= guard * guard)
                lo = lo - (guard + 1) * vox
        # (b) tách mảnh nối bằng cầu mỏng: co --split_vox ô, đánh nhãn, giữ mảnh >= min_part %; trả voxel gốc về mảnh gần nhất
        er = binary_erosion(grid, iterations=a.split_vox) if a.split_vox > 0 else grid
        lab_, nl = label(er)
        sizes = np.bincount(lab_.ravel())[1:] if nl else np.zeros(0)
        big = [i + 1 for i, sz in enumerate(sizes) if sz >= a.min_part / 100 * sizes.sum()] if nl else []
        comps = []
        if len(big) > 1:
            seed = np.isin(lab_, big)
            _, ind = distance_transform_edt(~seed, return_indices=True)
            near = lab_[ind[0], ind[1], ind[2]]
            for b in big:
                comps.append(grid & (near == b))
        else:
            comps = [grid]
        meshes = []
        for cg in comps:
            sm = gaussian_filter(cg.astype(np.float32), a.sigma)
            if sm.max() < 0.5:
                continue
            verts, faces, _, _ = marching_cubes(sm, 0.5)
            Pw = lo + verts * vox
            Pt = ((Pw - p0) @ R.T) * mpu
            m_ = trimesh.Trimesh(Pt, faces[:, ::-1], process=True)
            if m_.volume < 0:
                m_.invert()
            meshes.append(m_)
        if not meshes:
            rep[n] = {"skipped": "khung qua mong sau lam min"}
            continue
        resting = bool(min(m_.vertices[:, 2].min() for m_ in meshes) > -0.02)
        parts = []
        for m_ in meshes:
            # khoá phẳng đáy: vật đặt trên bàn thì cắt phẳng ở z = 0 và đóng nắp
            if resting and m_.vertices[:, 2].min() < 0.0:
                m_ = m_.slice_plane([0, 0, 0], [0, 0, 1], cap=True)
            parts += m_.split(only_watertight=False)
        vols = np.array([abs(p.volume) for p in parts])
        keep = [p for p, v in zip(parts, vols) if v >= a.min_part / 100 * vols.sum()]
        od = out / n
        od.mkdir(exist_ok=True)
        info = {"rests_on_table_mesh": resting, "support_parent": parent.get(n, "?"), "voxel_cm": vox * mpu * 100,
                "voxels_raw": n_raw, "voxels_after_close": int(grid.sum()), "thick_guard_dilate_vox": guard, "split_components": len(comps),
                "parts_total": len(parts), "parts_kept": len(keep), "parts": []}
        paths = []
        for k, pm in enumerate(keep):
            V, F = pm.vertices, pm.faces
            E, nb, nnm = edge_stats(F)
            chi = len(V) - E + len(F)
            wt = nb == 0 and nnm == 0
            pi = {"vertices": int(len(V)), "faces": int(len(F)), "boundary_edges": nb, "nonmanifold_edges": nnm,
                  "euler": int(chi), "genus": (1 - chi / 2) if wt else None, "watertight_edgehash": bool(wt),
                  "watertight_trimesh": bool(pm.is_watertight), "volume_cm3": float(pm.volume * 1e6),
                  "convex_hull_cm3": float(pm.convex_hull.volume * 1e6), "bbox_cm": (pm.extents * 100).round(1).tolist(),
                  "bottom_z_mm": float(V[:, 2].min() * 1000)}
            pm.export(od / f"scene_part_{k}.obj")                          # đúng vị trí trong cảnh (hệ bàn, mét)
            cparts = convex_parts(pm, a.coacd_threshold)
            pi["coacd_parts"] = len(cparts)
            pi["coacd_volume_cm3"] = float(sum(c.volume for c in cparts) * 1e6)
            for j, c in enumerate(cparts):
                c.export(od / f"coll_{k}_{j}.obj")
            p = od / f"part_{k}.obj"
            pm2 = pm.copy()
            pm2.apply_translation([0, 0, -V[:, 2].min()])               # đặt đáy lên z = 0 cho thử MuJoCo riêng
            pm2.export(p)
            paths.append(p)
            info["parts"].append(pi)
        if resting and parent.get(n) == "TABLE":
            try:
                info["mujoco_rest"] = mujoco_rest(paths, od / "rest.xml")
            except Exception as ex:
                info["mujoco_rest"] = {"error": str(ex)[:300]}
        else:
            info["mujoco_rest"] = {"skipped": f"khong thu rieng tren ban (vat do: {parent.get(n, '?')}, dat tren ban: {resting})"}
        rep[n] = info
        print(n, json.dumps({k: v for k, v in info.items() if k != "parts"}), [(q["watertight_edgehash"], q["euler"], round(q["volume_cm3"], 1)) for q in info["parts"]])
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    L = ["# Mesh kin tu khung chat", "", "| vat | manh | kin (canh bien / non-manifold) | Euler | genus | the tich cm3 | bao loi cm3 | MuJoCo truot mm / xoay do / lun mm |",
         "|---|---|---|---|---|---|---|---|"]
    for n, r in rep.items():
        if "parts" not in r:
            L.append(f"| {n} | - | {r.get('skipped')} | | | | | |")
            continue
        mj = r["mujoco_rest"]
        mjs = ("loi" if "error" in mj else mj["skipped"] if "skipped" in mj
               else f"{mj['max_slide_mm']:.2f} / {mj['max_rot_deg']:.2f} / {mj['max_penetration_mm']:.2f}")
        for k, q in enumerate(r["parts"]):
            L.append(f"| {n} | {k} | {'co' if q['watertight_edgehash'] else 'KHONG'} ({q['boundary_edges']} / {q['nonmanifold_edges']}) | {q['euler']} | "
                     f"{q['genus']} | {q['volume_cm3']:.1f} | {q['convex_hull_cm3']:.1f} | {mjs if k == 0 else ''} |")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
