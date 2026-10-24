"""
S6: mesh va chạm cho từng vật + mặt đỡ, theo đơn vị mét, hệ toạ độ mô phỏng.

Hệ toạ độ mô phỏng ("sim"): trục z = pháp tuyến mặt bàn (hướng lên), mặt bàn ở z = 0, gốc xy = tâm mặt bàn,
đơn vị mét (meters_per_unit từ S3: chiều cao bàn giả định / khoảng cách bàn-sàn). Lưu world_to_sim.json.

Mỗi vật:
  1. Trường opacity GOF của RIÊNG Gaussian vật trên lưới bao vật (grid_res ô theo cạnh dài), qua các camera train,
     marching cubes mức 0.5 (= mặt SDF 0 của CoMe/GOF). (Bản đầu dùng TSDF Open3D nhưng ra rỗng ở mọi biến thể:
     pipeline/debug_tsdf.py.) Phần dưới đáy vật không camera nào thấy nên trường tự đóng -> khối kín, cắt ở bước 2.
  2. Giữ thành phần lớn, đổi sang hệ sim, cắt tại z = cut_offset và đóng nắp (trimesh slice_plane cap=True) = đáy phẳng.
  3. Kiểm tra kín; không kín thì thử fill_holes, vẫn không thì dùng bao lồi làm mesh va chạm (ghi cờ).
  4. CoACD -> các mảnh lồi collision_XX.obj.
Mặt đỡ: bàn = hộp (mép xy theo percentile 1-99 Gaussian mặt đỡ), mặt trên z = 0, dày 3 cm; sàn = mặt phẳng z = -cao bàn.
Checkpoint: vật nào đã có info.json thì bỏ qua.
Xuất: objects/<tên>/{visual.ply, collision_*.obj, info.json, preview.png}, support/{table_box.json, floor.json},
      world_to_sim.json, scene.glb (mọi mesh + bàn + sàn để mở xem), preview_scene.png
"""
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from common import log, make_parser, load_scene, read_ply, gaussians_from_ply, load_cfg, opacity_field
from arguments import get_combined_args


def world_to_sim(n, d, center_world, mpu):
    z = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)                       # hàng = trục sim trong toạ độ thế giới
    c = center_world - (center_world @ z + d) * z    # chiếu tâm xuống mặt bàn
    T = np.eye(4)
    T[:3, :3] = R * mpu
    T[:3, 3] = -(R @ c) * mpu
    return T


def field_mesh(g_o, S, views, xyz, res):
    """Marching cubes mức 0.5 trên trường opacity GOF của RIÊNG Gaussian vật (mặt SDF = 0 của CoMe/GOF).
    Lưới bao vật (percentile 1-99, nới 10%), res ô theo cạnh dài nhất."""
    from skimage.measure import marching_cubes
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    pad = 0.1 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    step = float((hi - lo).max()) / res
    dims = np.ceil((hi - lo) / step).astype(int) + 1
    axes = [lo[i] + step * np.arange(dims[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    # chỉ giữ view thấy vật (integrate lỗi CUDA "invalid configuration" khi không Gaussian nào trong khung hình)
    X = torch.from_numpy(xyz.astype(np.float32)).cuda()
    Xh = torch.cat([X, torch.ones(len(X), 1, device="cuda")], 1)
    keep = []
    for v in views:
        pp = Xh @ v.full_proj_transform
        z = (Xh @ v.world_view_transform)[:, 2]
        ndc = pp[:, :2] / pp[:, 3:4].clamp(min=1e-8)
        inside = (z > 0.2) & (ndc.abs() <= 1).all(1)
        if int(inside.sum()) >= 20:
            keep.append(v)
    views = keep
    vals = np.empty(len(grid), np.float32)
    B = 1_000_000
    for s in range(0, len(grid), B):
        pts = torch.from_numpy(grid[s:s + B]).cuda().contiguous()
        vals[s:s + B] = opacity_field(pts, views, g_o, S).cpu().numpy()
    F = vals.reshape(dims)
    stats = {"views_used": len(views), "grid_dims": dims.tolist(), "step_units": step, "field_max": float(F.max()),
             "frac_inside": float((F >= 0.5).mean())}
    if F.max() < 0.5:
        return None, stats
    verts, faces, _, _ = marching_cubes(F, level=0.5, spacing=(step, step, step))   # pháp tuyến hướng ra ngoài
    return trimesh.Trimesh(verts + lo, faces, process=True), stats


def preview_png(meshes, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 4))
    for k, (el, az) in enumerate([(25, -60), (25, 30), (85, -90)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        allv = []
        for m, color in meshes:
            f = m.faces
            if len(f) > 6000:
                f = f[np.random.default_rng(0).choice(len(f), 6000, replace=False)]
            ax.add_collection3d(Poly3DCollection(m.vertices[f], facecolor=color, edgecolor="none", alpha=0.9))
            allv.append(m.vertices)
        V = np.concatenate(allv)
        c, r = V.mean(0), (V.max(0) - V.min(0)).max() / 2 + 1e-6
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(el, az); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    p, model, pipe, ss, _, _ = make_parser("S6 mesh")
    p.add_argument("--s3", type=str, required=True)
    p.add_argument("--s5", type=str, required=True)
    args = get_combined_args(p)
    cfg_all = load_cfg(args.cfg); cfg = cfg_all["S6_mesh"]
    s3, s5, out = Path(args.s3), Path(args.s5), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sup = json.loads((s3 / "support.json").read_text())
    groups = json.loads((s3 / "groups.json").read_text())
    n = np.array(sup["table"]["normal"]); d = sup["table"]["d"]; mpu = sup["meters_per_unit"]
    assert mpu, "S3 không tìm được sàn -> chưa có scale mét (support.json)"

    # hệ sim: tâm = tâm Gaussian mặt đỡ
    sup_el = read_ply(s3 / "gaussians" / "background_with_support.ply")
    sxyz = np.stack([sup_el["x"], sup_el["y"], sup_el["z"]], 1)
    on_table = np.abs(sxyz @ n + d) < 0.01 / mpu
    T = world_to_sim(n, d, sxyz[on_table].mean(0), mpu)
    (out / "world_to_sim.json").write_text(json.dumps({"T": T.tolist(), "meters_per_unit": mpu,
                                                       "note": "p_sim = T @ [p_world, 1]"}, indent=2))
    to_sim = lambda P: P @ T[:3, :3].T + T[:3, 3]

    # mặt đỡ
    (out / "support").mkdir(exist_ok=True)
    tab = to_sim(sxyz[on_table])
    lo, hi = np.percentile(tab[:, :2], 1, 0), np.percentile(tab[:, :2], 99, 0)
    box = {"center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), -0.015],
           "half_size": [float((hi[0] - lo[0]) / 2), float((hi[1] - lo[1]) / 2), 0.015], "top_z": 0.0}
    (out / "support" / "table_box.json").write_text(json.dumps(box, indent=2))
    floor = {"z": -cfg["table_height_m"], "note": "theo giả định chiều cao bàn"}
    (out / "support" / "floor.json").write_text(json.dumps(floor, indent=2))

    S = load_scene(args, model, pipe, ss)
    import coacd
    scene_meshes = []
    for name in groups["objects"]:
        od = out / "objects" / name
        src = s5 / "objects" / name / "gaussians.ply"
        if (od / "info.json").exists():
            info = json.loads((od / "info.json").read_text())
            if (od / "visual.ply").exists():
                scene_meshes.append(trimesh.load(od / "visual.ply"))
            log(f"{name}: đã có, bỏ qua"); continue
        od.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            (od / "info.json").write_text(json.dumps({"skipped": "không có Gaussian"})); continue
        el = read_ply(src)
        xyz = np.stack([el["x"], el["y"], el["z"]], 1)
        g_o = gaussians_from_ply(src, S["dataset"].sh_degree, S["train"])
        m, fstats = field_mesh(g_o, S, S["train"], xyz, cfg["grid_res"])
        if m is None or len(m.faces) == 0:
            (od / "info.json").write_text(json.dumps({"skipped": "trường opacity không vượt 0.5", **fstats})); continue
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            areas = np.array([q.area for q in parts])
            m = trimesh.util.concatenate([q for q, a in zip(parts, areas) if a >= 0.1 * areas.max()])
        m.apply_transform(T)
        cut = cfg["cut_offset"]
        below = float((m.vertices[:, 2] < cut).mean())
        try:
            mc = m.slice_plane([0, 0, cut], [0, 0, 1], cap=True)
        except Exception as e:
            log(f"{name}: slice_plane cap lỗi ({e}), cắt không đóng nắp"); mc = m.slice_plane([0, 0, cut], [0, 0, 1])
        if not mc.is_watertight:
            trimesh.repair.fill_holes(mc)
        watertight = bool(mc.is_watertight)
        coll_src = mc if watertight else mc.convex_hull
        mc.export(od / "visual.ply")
        cm = coacd.Mesh(coll_src.vertices, coll_src.faces)
        hulls = coacd.run_coacd(cm, threshold=cfg["coacd_threshold"])
        for k, (vv, ff) in enumerate(hulls):
            trimesh.Trimesh(vv, ff).export(od / f"collision_{k:02d}.obj")
        info = {**fstats, "vertices": int(len(mc.vertices)),
                "faces": int(len(mc.faces)), "watertight": watertight,
                "collision_from": "mesh" if watertight else "convex_hull (mesh không kín)",
                "volume_m3": float(coll_src.volume), "bbox_m": coll_src.extents.tolist(),
                "fraction_below_table_cut": below, "coacd_parts": len(hulls)}
        (od / "info.json").write_text(json.dumps(info, indent=2))
        preview_png([(mc, (0.8, 0.5, 0.3))], od / "preview.png", f"{name} (m)")
        scene_meshes.append(mc)
        log(f"{name}: {info}")

    # cảnh tổng
    tb = trimesh.creation.box(extents=np.array(box["half_size"]) * 2)
    tb.apply_translation(box["center"])
    fl = trimesh.creation.box(extents=[box["half_size"][0] * 4, box["half_size"][1] * 4, 0.01])
    fl.apply_translation([box["center"][0], box["center"][1], floor["z"] - 0.005])
    sc = trimesh.Scene([tb, fl] + scene_meshes)
    sc.export(out / "scene.glb")
    if scene_meshes:
        preview_png([(tb, (0.6, 0.6, 0.6))] + [(q, (0.8, 0.5, 0.3)) for q in scene_meshes], out / "preview_scene.png",
                    "vật + bàn (m)")
    log(f"S6 xong -> {out}")


if __name__ == "__main__":
    main()
