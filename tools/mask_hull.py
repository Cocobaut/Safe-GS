"""
Vùng 3D của từng vật CHỈ từ mask ID + camera SfM (visual hull / space carving), xuất PLY để xoay xem.

Mỗi voxel trong vùng quanh bàn được chiếu vào mọi ảnh (camera COLMAP). Ở mỗi ảnh nó rơi vào:
  - mask của vật k        -> 1 phiếu "trong" cho vật k
  - mask của vật khác      -> trung lập (có thể bị vật đó che)
  - nền (bàn, sàn, ghế, 0) -> 1 phiếu "ngoài"
Voxel thuộc vật k nếu phiếu "trong" >= --min_views và trong / (trong + ngoài) >= --ratio.

    python tools/mask_hull.py --colmap Data/teatime/colmap/sparse/0 --labels Data/teatime/sam2_masks/labels \
        --out Data/teatime/mask_hull
Xuất: hull_objects.ply (mỗi vật 1 màu + điểm SfM xám), hull_<vật>.ply, hull_consistency.ply (màu theo tỉ lệ đồng
thuận, thấy rõ vùng "mập mờ" quanh vật), stats.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Based_Model" / "CoMe" / "scene"))
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, qvec2rotmat  # noqa: E402

# ID mask SAM2 của team (config S3_label / S10_eval)
OBJECTS = {
    "coc_su": ([22], (230, 230, 230)), "ly_tra": ([29], (255, 170, 0)), "dia": ([17], (80, 160, 255)),
    "banh_quy": ([99, 61, 75], (160, 90, 30)), "tao": ([44], (220, 20, 40)), "tui_banh": ([15], (150, 110, 70)),
    "hop_kem_sua": ([67], (0, 220, 220)), "khan_giay": ([11], (255, 255, 120)),
    "gau": ([2, 9], (90, 50, 30)), "cuu": ([4, 27], (255, 140, 200)),
}
# ID của SAM 3 (modal_sam3.py, bản đã gộp: 140->80, 141->70, 101-104->100; bàn chỉ giữ 13)
OBJECTS_SAM3 = {
    "coc_su": ([80], (230, 230, 230)), "ly_tra": ([70], (255, 170, 0)), "dia": ([60], (80, 160, 255)),
    "banh_quy": ([100], (160, 90, 30)), "tao": ([90], (220, 20, 40)), "tui_banh": ([50], (150, 110, 70)),
    "khan_giay": ([130], (255, 255, 120)), "gau": ([20], (90, 50, 30)), "cuu": ([30], (255, 140, 200)),
    "thia": ([110], (120, 120, 255)), "vung_tra": ([170, 171], (200, 120, 0)),
    "goi_keo": ([180, 181], (0, 200, 100)), "coc_kem_sua": ([160, 161], (255, 0, 255)),
}
MPU = 0.247                                                    # m / đơn vị COLMAP (S3: bàn cao 0.75 m)


def write_ply(path, xyz, rgb, extra=None):
    n = len(xyz)
    props = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if extra is not None:
        props.append(("object", "u1"))
    a = np.empty(n, dtype=props)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if extra is not None:
        a["object"] = extra
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    head += [f"property {'float' if t == 'f4' else 'uchar'} {p}" for p, t in props] + ["end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(head).encode())
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=320, help="số voxel theo cạnh dài nhất")
    ap.add_argument("--pad", type=float, default=0.5, help="nới vùng quanh các vật trên bàn (tỉ lệ kích thước)")
    ap.add_argument("--min_views", type=int, default=15)
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--scheme", choices=["sam2", "sam3"], default="sam2", help="bảng ID mask")
    a = ap.parse_args()
    global OBJECTS
    if a.scheme == "sam3":
        OBJECTS = OBJECTS_SAM3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sp = Path(a.colmap)
    intr = read_intrinsics_binary(str(sp / "cameras.bin"))
    ext = read_extrinsics_binary(str(sp / "images.bin"))
    xyz_sfm, rgb_sfm, _ = read_points3D_binary(str(sp / "points3D.bin"))
    names = list(OBJECTS)
    K = len(names)
    lut = np.full(256, K + 1, np.int64)                         # mặc định: nền -> phiếu "ngoài"
    for k, n in enumerate(names):
        lut[OBJECTS[n][0]] = k
    lut_t = torch.tensor(lut, device=dev)

    views = []
    for e in ext.values():
        lp = Path(a.labels) / (Path(e.name).stem + ".png")
        if not lp.exists():
            continue
        c = intr[e.camera_id]
        p = c.params
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if c.model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
        lab = np.array(Image.open(lp))
        if lab.shape != (c.height, c.width):
            lab = np.array(Image.open(lp).resize((c.width, c.height), Image.NEAREST))
        views.append((torch.tensor(qvec2rotmat(e.qvec), dtype=torch.float32, device=dev),
                      torch.tensor(e.tvec, dtype=torch.float32, device=dev), fx, fy, cx, cy,
                      lut_t[torch.tensor(lab.astype(np.int64), device=dev)]))
    print(f"{len(views)} view có mask")

    def vote(P):
        """P [N,3] -> cnt [N,K] (phiếu trong theo vật), against [N], seen [N]."""
        cnt = torch.zeros(len(P), K, dtype=torch.int16, device=dev)
        against = torch.zeros(len(P), dtype=torch.int16, device=dev)
        seen = torch.zeros(len(P), dtype=torch.int16, device=dev)
        for R, t, fx, fy, cx, cy, g in views:
            X = P @ R.T + t
            z = X[:, 2]
            u = (fx * X[:, 0] / z + cx).long(); v = (fy * X[:, 1] / z + cy).long()
            H, W = g.shape
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            gi = torch.full((len(P),), K, dtype=torch.long, device=dev)
            gi[ok] = g[v[ok], u[ok]]
            seen += ok.short()
            against += (gi == K + 1).short()
            m = gi < K
            cnt[m.nonzero(as_tuple=True)[0], gi[m]] += 1
        return cnt, against, seen

    # 1. giới hạn vùng: điểm SfM của các vật trên bàn (bỏ gấu / cừu vì quá to), phân vị 2-98 %, nới --pad
    P = torch.tensor(xyz_sfm, dtype=torch.float32, device=dev)
    cnt, against, _ = vote(P)
    best = cnt.max(1)
    table_k = torch.tensor([k for k, n in enumerate(names) if n not in ("gau", "cuu")], device=dev)
    keep = (best.values >= a.min_views) & (best.values.float() / (best.values + against).float() >= 0.6) \
        & torch.isin(best.indices, table_k)
    obj_pts = P[keep]
    lo, hi = torch.quantile(obj_pts, 0.02, dim=0), torch.quantile(obj_pts, 0.98, dim=0)
    pad = (hi - lo) * a.pad
    lo, hi = lo - pad, hi + pad
    vs = float((hi - lo).max()) / a.res
    print(f"vùng: {((hi - lo) * MPU * 100).cpu().numpy().round(1)} cm, voxel {vs * MPU * 1000:.1f} mm")

    # 2. bỏ phiếu cho từng voxel (theo lát z để vừa bộ nhớ)
    nx, ny, nz = [int(np.ceil(float(x) / vs)) for x in (hi - lo)]
    xs = lo[0] + (torch.arange(nx, device=dev) + 0.5) * vs
    ys = lo[1] + (torch.arange(ny, device=dev) + 0.5) * vs
    res_xyz, res_k, res_ratio = [], [], []
    for iz in range(nz):
        zc = lo[2] + (iz + 0.5) * vs
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        Q = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), float(zc), device=dev)], 1)
        cnt, against, _ = vote(Q)
        b = cnt.max(1)
        r = b.values.float() / (b.values + against).float().clamp(min=1)
        m = (b.values >= a.min_views) & (r >= 0.5)
        res_xyz.append(Q[m].cpu()); res_k.append(b.indices[m].cpu()); res_ratio.append(r[m].cpu())
    Vxyz = torch.cat(res_xyz).numpy(); Vk = torch.cat(res_k).numpy(); Vr = torch.cat(res_ratio).numpy()

    # 3. xuất
    sfm_rgb = np.full((len(xyz_sfm), 3), 110, np.uint8)
    hull = Vr >= a.ratio
    col = np.array([OBJECTS[n][1] for n in names], np.uint8)
    write_ply(out / "hull_objects.ply", np.concatenate([Vxyz[hull], xyz_sfm]),
              np.concatenate([col[Vk[hull]], sfm_rgb]), np.concatenate([Vk[hull] + 1, np.zeros(len(xyz_sfm), int)]))
    t = np.clip((Vr - 0.5) / 0.5, 0, 1)[:, None]                  # 0.5 xanh -> 1.0 đỏ
    heat = (np.array([40, 80, 255]) * (1 - t) + np.array([255, 40, 40]) * t).astype(np.uint8)
    write_ply(out / "hull_consistency.ply", np.concatenate([Vxyz, xyz_sfm]), np.concatenate([heat, sfm_rgb]))
    stats = {"views": len(views), "voxel_mm": vs * MPU * 1000, "min_views": a.min_views, "ratio": a.ratio, "objects": {}}
    for k, n in enumerate(names):
        m = hull & (Vk == k)
        if m.sum() == 0:
            stats["objects"][n] = {"voxels": 0}
            continue
        p = Vxyz[m]
        write_ply(out / f"hull_{n}.ply", p, np.repeat(col[k][None], len(p), 0))
        stats["objects"][n] = {"voxels": int(m.sum()), "volume_cm3": float(m.sum() * (vs * MPU * 100) ** 3),
                               "bbox_cm": ((p.max(0) - p.min(0)) * MPU * 100).round(1).tolist(),
                               "uncertain_voxels_0.5_to_ratio": int(((Vk == k) & ~hull).sum())}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
