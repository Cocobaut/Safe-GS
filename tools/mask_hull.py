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
    ap.add_argument("--voxel_mm", type=float, default=0.0, help="> 0: cạnh voxel cố định (mm), bỏ qua --res")
    ap.add_argument("--pad", type=float, default=0.5, help="nới vùng quanh các vật trên bàn (tỉ lệ kích thước)")
    ap.add_argument("--min_views", type=int, default=15)
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--scheme", choices=["sam2", "sam3"], default="sam2", help="bảng ID mask")
    ap.add_argument("--adaptive_min_views", type=int, default=0, help="1: min_views_k = min(min_views, max(4, 0.35*frames_k))")
    ap.add_argument("--soft_empty_weight", type=float, default=1.0, help="trong so phieu ngoai o view khong thay vat k (1.0=cu, 0.25=mem)")
    ap.add_argument("--clean_3d", type=int, default=0, help="1: binary_closing 1 voxel + loc nhieu thanh phan nho 3D")
    ap.add_argument("--objects_json", default="", help="canh khac teatime: JSON {ten_vat: [id mask, ...]} (vd S3_label.objects cua config)")
    ap.add_argument("--vox_rel", type=float, default=0.0,
                    help="> 0: canh voxel = trung vi kich thuoc vat (phan vi 2-98 %% diem SfM) / vox_rel (khong phu thuoc vat ngoai lai lam phinh vung)")
    ap.add_argument("--mpu", type=float, default=0.247, help="m / don vi COLMAP (chi de bao cao cm, voxel_mm)")
    ap.add_argument("--assign_mode", default="count", choices=["count", "ratio"],
                    help="count: voxel thuoc vat co nhieu phieu trong nhat (cu); ratio: vat co ti le phieu trong / so view CO nhan vat do "
                         "va nhin thay voxel cao nhat (prompt chung bat nham vat cu the trong mot so frame khong nuot duoc vat cu the)")
    ap.add_argument("--sfm_anchor", type=int, default=0, help="1: chi giu thanh phan lien thong 3D co it nhat 1 diem SfM cua vat (bo non ma)")
    a = ap.parse_args()
    global OBJECTS, MPU
    if a.scheme == "sam3":
        OBJECTS = OBJECTS_SAM3
    if a.objects_json:
        objs = json.loads(Path(a.objects_json).read_text(encoding="utf-8"))
        objs = objs.get("S3_label", {}).get("objects", objs)
        rng = np.random.default_rng(0)
        OBJECTS = {n: (list(ids), tuple(int(c) for c in rng.integers(60, 255, 3))) for n, ids in objs.items()}
    MPU = a.mpu
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
    frames_with_k = np.zeros(K, dtype=int)
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
        mapped = lut_t[torch.tensor(lab.astype(np.int64), device=dev)]
        has_k = torch.tensor([int((mapped == k).sum()) >= 20 for k in range(K)], dtype=torch.bool, device=dev)
        frames_with_k += has_k.cpu().numpy().astype(int)
        views.append((torch.tensor(qvec2rotmat(e.qvec), dtype=torch.float32, device=dev),
                      torch.tensor(e.tvec, dtype=torch.float32, device=dev), fx, fy, cx, cy,
                      mapped, has_k))
    print(f"{len(views)} view có mask")

    min_views_per_k = torch.full((K,), a.min_views, dtype=torch.float32, device=dev)
    if a.adaptive_min_views:
        for k in range(K):
            min_views_per_k[k] = min(a.min_views, max(4, int(round(0.35 * frames_with_k[k])))) if frames_with_k[k] > 0 else a.min_views

    def vote(P):
        """P [N,3] -> cnt [N,K] (phiếu trong theo vật), against [N,K] (phiếu ngoài theo vật), seen [N]."""
        cnt = torch.zeros(len(P), K, dtype=torch.float32, device=dev)
        visk = torch.zeros(len(P), K, dtype=torch.float32, device=dev)
        against = torch.zeros(len(P), K, dtype=torch.float32, device=dev)
        seen = torch.zeros(len(P), dtype=torch.int16, device=dev)
        for R, t, fx, fy, cx, cy, g, has_k in views:
            X = P @ R.T + t
            z = X[:, 2]
            u = (fx * X[:, 0] / z + cx).long(); v = (fy * X[:, 1] / z + cy).long()
            H, W = g.shape
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            gi = torch.full((len(P),), K, dtype=torch.long, device=dev)
            gi[ok] = g[v[ok], u[ok]]
            seen += ok.short()
            bg_hit = (gi == K + 1).float()[:, None]
            w_k = torch.where(has_k, 1.0, float(a.soft_empty_weight))[None, :]
            against += bg_hit * w_k
            visk += ok.float()[:, None] * has_k.float()[None, :]
            m = gi < K
            cnt[m.nonzero(as_tuple=True)[0], gi[m]] += 1.0
        return cnt, against, seen, visk

    def pick(cnt, visk):
        if a.assign_mode == "ratio":
            sc = torch.where(cnt >= min_views_per_k[None, :], cnt / visk.clamp(min=1), torch.full_like(cnt, -1.0))
            idx = sc.argmax(1)
            return cnt.gather(1, idx[:, None]).squeeze(1), idx
        b = cnt.max(1)
        return b.values, b.indices

    # 1. giới hạn vùng: điểm SfM của MỌI vật (không bỏ vật nào theo tên; bản cũ bỏ gấu / cừu nên hull của chúng bị cụt),
    #    phân vị 2-98 % theo từng vật rồi hợp lại, nới --pad
    P = torch.tensor(xyz_sfm, dtype=torch.float32, device=dev)
    cnt, against, _, visk = vote(P)
    bv, bi = pick(cnt, visk)
    ag_best = against.gather(1, bi[:, None]).squeeze(1)
    mv_best = min_views_per_k[bi]
    keep = (bv >= mv_best) & (bv / (bv + ag_best).clamp(min=1) >= 0.6)
    los, his = [], []
    for k in range(K):
        q = P[keep & (bi == k)]
        if len(q) >= 20:
            los.append(torch.quantile(q, 0.02, dim=0)); his.append(torch.quantile(q, 0.98, dim=0))
    ext_med = float(torch.median(torch.stack([(h - l).max() for l, h in zip(los, his)])))
    lo, hi = torch.stack(los).min(0).values, torch.stack(his).max(0).values
    pad = (hi - lo) * a.pad
    lo, hi = lo - pad, hi + pad
    vs = float((hi - lo).max()) / a.res if a.voxel_mm <= 0 else a.voxel_mm / 1000 / MPU
    if a.vox_rel > 0:
        vs = ext_med / a.vox_rel
        print(f"vox_rel: trung vi kich thuoc vat {ext_med:.4f} don vi -> voxel {vs:.5f}")
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
        cnt, against, _, visk = vote(Q)
        bv_, bi_ = pick(cnt, visk)
        ag_b = against.gather(1, bi_[:, None]).squeeze(1)
        mv_b = min_views_per_k[bi_]
        r = bv_ / (bv_ + ag_b).clamp(min=1)
        m = (bv_ >= mv_b) & (r >= 0.5)
        res_xyz.append(Q[m].cpu()); res_k.append(bi_[m].cpu()); res_ratio.append(r[m].cpu())
    Vxyz = torch.cat(res_xyz).numpy(); Vk = torch.cat(res_k).numpy(); Vr = torch.cat(res_ratio).numpy()

    # 3. xuất
    sfm_rgb = np.full((len(xyz_sfm), 3), 110, np.uint8)
    hull = Vr >= a.ratio
    if a.clean_3d and hull.any():
        from scipy.ndimage import binary_closing, label as cc3_label
        ijk = np.round((Vxyz - lo.cpu().numpy()) / vs - 0.5).astype(int)
        keep_mask = np.zeros(len(Vxyz), dtype=bool)
        for k in range(K):
            mk = hull & (Vk == k)
            if mk.sum() == 0:
                continue
            pts_ijk = ijk[mk]
            bmin = pts_ijk.min(0)
            rel = pts_ijk - bmin + 1
            sz = rel.max(0) + 2
            grid = np.zeros(sz, dtype=bool)
            grid[rel[:, 0], rel[:, 1], rel[:, 2]] = True
            grid = binary_closing(grid, iterations=1)
            lab, ncc = cc3_label(grid)
            if ncc > 1:
                counts = np.bincount(lab.ravel())[1:]
                max_c = counts.max()
                sfm_k = P[keep & (bi == k)].cpu().numpy()
                has_sfm = np.zeros(ncc + 1, dtype=bool)
                if a.sfm_anchor and len(sfm_k) > 0:
                    s_rel = np.round((sfm_k - lo.cpu().numpy()) / vs - 0.5).astype(int) - bmin + 1
                    in_b = np.all((s_rel >= 0) & (s_rel < sz), axis=1)
                    for sx, sy, sz_ in s_rel[in_b]:
                        cid_s = int(lab[sx, sy, sz_])
                        if cid_s > 0:
                            has_sfm[cid_s] = True
                keep_largest_only = a.sfm_anchor and not has_sfm.any()
                for c_id, cnt_c in enumerate(counts, 1):
                    if cnt_c < max(8, 0.01 * max_c):
                        grid[lab == c_id] = False
                    elif a.sfm_anchor and has_sfm.any() and not has_sfm[c_id] and cnt_c < 0.25 * max_c:
                        grid[lab == c_id] = False
                    elif keep_largest_only and cnt_c < max_c:
                        grid[lab == c_id] = False
            # Map back to existing Vxyz or append closed voxels
            gx, gy, gz = np.where(grid)
            new_ijk = np.stack([gx, gy, gz], 1) + bmin - 1
            new_xyz = lo.cpu().numpy() + (new_ijk + 0.5) * vs
            # Keep original voxels that survived + newly closed voxels
            orig_rel = ijk[mk] - bmin + 1
            orig_ok = grid[orig_rel[:, 0], orig_rel[:, 1], orig_rel[:, 2]]
            keep_mask[np.where(mk)[0][orig_ok]] = True
            # Append newly filled 1-voxel holes if any
            orig_set = set(map(tuple, pts_ijk.tolist()))
            added = [pt for i_row, pt in zip(new_ijk.tolist(), new_xyz) if tuple(i_row) not in orig_set]
            if added:
                added_arr = np.array(added, dtype=Vxyz.dtype)
                Vxyz = np.concatenate([Vxyz, added_arr], 0)
                Vk = np.concatenate([Vk, np.full(len(added_arr), k, dtype=Vk.dtype)], 0)
                Vr = np.concatenate([Vr, np.full(len(added_arr), 1.0, dtype=Vr.dtype)], 0)
                hull = np.concatenate([hull, np.ones(len(added_arr), dtype=bool)], 0)
                keep_mask = np.concatenate([keep_mask, np.ones(len(added_arr), dtype=bool)], 0)
                ijk = np.concatenate([ijk, np.array([i_row for i_row in new_ijk.tolist() if tuple(i_row) not in orig_set], dtype=int)], 0)
        hull = hull & keep_mask
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
