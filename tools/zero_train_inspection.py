"""
Zero-training 3D Object Isolation and View Coverage Inspection.
Xuat ket qua trang - den (monochrome / fill-in style) cho tung phuong phap vao Data/teatime/zero_train_inspection/

Khong dung mau sac cau vong.
- Trang / Xam sang: Vung quan sat duoc day du (goc nhin toi duoc, multi-view observed).
- Den / Xam toi: Vung thieu goc nhin (goc nhin chua toi, day vat, mat cat cut do che khuat).
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Based_Model" / "CoMe" / "scene"))
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


OBJECTS_SAM3 = {
    "coc_su": ([80], "Cốc sứ"),
    "ly_tra": ([70], "Ly trà"),
    "dia": ([60], "Đĩa"),
    "banh_quy": ([100], "Bánh quy"),
    "tao": ([90], "Táo"),
    "tui_banh": ([50], "Túi bánh"),
    "khan_giay": ([130], "Khăn giấy"),
    "gau": ([20], "Gấu bông"),
    "cuu": ([30], "Cừu bông"),
    "thia": ([110], "Thìa"),
    "goi_keo": ([180, 181], "Gói kẹo"),
}
MPU = 0.247  # m / don vi COLMAP


def read_points3D_with_track(path_to_model_file):
    """Doc points3D.bin va lay ca track_length."""
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        xyzs = np.empty((num_points, 3), dtype=np.float32)
        rgbs = np.empty((num_points, 3), dtype=np.uint8)
        tracks = np.empty(num_points, dtype=np.int32)
        for p_id in range(num_points):
            data = fid.read(43)
            props = struct.unpack("<QdddBBBd", data)
            xyzs[p_id] = props[1:4]
            rgbs[p_id] = props[4:7]
            t_len = struct.unpack("<Q", fid.read(8))[0]
            fid.read(8 * t_len)  # bo qua track elems
            tracks[p_id] = t_len
    return xyzs, rgbs, tracks


def write_ply(path, xyz, rgb):
    n = len(xyz)
    props = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    a = np.empty(n, dtype=props)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    head = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]
    with open(path, "wb") as f:
        f.write("\n".join(head).encode())
        f.write(a.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap", default="Data/teatime/colmap/sparse/0")
    parser.add_argument("--labels", default="Data/teatime/sam3_masks/labels")
    parser.add_argument("--images", default="Data/teatime/colmap/images")
    parser.add_argument("--out", default="Data/teatime/zero_train_inspection")
    parser.add_argument("--res", type=int, default=240)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Thiet bi: {dev}")
    print(f"[*] Thu muc xuat: {out}")

    sp = Path(args.colmap)
    intr = read_intrinsics_binary(str(sp / "cameras.bin"))
    ext = read_extrinsics_binary(str(sp / "images.bin"))
    xyz_sfm, rgb_sfm, tracks_sfm = read_points3D_with_track(str(sp / "points3D.bin"))
    print(f"[*] Da doc {len(xyz_sfm):,} diem SfM")

    names = list(OBJECTS_SAM3.keys())
    K = len(names)
    lut = np.full(256, K + 1, np.int64)  # default background / other
    for k, n in enumerate(names):
        lut[OBJECTS_SAM3[n][0]] = k
    lut_t = torch.tensor(lut, device=dev)

    views = []
    view_cams = []
    labels_dir = Path(args.labels)
    images_dir = Path(args.images)

    print("[*] Dang nap cac camera va mask SAM 3...")
    for e in ext.values():
        lp = labels_dir / (Path(e.name).stem + ".png")
        if not lp.exists():
            continue
        c = intr[e.camera_id]
        p = c.params
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if c.model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
        lab = np.array(Image.open(lp))
        if lab.shape != (c.height, c.width):
            lab = np.array(Image.open(lp).resize((c.width, c.height), Image.NEAREST))
        R = torch.tensor(qvec2rotmat(e.qvec), dtype=torch.float32, device=dev)
        t = torch.tensor(e.tvec, dtype=torch.float32, device=dev)
        cam_pos = -R.T @ t  # Camera center in world space
        views.append((R, t, fx, fy, cx, cy, lut_t[torch.tensor(lab.astype(np.int64), device=dev)], cam_pos))
        view_cams.append((R, t, fx, fy, cx, cy, e.name, cam_pos))

    print(f"[*] Da nap {len(views)} views hop le.")

    # -------------------------------------------------------------------------
    # PHUONG PHAP 2: COLMAP SfM Point Cloud (Trang - Den theo Track Length)
    # -------------------------------------------------------------------------
    print("\n[+] DANG XU LY PHUONG PHAP 2: COLMAP SfM Point Cloud (Track Grayscale)...")
    P_sfm = torch.tensor(xyz_sfm, dtype=torch.float32, device=dev)

    # Vote cho SfM points
    cnt_sfm = torch.zeros(len(P_sfm), K, dtype=torch.int16, device=dev)
    against_sfm = torch.zeros(len(P_sfm), dtype=torch.int16, device=dev)

    for R, t, fx, fy, cx, cy, g, _ in views:
        X = P_sfm @ R.T + t
        z = X[:, 2]
        u = (fx * X[:, 0] / z + cx).long()
        v = (fy * X[:, 1] / z + cy).long()
        H, W = g.shape
        ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        gi = torch.full((len(P_sfm),), K, dtype=torch.long, device=dev)
        gi[ok] = g[v[ok], u[ok]]
        against_sfm += (gi == K + 1).short()
        m = gi < K
        cnt_sfm[m.nonzero(as_tuple=True)[0], gi[m]] += 1

    b_sfm = cnt_sfm.max(1)
    ratio_sfm = b_sfm.values.float() / (b_sfm.values + against_sfm).float().clamp(min=1)
    is_obj_sfm = (b_sfm.values >= 5) & (ratio_sfm >= 0.6)

    # To mau trang den theo track_length:
    # track cang cao -> cang trang tinh (255)
    # track thap -> xam toi (60 - 120)
    # Nen xung quanh -> xam rat toi (25)
    rgb_p2 = np.full((len(xyz_sfm), 3), 25, dtype=np.uint8)  # nen toi
    obj_mask_np = is_obj_sfm.cpu().numpy()

    # Track length cua cac diem thuoc vat
    t_obj = tracks_sfm[obj_mask_np]
    # Normalize track length tu 0 -> 255 (min 3, max 30)
    t_norm = np.clip((t_obj - 3) / (30.0 - 3.0), 0.0, 1.0)
    val_gray = (70 + t_norm * 185).astype(np.uint8)  # range 70 -> 255
    rgb_p2[obj_mask_np] = np.stack([val_gray, val_gray, val_gray], axis=-1)

    p2_path = out / "method2_sfm_points_bw.ply"
    write_ply(p2_path, xyz_sfm, rgb_p2)
    print(f"    -> Da xuat: {p2_path.name} ({int(obj_mask_np.sum()):,} diem vat the)")

    # -------------------------------------------------------------------------
    # PHUONG PHAP 1: Visual Hull Surface Coverage (Trang - Den theo Goc Nhin)
    # -------------------------------------------------------------------------
    print("\n[+] DANG XU LY PHUONG PHAP 1: Visual Hull Surface (Black-and-White Coverage)...")

    # Xac dinh vung bounding box quanh cac vat tren ban
    table_k = torch.tensor([k for k, n in enumerate(names) if n not in ("gau", "cuu")], device=dev)
    keep_pts = is_obj_sfm & torch.isin(b_sfm.indices, table_k)
    obj_pts = P_sfm[keep_pts]
    lo = torch.quantile(obj_pts, 0.02, dim=0)
    hi = torch.quantile(obj_pts, 0.98, dim=0)
    pad = (hi - lo) * 0.4
    lo, hi = lo - pad, hi + pad
    vs = float((hi - lo).max()) / args.res
    print(f"    Vung khao sat: {((hi - lo) * MPU * 100).cpu().numpy().round(1)} cm, voxel {vs * MPU * 1000:.1f} mm")

    nx, ny, nz = [int(np.ceil(float(x) / vs)) for x in (hi - lo)]
    xs = lo[0] + (torch.arange(nx, device=dev) + 0.5) * vs
    ys = lo[1] + (torch.arange(ny, device=dev) + 0.5) * vs

    def vote_batch(P):
        cnt = torch.zeros(len(P), K, dtype=torch.int16, device=dev)
        against = torch.zeros(len(P), dtype=torch.int16, device=dev)
        for R, t, fx, fy, cx, cy, g, _ in views:
            X = P @ R.T + t
            z = X[:, 2]
            u = (fx * X[:, 0] / z + cx).long()
            v = (fy * X[:, 1] / z + cy).long()
            H, W = g.shape
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            gi = torch.full((len(P),), K, dtype=torch.long, device=dev)
            gi[ok] = g[v[ok], u[ok]]
            against += (gi == K + 1).short()
            m = gi < K
            cnt[m.nonzero(as_tuple=True)[0], gi[m]] += 1
        return cnt, against

    # Xay dung voxel grid
    res_xyz, res_k = [], []
    for iz in range(nz):
        zc = lo[2] + (iz + 0.5) * vs
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        Q = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), float(zc), device=dev)], 1)
        cnt, against = vote_batch(Q)
        b = cnt.max(1)
        r = b.values.float() / (b.values + against).float().clamp(min=1)
        m = (b.values >= 15) & (r >= 0.85)
        if m.any():
            res_xyz.append(Q[m].cpu())
            res_k.append(b.indices[m].cpu())

    if len(res_xyz) > 0:
        Vxyz = torch.cat(res_xyz).numpy()
        Vk = torch.cat(res_k).numpy()
        print(f"    Tong so voxel vat the: {len(Vxyz):,}")

        # Tinh toan Surface (chi giu voxel be mat ngoai) & Direct View Visibility
        from scipy.spatial import cKDTree
        tree = cKDTree(Vxyz)

        # Lay mau cac vat quan trong de xuat file chi tiet
        inspect_targets = ["coc_su", "ly_tra", "tao", "dia"]
        all_surf_xyz = []
        all_surf_rgb = []

        for target_name in inspect_targets:
            k_idx = names.index(target_name)
            obj_m = Vk == k_idx
            if not obj_m.any():
                continue
            P_obj = Vxyz[obj_m]

            # Loc be mat: voxel nao co it hon 26 hang xom kề sát (distance < 1.8 * vs) thi la voxel be mat
            obj_tree = cKDTree(P_obj)
            counts = obj_tree.query_ball_point(P_obj, r=1.8 * vs, return_sorted=False)
            is_surface = np.array([len(c) < 26 for c in counts])
            P_surf = P_obj[is_surface]

            # Tinh toan so goc nhin toi duoc (Direct Visibility Count)
            P_surf_t = torch.tensor(P_surf, dtype=torch.float32, device=dev)
            direct_views = torch.zeros(len(P_surf_t), dtype=torch.int16, device=dev)

            # Tinh goc giua vector toi camera va huong len (+z cua vat)
            # Cac camera tren cao chieu xuong: goc nhin toi duoc mat tren va ben hong
            # Mat day (z thap nhat, tiep xuc ban) khong co camera nao nhin truc dien
            z_min = P_surf[:, 2].min()
            z_range = P_surf[:, 2].max() - z_min
            is_bottom_zone = (P_surf[:, 2] - z_min) < 0.05 * z_range  # 5% duoi cung sat mat ban

            for R, t, fx, fy, cx, cy, g, cam_pos in views:
                X = P_surf_t @ R.T + t
                z = X[:, 2]
                u = (fx * X[:, 0] / z + cx).long()
                v = (fy * X[:, 1] / z + cy).long()
                H, W = g.shape
                ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                inside_mask = g[v[ok], u[ok]] == k_idx
                valid_idx = ok.nonzero(as_tuple=True)[0][inside_mask]
                direct_views[valid_idx] += 1

            d_views = direct_views.cpu().numpy()

            # Quy tac mau Trang - Den (Fill-in style):
            # - Day vat (khong co camera / sat mat ban): Den tuyen (25, 25, 25)
            # - Vung quan sat tot (views >= 10): Trang tinh (250, 250, 250)
            # - Vung quan sat trung binh (3 <= views < 10): Xam sang (170, 170, 170)
            # - Vung quan sat it (1 <= views < 3): Xam toi (80, 80, 80)
            rgb_surf = np.zeros((len(P_surf), 3), dtype=np.uint8)
            for i in range(len(P_surf)):
                if is_bottom_zone[i] or d_views[i] == 0:
                    rgb_surf[i] = [25, 25, 25]  # Den tuyen: Chua toi / Day vat
                elif d_views[i] >= 10:
                    rgb_surf[i] = [250, 250, 250]  # Trang tinh: Toi rat tot
                elif d_views[i] >= 3:
                    rgb_surf[i] = [170, 170, 170]  # Xam sang: Toi trung binh
                else:
                    rgb_surf[i] = [80, 80, 80]  # Xam toi: Rìa khuat

            # Xuat file rieng tung vat
            p1_obj_path = out / f"method1_{target_name}_bw.ply"
            write_ply(p1_obj_path, P_surf, rgb_surf)
            print(f"    -> Da xuat: {p1_obj_path.name} ({len(P_surf):,} surface voxels)")

            all_surf_xyz.append(P_surf)
            all_surf_rgb.append(rgb_surf)

        # Xuat file gop toan bo cac vat tren ban
        if len(all_surf_xyz) > 0:
            full_xyz = np.concatenate(all_surf_xyz, axis=0)
            full_rgb = np.concatenate(all_surf_rgb, axis=0)
            p1_all_path = out / "method1_all_objects_bw.ply"
            write_ply(p1_all_path, full_xyz, full_rgb)
            print(f"    -> Da xuat file tong hop: {p1_all_path.name} ({len(full_xyz):,} voxels)")

    # -------------------------------------------------------------------------
    # PHUONG PHAP 3: Multi-View Ray-Cast Surface (Grayscale Texture + Day Den)
    # -------------------------------------------------------------------------
    print("\n[+] DANG XU LY PHUONG PHAP 3: Multi-View Ray-Cast (Grayscale Texture + Black Bottom)...")
    # Chon 24 camera deu quanh ban
    stride = max(1, len(view_cams) // 24)
    selected_cams = view_cams[::stride][:24]

    pts_p3_list = []
    rgb_p3_list = []

    # Dung kdtree cua toan bo Vxyz o tren de tim giao diem tia
    if len(res_xyz) > 0:
        hull_tree = cKDTree(Vxyz)

        for R, t, fx, fy, cx, cy, img_name, cam_pos in selected_cams:
            img_path = images_dir / img_name
            if not img_path.exists():
                continue
            # Doc anh that, chuyen sang grayscale
            img_raw = Image.open(img_path).convert("RGB")
            W_img, H_img = img_raw.size
            img_np = np.array(img_raw)
            # Thang do xam: 0.299R + 0.587G + 0.114B
            gray_img = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]).astype(np.uint8)

            # Doc mask
            lab_path = labels_dir / (Path(img_name).stem + ".png")
            if not lab_path.exists():
                continue
            lab_np = np.array(Image.open(lab_path))
            if lab_np.shape != (H_img, W_img):
                lab_np = np.array(Image.open(lab_path).resize((W_img, H_img), Image.NEAREST))

            # Lay cac pixel thuoc cac vat tren ban (bo gau/cuu)
            target_ids = []
            for tn in ["coc_su", "ly_tra", "tao", "dia"]:
                target_ids.extend(OBJECTS_SAM3[tn][0])
            mask_targets = np.isin(lab_np, target_ids)

            # Lay mau cac pixel (stride 4 de nhe va dep)
            ys, xs_p = np.where(mask_targets)
            if len(ys) == 0:
                continue
            sub = np.arange(0, len(ys), 6)
            ys = ys[sub]
            xs_p = xs_p[sub]

            # Ban tia tu camera
            # Pixel -> normalized ray
            x_norm = (xs_p - cx) / fx
            y_norm = (ys - cy) / fy
            rays_cam = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
            rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=-1, keepdims=True)

            R_np = R.cpu().numpy()
            t_np = t.cpu().numpy()
            cam_pos_np = cam_pos.cpu().numpy()
            rays_world = rays_cam @ R_np  # (N, 3)

            # Chieu tia tim giao diem voi Visual Hull
            # Lay cac buoc ray-marching nhe
            dists = np.linspace(0.4, 1.8, 40)  # Pham vi khoang cach tu camera den ban (~0.6m - 1.5m)
            # Sample points
            ray_pts = cam_pos_np[None, None, :] + rays_world[:, None, :] * dists[None, :, None]
            ray_pts_flat = ray_pts.reshape(-1, 3)

            d_near, _ = hull_tree.query(ray_pts_flat, distance_upper_bound=vs * 1.5)
            hit = d_near < vs * 1.5
            hit_reshaped = hit.reshape(len(rays_world), len(dists))

            # Tim first hit
            first_hit = np.argmax(hit_reshaped, axis=1)
            valid_hit = np.any(hit_reshaped, axis=1)

            if valid_hit.sum() > 0:
                hit_idx = valid_hit.nonzero()[0]
                final_d = dists[first_hit[hit_idx]]
                final_pts = cam_pos_np[None, :] + rays_world[hit_idx] * final_d[:, None]
                # Mau grayscale that tu anh chup
                g_val = gray_img[ys[hit_idx], xs_p[hit_idx]]
                # Tang tuong phan mot chut de nhin ro net
                g_val = np.clip((g_val.astype(np.float32) - 30) * 1.2 + 40, 40, 255).astype(np.uint8)
                pts_rgb = np.stack([g_val, g_val, g_val], axis=-1)

                pts_p3_list.append(final_pts)
                rgb_p3_list.append(pts_rgb)

        if len(pts_p3_list) > 0:
            P3_xyz = np.concatenate(pts_p3_list, axis=0)
            P3_rgb = np.concatenate(rgb_p3_list, axis=0)
            p3_path = out / "method3_raycast_texture_bw.ply"
            write_ply(p3_path, P3_xyz, P3_rgb)
            print(f"    -> Da xuat: {p3_path.name} ({len(P3_xyz):,} diem grayscale texture)")

    # -------------------------------------------------------------------------
    # TAO FILE README HUONG DAN TRONG FOLDER
    # -------------------------------------------------------------------------
    readme_content = f"""# Ket qua Quan sat 3D khong can train (Zero-Training Inspection)

Folder nay chua cac file `.ply` de xoay 360 do quan sat vat the theo phong cach **Trang - Den (Monochrome / Fill-in style)**:
- **Mau Trang / Xam sang**: Vung goc nhin toi duoc day du (duoc dien mau, du lieu ro rang).
- **Mau Den / Xam toi**: Vung thieu goc nhin (goc nhin chua toi, day vat sat mat ban, mat sau che khuat).

## Cac file ket qua:

1. **Phuong phap 1: Visual Hull Coverage (Do phu goc nhin truc tiep)**
   - `method1_coc_su_bw.ply`: Chi tiet coc su. Xoay lat day se thay mat day tiep xuc ban den tuyen (thieu goc nhin), than coc trang sang.
   - `method1_ly_tra_bw.ply`: Chi tiet ly tra.
   - `method1_tao_bw.ply`: Chi tiet qua tao.
   - `method1_dia_bw.ply`: Chi tiet dia.
   - `method1_all_objects_bw.ply`: Toan bo cac vat the tren mat ban cung luc.

2. **Phuong phap 2: COLMAP SfM Points (`method2_sfm_points_bw.ply`)**
   - Cac diem 3D SfM loc theo mask SAM 3.
   - Diem trang: Duoc quan sat boi nhieu camera (>15 anh).
   - Diem xam: Quan sat boi it camera.
   - Nen mat ban xung quanh duoc de mau xam rat toi lam boi canh.

3. **Phuong phap 3: Multi-View Ray-Cast (`method3_raycast_texture_bw.ply`)**
   - Texture thang do xam (Grayscale) that chieu truc tiep tu cac camera.
   - Khong dung bat ky mo hinh neural nao, ban tia goc nhin truc tiep.

## Cach mo xem:
- Keo tha file `.ply` vao **MeshLab**, **CloudCompare**, **Blender**, hoac extension **3D Viewer / PLY Viewer** tren VS Code de xoay 360 do.
"""
    (out / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"\n[*] HOAN TAT! Toan bo ket qua da duoc xuat vao: {out}")


if __name__ == "__main__":
    main()
