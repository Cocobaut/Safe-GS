"""
S3: gán từng Gaussian cho vật / mặt đỡ / nền + đồng bộ ID trong 3D + tìm mặt bàn, sàn.

Vote theo ĐÓNG GÓP RENDER (ý tưởng Gradient-Driven Seg. / FlashSplat): render với màu giả c (N x 3, requires_grad),
loss = sum(ảnh_render * mask), thì dL/dc_i = tổng (alpha_i * T_i) của Gaussian i trên vùng mask. Một lần backward cho 3
kênh mask. W[i, g] = tổng qua các view; W_all = trọng số trên toàn ảnh; tỉ lệ W[i, g] / W_all quyết định nhóm.

Bước:
  1. (merge_graph) vote cho từng ID SAM2 thô trên 1/view_stride số view -> đồ thị đồng xuất hiện: 2 ID cùng có tỉ lệ
     >= share trên nhiều Gaussian -> đề xuất gộp; gộp vào nhóm vật nếu overlap >= overlap (không gộp vào mặt đỡ).
  2. vote cho các nhóm (vật + mặt đỡ) trên mọi view -> nhãn.
  3. (interior_knn) Gaussian không đóng góp gì (W_all ~ 0) gần vật -> nhãn theo đa số k láng giềng.
  4. mặt bàn = RANSAC trên Gaussian mặt đỡ; sàn = RANSAC trên Gaussian nền nằm dưới, song song mặt bàn; scale m/đơn vị.
Checkpoint: W cộng dồn được lưu mỗi 20 view (partial_*.pt), chạy lại thì tiếp.
Xuất (--out): labels.npy, ratios.npy, groups.json, merge_report.json, support.json, gaussians/<nhóm>.ply,
              gaussians/background.ply, preview/<frame>.jpg [toàn cảnh | nền + mặt đỡ | vật tô màu theo nhóm], stats.json
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (log, make_parser, load_scene, view_name, freeze, read_ply, write_ply, gaussians_from_ply,
                    render_rgbd, to_u8, load_cfg, ransac_plane)
from gaussian_renderer import render

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40]],
                   np.float32) / 255.0


def accumulate(S, g, views, mask_fn, n_ch, partial, every=20):
    """W[N, n_ch]: trọng số đóng góp render của mỗi Gaussian vào từng kênh mask, cộng qua các view."""
    N = g.get_xyz.shape[0]
    W, start = torch.zeros(N, n_ch, device="cuda"), 0
    if partial.exists():
        ck = torch.load(partial)
        if ck["W"].shape == (N, n_ch):
            W, start = ck["W"].cuda(), ck["next"]
            log(f"♻️ tiếp tục vote từ view {start}/{len(views)} ({partial.name})")
    for k in range(start, len(views)):
        view = views[k]
        M = mask_fn(view)
        if M is not None:
            for c0 in range(0, n_ch, 3):
                m = M[c0:c0 + 3]
                nc = m.shape[0]
                if nc < 3:
                    m = torch.cat([m, torch.zeros(3 - nc, *m.shape[1:], device="cuda")], 0)
                col = torch.zeros(N, 3, device="cuda", requires_grad=True)
                out = render(view, g, S["pipe"], S["bg"], splat_args=S["splat_args"], override_color=col)["render"][:3]
                (out * m).sum().backward()
                W[:, c0:c0 + nc] += col.grad[:, :nc]
        if (k + 1) % every == 0 or k + 1 == len(views):
            torch.save({"W": W.cpu(), "next": k + 1}, partial)
            log(f"vote {partial.stem}: {k + 1}/{len(views)}")
    return W


def main():
    p, model, pipe, ss, _, _ = make_parser("S3 label")
    p.add_argument("--labels", type=str, required=True)
    from arguments import get_combined_args
    args = get_combined_args(p)
    full_cfg = load_cfg(args.cfg)
    cfg = full_cfg["S3_label"]
    out = Path(args.out); (out / "preview").mkdir(parents=True, exist_ok=True); (out / "gaussians").mkdir(exist_ok=True)
    S = load_scene(args, model, pipe, ss)
    g = S["g"]; freeze(g)
    N = g.get_xyz.shape[0]
    labels_dir = Path(args.labels)
    lab_cache = {}

    def lab_of(view):
        nm = view_name(view)
        if nm not in lab_cache:
            pth = labels_dir / f"{nm}.png"
            lab_cache[nm] = torch.from_numpy(cv2.imread(str(pth), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda() \
                if pth.exists() else None
        return lab_cache[nm]

    objects = {k: list(v) for k, v in cfg["objects"].items()}
    support = {k: list(v) for k, v in cfg["support"].items()}

    # ---------------------------------------------------------------- 1. merge graph trên ID thô
    merge_report = {}
    if cfg["merge_graph"]["enabled"]:
        mg = cfg["merge_graph"]
        info = json.loads((labels_dir.parent / "objects.json").read_text())
        raw_ids = sorted(int(k) for k, v in info.items() if v["frames_visible"] >= mg["min_frames_visible"])
        sub_views = S["views"][::mg["view_stride"]]
        ids_t = torch.tensor(raw_ids, device="cuda")

        def raw_masks(view):
            lab = lab_of(view)
            if lab is None:
                return None
            return torch.cat([(lab[None] == ids_t[:, None, None]).float(), torch.ones(1, *lab.shape, device="cuda")], 0)

        Wr = accumulate(S, g, sub_views, raw_masks, len(raw_ids) + 1, out / "partial_raw.pt")
        tot = Wr[:, -1:].clamp(min=1e-12)
        R = (Wr[:, :-1] / tot).cpu().numpy()
        has = (Wr[:, -1] > cfg["min_total_weight"]).cpu().numpy()
        A = (R[has] >= mg["share"]).astype(np.float32)                 # Gaussian x ID: ID chiếm >= share
        cnt = A.sum(0)
        co = A.T @ A                                                   # số Gaussian dùng chung
        pairs = []
        for a in range(len(raw_ids)):
            for b in range(a + 1, len(raw_ids)):
                if co[a, b] > 0:
                    ov = co[a, b] / max(min(cnt[a], cnt[b]), 1)
                    if ov >= mg["overlap"]:
                        pairs.append((raw_ids[a], raw_ids[b], float(ov), int(co[a, b])))
        pairs.sort(key=lambda x: -x[2])
        merge_report = {"raw_ids": raw_ids, "gaussians_per_id": {int(i): int(c) for i, c in zip(raw_ids, cnt)},
                        "pairs_overlap>=thr": [{"a": a, "b": b, "overlap": o, "shared": s} for a, b, o, s in pairs]}
        # gộp vào nhóm vật (union-find đơn giản), không kéo ID của mặt đỡ / vật khác vào
        if mg["auto_merge_objects"]:
            owner = {i: n for n, ids in objects.items() for i in ids}
            sup_ids = {i for ids in support.values() for i in ids}
            added = []
            changed = True
            while changed:
                changed = False
                for a, b, o, s in pairs:
                    for x, y in ((a, b), (b, a)):
                        if x in owner and y not in owner and y not in sup_ids:
                            owner[y] = owner[x]; objects[owner[x]].append(y); added.append((y, owner[x], o))
                            changed = True
            merge_report["auto_added"] = [{"id": i, "to": n, "overlap": o} for i, n, o in added]
        (out / "merge_report.json").write_text(json.dumps(merge_report, indent=2))
        log(f"merge graph: {len(pairs)} cặp gợi ý gộp, thêm vào vật: {merge_report.get('auto_added', [])}")

    # ---------------------------------------------------------------- 2. vote theo nhóm
    groups = list(objects.items()) + list(support.items())
    names = [n for n, _ in groups]
    gid_t = [torch.tensor(ids, device="cuda") for _, ids in groups]

    def group_masks(view):
        lab = lab_of(view)
        if lab is None:
            return None
        ms = [torch.isin(lab, t).float() for t in gid_t]
        if present_norm:      # them K kenh "vat k co nhan trong view nay" (>= 20 px): mau so rieng cho tung vat
            ms += [torch.ones_like(m) if float(m.sum()) >= 20 else torch.zeros_like(m) for m in ms]
        return torch.stack(ms + [torch.ones_like(lab, dtype=torch.float32)], 0)

    # opt-in S3_label.vote_norm = "present": ti le vat k = dong gop trong mask k / dong gop o cac view CO nhan vat k (thay vi moi view),
    # va can cac view do chiem >= 20 % dong gop cua Gaussian. Prompt chung (vd 'yellow rubber duck') bat nham vat cu the trong mot so
    # frame khong con thang phieu chi vi xuat hien o nhieu frame hon (cung y tuong tools/mask_hull.py --assign_mode ratio).
    # "present_gated": nhu present nhung chi xet vat k co W_k / tong >= present_min_share (mac dinh 0.2): tranh Gaussian nen sat vat
    # nho (it view co nhan) duoc gan vao vat do chi nho mask lem o vai view.
    # "fg_present": chi Gaussian co tong dong gop vao tap vat tien canh (0..n_obj-1) >= assign_ratio moi duoc dung ty le present;
    # tranh vat nho/nhieu cuop Gaussian nen, dong thoi giu uu the cua present cho vat bi vat khac chong lan (vit mu do, vat long nhau).
    vote_norm = cfg.get("vote_norm", "all")
    present_norm = vote_norm in ("present", "present_gated", "fg_present", "all_then_fg_present", "fg_present_hull")
    n_obj = len(objects)
    views = S["views"][::cfg["view_stride"]]
    K_ = len(groups)
    Wg = accumulate(S, g, views, group_masks, (2 * K_ if present_norm else K_) + 1,
                    out / ("partial_groups_present.pt" if present_norm else "partial_groups.pt"))
    tot = Wg[:, -1]
    if present_norm:
        Wp = Wg[:, K_:2 * K_]
        if vote_norm in ("fg_present", "all_then_fg_present", "fg_present_hull"):
            fg_share = Wg[:, :n_obj].sum(1) / tot.clamp(min=1e-12)
            obj_share = Wg[:, :n_obj] / tot[:, None].clamp(min=1e-12)
            min_obj = float(cfg.get("fg_min_obj_share", 0.05))
            ratio_fg = torch.where(
                (fg_share[:, None] >= float(cfg["assign_ratio"]))
                & (Wp[:, :n_obj] >= 0.2 * tot[:, None])
                & (obj_share >= min_obj),
                Wg[:, :n_obj] / Wp[:, :n_obj].clamp(min=1e-12),
                torch.zeros_like(Wp[:, :n_obj]),
            )
            if vote_norm == "fg_present_hull":
                from scipy.spatial import cKDTree
                xyz_np = g.get_xyz.detach().cpu().numpy()
                hull_rel = full_cfg.get("S1_masks", {}).get("hull_rel", "mask_hull_sam3_adap")
                hull_dir = out.parents[2] / hull_rel
                gate_vox = float(cfg.get("hull_gate_vox", 2.5))
                in_hull_t = torch.zeros((len(xyz_np), n_obj), dtype=torch.bool, device="cuda")
                out_hull_t = torch.zeros((len(xyz_np), n_obj), dtype=torch.bool, device="cuda")
                for k_idx, n_k in enumerate(names[:n_obj]):
                    hpth = hull_dir / f"hull_{n_k}.ply"
                    if not hpth.exists():
                        in_hull_t[:, k_idx] = True
                        continue
                    v_h = read_ply(hpth)
                    if len(v_h) < 4:
                        in_hull_t[:, k_idx] = True
                        continue
                    hpts = np.stack([np.asarray(v_h["x"], np.float32),
                                     np.asarray(v_h["y"], np.float32),
                                     np.asarray(v_h["z"], np.float32)], 1)
                    ux = np.unique(np.round(hpts[:, 0], 5))
                    dx = np.diff(ux)
                    vs = float(np.median(dx[dx > 1e-5])) if (dx > 1e-5).any() else 0.015
                    if len(hpts) > 250000:
                        hpts = hpts[::max(1, len(hpts) // 250000)]
                    d_k, _ = cKDTree(hpts).query(xyz_np, k=1, workers=-1)
                    d_vox = torch.from_numpy((d_k / max(vs, 1e-5)).astype(np.float32)).cuda()
                    in_hull_t[:, k_idx] = d_vox <= gate_vox
                    out_hull_t[:, k_idx] = d_vox > gate_vox
                # Only allow fg_present boost when inside/near object k's 3D hull; otherwise fall back to ratio_all
                ratio_all_fg = Wg[:, :n_obj] / tot[:, None].clamp(min=1e-12)
                ratio_fg = torch.where(in_hull_t, ratio_fg, ratio_all_fg)
                # Cross-object hull disambiguation: if Gaussian is outside winner k's hull but inside another candidate j's hull
                any_in_other = (in_hull_t & (ratio_fg >= 0.20)).any(1, keepdim=True)
                ratio_fg = torch.where(out_hull_t & any_in_other, ratio_fg * 0.25, ratio_fg)
                log(f"fg_present_hull: gate_vox={gate_vox}, hull_dir={hull_dir}")
            if K_ > n_obj:
                ratio_sup = Wg[:, n_obj:K_] / tot[:, None].clamp(min=1e-12)
                ratio_t = torch.cat([ratio_fg, ratio_sup], 1)
            else:
                ratio_t = ratio_fg
            if vote_norm == "all_then_fg_present":
                ratio_all = Wg[:, :K_] / tot[:, None].clamp(min=1e-12)
                has_all = ratio_all.max(1, keepdim=True).values >= float(cfg["assign_ratio"])
                ratio_t = torch.where(has_all, ratio_all, ratio_t)
        else:
            ratio_t = torch.where(Wp >= 0.2 * tot[:, None], Wg[:, :K_] / Wp.clamp(min=1e-12), torch.zeros_like(Wp))
            if vote_norm == "present_gated":
                share = Wg[:, :K_] / tot[:, None].clamp(min=1e-12)
                ratio_t = torch.where(share >= float(cfg.get("present_min_share", 0.2)), ratio_t, torch.zeros_like(ratio_t))
        ratio = ratio_t.cpu().numpy()
        Wg = torch.cat([Wg[:, :K_], Wg[:, -1:]], 1)
    else:
        ratio = (Wg[:, :-1] / tot[:, None].clamp(min=1e-12)).cpu().numpy()
    tot_np = tot.cpu().numpy()
    best = ratio.argmax(1)
    lab = np.where((ratio.max(1) >= cfg["assign_ratio"]) & (tot_np > cfg["min_total_weight"]), best, -1).astype(np.int16)
    log("vote xong: " + ", ".join(f"{n}={int((lab == i).sum())}" for i, n in enumerate(names)) +
        f", nền={int((lab == -1).sum())}, không đóng góp={int((tot_np <= cfg['min_total_weight']).sum())}")

    # ---------------------------------------------------------------- 3. interior kNN
    xyz = g.get_xyz.detach().cpu().numpy()
    n_obj = len(objects)
    knn_added = 0
    if cfg["interior_knn"]["enabled"]:
        from scipy.spatial import cKDTree
        contrib = tot_np > cfg["min_total_weight"]
        smax = g.get_scaling.max(1).values.detach().cpu().numpy()
        objm = (lab >= 0) & (lab < n_obj)
        r = cfg["interior_knn"]["radius_scale"] * float(np.median(smax[objm])) if objm.any() else 0.0
        cand = np.where(~contrib)[0]
        if r > 0 and len(cand):
            tree = cKDTree(xyz[contrib])
            lab_c = lab[contrib]
            dist, idx = tree.query(xyz[cand], k=cfg["interior_knn"]["k"], distance_upper_bound=r)
            valid = np.isfinite(dist)
            nb = np.where(valid, lab_c[np.minimum(idx, len(lab_c) - 1)], -2)
            for j, c in enumerate(cand):
                v = nb[j][nb[j] >= 0]
                v = v[v < n_obj]
                if len(v) and np.bincount(v).max() > cfg["interior_knn"]["k"] // 2:
                    lab[c] = np.bincount(v).argmax(); knn_added += 1
        log(f"interior kNN: bán kính {r:.4f}, gán thêm {knn_added}")

    np.save(out / "labels.npy", lab)
    np.save(out / "ratios.npy", ratio.astype(np.float16))

    # ---------------------------------------------------------------- 4. mặt bàn, sàn, scale
    op = g.get_opacity_with_3D_filter.squeeze(-1).detach().cpu().numpy()
    sup_idx = [i for i, n in enumerate(names) if n in support]
    supm = np.isin(lab, sup_idx) & (op >= 0.5)
    support_info = {}
    if supm.sum() > 100:
        n_t, d_t, inl = ransac_plane(xyz[supm], thresh=float(np.median(g.get_scaling.max(1).values.cpu().numpy())) * 2)
        objp = xyz[(lab >= 0) & (lab < n_obj)]
        if len(objp) and np.median(objp @ n_t + d_t) < 0:          # pháp tuyến hướng lên phía vật
            n_t, d_t = -n_t, -d_t
        support_info["table"] = {"normal": n_t.tolist(), "d": float(d_t), "inliers": int(inl.sum())}
        bgm = (lab == -1) & (op >= 0.5)
        sd = xyz @ n_t + d_t
        below = bgm & (sd < -0.05 * np.ptp(sd[supm]) - 1e-3)
        if below.sum() > 1000:
            n_f, d_f, inl_f = ransac_plane(xyz[below], thresh=float(np.median(g.get_scaling.max(1).values.cpu().numpy())) * 3,
                                           normal_hint=n_t, min_cos=0.97)
            if n_f @ n_t < 0:
                n_f, d_f = -n_f, -d_f
            # khoảng cách 2 mặt (song song gần đúng) = d_f - d_t theo pháp tuyến bàn
            p_floor = xyz[below][inl_f].mean(0)
            height = float(-(p_floor @ n_t + d_t))
            support_info["floor"] = {"normal": n_f.tolist(), "d": float(d_f), "inliers": int(inl_f.sum()),
                                     "table_to_floor_units": height}
            h_m = load_cfg(args.cfg)["S6_mesh"]["table_height_m"]
            support_info["meters_per_unit"] = h_m / height if height > 0 else None
            support_info["scale_assumption"] = f"bàn cao {h_m} m"
        # opt-in hieu chinh ty le (S6_mesh): "meters_per_unit" (so cho san) > "scale_reference" {"object", "size_m"} (vat co kich
        # thuoc biet truoc: chieu cao p98 - p2 theo phap tuyen ban cua Gaussian vat, opacity >= 0.5) > khoang cach ban-san.
        # Gia tri theo san van luu o "meters_per_unit_floor" de doi chieu.
        m6 = load_cfg(args.cfg).get("S6_mesh", {})
        ref = m6.get("scale_reference")
        if m6.get("meters_per_unit") or (ref and ref.get("object") in names):
            support_info["meters_per_unit_floor"] = support_info.get("meters_per_unit")
            if m6.get("meters_per_unit"):
                support_info["meters_per_unit"] = float(m6["meters_per_unit"])
                support_info["scale_assumption"] = "S6_mesh.meters_per_unit (cho san)"
            else:
                om = (lab == names.index(ref["object"])) & (op >= 0.5)
                if om.sum() > 50:     # bo Gaussian lac (floater) gan nhan vat: giu diem cach trung vi < 3 x trung vi khoang cach
                    pts = xyz[om]
                    dc = np.linalg.norm(pts - np.median(pts, 0), axis=1)
                    om_idx = np.nonzero(om)[0]
                    om = np.zeros_like(om); om[om_idx[dc < 3.0 * np.median(dc)]] = True
                hs = xyz[om] @ n_t + d_t
                if ref.get("stat", "height") == "length":     # vat dai nam tren ban: p98 - p2 theo truc chinh trong mat ban
                    q = xyz[om] - np.outer(hs, n_t)
                    q = q - q.mean(0)
                    ax = np.linalg.svd(q, full_matrices=False)[2][0] if om.sum() > 50 else n_t
                    pr = q @ ax
                    h_u = float(np.percentile(pr, 98) - np.percentile(pr, 2)) if om.sum() > 50 else 0.0
                else:
                    h_u = float(np.percentile(hs, 98) - max(np.percentile(hs, 2), 0.0)) if om.sum() > 50 else 0.0
                if h_u > 0:
                    support_info["meters_per_unit"] = float(ref["size_m"]) / h_u
                    support_info["scale_assumption"] = f"{ref['object']} cao {ref['size_m']} m ({h_u:.4f} don vi, {int(om.sum())} Gaussian)"
            f_ = support_info.get("meters_per_unit_floor")
            if f_:
                support_info["scale_ratio_floor_over_used"] = f_ / support_info["meters_per_unit"]
    (out / "support.json").write_text(json.dumps(support_info, indent=2))
    log(f"mặt đỡ: {json.dumps(support_info)}")

    # ---------------------------------------------------------------- xuất ply theo nhóm + preview
    full = out / "_full.ply"
    g.save_ply(str(full))
    el = read_ply(full); full.unlink()
    for i, n in enumerate(names):
        write_ply(out / "gaussians" / f"{n}.ply", el[lab == i])
    write_ply(out / "gaussians" / "background.ply", el[lab == -1])
    write_ply(out / "gaussians" / "background_with_support.ply", el[(lab == -1) | np.isin(lab, sup_idx)])
    write_ply(out / "gaussians" / "objects_all.ply", el[(lab >= 0) & (lab < n_obj)])

    cams = S["train"]
    g_bg = gaussians_from_ply(out / "gaussians" / "background_with_support.ply", S["dataset"].sh_degree, cams)
    g_ob = gaussians_from_ply(out / "gaussians" / "objects_all.ply", S["dataset"].sh_degree, cams)
    lab_ob = lab[(lab >= 0) & (lab < n_obj)]
    col_ob = torch.from_numpy(PALETTE[lab_ob % len(PALETTE)]).cuda()
    for view in S["views"]:
        nm = view_name(view)
        if nm not in cfg["preview_frames"]:
            continue
        a, _ = render_rgbd(view, g, S)
        b, _ = render_rgbd(view, g_bg, S)
        c, _ = render_rgbd(view, g_ob, S, override_color=col_ob)
        sep = torch.ones(3, a.shape[1], 8, device="cuda")
        img = to_u8(torch.cat([a, sep, b, sep, c], 2))[..., ::-1]
        cv2.imwrite(str(out / "preview" / f"{nm}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    groups_json = {"objects": {n: {"label": i, "ids": objects[n], "gaussians": int((lab == i).sum())}
                               for i, n in enumerate(objects)},
                   "support": {n: {"label": len(objects) + j, "ids": support[n],
                                   "gaussians": int((lab == len(objects) + j).sum())} for j, n in enumerate(support)},
                   "background_label": -1, "palette_rgb": PALETTE.tolist()}
    (out / "groups.json").write_text(json.dumps(groups_json, indent=2, ensure_ascii=False))
    stats = {"gaussians": int(N), "background": int((lab == -1).sum()), "interior_knn_added": knn_added,
             "no_contribution": int((tot_np <= cfg["min_total_weight"]).sum()), "views_voted": len(views)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    for f in out.glob("partial_*.pt"):
        f.unlink()
    log(f"S3 xong -> {out}")


if __name__ == "__main__":
    main()
