"""
Ghi ket qua trung gian cho buoc 2 (khung 3D) va buoc 3 (dong bo mask) cua Split.
Duoc goi tu ban sao co chen log cua sam2/mask_propagation_scanet.py qua _dbg.at(<tag>, locals()) va
_dbg.mask_saved(...). Chi DOC bien cua code goc (khong ghi, khong goi RNG toan cuc) -> khong doi ket qua.
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import id_color

CFG = {}
S = {
    "frame_masks": {},      # image_id -> [ten file mask]
    "decisions": {},        # (image_id, mask_idx) -> dict
    "decision_order": [],
    "label_growth": [],
    "prev_labels": set(),
    "depth_rows": [],
    "cluster_rows": {},
    "mask_source": {},      # (label, "0000.png") -> nguon
    "labeled_sizes": {},
    "pcd_written": False,
    "palette": {},
}


def configure(out_root, work, scene):
    CFG.update(out=out_root, work=work, scene=scene)
    for d in ["02_skeleton_3d/depth_check", "03_propagation/virtual_vs_sam", "03_propagation/votes",
              "03_propagation/clusters", "04_final/masks", "04_final/id_map"]:
        os.makedirs(os.path.join(out_root, d), exist_ok=True)


def P(*parts):
    return os.path.join(CFG["out"], *parts)


def pal(i):
    i = int(i)
    if i not in S["palette"]:
        S["palette"][i] = id_color(i)
    return S["palette"][i]


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def label_text(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([0, 0, 8 * len(text) + 10, 22], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((5, 5), text, fill=(255, 255, 255))
    return np.array(im)


def dilate_colors(color_img, mask, r=2):
    """Lam to cac diem le (nhan 3D chieu vao anh) de nhin thay; chi phuc vu hien thi."""
    import cv2
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    m = cv2.dilate(mask.astype(np.uint8), k) > 0
    out = cv2.dilate(np.ascontiguousarray(color_img), k)
    return out, m


# ============================================================ dispatcher
def at(tag, g):
    fn = globals().get("h_" + tag)
    if fn is None:
        raise KeyError(tag)
    fn(g)


# ============================================================ BUOC 2
def h_frame_visible(g):
    image_id = int(g["image_id"])
    frame = f"{image_id:04d}"
    S["frame_masks"][image_id] = list(g["masks"])
    pts = np.asarray(g["pcd_points"])
    if not S["pcd_written"]:
        write_ply(P("02_skeleton_3d", "point_cloud_100k.ply"), pts, np.full((len(pts), 3), 200, np.uint8))
        S["pcd_written"] = True

    img, depth = np.asarray(g["img"]), np.asarray(g["depth"])
    extr, intr, eps = np.asarray(g["extr"]), np.asarray(g["intr"]), float(g["EPSILON"])
    H, W = img.shape[:2]
    w2c = np.linalg.inv(extr)
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > 0
    u = np.rint(intr[0, 0] * pc[front, 0] / z[front] + intr[0, 2]).astype(np.int64)
    v = np.rint(intr[1, 1] * pc[front, 1] / z[front] + intr[1, 2]).astype(np.int64)
    zf = z[front]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zf = u[inb], v[inb], zf[inb]
    ok = np.abs(depth[v, u] - zf) <= eps   # cung dieu kien voi kernel pcd_2D (fabsf(d - z) > thresh -> loai)

    vis = (img.astype(np.float32) * 0.55).astype(np.uint8)
    for sel, col in [(~ok, (230, 40, 40)), (ok, (40, 230, 40))]:
        layer = np.zeros_like(vis)
        msk = np.zeros((H, W), bool)
        layer[v[sel], u[sel]] = col
        msk[v[sel], u[sel]] = True
        layer, msk = dilate_colors(layer, msk, r=1)
        vis[msk] = layer[msk]
    n_in, n_ok = int(len(u)), int(ok.sum())
    pct = 100.0 * n_ok / max(n_in, 1)
    vis = label_text(vis, f"frame {frame}: {n_ok}/{n_in} diem qua kiem tra depth ({pct:.1f}%)")
    Image.fromarray(vis).save(P("02_skeleton_3d", "depth_check", f"{frame}.png"))
    S["depth_rows"].append([frame, n_in, n_ok, f"{pct:.2f}", int(len(g["visible_ids"]))])


# ============================================================ BUOC 3
def _decision(image_id, idx, **kw):
    key = (int(image_id), int(idx))
    if key not in S["decisions"]:
        S["decision_order"].append(key)
        S["decisions"][key] = {"frame": f"{int(image_id):04d}", "mask": "", "truong_hop": "", "nhan": "",
                               "cac_nhan_chong": "", "so_diem_3d": None, "ket_qua": None}
    S["decisions"][key].update(kw)
    return S["decisions"][key]


def h_first_frame_mask(g):
    _decision(g["image_id"], g["i"], mask=g["m"], truong_hop="frame_dau", nhan=int(g["gt_labels_idx"]) - 1,
              so_diem_3d=int(len(g["ids"])), ket_qua="ap_dung")


def h_virtual_mask(g):
    image_id = int(g["image_id"])
    frame = f"{image_id:04d}"
    img = np.asarray(g["img"])
    labels_map = np.asarray(g["mask_labels"])
    H, W = labels_map.shape
    left = np.zeros((H, W, 3), np.uint8)
    msk = labels_map >= 0
    for lab in np.unique(labels_map[msk]):
        left[labels_map == lab] = pal(lab)
    left, msk = dilate_colors(left, msk, r=2)
    base = (img.astype(np.float32) * 0.35).astype(np.uint8)
    lv = base.copy()
    lv[msk] = left[msk]
    rv = base.copy().astype(np.float32)
    masks = [np.array(Image.open(os.path.join(g["mask_folder"], g["mask_name"], m)).convert("L")) > 0 for m in g["masks"]]
    for k in sorted(range(len(masks)), key=lambda k: -masks[k].sum()):
        rv[masks[k]] = rv[masks[k]] * 0.3 + np.array(id_color(1000 + k)) * 0.7
    lv = label_text(lv, f"frame {frame} | mask ao (nhan 3D chieu vao, mau = ID)")
    rv = label_text(rv.clip(0, 255).astype(np.uint8), f"frame {frame} | mask SAM cua frame (mau = id rieng frame)")
    Image.fromarray(np.concatenate([lv, rv], axis=1)).save(P("03_propagation", "virtual_vs_sam", f"{frame}.png"))


def h_mask_case(g):
    lab = [int(x) for x in np.asarray(g["label_in_mask"]).ravel()]
    if len(lab) == 0:
        case, label = "vat_moi", int(max(g["labelPoints"])) + 1
    elif len(lab) == 1:
        case, label = "khop_1_nhan", lab[0]
    else:
        vals = np.asarray(g["mask_labels"])[np.asarray(g["mask_erosion"])].ravel()
        vals = vals[vals != -1]
        case, label = "chong_nhieu_nhan", (int(np.bincount(vals).argmax()) if vals.size else "")
    _decision(g["image_id"], g["i"], mask=g["m"], truong_hop=case, nhan=label,
              cac_nhan_chong=";".join(map(str, lab)) if len(lab) >= 2 else "")


def h_dbscan_new(g):
    _decision(g["image_id"], g["i"], so_diem_3d=int(len(g["ids"])), ket_qua="ap_dung")


def h_dbscan_multi(g):
    _decision(g["image_id"], g["i"], so_diem_3d=int(len(g["ids"])), ket_qua="ap_dung")


def h_dbscan_step4(g):
    _decision(g["image_id"], g["mask_idx"], so_diem_3d=int(len(g["ids"])), ket_qua="ap_dung")


def h_frame_end(g):
    image_id = int(g["image_id"])
    masks = S["frame_masks"].get(image_id, [])
    assigned = {int(k): int(v) for k, v in dict(g.get("assigned_labels", {}) or {}).items()} if int(g["iter"]) > 0 else {}
    for idx, m in enumerate(masks):
        key = (image_id, idx)
        if key not in S["decisions"]:
            _decision(image_id, idx, mask=m, truong_hop="frame_dau" if int(g["iter"]) == 0 else "",
                      ket_qua="bo_qua (khong co diem 3D)")
            continue
        d = S["decisions"][key]
        if d["ket_qua"] is None:
            if d["truong_hop"] == "khop_1_nhan" and assigned.get(int(d["nhan"])) != idx:
                d["ket_qua"] = "bi_de (mask khac cung nhan duoc chon)"
            else:
                d["ket_qua"] = "bo_qua (khong co diem 3D)"
    keys = set(int(k) for k in g["labelPoints"].keys())
    S["label_growth"].append([f"{image_id:04d}", len(keys), len(keys - S["prev_labels"]), len(S["prev_labels"] - keys)])
    S["prev_labels"] = keys
    if len(S["label_growth"]) % 10 == 0:
        flush_tables()


def h_votes_final(g):
    pts = np.asarray(g["pcd"].points)
    votes, final = g["pointsLabels"], g["pointsLabels_def"]
    ids, conf, best = [], [], []
    for pid, labs in votes.items():
        if not labs:
            continue
        w = np.array(list(labs.values()), np.float64)
        k = int(np.argmax(w))
        ids.append(int(pid))
        conf.append(float(w[k] / w.sum()))
        best.append(int(list(labs.keys())[k]))
    ids, conf = np.array(ids, np.int64), np.array(conf)
    if len(ids):
        c = np.clip(conf, 0, 1)[:, None]
        rgb = (np.array([230, 40, 40]) * (1 - c) + np.array([40, 230, 40]) * c).astype(np.uint8)
        write_ply(P("03_propagation", "votes", "confidence.ply"), pts[ids], rgb)
        rej = np.array([p not in final for p in ids])
        write_ply(P("03_propagation", "votes", "rejected.ply"), pts[ids[rej]],
                  np.tile([230, 40, 40], (int(rej.sum()), 1)))
    fid = np.array(list(final.keys()), np.int64)
    write_ply(P("03_propagation", "votes", "labels_final.ply"), pts[fid] if len(fid) else np.zeros((0, 3)),
              np.array([pal(final[p]) for p in fid], np.uint8).reshape(-1, 3))
    S["votes_summary"] = {"diem_co_phieu": int(len(ids)), "diem_chot_nhan": int(len(fid)),
                          "diem_bi_loai_duoi_nguong": int(len(ids) - len(fid))}


def h_cluster_label(g):
    label = int(g["label"])
    counts = np.asarray(g["counts"])
    frag = len(g["unique"]) > 3
    S["cluster_rows"][label] = [label, len(g["p3D_ids"]), len(g["valid_points"]), len(g["unique"]),
                                int(counts.max()), "co" if frag else "khong"]


def h_clusters(g):
    for label, pts in dict(g["labeled_clusters"]).items():
        S["labeled_sizes"][int(label)] = len(pts)


def mask_saved(source, label, save_name):
    S["mask_source"][(int(label), str(save_name))] = source


# ============================================================ ghi bang + buoc cuoi
def flush_tables():
    order = S["decision_order"]
    write_csv(P("03_propagation", "decisions.csv"),
              ["frame", "mask", "truong_hop", "nhan", "cac_nhan_chong", "so_diem_3d", "ket_qua"],
              [[S["decisions"][k][c] if S["decisions"][k][c] is not None else "" for c in
                ["frame", "mask", "truong_hop", "nhan", "cac_nhan_chong", "so_diem_3d", "ket_qua"]] for k in order])
    write_csv(P("03_propagation", "label_growth.csv"), ["frame", "tong_so_nhan", "nhan_moi", "nhan_bi_xoa"],
              S["label_growth"])
    write_csv(P("02_skeleton_3d", "depth_check.csv"),
              ["frame", "so_diem_trong_khung", "so_diem_qua_depth", "ti_le_qua_pct", "so_diem_zbuffer_thay"],
              S["depth_rows"])


def finalize():
    import cv2
    flush_tables()
    scene, work = CFG["scene"], CFG["work"]
    src_masks = os.path.join(work, "output", f"{scene}_masks")

    rows = []
    for label, r in sorted(S["cluster_rows"].items()):
        kept = label in S["labeled_sizes"]
        rows.append(r + ["co" if kept else "khong", S["labeled_sizes"].get(label, 0)])
    write_csv(P("03_propagation", "clusters", "clusters.csv"),
              ["nhan", "so_diem_truoc_dbscan", "so_diem_hop_le", "so_cum", "cum_lon_nhat", "phan_manh_gt3_cum",
               "duoc_giu", "so_diem_giu"], rows)
    write_csv(P("03_propagation", "mask_source.csv"), ["nhan", "frame", "nguon"],
              [[l, f.replace(".png", ""), s] for (l, f), s in sorted(S["mask_source"].items())])

    # copy output nguyen goc cua tac gia
    per_frame = {}
    if os.path.isdir(src_masks):
        for lab in sorted(os.listdir(src_masks), key=lambda x: int(x) if x.isdigit() else 1e9):
            d = os.path.join(src_masks, lab)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".ply"):
                    shutil.copy2(os.path.join(d, f), P("03_propagation", "clusters", f))
                elif f.endswith(".png"):
                    os.makedirs(P("04_final", "masks", lab), exist_ok=True)
                    shutil.copy2(os.path.join(d, f), P("04_final", "masks", lab, f))
                    per_frame.setdefault(f.replace(".png", ""), []).append(int(lab))

    image_dir = os.path.join(work, "data", scene, "images")
    frames = sorted(f.replace(".JPEG", "") for f in os.listdir(image_dir) if f.endswith(".JPEG"))
    vid_dir = os.path.join(work, "video_frames")
    shutil.rmtree(vid_dir, ignore_errors=True)
    os.makedirs(vid_dir)
    inst = {}
    overlays = {}
    for fr in frames:
        img = np.array(Image.open(os.path.join(image_dir, fr + ".JPEG")).convert("RGB"))
        H, W = img.shape[:2]
        items = []
        for lab in per_frame.get(fr, []):
            m = np.array(Image.open(P("04_final", "masks", str(lab), fr + ".png")).convert("L")) > 0
            items.append((lab, m, int(m.sum())))
            st = inst.setdefault(lab, {"frames": [], "areas": []})
            st["frames"].append(fr)
            st["areas"].append(100.0 * m.sum() / (H * W))
        idmap = np.zeros_like(img)
        for lab, m, _ in sorted(items, key=lambda t: -t[2]):  # lon ve truoc, nho de len
            idmap[m] = pal(lab)
        Image.fromarray(idmap).save(P("04_final", "id_map", fr + ".png"))
        cov = idmap.any(axis=2)
        ov = img.astype(np.float32) * 0.5
        ov[cov] += idmap[cov].astype(np.float32) * 0.5
        ov = label_text(ov.clip(0, 255).astype(np.uint8), f"frame {fr}")
        Image.fromarray(ov).save(os.path.join(vid_dir, fr + ".png"))
        overlays[fr] = ov

    pick = [frames[int(round(k))] for k in np.linspace(0, len(frames) - 1, 16)] if frames else []
    if pick:
        tiles = [cv2.resize(overlays[f], (overlays[f].shape[1] // 2, overlays[f].shape[0] // 2)) for f in pick]
        grid = np.concatenate([np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1) for r in range(4)], axis=0)
        Image.fromarray(grid).save(P("04_final", "consistency_grid.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-pattern_type", "glob", "-i",
                        os.path.join(vid_dir, "*.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-pix_fmt", "yuv420p", P("04_final", "consistency.mp4")], check=False)

    write_csv(P("04_final", "instances.csv"),
              ["id", "so_frame_xuat_hien", "so_diem_3d", "dien_tich_tb_pct", "frame_dau", "frame_cuoi"],
              [[lab, len(st["frames"]), S["labeled_sizes"].get(lab, 0), f"{np.mean(st['areas']):.3f}",
                min(st["frames"]), max(st["frames"])] for lab, st in sorted(inst.items())])
    with open(P("04_final", "palette.json"), "w") as f:
        json.dump({str(k): list(v) for k, v in sorted(S["palette"].items()) if k in inst}, f, indent=1)
    return {"so_id_cuoi": len(inst), **S.get("votes_summary", {})}
