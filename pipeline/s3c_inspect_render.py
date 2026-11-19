"""
Anh / video quay vong (orbit) cua Gaussian VAT sau S3c, de nhin truc tiep Gaussian nam trong khung.

Moi run (--run NHAN=THU_MUC_S3c:THU_MUC_S3m): doc final.ply + oid_final.npy + metrics.json (thu tu vat = khoa per_object),
khung S3m/frame/<vat>.npz. Voi moi vat (va 1 canh "all" gom moi vat tren ban, khong nen):
  - hang 1: mau Gaussian (SH day du, nen trang) cua tung run
  - hang 2: cung goc nhin, Gaussian ngoai khung > --out_cm to DO (floater), trong khung giu mau goc
Camera quay quanh tam vat (median Gaussian co opacity > 0.1), ban kinh theo p95 kich thuoc, nghieng --elev_deg so voi mat ban,
truc len = phap tuyen ban (support.json). Xuat JPEG tung frame: <out>/<vat>/f_###.jpg (ghep mp4 o may local),
<out>/<vat>_sheet.jpg (4 goc), <out>/stats.json (so Gaussian, so floater theo vat va run).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from common import log, make_parser, load_scene, read_ply, write_ply, gaussians_from_ply, to_u8
from arguments import get_combined_args
from gaussian_renderer import render
from s3i_mold_fill import look_at_cam
from s3c_constrain import FrameDist


def main():
    p, model, pipe, ss, _, _ = make_parser("S3c inspect render")
    p.add_argument("--s3", required=True, help="S3_label (support.json)")
    p.add_argument("--run", action="append", required=True, help="NHAN=THU_MUC_S3c:THU_MUC_S3m")
    p.add_argument("--objects", default="ly_tra,khan_giay,coc_su,dia,banh_quy,tao,tui_banh,goi_keo,thia,all")
    p.add_argument("--n_frames", type=int, default=72)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--elev_deg", type=float, default=30.0)
    p.add_argument("--out_cm", type=float, default=0.5)
    args = get_combined_args(p)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sup = json.loads((Path(args.s3) / "support.json").read_text())
    n_t = np.array(sup["table"]["normal"], np.float64); n_t /= np.linalg.norm(n_t)
    cm = 100 * float(sup["meters_per_unit"])

    S = load_scene(args, model, pipe, ss)
    white = torch.ones(3, device="cuda")
    runs = []
    for spec in args.run:
        lab, _, dirs = spec.partition("=")
        d_c, _, d_m = dirs.partition(":")
        d_c, d_m = Path(d_c), Path(d_m)
        names = list(json.loads((d_c / "metrics.json").read_text())["after"]["per_object"].keys())
        el = read_ply(d_c / "final.ply")
        oid = np.load(d_c / "oid_final.npy")
        assert len(oid) == len(el), f"{lab}: oid {len(oid)} != ply {len(el)}"
        runs.append({"label": lab, "names": names, "el": el, "oid": oid, "mold": d_m})
        log(f"{lab}: {len(el)} Gaussian, {int((oid >= 0).sum())} Gaussian vat, vat = {names}")

    stats = {}
    want = args.objects.split(",")
    for obj in want:
        per_run = []
        for r in runs:
            if obj == "all":
                sel_names = [n for n in r["names"] if n not in ("gau", "cuu")]   # vat tren ban; gau / cuu che het canh
            else:
                sel_names = [obj] if obj in r["names"] else []
            if not sel_names:
                per_run.append(None); continue
            ids = [r["names"].index(n) for n in sel_names]
            m = np.isin(r["oid"], ids)
            sub = r["el"][m]
            tmp = out / f"_tmp_{r['label']}.ply"
            write_ply(tmp, sub)
            g = gaussians_from_ply(tmp, S["dataset"].sh_degree, S["train"])
            tmp.unlink(missing_ok=True)
            xyz = g.get_xyz
            with torch.no_grad():
                dout = torch.zeros(len(xyz), device="cuda")
                sub_oid = torch.tensor(r["oid"][m], device="cuda")
                for i, n in zip(ids, sel_names):
                    f = r["mold"] / "frame" / f"{n}.npz"
                    if not f.exists():
                        continue
                    z = np.load(f)
                    fd = FrameDist(z["xyz"].astype(np.float64), float(z["voxel"]), 10.0 / cm)
                    k = sub_oid == i
                    dout[k] = fd(xyz[k]) * cm
                op = g.get_opacity.squeeze(-1)
                outside = dout > args.out_cm
                c0 = (0.5 + 0.28209479 * g._features_dc[:, 0, :]).clamp(0, 1)
                col_out = c0.clone(); col_out[outside] = torch.tensor([1.0, 0.05, 0.05], device="cuda")
            st = {"gaussians": int(len(xyz)), "outside": int(outside.sum()),
                  "outside_pct": 100 * float(outside.float().mean()) if len(xyz) else 0.0,
                  "outside_opacity_sum": float(op[outside].sum()),
                  "outside_visible": int((outside & (op > 0.1)).sum())}
            stats.setdefault(obj, {})[r["label"]] = st
            if obj != "all":                         # Gaussian vat cho trinh xem WebGL (tools/build_mold_inspector.py --gauss_layer)
                gd = out / "gauss" / r["label"]; gd.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(gd / f"{obj}.npz", xyz=xyz.detach().cpu().numpy().astype(np.float32),
                                    rgb=c0.detach().cpu().numpy().astype(np.float32), opacity=op.detach().cpu().numpy().astype(np.float32),
                                    outside=outside.cpu().numpy(), dout_cm=dout.cpu().numpy().astype(np.float32))
            per_run.append({"g": g, "col_out": col_out, "xyz": xyz.detach().cpu().numpy(), "op": op.detach().cpu().numpy()})
        ok = [x for x in per_run if x is not None]
        if not ok:
            continue
        # tam + ban kinh: tu run dau tien co vat (cung camera cho moi run)
        X = np.concatenate([x["xyz"][x["op"] > 0.1] for x in ok[:1]])
        c = np.median(X, 0)
        rad = float(np.percentile(np.linalg.norm(X - c, axis=1), 95))
        dist = max(rad * 2.6, 6.0 / cm)
        e1 = np.cross(n_t, [1.0, 0, 0] if abs(n_t[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_t, e1)
        el_r = math.radians(args.elev_deg)
        od = out / obj; od.mkdir(exist_ok=True)
        sheet = []
        for k in range(args.n_frames):
            a = 2 * math.pi * k / args.n_frames
            C = c + dist * (math.cos(el_r) * (math.cos(a) * e1 + math.sin(a) * e2) + math.sin(el_r) * n_t)
            cam, _, _ = look_at_cam(C, c, n_t, W=args.size, H=args.size, fov_deg=40.0)
            row1, row2 = [], []
            for x in per_run:
                if x is None:
                    blank = np.full((args.size, args.size, 3), 235, np.uint8)
                    row1.append(blank); row2.append(blank); continue
                with torch.no_grad():
                    im = render(cam, x["g"], S["pipe"], white, splat_args=S["splat_args"])["render"][:3]
                    im2 = render(cam, x["g"], S["pipe"], white, splat_args=S["splat_args"], override_color=x["col_out"])["render"][:3]
                row1.append(to_u8(im)); row2.append(to_u8(im2))
            sep = np.full((args.size, 4, 3), 60, np.uint8)
            top = np.hstack(sum([[r_, sep] for r_ in row1], [])[:-1])
            bot = np.hstack(sum([[r_, sep] for r_ in row2], [])[:-1])
            tile = np.vstack([top, np.full((4, top.shape[1], 3), 60, np.uint8), bot])
            for j, r in enumerate(runs):
                s_ = stats.get(obj, {}).get(r["label"])
                txt = f"{r['label']}" + (f"  n={s_['gaussians']}" if s_ else "  (khong co)")
                cv2.putText(tile, txt, (8 + j * (args.size + 4), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                if s_:
                    cv2.putText(tile, f"ngoai khung >{args.out_cm}cm: {s_['outside']} ({s_['outside_pct']:.1f}%)",
                                (8 + j * (args.size + 4), args.size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
            cv2.putText(tile, obj, (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
            cv2.imwrite(str(od / f"f_{k:03d}.jpg"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            if k % (args.n_frames // 4) == 0:
                sheet.append(cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{obj}_sheet.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 90])
        log(f"{obj}: {stats.get(obj)}")
        del per_run, ok
        torch.cuda.empty_cache()
    (out / "stats.json").write_text(json.dumps({"runs": [r["label"] for r in runs], "out_cm": args.out_cm, "objects": stats},
                                               indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
