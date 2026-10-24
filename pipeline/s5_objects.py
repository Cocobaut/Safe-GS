"""
S5: chuẩn bị từng vật.
v1 (completion = plane_cap): chưa sinh phần không thấy; ghi vết tiếp xúc trên mặt bàn để S6 đóng đáy phẳng.
Nâng cấp sau: SAM 3D Objects (sinh vật đầy đủ từ 1 ảnh + mask) rồi căn về cảnh.
Xuất mỗi vật objects/<tên>/: gaussians.ply, info.json (số Gaussian, chiều cao trên mặt bàn, khoảng cách đáy tới mặt bàn,
diện tích vết tiếp xúc), preview.jpg (vật render riêng trên nền trắng ở các frame preview mà vật xuất hiện).
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from common import log, make_parser, load_scene, view_name, read_ply, gaussians_from_ply, render_rgbd, to_u8, load_cfg
from arguments import get_combined_args


def main():
    p, model, pipe, ss, _, _ = make_parser("S5 objects")
    p.add_argument("--s3", type=str, required=True)
    args = get_combined_args(p)
    cfg_all = load_cfg(args.cfg)
    s3, out = Path(args.s3), Path(args.out)
    groups = json.loads((s3 / "groups.json").read_text())
    sup = json.loads((s3 / "support.json").read_text())
    n = np.array(sup["table"]["normal"]); d = sup["table"]["d"]; mpu = sup["meters_per_unit"]
    S = load_scene(args, model, pipe, ss)
    white = torch.ones(3, device="cuda")
    S_white = dict(S, bg=white)
    summary = {}
    for name, gi in groups["objects"].items():
        od = out / "objects" / name
        if (od / "info.json").exists():
            summary[name] = json.loads((od / "info.json").read_text()); continue
        od.mkdir(parents=True, exist_ok=True)
        src = s3 / "gaussians" / f"{name}.ply"
        el = read_ply(src)
        if len(el) == 0:
            info = {"gaussians": 0, "note": "S3 không gán được Gaussian nào"}
            (od / "info.json").write_text(json.dumps(info, indent=2)); summary[name] = info; continue
        shutil.copy2(src, od / "gaussians.ply")
        xyz = np.stack([el["x"], el["y"], el["z"]], 1)
        sd = xyz @ n + d
        lo, hi = np.percentile(sd, 1), np.percentile(sd, 99)
        info = {"gaussians": int(len(el)), "mask_ids": gi["ids"],
                "bottom_above_table_m": float(lo * mpu), "top_above_table_m": float(hi * mpu),
                "height_m": float((hi - lo) * mpu),
                "contact_gaussians": int((sd < lo + 0.05 * (hi - lo)).sum()),
                "completion": cfg_all["S5_objects"]["completion"]}
        g_o = gaussians_from_ply(od / "gaussians.ply", S["dataset"].sh_degree, S["train"])
        tiles = []
        for v in S["views"]:
            if view_name(v) in cfg_all["S3_label"]["preview_frames"]:
                rgb, depth = render_rgbd(v, g_o, S_white)
                if (depth > 0).sum() > 50:
                    tiles.append(to_u8(rgb)[..., ::-1])
        if tiles:
            h = 360
            tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
            cv2.imwrite(str(od / "preview.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
        (od / "info.json").write_text(json.dumps(info, indent=2))
        summary[name] = info
        log(f"{name}: {info}")
    (out / "objects_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"S5 xong -> {out}")


if __name__ == "__main__":
    main()
