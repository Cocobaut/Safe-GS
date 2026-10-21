"""
Quan sát mặt SDF = 0 của CoMe/GOF (opacity tích luỹ = 0.5) từ đúng các góc camera COLMAP.

Chạy BÊN TRONG thư mục CoMe (cần CUDA + extension của CoMe), setup giống hệt render.py:
    python come_sdf_vis.py -m <model_path> -s <colmap_dir> --iteration 30000 --out <thư mục>

Kênh depth (kênh 6) của rasterizer CoMe là khoảng cách dọc tia tới chỗ T giảm qua 0.5
(hierarchical_render.cuh, nhánh EXACT_DEPTH, giải chính xác trên Gaussian) = mặt sdf = alpha - 0.5 = 0 nhìn từ camera.
Xuất cho mỗi camera (tên frame_XXXXX.png):
    normal/   normal của mặt đó (utils.depth_utils.depth_to_normal, đổi sang hệ camera), đen = tia không cắt mặt 0.5
    contour/  ảnh render + viền mặt SDF tô đậm: vàng = viền khuất (depth nhảy), đỏ = nếp gấp (normal đổi > CREASE_DEG),
              xanh dương = tia không cắt mặt 0.5
và 2 video có số frame ở góc: sdf_contour.mp4, sdf_normal.mp4. Frame đã có file thì bỏ qua (chạy lại tiếp được).
"""
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

from arguments import ModelParams, PipelineParams, SplattingSettings, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.depth_utils import depth_to_normal

DEPTH_JUMP = 0.04          # |Δdepth| / depth giữa 2 pixel kề nhau -> viền khuất
CREASE_DEG = 40.0          # góc giữa normal 2 pixel kề nhau -> nếp gấp (--crease; mặc định tắt: normal từ depth theo
                           # từng pixel quá nhiễu ở lông thú / vân gỗ nên nếp gấp ra toàn chấm)
MIN_EDGE_PX = 40           # bỏ mẩu viền nhỏ hơn số pixel này (vụn ở nền xa)


def neighbor_max(x, f):
    """max của f(x, x_kề) qua 4 hướng (pixel biên = 0)."""
    out = np.zeros_like(x, dtype=np.float32)
    out[:, :-1] = np.maximum(out[:, :-1], f(x[:, :-1], x[:, 1:]))
    out[:, 1:] = np.maximum(out[:, 1:], f(x[:, 1:], x[:, :-1]))
    out[:-1, :] = np.maximum(out[:-1, :], f(x[:-1, :], x[1:, :]))
    out[1:, :] = np.maximum(out[1:, :], f(x[1:, :], x[:-1, :]))
    return out


def edges(depth, n_cam):
    # Chỉ dựa vào depth: depth > 0 <=> tia đã đi qua chỗ T = 0.5. KHÔNG dùng kênh 7: đó là giá trị trường opacity
    # tại điểm bề mặt (loss opacity field kéo nó về 0.5), không phải opacity tích luỹ (đã kiểm: trung vị 0.52).
    valid = np.isfinite(depth) & (depth > 0)
    d = np.where(valid, depth, 0).astype(np.float32)
    jump = neighbor_max(d, lambda a, b: np.abs(a - b) / np.maximum(np.minimum(a, b), 1e-6))
    occl = valid & (jump > DEPTH_JUMP)
    cos_min = np.ones(depth.shape, np.float32)
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        c = (n_cam[sl_a] * n_cam[sl_b]).sum(-1)
        cos_min[sl_a] = np.minimum(cos_min[sl_a], c)
    crease = valid & ~occl & (cos_min < math.cos(math.radians(CREASE_DEG)))
    sil = valid & ~cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return valid, occl | sil, crease


def write_video(files, path, fps):
    h, w = cv2.imread(str(files[0])).shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        txt = f"{i}/{len(files)}  {f.stem}"
        cv2.rectangle(img, (0, 0), (12 + 13 * len(txt), 36), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main():
    parser = ArgumentParser(description="SDF = 0 visualisation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dump", type=str, default="", help="chỉ chạy 1 frame, lưu 13 kênh render ra <out>/debug/<frame>.npy")
    parser.add_argument("--crease", action="store_true", help="vẽ thêm nếp gấp (đỏ), nhiễu")
    args = get_combined_args(parser)
    splat_args = ss.get_settings(args)
    safe_state(args.quiet)

    out = Path(args.out)
    (out / "normal").mkdir(parents=True, exist_ok=True)
    (out / "contour").mkdir(parents=True, exist_ok=True)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        train = scene.getTrainCameras()
        gaussians.compute_3D_filter(train.copy())                  # như render.py
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        views = sorted(train + scene.getTestCameras(), key=lambda c: c.image_name)
        for k, view in enumerate(views, 1):
            name = Path(view.image_name).stem
            if args.dump:
                if name != args.dump:
                    continue
                r = render(view, gaussians, pipe, bg, splat_args=splat_args)["render"]
                (out / "debug").mkdir(exist_ok=True)
                np.save(out / "debug" / f"{name}.npy", r.cpu().numpy())
                print(f"[sdf_vis] dump {name}: {tuple(r.shape)}", flush=True)
                return
            fn, fc = out / "normal" / f"{name}.png", out / "contour" / f"{name}.png"
            if fn.exists() and fc.exists():
                continue
            r = render(view, gaussians, pipe, bg, splat_args=splat_args)["render"]
            rgb = (r[0:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1].copy()
            depth = r[6:7]
            n_world, _ = depth_to_normal(view, depth)                 # (H, W, 3), hệ thế giới
            w2c = view.world_view_transform.T                         # 3DGS lưu world_view_transform đã chuyển vị
            n_cam = (n_world @ w2c[:3, :3].T).cpu().numpy()
            valid, outline, crease = edges(depth[0].cpu().numpy(), n_cam)

            nimg = ((-n_cam * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)[..., ::-1]
            nimg[~valid] = 0
            cv2.imwrite(str(fn), nimg)

            cimg = (rgb.astype(np.float32) * 0.8).astype(np.uint8)
            cimg[~valid] = (cimg[~valid] * 0.5 + np.array([255, 120, 0]) * 0.5).astype(np.uint8)
            k3 = np.ones((3, 3), np.uint8)
            if args.crease:
                cimg[cv2.dilate(crease.astype(np.uint8), k3).astype(bool)] = (0, 0, 255)
            ol = outline.astype(np.uint8)
            n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(ol, connectivity=8)
            keep = np.zeros(n_cc, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_EDGE_PX
            ol = keep[cc].astype(np.uint8)
            cimg[cv2.dilate(ol, np.ones((5, 5), np.uint8)).astype(bool)] = (0, 255, 255)
            cv2.imwrite(str(fc), cimg)
            if k % 10 == 0:
                print(f"[sdf_vis] {k}/{len(views)} {name}", flush=True)

    for sub in ("contour", "normal"):
        files = sorted((out / sub).glob("frame_*.png"))
        write_video(files, out / f"sdf_{sub}.mp4", args.fps)
        print(f"[sdf_vis] video {out / f'sdf_{sub}.mp4'} ({len(files)} frame)", flush=True)


if __name__ == "__main__":
    main()
