"""
3DGIC (3D Gaussian Inpainting with Depth-Guided Cross-view Consistency, CVPR 2025) trên LERF teatime – Modal GPU
==============================================================================================================
Workspace Modal: profile `tandat` (workspace datduongtandat385). Mọi dữ liệu/kết quả nằm trên volume `safe-gs-3dgic`.

Tận dụng lại dữ liệu LERF đã có (không chạy lại SAM2 / COLMAP):
  - Ảnh đã crop (principal point về tâm) + COLMAP sparse/0:  ../LERF Datasets/<scene>/<scene>/infusion_data
  - Label map SAM2 mọi vật, ID cố định qua các view:        ../LERF Datasets/<scene>/<scene>/mask/labels
    -> crop giống ảnh rồi dùng trực tiếp làm `object_mask` của Gaussian Grouping / 3DGIC (pixel = ID vật, 0 = nền).

Các bước (chạy lần lượt, bước sau dùng kết quả bước trước trên volume):
  1. upload       (local)   python -m modal run modal_3dgic.py::upload --scene teatime
  2. prepare      (CPU)     python -m modal run modal_3dgic.py::prepare --scene teatime
  3. verify       (L4)      python -m modal run modal_3dgic.py::verify --scene teatime
  4. train        (L4)      python -m modal run modal_3dgic.py::train --scene teatime
  5. removal      (L4)      python -m modal run modal_3dgic.py::removal --scene teatime --ids 11 --run obj11      <- CHỌN ID VẬT
  6. depth_mask   (L4)      python -m modal run modal_3dgic.py::depth_mask --scene teatime --run obj11
  7. inpaint_ref  (L4)      python -m modal run modal_3dgic.py::inpaint_ref --scene teatime --run obj11 --ref-ids 12,60  <- CHỌN ID ẢNH
  8. inpaint_3d   (L4)      python -m modal run modal_3dgic.py::inpaint_3d --scene teatime --run obj11

Layout trên volume:
  /vol/lerf_src/<scene>/{images, sparse/0, labels, transforms.json, objects.json}   dữ liệu upload
  /vol/data/<scene>/base                                                            source_path để train
  /vol/output/<scene>/3dgs                                                          3DGS + classifier đã train
  /vol/data/<scene>/<run>, /vol/output/<scene>/<run>                                mỗi lần xóa vật một thư mục riêng
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import modal

APP_NAME = "safe-gs-3dgic"
app = modal.App(APP_NAME)
vol = modal.Volume.from_name("safe-gs-3dgic", create_if_missing=True)

VOL = "/vol"
SRC_ROOT = f"{VOL}/lerf_src"
DATA_ROOT = f"{VOL}/data"
OUT_ROOT = f"{VOL}/output"
CODE = "/workspace/3DGIC"
ITER = 40000  # số iteration train 3DGS (giống script/run_bear.sh); các script removal/inpaint đọc iteration lớn nhất
NUM_CLASSES = 256
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_PATH = f"{VOL}/cache/lama/big-lama.pt"

repo_dir = Path(__file__).resolve().parent
code_local = repo_dir / "Based_Model" / "3DGIC"
EXTENSIONS = ["submodules/simple-knn", "bvh", "r3dg-rasterization", "diff-gaussian-rasterization-depth"]


def _ignore(p) -> bool:
    """Bỏ build cũ (Linux py3.8/3.10 từ máy tác giả), glm doc/test, cache và dữ liệu khi upload code."""
    parts = Path(p).parts
    s = str(p).replace("\\", "/")
    return (
        any(x in parts for x in ["build", "__pycache__", ".git", "data", "output", "nvdiffrast", "media", "img_src"])
        or any(x.endswith(".egg-info") for x in parts)
        or "glm/doc" in s or "glm/test" in s
        or s.endswith(".so")
    )


# Môi trường giống install.md của tác giả (torch 2.7.0 + CUDA 12.8), build extension cho L4/A10G/A100
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install("torch==2.7.0", "torchvision==0.22.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy==2.1.2", "plyfile==1.1.2", "imageio==2.37.0", "tqdm==4.67.1", "scipy==1.15.3",
        "matplotlib==3.10.5", "scikit-image==0.25.2", "scikit-learn==1.7.1", "opencv-python-headless==4.12.0.88",
        "lpips==0.1.4", "pillow==11.0.0", "tensorboard==2.20.0", "kornia==0.8.1", "ninja==1.13.0",
        "setuptools<75", "wheel",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9",
        "MAX_JOBS": "4",
        "CC": "gcc",  # python add_python mặc định link bằng clang++ (không có trong image CUDA)
        "CXX": "g++",
        "TORCH_HOME": f"{VOL}/cache/torch",  # cache trọng số VGG của LPIPS trên volume
        "PYTHONUNBUFFERED": "1",
    })
)
for ext in EXTENSIONS:
    image = image.add_local_dir(code_local / ext, remote_path=f"/opt/3dgic_ext/{ext}", copy=True, ignore=_ignore)
image = image.run_commands(*[f"pip install --no-build-isolation /opt/3dgic_ext/{ext}" for ext in EXTENSIONS])
# copy=True: code nằm trong image (ghi được), vì script ghi file tạm (test_point_1.ply, test_gt.png) vào thư mục chạy
image = image.add_local_dir(code_local, remote_path=CODE, copy=True,
                             ignore=lambda p: _ignore(p) or Path(p).parts[:1] == ("mesh",))

cpu_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "numpy==2.1.2", "opencv-python-headless==4.12.0.88"
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _run(cmd: str, cwd: str = CODE) -> None:
    print(f"\n[Modal] CWD: {cwd}\n[Modal] RUN: {cmd}\n" + "-" * 60, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
    print("-" * 60 + f"\n[Modal] EXIT CODE: {p.returncode}\n", flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")


def _check_name(name: str, what: str) -> None:
    # Loader của 3DGIC cắt đường dẫn theo '.' và '_' -> chỉ cho chữ, số, '-', '_' trong tên
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"{what} '{name}' chỉ được chứa chữ, số, '-', '_' (không dấu chấm / khoảng trắng)")


def _ids(ids: str) -> list[int]:
    sel = [int(x) for x in ids.replace(" ", "").split(",") if x]
    if not sel:
        raise ValueError("Cần danh sách ID, ví dụ --ids 11 hoặc --ids 2,6,9")
    return sel


def _need_run(run: str) -> str:
    if not run:
        raise ValueError("Cần --run (tên lần xóa vật đã tạo ở bước removal, ví dụ --run obj11)")
    return run


def _paths(scene: str, run: str = "") -> dict[str, Path]:
    _check_name(scene, "scene")
    if run:
        _check_name(run, "run")
    p = {
        "src": Path(SRC_ROOT) / scene,
        "base_data": Path(DATA_ROOT) / scene / "base",
        "base_model": Path(OUT_ROOT) / scene / "3dgs",
    }
    if run:
        p["data"] = Path(DATA_ROOT) / scene / run
        p["model"] = Path(OUT_ROOT) / scene / run
        p["removal_out"] = p["model"] / "train" / "ours_object_removal" / f"iteration_{ITER}"
    return p


def _crop_boxes(transforms: Path) -> dict[str, tuple[int, int, int, int]]:
    """Crop box giống modal_lerf_prep._frames: đưa principal point về tâm ảnh (rasterizer bỏ qua cx, cy)."""
    t = json.loads(transforms.read_text())
    boxes = {}
    for f in t["frames"]:
        w, h, cx, cy = int(f["w"]), int(f["h"]), float(f["cx"]), float(f["cy"])
        half_w, half_h = math.floor(min(cx, w - cx)), math.floor(min(cy, h - cy))
        x0 = min(max(int(round(cx - half_w)), 0), w - 2 * half_w)
        y0 = min(max(int(round(cy - half_h)), 0), h - 2 * half_h)
        boxes[Path(f["file_path"]).stem] = (x0, y0, x0 + 2 * half_w, y0 + 2 * half_h)
    return boxes


def _write_config(path: Path, cfg: dict) -> Path:
    path.write_text(json.dumps(cfg, indent=2))
    print(f"[i] Config -> {path}\n{json.dumps(cfg, indent=2)}")
    return path


# -----------------------------------------------------------------------------
# 1. Upload dữ liệu LERF đã có từ máy lên volume (chạy local)
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def upload(scene: str = "teatime", lerf_dir: str = "../LERF Datasets"):
    # Đường dẫn tương đối tính từ thư mục Safe-GS (repo_dir)
    root = (repo_dir / lerf_dir).resolve() / scene / scene
    items = {
        root / "infusion_data" / "images": "images",
        root / "infusion_data" / "sparse" / "0": "sparse/0",
        root / "mask" / "labels": "labels",
        root / "transforms.json": "transforms.json",
        root / "mask" / "objects.json": "objects.json",
    }
    for local in items:
        if not local.exists():
            raise FileNotFoundError(f"Không thấy {local}")
    print(f"[*] Upload {root} -> volume safe-gs-3dgic:/lerf_src/{scene}")
    with vol.batch_upload(force=True) as batch:
        for local, remote in items.items():
            dst = f"/lerf_src/{scene}/{remote}"
            if local.is_dir():
                batch.put_directory(str(local), dst)
            else:
                batch.put_file(str(local), dst)
            print(f"    {local} -> {dst}")
    print("[✓] Upload xong")


# -----------------------------------------------------------------------------
# 2. Chuẩn bị source_path cho 3DGIC (crop label SAM2 giống ảnh)
# -----------------------------------------------------------------------------
@app.function(image=cpu_image, volumes={VOL: vol}, cpu=4, timeout=1800)
def prepare(scene: str = "teatime"):
    import cv2
    import numpy as np

    vol.reload()
    p = _paths(scene)
    src, data = p["src"], p["base_data"]
    if (data / "object_mask").exists():
        raise FileExistsError(f"{data} đã có. Xóa trước nếu muốn làm lại: python -m modal volume rm -r safe-gs-3dgic data/{scene}/base")
    boxes = _crop_boxes(src / "transforms.json")
    shutil.copytree(src / "images", data / "images")
    shutil.copytree(src / "sparse" / "0", data / "sparse" / "0")
    (data / "sparse" / "0" / "points3D.ply").unlink(missing_ok=True)

    obj_dir = data / "object_mask"
    obj_dir.mkdir(parents=True)
    ids_seen: set[int] = set()
    for img_path in sorted((data / "images").glob("*.png")):
        stem = img_path.stem
        label = cv2.imread(str(src / "labels" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"Thiếu label cho {stem}")
        x0, y0, x1, y1 = boxes[stem]
        crop = label[y0:y1, x0:x1]
        h, w = cv2.imread(str(img_path)).shape[:2]
        if crop.shape != (h, w):
            raise ValueError(f"{stem}: label crop {crop.shape} khác ảnh {(h, w)}")
        if crop.max() >= NUM_CLASSES:
            raise ValueError(f"{stem}: ID {crop.max()} >= num_classes {NUM_CLASSES}")
        ids_seen.update(np.unique(crop).tolist())
        cv2.imwrite(str(obj_dir / f"{stem}.png"), crop.astype(np.uint8))
    vol.commit()
    n = len(list(obj_dir.glob("*.png")))
    print(f"[✓] {n} ảnh + object_mask ({len(ids_seen) - (0 in ids_seen)} ID vật, max {max(ids_seen)}) -> {data}")
    return {"frames": n, "num_ids": len(ids_seen)}


# -----------------------------------------------------------------------------
# 3. Kiểm tra: extension CUDA, đọc scene, 1 bước forward/backward (không train)
# -----------------------------------------------------------------------------
@app.function(image=image, gpu="L4", volumes={VOL: vol}, timeout=1800)
def verify(scene: str = "teatime"):
    import os
    from argparse import ArgumentParser

    import torch

    vol.reload()
    os.chdir(CODE)
    sys.path.insert(0, CODE)
    from diff_gaussian_rasterization_depth import GaussianRasterizer  # noqa: F401
    from r3dg_rasterization import _C  # noqa: F401
    from simple_knn._C import distCUDA2  # noqa: F401
    import bvh_tracing._C  # noqa: F401
    print(f"[✓] Extensions OK | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    from arguments import ModelParams, OptimizationParams, PipelineParams
    from gaussian_renderer import render_fn_dict
    from scene import GaussianModel, Scene

    p = _paths(scene)
    parser = ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args(["-s", str(p["base_data"]), "-m", "/tmp/verify",
                              "--lambda_normal_render_depth", "0.01", "--lambda_mask_entropy", "0.1"])
    os.makedirs("/tmp/verify", exist_ok=True)
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, render_type="render")
    scene_obj = Scene(dataset, gaussians, shuffle=False)
    gaussians.create_from_pcd(scene_obj.scene_info.point_cloud, scene_obj.cameras_extent)
    gaussians.training_setup(opt)
    cams = scene_obj.getTrainCameras()
    cam = cams[0]
    print(f"[i] {len(cams)} camera | ảnh {tuple(cam.original_image.shape)} | objects {tuple(cam.objects.shape)} "
          f"| {gaussians.get_xyz.shape[0]} điểm khởi tạo | extent {scene_obj.cameras_extent:.3f}")
    missing = [c.image_name for c in cams if c.objects is None]
    if missing:
        raise RuntimeError(f"{len(missing)} ảnh không có object_mask, ví dụ {missing[:3]}")

    classifier = torch.nn.Conv2d(gaussians.num_objects, NUM_CLASSES, kernel_size=1).cuda()
    bg = torch.zeros(3, device="cuda")
    pkg = render_fn_dict["render"](cam, gaussians, pipe, bg, opt=opt, is_training=True, dict_params={"iteration": 1})
    logits = classifier(pkg["render_object"])
    loss_obj = torch.nn.functional.cross_entropy(logits.unsqueeze(0), cam.objects.cuda().long())
    loss = pkg["loss"] + loss_obj
    loss.backward()
    print(f"[✓] Render {tuple(pkg['render'].shape)} | render_object {tuple(pkg['render_object'].shape)} "
          f"| loss {pkg['loss'].item():.4f} + obj {loss_obj.item():.4f} | backward OK")
    print(f"[i] ID trong object_mask ảnh {cam.image_name}: {torch.unique(cam.objects).long().tolist()}")
    return {"cameras": len(cams), "points": int(gaussians.get_xyz.shape[0])}


# -----------------------------------------------------------------------------
# 4. Train 3DGS + object features (Gaussian Grouping) – giống script/run_bear.sh
# -----------------------------------------------------------------------------
@app.function(image=image, gpu="L4", volumes={VOL: vol}, timeout=3600 * 24)
def train(scene: str = "teatime", iterations: int = ITER):
    vol.reload()
    p = _paths(scene)
    model = p["base_model"]
    if (model / "point_cloud").exists():
        raise FileExistsError(f"{model} đã có kết quả train. Không ghi đè; xóa thủ công nếu thật sự muốn train lại.")
    if iterations != ITER:
        print(f"[!] iterations={iterations} khác {ITER}: các bước removal sẽ không tìm thấy thư mục iteration_{ITER}")
    _run(
        f"python train.py -s {p['base_data']} -m {model} "
        f"--lambda_normal_render_depth 0.01 --lambda_mask_entropy 0.1 "
        f"--config_file configs/gaussian_dataset/train.json --densification_interval 500 "
        f"--save_training_vis --iterations {iterations}"
    )
    vol.commit()
    print(f"[✓] Model -> {model}/point_cloud/iteration_{iterations}, {model}/chkpnt{iterations}.pth")


# -----------------------------------------------------------------------------
# 5. Xóa vật đã chọn (theo ID SAM2) + render depth trước/sau khi xóa
# -----------------------------------------------------------------------------
@app.function(image=image, gpu="L4", volumes={VOL: vol}, timeout=3600 * 4)
def removal(scene: str = "teatime", ids: str = "", run: str = "", removal_thresh: float = 0.99):
    vol.reload()
    sel = _ids(ids)
    run = run or "obj" + "-".join(map(str, sel))
    p = _paths(scene, run)
    base_model, data, model = p["base_model"], p["data"], p["model"]
    ckpt_dir = base_model / "point_cloud" / f"iteration_{ITER}"
    if not (ckpt_dir / "classifier.pth").exists():
        raise FileNotFoundError(f"Chưa có {ckpt_dir}/classifier.pth. Chạy bước train trước.")
    if model.exists() or data.exists():
        raise FileExistsError(f"Run '{run}' đã tồn tại ({model}). Dùng --run tên khác để không ghi đè kết quả cũ.")

    # Mỗi run có bản sao data + model riêng (bước sau ghi thêm depth/mask vào data)
    for sub in ["images", "sparse", "object_mask"]:
        shutil.copytree(p["base_data"] / sub, data / sub)
    shutil.copytree(ckpt_dir, model / "point_cloud" / f"iteration_{ITER}")
    shutil.copy(base_model / "cfg_args", model / "cfg_args")
    (model / "selected_ids.txt").write_text(",".join(map(str, sel)))

    cfg = _write_config(model / "removal.json", {
        "num_classes": NUM_CLASSES, "removal_thresh": removal_thresh, "select_obj_id": sel,
        "object_path": "object_mask", "depth_path": "depth_removal", "source_path": str(data),
    })
    _run(f"python edit_object_removal.py -m {model} --config_file {cfg} --skip_test --type render")

    # = script/find_depth_guided_mask.sh (phần copy depth)
    shutil.copytree(p["removal_out"] / "depth", data / "object_mask" / "depth_removal")
    vol.commit()
    print(f"[✓] Đã xóa ID {sel} -> {model}/point_cloud_object_removal")
    print(f"[✓] Video kiểm tra: {p['removal_out']}/concat/result.mp4  (gt | đã xóa | gt obj | pred obj | feature)")
    print(f"[→] Bước tiếp: python -m modal run modal_3dgic.py::depth_mask --scene {scene} --run {run}")
    return {"run": run}


def _ref_candidates(scene: str, run: str, top_k: int) -> dict:
    """Xếp hạng view tham chiếu theo diện tích vùng không ai nhìn thấy -> ref_candidates.{json,jpg}."""
    import cv2
    import numpy as np

    p = _paths(scene, _need_run(run))
    model, rem, inp = p["model"], p["removal_out"], p["data"] / "inpaint_2d_unseen_mask"
    # Gợi ý view tham chiếu: vùng "không ai nhìn thấy" lớn, không chạm mép ảnh
    stats = []
    for mpath in sorted(inp.glob("[0-9]*.png")):
        m = cv2.imread(str(mpath), 0) > 127
        ref_id = int(mpath.stem) + 1  # ref_id trong config là số thứ tự 1-based = số frame
        obj = cv2.imread(str(rem / "remove_objects_pred" / mpath.name), 0) > 127
        ys, xs = np.nonzero(m)
        border = len(xs) > 0 and (xs.min() == 0 or ys.min() == 0 or xs.max() == m.shape[1] - 1 or ys.max() == m.shape[0] - 1)
        stats.append({"ref_id": ref_id, "frame": f"frame_{ref_id:05d}", "unseen_pct": round(float(100 * m.mean()), 3),
                      "object_pct": round(float(100 * obj.mean()), 3), "touches_border": bool(border)})
    ranked = sorted([s for s in stats if s["unseen_pct"] > 0], key=lambda s: (s["touches_border"], -s["unseen_pct"]))
    (model / "ref_candidates.json").write_text(json.dumps({"ranked": ranked, "all": stats}, indent=2))

    tiles = []
    for s in ranked[:top_k]:
        name = f"{s['ref_id'] - 1:05d}.png"
        img = cv2.imread(str(rem / "renders" / name))
        orig = cv2.imread(str(rem / "gt" / name))
        m = cv2.imread(str(inp / name), 0) > 127
        over = img.copy()
        over[m] = (0.45 * over[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        tile = np.hstack([orig, over])
        txt = f"ref_id {s['ref_id']}  unseen {s['unseen_pct']}%" + ("  (cham mep)" if s["touches_border"] else "")
        cv2.putText(tile, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(tile, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(tile, (992, 345)))  # LERF crops differ by a few px per frame
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(model / "ref_candidates.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    vol.commit()

    print(f"[i] {len(ranked)}/{len(stats)} view còn vùng không ai nhìn thấy. Top {min(top_k, len(ranked))} gợi ý làm view tham chiếu:")
    for s in ranked[:top_k]:
        print(f"    ref_id {s['ref_id']:>3} ({s['frame']}): unseen {s['unseen_pct']:6.2f}% | vật {s['object_pct']:6.2f}%"
              + (" | chạm mép" if s["touches_border"] else ""))
    print(f"[→] Tải ảnh gợi ý: python -m modal volume get safe-gs-3dgic output/{scene}/{run}/ref_candidates.jpg outputs/3dgic/{scene}/{run}/")
    print(f"[→] CHỌN 1-3 ref_id ở các góc khác nhau rồi chạy: python -m modal run modal_3dgic.py::inpaint_ref --scene {scene} --run {run} --ref-ids <id1,id2>")
    return {"candidates": ranked[:top_k]}


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=1800)
def ref_candidates(scene: str = "teatime", run: str = "", top_k: int = 12):
    """Chỉ tạo lại ảnh gợi ý view tham chiếu (không chạy lại depth-guided mask)."""
    vol.reload()
    return _ref_candidates(scene, run, top_k)


# -----------------------------------------------------------------------------
# 6. Depth-guided mask (vùng không view nào nhìn thấy) + chuẩn bị dữ liệu inpaint + gợi ý view tham chiếu
# -----------------------------------------------------------------------------
@app.function(image=image, gpu="L4", volumes={VOL: vol}, timeout=3600 * 6)
def depth_mask(scene: str = "teatime", run: str = "", top_k: int = 12):
    import cv2
    import numpy as np

    vol.reload()
    p = _paths(scene, _need_run(run))
    data, model, rem = p["data"], p["model"], p["removal_out"]
    if not (data / "object_mask" / "depth_removal").exists():
        raise FileNotFoundError(f"Chưa chạy bước removal cho run '{run}'")
    if (data / "inpaint_2d_unseen_mask").exists():
        raise FileExistsError(f"{data}/inpaint_2d_unseen_mask đã có (bước này đã chạy cho run '{run}').")

    _run(f"python edit_object_removal.py -m {model} --config_file {model / 'removal.json'} --skip_test --render_intersect")

    # = script/move_data.sh
    inp = data / "inpaint_2d_unseen_mask"
    shutil.copytree(rem / "inpaint_mask_pred", inp)
    shutil.copytree(rem / "renders", inp / "images")
    shutil.copytree(rem / "depth_removal", inp / "depth_removal")
    shutil.copytree(rem / "gt_objects", inp / "obj_original")
    shutil.copytree(rem / "inpaint_mask_pred", data / "intersect_mask")

    vol.commit()
    print(f"[✓] Depth-guided mask + dữ liệu inpaint -> {inp}")
    return _ref_candidates(scene, run, top_k)


# -----------------------------------------------------------------------------
# 7. Inpaint 2D RGB + depth của view tham chiếu bằng big-LaMa (README: "use any inpainting model you like")
# -----------------------------------------------------------------------------
def _lama(model, img, mask):
    import numpy as np
    import torch

    H, W = mask.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    img_p = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m_p = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
    t = torch.from_numpy(img_p).float().permute(2, 0, 1)[None].cuda() / 255.0
    mt = torch.from_numpy(m_p.astype(np.float32))[None, None].cuda()
    with torch.no_grad():
        out = model(t, mt)[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)[:H, :W]
    return np.where(mask[..., None], out, img)


def _clone_run(scene: str, src: str, dst: str) -> None:
    """Run mới dùng lại kết quả removal + depth-guided mask của run cũ (không chạy lại bước ~1 giờ)."""
    ps, pd = _paths(scene, src), _paths(scene, dst)
    if pd["data"].exists() or pd["model"].exists():
        raise FileExistsError(f"Run '{dst}' đã tồn tại. Dùng --run tên khác.")
    if not (ps["data"] / "inpaint_2d_unseen_mask").exists():
        raise FileNotFoundError(f"Run nguồn '{src}' chưa có depth_mask")
    print(f"[*] Copy dữ liệu run '{src}' -> '{dst}' (bỏ substitude, khôi phục ảnh tham chiếu gốc)")
    shutil.copytree(ps["data"], pd["data"], ignore=shutil.ignore_patterns("substitude", "ref_backup"))
    backup = ps["data"] / "ref_backup"
    for f in backup.glob("**/*.png") if backup.exists() else []:
        shutil.copy(f, pd["data"] / "inpaint_2d_unseen_mask" / f.relative_to(backup))
    pd["model"].mkdir(parents=True)
    shutil.copytree(ps["model"] / "point_cloud", pd["model"] / "point_cloud")
    for name in ["cfg_args", "selected_ids.txt", "removal.json", "ref_candidates.json", "ref_candidates.jpg"]:
        if (ps["model"] / name).exists():
            shutil.copy(ps["model"] / name, pd["model"] / name)
    shutil.copytree(ps["removal_out"] / "remove_objects_pred", pd["removal_out"] / "remove_objects_pred")
    (pd["model"] / "cloned_from.txt").write_text(src)
    vol.commit()


@app.function(image=image, gpu="L4", volumes={VOL: vol}, timeout=3600)
def inpaint_ref(scene: str = "teatime", run: str = "", ref_ids: str = "", dilate: int = 15, with_object: bool = True,
                from_run: str = ""):
    import urllib.request

    import cv2
    import numpy as np
    import torch

    vol.reload()
    refs = _ids(ref_ids)
    if from_run:
        _clone_run(scene, from_run, _need_run(run))
    p = _paths(scene, _need_run(run))
    inp, model = p["data"] / "inpaint_2d_unseen_mask", p["model"]
    if not inp.exists():
        raise FileNotFoundError(f"Chưa chạy bước depth_mask cho run '{run}'")
    if (p["data"] / "substitude").exists():
        raise FileExistsError(f"Run '{run}' đã chạy inpaint_3d với ảnh tham chiếu cũ. Tạo run mới để đổi view tham chiếu.")

    if not Path(LAMA_PATH).exists():
        Path(LAMA_PATH).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LAMA_URL, LAMA_PATH)
    lama = torch.jit.load(LAMA_PATH, map_location="cuda").eval()

    # Bản gốc (chưa inpaint) được giữ ở ref_backup để chạy lại bước này với ref khác / dilate khác
    backup = p["data"] / "ref_backup"
    preview = model / "ref_inpaint"
    preview.mkdir(parents=True, exist_ok=True)
    for f in backup.glob("**/*.png") if backup.exists() else []:
        shutil.copy(f, inp / f.relative_to(backup))  # khôi phục lần chạy trước
    kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    for rid in refs:
        name = f"{rid - 1:05d}.png"
        files = [Path("images") / name, Path("images") / f"frame_{rid:05d}.png", Path("depth_removal") / name]
        for rel in files:
            if not (inp / rel).exists():
                raise FileNotFoundError(f"Không thấy {inp / rel} (ref_id {rid} sai?)")
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            if not (backup / rel).exists():
                shutil.copy(inp / rel, backup / rel)
        m = (cv2.imread(str(inp / name), 0) > 127).astype(np.uint8)
        if with_object:
            # thêm mask vật: vùng vật đã được view khác nhìn thấy vẫn còn rác trong ảnh render của view tham chiếu
            m |= (cv2.imread(str(p["removal_out"] / "remove_objects_pred" / name), 0) > 127).astype(np.uint8)
        if kernel is not None:
            m = cv2.dilate(m, kernel)
        m = m.astype(bool)
        if not m.any():
            print(f"[!] ref_id {rid}: mask vùng không thấy rỗng -> không có gì để inpaint, vẫn dùng làm view tham chiếu")

        rgb = cv2.cvtColor(cv2.imread(str(inp / "images" / name)), cv2.COLOR_BGR2RGB)
        rgb_out = cv2.cvtColor(_lama(lama, rgb, m), cv2.COLOR_RGB2BGR)
        depth = cv2.cvtColor(cv2.imread(str(inp / "depth_removal" / name)), cv2.COLOR_BGR2RGB)
        d = _lama(lama, depth, m).mean(axis=2).round().astype(np.uint8)
        depth_out = np.repeat(d[..., None], 3, axis=2)  # depth PNG là ảnh xám 3 kênh

        cv2.imwrite(str(inp / "images" / name), rgb_out)
        cv2.imwrite(str(inp / "images" / f"frame_{rid:05d}.png"), rgb_out)
        cv2.imwrite(str(inp / "depth_removal" / name), depth_out)

        over = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        over[m] = (0.45 * over[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(preview / f"ref_{rid:03d}.jpg"), np.hstack([over, rgb_out, depth_vis]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[✓] ref_id {rid}: inpaint {100 * m.mean():.2f}% ảnh")
    (model / "ref_ids.txt").write_text(",".join(map(str, refs)))
    vol.commit()
    print(f"[✓] Ảnh kiểm tra (mask | RGB đã inpaint | depth đã inpaint) -> {preview}")
    print(f"[→] Tải về: python -m modal volume get safe-gs-3dgic output/{scene}/{run}/ref_inpaint outputs/3dgic/{scene}/{run}/")
    print(f"[→] Bước tiếp: python -m modal run modal_3dgic.py::inpaint_3d --scene {scene} --run {run}")


# -----------------------------------------------------------------------------
# 8. 3D inpainting (edit_object_inpaint_spin.py) -> 3DGS hoàn chỉnh
# -----------------------------------------------------------------------------
@app.function(image=image, gpu="L4", volumes={VOL: vol}, timeout=3600 * 8)
def inpaint_3d(scene: str = "teatime", run: str = "", exp: str = "3dgic", finetune_iteration: int = 10000,
               removal_thresh: float = 0.5, lambda_dlpips: float = 0.5):
    vol.reload()
    p = _paths(scene, _need_run(run))
    _check_name(exp, "exp")
    data, model = p["data"], p["model"]
    if not (model / "ref_ids.txt").exists():
        raise FileNotFoundError(f"Chưa chạy bước inpaint_ref cho run '{run}'")
    if list((model / "point_cloud_object_inpaint").glob(f"{exp}_iteration_*")):
        raise FileExistsError(f"Đã có kết quả exp '{exp}' trong run '{run}'. Dùng --exp tên khác.")
    refs = _ids((model / "ref_ids.txt").read_text())
    sel = _ids((model / "selected_ids.txt").read_text())

    # = script/edit_object_inpaint_spin.sh
    sub = data / "substitude"
    if not sub.exists():
        shutil.copytree(data / "inpaint_2d_unseen_mask" / "images", sub)
    cfg = _write_config(model / f"inpaint_{exp}.json", {
        "num_classes": NUM_CLASSES, "removal_thresh": removal_thresh, "select_obj_id": sel,
        "images": "substitude", "object_path": "inpaint_2d_unseen_mask", "depth_path": "depth_removal",
        "r": 1, "lambda_dlpips": lambda_dlpips, "finetune_iteration": finetune_iteration, "exp_setting": exp,
        "source_path": str(data), "reg3d_interval": 5, "reg3d_k": 5, "reg3d_lambda_val": 2,
        "reg3d_max_points": 10000, "reg3d_sample_size": 100, "use_ref": 1, "use_base": 0,
        "skip_init": "False", "ref_id": refs,
    })
    _run(f"python edit_object_inpaint_spin.py -m {model} --config_file {cfg} --skip_test")
    vol.commit()
    it = finetune_iteration - 1
    print(f"[✓] 3DGS đã inpaint -> {model}/point_cloud_object_inpaint/{exp}_iteration_{it}/point_cloud.ply")
    print(f"[✓] Render -> {model}/train/ours_object_inpaint/{exp}_iteration_{it}/renders")
    print(f"[→] Tải về: python -m modal volume get safe-gs-3dgic output/{scene}/{run}/point_cloud_object_inpaint/{exp}_iteration_{it} "
          f"Based_Model/3DGIC/output/{scene}/{run}/")


# -----------------------------------------------------------------------------
# 9. Trích mesh bằng GOF (marching tetrahedra + binary search) từ 3DGS đã inpaint – code ở Based_Model/3DGIC/mesh
#    Image riêng: rasterizer GOF (có integrate) + tetra-triangulation (CGAL), giống modal_infusion.py::mesh_image
# -----------------------------------------------------------------------------
mesh_local = code_local / "mesh"
MESH_REMOTE = "/workspace/3DGIC_mesh"
mesh_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libcgal-dev", "libgmp-dev", "libmpfr-dev",
                 "libboost-dev", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.1.2", "torchvision==0.16.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "plyfile==1.0.3", "trimesh==4.4.9", "einops==0.8.0", "tqdm", "wheel", "setuptools<70")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", "MAX_JOBS": "4",
          "CC": "gcc", "CXX": "g++", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(mesh_local / "submodules", remote_path="/opt/gof_submodules", copy=True, ignore=_ignore)
    .run_commands(
        "pip install --no-build-isolation /opt/gof_submodules/diff-gaussian-rasterization",
        "cd /opt/gof_submodules/tetra-triangulation && export CPLUS_INCLUDE_PATH=/usr/local/cuda/include && "
        "cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') && "
        "make -j8 && pip install -e .",
        "python -c \"from tetranerf.utils.extension import cpp; import diff_gaussian_rasterization; print('GOF OK')\"",
    )
    .add_local_dir(mesh_local, remote_path=MESH_REMOTE, copy=False,
                   ignore=lambda p: "submodules" in Path(p).parts or "__pycache__" in Path(p).parts)
)


@app.function(image=mesh_image, gpu="L4", cpu=8, memory=65536, volumes={VOL: vol}, timeout=3600 * 4)
def extract_mesh(scene: str = "teatime", run: str = "", exp: str = "3dgic", texture_mesh: bool = True,
                 filter_mesh: bool = False, view_stride: int = 1, opacity_threshold: float = 0.1):
    """Mesh GOF từ 3DGS cuối cùng của 3DGIC (point_cloud_object_inpaint/<exp>_iteration_*)."""
    vol.reload()
    p = _paths(scene, _need_run(run))
    plys = sorted((p["model"] / "point_cloud_object_inpaint").glob(f"{exp}_iteration_*/point_cloud.ply"))
    if not plys:
        raise FileNotFoundError(f"Chưa có 3DGS inpaint '{exp}' trong run '{run}' (chạy inpaint_3d trước)")
    ply = plys[-1]
    mesh_dir = p["model"] / "mesh" / ply.parent.name
    _run(
        f"python -u extract_mesh.py --ply_path {ply} --cameras_json {p['base_model'] / 'cameras.json'} "
        f"--output_dir {mesh_dir} --view_stride {view_stride} --opacity_threshold {opacity_threshold}"
        f"{' --texture_mesh' if texture_mesh else ''}{' --filter_mesh' if filter_mesh else ''}",
        cwd=MESH_REMOTE,
    )
    vol.commit()
    # extract_mesh.py không ghi đè (mesh_gof.ply -> mesh_gof_v2.ply ...): trả về file mới nhất
    mesh = max(mesh_dir.glob("mesh_gof*.ply"), key=lambda f: f.stat().st_mtime)
    print(f"[✓] Mesh -> {mesh} ({mesh.stat().st_size / 1024**2:.1f} MB)")
    return {"mesh": str(mesh)}


crop_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "scipy==1.13.1", "trimesh==4.4.9", "pillow==10.4.0")
    .add_local_dir(mesh_local, remote_path=MESH_REMOTE, copy=False,
                   ignore=lambda p: "submodules" in Path(p).parts or "__pycache__" in Path(p).parts)
)


@app.function(image=crop_image, cpu=4, memory=32768, volumes={VOL: vol}, timeout=3600)
def crop_mesh(scene: str = "teatime", run: str = "", mesh: str = "", frame_json: str = "",
              extra_margin: float = 0.08, extra_up: float = 0.08, extra_down: float = 0.08):
    """Cắt mesh GOF quanh bàn bằng vùng cắt *_crop.json (đường dẫn trên volume, tương đối với /vol)."""
    vol.reload()
    p = _paths(scene, _need_run(run))
    src = Path(VOL) / mesh if mesh else max((p["model"] / "mesh").glob("*/mesh_gof*.ply"), key=lambda f: f.stat().st_mtime)
    _run(f"python -u crop_with_frame.py --mesh {src} --frame_json {Path(VOL) / frame_json} --output_dir {src.parent} "
         f"--extra_margin {extra_margin} --extra_up {extra_up} --extra_down {extra_down}", cwd=MESH_REMOTE)
    vol.commit()


@app.function(image=crop_image, cpu=2, memory=16384, volumes={VOL: vol}, timeout=1800)
def preview_mesh(mesh: str = "", crop_json: str = "", out_png: str = ""):
    """Ảnh nhìn từ trên xuống của mesh (đường dẫn tương đối với /vol)."""
    vol.reload()
    _run(f"python -u preview_mesh.py {Path(VOL) / mesh} {Path(VOL) / crop_json} {Path(VOL) / out_png}", cwd=MESH_REMOTE)
    vol.commit()
