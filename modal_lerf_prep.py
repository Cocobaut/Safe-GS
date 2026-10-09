"""
Chuẩn bị dữ liệu LERF (figurines / teatime) cho InFusion trên Modal (modal_lerf_prep.py)
=======================================================================================
LERF chỉ có ảnh + transforms.json (nerfstudio), thiếu 2 thứ InFusion cần:
  1. lerf_sam2_masks : mask vật cần xóa cho MỌI ảnh -> SAM 2.1 video predictor lan mask từ các điểm bạn click
                       (scripts/lerf_select_object.py) ra toàn bộ chuỗi ảnh.             [GPU L4]
  2. lerf_colmap     : COLMAP sparse/0 (camera + point cloud khởi tạo) -> dùng pose có sẵn trong transforms.json,
                       chỉ trích SIFT + match + triangulate (không chạy lại SfM).          [CPU]

Cả hai ghi vào outputs/infusion/<scene>/data/{images,seg,sparse/0} -> modal_infusion.py dùng trực tiếp.
Ảnh được crop để principal point nằm đúng tâm (rasterizer 3DGS bỏ qua cx, cy; LERF lệch ~23px theo trục dọc).

Lệnh:
  python -m modal run modal_lerf_prep.py::lerf_colmap --scene teatime
  python -m modal run modal_lerf_prep.py::lerf_sam2_masks --scene teatime
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import modal

APP_NAME = "safe-gs-lerf-prep"
app = modal.App(APP_NAME)

dataset_vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
outputs_vol = modal.Volume.from_name("safe-gs-outputs", create_if_missing=True)

VOLUMES_MAPPING = {
    "/data": dataset_vol,
    "/workspace/Safe-GS/outputs": outputs_vol,
}
LERF_ROOT = "/data/LERF"
OUTPUTS_ROOT = "/workspace/Safe-GS/outputs/infusion"
SAM2_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

colmap_image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.10")
    .apt_install("colmap")
    .pip_install("numpy==1.26.4", "pillow==10.4.0")
    .env({"QT_QPA_PLATFORM": "offscreen", "PYTHONUNBUFFERED": "1"})
)

sam2_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "wget", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==1.26.4", "pillow==11.1.0", "opencv-python-headless==4.10.0.84", "tqdm",
                 "hydra-core==1.3.2", "iopath==0.1.10", "setuptools<75", "wheel")
    .env({"SAM2_BUILD_CUDA": "0", "SAM2_CHECKPOINT": "/opt/checkpoints/sam2.1_hiera_large.pt", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        "SAM2_BUILD_CUDA=0 pip install --no-build-isolation 'git+https://github.com/facebookresearch/sam2.git'",
        f"mkdir -p /opt/checkpoints && wget -q -O /opt/checkpoints/sam2.1_hiera_large.pt {SAM2_CKPT_URL}",
    )
)


# -----------------------------------------------------------------------------
# Helpers dùng chung
# -----------------------------------------------------------------------------
def _run_shell(cmd: str, cwd: str = "/tmp") -> None:
    print(f"\n[Modal] RUN: {cmd}\n" + "-" * 60)
    process = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    if process.stdout:
        for line in process.stdout:
            sys.stdout.write(line)
    process.wait()
    print("-" * 60 + f"\n[Modal] EXIT CODE: {process.returncode}\n")
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd}")


def _scene_src(scene: str) -> Path:
    src = Path(LERF_ROOT) / scene
    if not (src / "transforms.json").exists():
        raise FileNotFoundError(f"Không thấy {src}/transforms.json. Upload trước: python modal_sync.py upload-lerf --scene {scene}")
    return src


def _frames(scene: str) -> list[dict]:
    """Đọc transforms.json, thêm crop box để principal point về đúng tâm ảnh."""
    t = json.loads((_scene_src(scene) / "transforms.json").read_text())
    frames = []
    for f in t["frames"]:
        w, h, cx, cy = int(f["w"]), int(f["h"]), float(f["cx"]), float(f["cy"])
        half_w = math.floor(min(cx, w - cx))
        half_h = math.floor(min(cy, h - cy))
        x0 = min(max(int(round(cx - half_w)), 0), w - 2 * half_w)
        y0 = min(max(int(round(cy - half_h)), 0), h - 2 * half_h)
        frames.append({
            "name": Path(f["file_path"]).name,
            "stem": Path(f["file_path"]).stem,
            "c2w": f["transform_matrix"],
            "fx": float(f["fl_x"]), "fy": float(f["fl_y"]),
            "box": (x0, y0, x0 + 2 * half_w, y0 + 2 * half_h),
            "cx": cx - x0, "cy": cy - y0,
        })
    return sorted(frames, key=lambda x: x["name"])


def _rotmat2qvec(R):
    import numpy as np

    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    return -qvec if qvec[0] < 0 else qvec


# -----------------------------------------------------------------------------
# 1. COLMAP từ pose có sẵn
# -----------------------------------------------------------------------------
@app.function(image=colmap_image, cpu=16, memory=16384, volumes=VOLUMES_MAPPING, timeout=3600 * 2)
def lerf_colmap(scene: str = "teatime", max_features: int = 8192):
    """Crop ảnh + tạo COLMAP sparse/0 với pose của transforms.json (feature -> exhaustive match -> triangulate)."""
    import numpy as np
    from PIL import Image

    dataset_vol.reload()
    outputs_vol.reload()
    src = _scene_src(scene)
    frames = _frames(scene)
    data = Path(OUTPUTS_ROOT) / scene / "data"
    img_out = data / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    print(f"[*] Crop {len(frames)} ảnh (principal point về tâm) -> {img_out}")
    for f in frames:
        Image.open(src / "images" / f["name"]).convert("RGB").crop(f["box"]).save(img_out / f"{f['stem']}.png")
    outputs_vol.commit()

    # COLMAP làm việc trên disk local của container rồi mới copy kết quả vào volume
    work = Path("/tmp/colmap")
    shutil.rmtree(work, ignore_errors=True)
    (work / "known").mkdir(parents=True)
    db = work / "database.db"
    # Feature + match là bước tốn thời gian nhất (~15 phút) -> cache database lên volume để chạy lại không phải làm lại
    db_cache = Path(OUTPUTS_ROOT) / scene / "colmap" / "database.db"
    if db_cache.exists():
        print(f"[SKIP] Dùng lại feature/match đã cache: {db_cache}")
        shutil.copy(db_cache, db)
    else:
        _run_shell(
            f"colmap feature_extractor --database_path {db} --image_path {img_out} "
            f"--ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 0 "
            f"--SiftExtraction.max_num_features {max_features} --SiftExtraction.num_threads 16"
        )
        _run_shell(
            f"colmap exhaustive_matcher --database_path {db} --SiftMatching.use_gpu 0 --SiftMatching.num_threads 16"
        )
        db_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(db, db_cache)
        outputs_vol.commit()

    # Mỗi ảnh 1 camera PINHOLE với intrinsics riêng của nó (LERF có intrinsics theo từng frame)
    by_name = {f"{f['stem']}.png": f for f in frames}
    con = sqlite3.connect(db)
    rows = con.execute("SELECT image_id, name FROM images").fetchall()
    con.execute("DELETE FROM cameras")
    cam_lines, img_lines = [], []
    for image_id, name in sorted(rows):
        f = by_name[name]
        w, h = f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]
        params = np.array([f["fx"], f["fy"], f["cx"], f["cy"]], dtype=np.float64)
        con.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)", (image_id, 1, w, h, params.tobytes(), 1))
        con.execute("UPDATE images SET camera_id = ? WHERE image_id = ?", (image_id, image_id))
        cam_lines.append(f"{image_id} PINHOLE {w} {h} {f['fx']} {f['fy']} {f['cx']} {f['cy']}")

        # nerfstudio c2w (OpenGL) -> OpenCV -> world-to-camera
        c2w = np.array(f["c2w"], dtype=np.float64)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        q = _rotmat2qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        # images.txt: 2 dòng / ảnh (pose + POINTS2D), dòng POINTS2D để trống
        img_lines.append(f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {image_id} {name}\n\n")
    con.commit()
    con.close()
    (work / "known" / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (work / "known" / "images.txt").write_text("".join(img_lines))
    (work / "known" / "points3D.txt").write_text("")

    (work / "sparse").mkdir()
    _run_shell(
        f"colmap point_triangulator --database_path {db} --image_path {img_out} "
        f"--input_path {work / 'known'} --output_path {work / 'sparse'}"
    )
    _run_shell(f"colmap model_analyzer --path {work / 'sparse'}")

    dst = data / "sparse" / "0"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
        shutil.copy(work / "sparse" / fn, dst / fn)
    # Reader của InFusion cache points3D.ply, xóa để sinh lại từ points3D.bin mới
    (dst / "points3D.ply").unlink(missing_ok=True)
    outputs_vol.commit()
    print(f"[✓] COLMAP sparse/0 -> {dst}")


# -----------------------------------------------------------------------------
# 2. SAM 2.1 video masks
# -----------------------------------------------------------------------------
@app.function(image=sam2_image, gpu="L4", volumes=VOLUMES_MAPPING, timeout=3600)
def lerf_sam2_masks(scene: str = "teatime", dilate: int = 5, preview_every: int = 10):
    """Lan mask từ prompts.json ra toàn bộ ảnh bằng SAM 2.1 large, ghi data/seg/*.png (0/255, đã crop)."""
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    dataset_vol.reload()
    outputs_vol.reload()
    src = _scene_src(scene)
    prompts_file = src / "prompts.json"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Thiếu {prompts_file}. Chạy scripts/lerf_select_object.py rồi upload-lerf --prompts-only")
    prompts = json.loads(prompts_file.read_text(encoding="utf-8"))
    frames = _frames(scene)
    index = {f["name"]: i for i, f in enumerate(frames)}

    # SAM2 cần thư mục JPEG đặt tên theo số thứ tự
    tmp = Path("/tmp/sam2_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for i, f in enumerate(frames):
        shutil.copy(src / "images" / f["name"], tmp / f"{i:05d}.jpg")

    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", os.environ["SAM2_CHECKPOINT"], device="cuda")
    masks = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(tmp), offload_video_to_cpu=True)
        for obj in prompts["objects"]:
            for p in obj["prompts"]:
                predictor.add_new_points_or_box(
                    state, frame_idx=index[p["frame"]], obj_id=int(obj["id"]),
                    points=np.array(p["points"], dtype=np.float32), labels=np.array(p["labels"], dtype=np.int32),
                )
                print(f"[i] Vật #{obj['id']}: {len(p['points'])} điểm trên {p['frame']}")
        H, W = state["video_height"], state["video_width"]
        masks = np.zeros((len(frames), H, W), dtype=bool)
        for reverse in (False, True):
            for fidx, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                masks[fidx] |= (logits > 0.0).any(dim=0).squeeze(0).cpu().numpy()

    data = Path(OUTPUTS_ROOT) / scene / "data"
    seg = data / "seg"
    preview = Path(OUTPUTS_ROOT) / scene / "sam2_preview"
    shutil.rmtree(seg, ignore_errors=True)
    shutil.rmtree(preview, ignore_errors=True)
    seg.mkdir(parents=True)
    preview.mkdir(parents=True)
    kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    areas = []
    for i, f in enumerate(frames):
        m = masks[i].astype(np.uint8) * 255
        if kernel is not None:
            m = cv2.dilate(m, kernel)
        x0, y0, x1, y1 = f["box"]
        cv2.imwrite(str(seg / f"{f['stem']}.png"), m[y0:y1, x0:x1])
        areas.append(float((m > 0).mean()))
        if i % preview_every == 0 or f["name"] in {p["frame"] for o in prompts["objects"] for p in o["prompts"]}:
            img = cv2.imread(str(src / "images" / f["name"]))
            over = img.copy()
            over[m > 0] = (0.4 * over[m > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.imwrite(str(preview / f"{f['stem']}.jpg"), over)

    (data / "selected_view.txt").write_text(prompts["view"])
    outputs_vol.commit()
    empty = sum(a == 0 for a in areas)
    print(f"[✓] {len(frames)} mask -> {seg} | ảnh không thấy vật: {empty} | diện tích mask TB: {100 * np.mean(areas):.1f}%")
    print(f"[✓] Ảnh kiểm tra overlay -> {preview}")
    print(f"[✓] View inpaint cho InFusion: {prompts['view']}")
    return {"frames": len(frames), "empty_masks": empty, "mean_area": float(np.mean(areas)), "view": prompts["view"]}


# -----------------------------------------------------------------------------
# 3. SAM 2.1: mask MỌI vật (kiểu Segment-and-Track-Anything mà README InFusion khuyên dùng)
#    Automatic mask generator trên các keyframe + video predictor bám vật qua toàn bộ ảnh (ID cố định).
# -----------------------------------------------------------------------------
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# Cấu hình mask dày đặc, chất lượng cao từ notebook chính thức automatic_mask_generator_example.ipynb của SAM2
AMG_CFG = dict(
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)
MASKS_ROOT = "/workspace/Safe-GS/outputs/lerf_masks"


def _palette(n: int):
    import colorsys

    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.75, 1.0))[::-1] for i in range(n + 1)]


@app.function(image=sam2_image, gpu="L4", memory=32768, volumes=VOLUMES_MAPPING, timeout=3600 * 3)
def lerf_sam2_all_objects(
    scene: str = "teatime",
    num_keyframes: int = 6,
    max_new_per_keyframe: int = 30,
    min_area: float = 0.001,
    max_area: float = 0.6,
    new_object_max_overlap: float = 0.3,
):
    """
    Mask tất cả các vật của scene với ID cố định qua mọi ảnh:
      1. Automatic mask generator (SAM 2.1 large, cấu hình dày đặc) trên keyframe giữa video -> vật ban đầu
      2. Video predictor bám các vật đó qua toàn bộ ảnh (xuôi + ngược)
      3. Ở các keyframe còn lại: mask nào chưa được bám (overlap < new_object_max_overlap) là vật mới -> bám tiếp
    Kết quả: outputs/lerf_masks/<scene>/{labels/*.png, vis/*.jpg, overview.jpg, objects.json, sam2_config.json}
    """
    import cv2
    import numpy as np
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    dataset_vol.reload()
    src = _scene_src(scene)
    frames = _frames(scene)
    T = len(frames)
    tmp = Path("/tmp/sam2_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for i, f in enumerate(frames):
        shutil.copy(src / "images" / f["name"], tmp / f"{i:05d}.jpg")
    H, W = cv2.imread(str(tmp / "00000.jpg")).shape[:2]
    HW = H * W

    # Keyframe giữa trước (thường thấy nhiều vật nhất), sau đó các keyframe khác
    kfs = sorted(set(np.linspace(0, T - 1, num_keyframes).round().astype(int).tolist()))
    mid = kfs[len(kfs) // 2]
    kfs = [mid] + [k for k in kfs if k != mid]
    print(f"[i] {T} ảnh {W}x{H} | keyframes: {[frames[k]['name'] for k in kfs]}")

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # --- 1. Automatic masks trên các keyframe ---
    sam = build_sam2(SAM2_CFG, os.environ["SAM2_CHECKPOINT"], device="cuda", apply_postprocessing=False)
    amg = SAM2AutomaticMaskGenerator(sam, **AMG_CFG)
    kf_masks = {}
    with torch.inference_mode():
        for k in kfs:
            img = cv2.cvtColor(cv2.imread(str(tmp / f"{k:05d}.jpg")), cv2.COLOR_BGR2RGB)
            anns = [a for a in amg.generate(img) if min_area <= a["area"] / HW <= max_area]
            anns.sort(key=lambda a: a["predicted_iou"] * a["stability_score"], reverse=True)
            kept = []
            for a in anns:  # bỏ mask gần trùng nhau (IoU > 0.8)
                m = a["segmentation"]
                if all((m & k2).sum() / (m | k2).sum() < 0.8 for k2 in kept):
                    kept.append(m)
            kf_masks[k] = kept
            print(f"[i] AMG {frames[k]['name']}: {len(anns)} mask hợp lệ -> {len(kept)} sau khi lọc trùng")
    del amg, sam
    torch.cuda.empty_cache()

    # --- 2+3. Bám vật qua video, mỗi nhóm vật mới bắt đầu từ keyframe của nó ---
    predictor = build_sam2_video_predictor(SAM2_CFG, os.environ["SAM2_CHECKPOINT"], device="cuda")
    tracks: dict[int, np.ndarray] = {}  # obj_id -> (T, HW/8) packbits
    seed_frame: dict[int, int] = {}
    union = np.zeros((T, (HW + 7) // 8), dtype=np.uint8)
    next_id = 1
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(tmp), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for k in kfs:
            covered = np.unpackbits(union[k])[:HW].reshape(H, W).astype(bool)
            new = [m for m in kf_masks[k] if (m & covered).sum() / m.sum() < new_object_max_overlap]
            new = sorted(new, key=lambda m: m.sum(), reverse=True)[:max_new_per_keyframe]
            if not new:
                print(f"[i] {frames[k]['name']}: không có vật mới")
                continue
            predictor.reset_state(state)
            ids = list(range(next_id, next_id + len(new)))
            next_id += len(new)
            for oid, m in zip(ids, new):
                predictor.add_new_mask(state, frame_idx=k, obj_id=oid, mask=m)
                tracks[oid] = np.zeros((T, (HW + 7) // 8), dtype=np.uint8)
                seed_frame[oid] = k
            print(f"[*] {frames[k]['name']}: bám {len(new)} vật mới (ID {ids[0]}-{ids[-1]}) qua {T} ảnh...")
            for reverse in (False, True):
                for fidx, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=k, reverse=reverse):
                    ms = (logits > 0.0)[:, 0].cpu().numpy()
                    for oid, m in zip(obj_ids, ms):
                        packed = np.packbits(m.reshape(-1))
                        tracks[oid][fidx] = packed
                        union[fidx] |= packed

    # --- Label map: vật nhỏ vẽ sau (nằm trên) vật lớn, ví dụ bánh nằm trên đĩa ---
    out = Path(MASKS_ROOT) / scene
    shutil.rmtree(out, ignore_errors=True)
    (out / "labels").mkdir(parents=True)
    (out / "vis").mkdir()
    ids = sorted(tracks)
    colors = _palette(max(ids) if ids else 0)
    dtype = np.uint8 if len(ids) < 255 else np.uint16
    stats = {oid: {"frames": 0, "area_sum": 0.0, "max_area": 0.0, "best_frame": None} for oid in ids}
    overview = []
    for fidx, f in enumerate(frames):
        masks = {oid: np.unpackbits(tracks[oid][fidx])[:HW].reshape(H, W).astype(bool) for oid in ids}
        masks = {oid: m for oid, m in masks.items() if m.sum() >= 50}
        label = np.zeros((H, W), dtype=dtype)
        for oid in sorted(masks, key=lambda o: masks[o].sum(), reverse=True):
            label[masks[oid]] = oid
        cv2.imwrite(str(out / "labels" / f"{f['stem']}.png"), label)

        vis = cv2.imread(str(tmp / f"{fidx:05d}.jpg"))
        for oid in masks:
            sel = label == oid
            vis[sel] = (0.45 * vis[sel] + 0.55 * np.array(colors[oid])).astype(np.uint8)
        for oid in masks:
            m = (label == oid).astype(np.uint8)
            area = m.sum() / HW
            if area == 0:
                continue
            s = stats[oid]
            s["frames"] += 1
            s["area_sum"] += area
            if area > s["max_area"]:
                s["max_area"], s["best_frame"] = area, f["stem"]
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
            if m.sum() >= 150:  # số ID đặt ở điểm "sâu" nhất trong mask
                dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
                y, x = np.unravel_index(np.argmax(dist), dist.shape)
                txt = str(oid)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                org = (int(x - tw / 2), int(y + th / 2))
                cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f["stem"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, f["stem"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out / "vis" / f"{f['stem']}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if fidx in kfs:
            overview.append((fidx, vis))

    # Ảnh tổng quan: các keyframe theo thứ tự thời gian, 3 cột
    tiles = [cv2.resize(v, (W // 2, H // 2)) for _, v in sorted(overview, key=lambda x: x[0])]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out / "overview.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    objects = {
        str(oid): {
            "seed_frame": frames[seed_frame[oid]]["stem"],
            "frames_visible": s["frames"],
            "mean_area_pct": round(100 * s["area_sum"] / max(s["frames"], 1), 3),
            "max_area_pct": round(100 * s["max_area"], 3),
            "best_frame": s["best_frame"],
        }
        for oid, s in stats.items() if s["frames"] > 0
    }
    (out / "objects.json").write_text(json.dumps(objects, indent=2))
    (out / "sam2_config.json").write_text(json.dumps({
        "model": "sam2.1_hiera_large", "config": SAM2_CFG, "automatic_mask_generator": AMG_CFG,
        "keyframes": [frames[k]["stem"] for k in kfs], "min_area": min_area, "max_area": max_area,
        "max_new_per_keyframe": max_new_per_keyframe, "new_object_max_overlap": new_object_max_overlap,
    }, indent=2))
    outputs_vol.commit()
    print(f"[✓] {len(objects)} vật, {T} label map -> {out}")
    return {"objects": len(objects), "frames": T}


@app.function(image=sam2_image, volumes=VOLUMES_MAPPING, timeout=1800)
def lerf_masks_from_ids(scene: str = "teatime", ids: str = "", view: str = "", dilate: int = 5, out_scene: str = ""):
    """
    Tạo data/seg cho InFusion từ các ID vật cần xóa (xem mask/vis), ví dụ --ids 5,12.
    --out-scene teatime_v2: tạo một lần chạy InFusion riêng (dùng chung ảnh crop + COLMAP của <scene>),
    không đụng tới kết quả cũ của <scene>; sau đó chạy modal_infusion.py với --scene teatime_v2.
    """
    import cv2
    import numpy as np

    outputs_vol.reload()
    dataset_vol.reload()
    sel = [int(x) for x in ids.replace(" ", "").split(",") if x]
    if not sel:
        raise ValueError("Cần --ids, ví dụ --ids 5,12 (xem số ID trong mask/vis/*.jpg)")
    labels_dir = Path(MASKS_ROOT) / scene / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Chưa có {labels_dir}. Chạy lerf_sam2_all_objects --scene {scene} trước.")
    frames = _frames(scene)
    data = Path(OUTPUTS_ROOT) / (out_scene or scene) / "data"
    if out_scene and out_scene != scene:
        src_data = Path(OUTPUTS_ROOT) / scene / "data"
        for sub in ["images", "sparse"]:
            if not (data / sub).exists():
                shutil.copytree(src_data / sub, data / sub)
        (data / "sparse" / "0" / "points3D.ply").unlink(missing_ok=True)
        print(f"[i] Lần chạy mới '{out_scene}': dùng ảnh + COLMAP của '{scene}' -> {data}")
    seg = data / "seg"
    shutil.rmtree(seg, ignore_errors=True)
    seg.mkdir(parents=True)
    kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    best, best_area, empty = None, -1.0, 0
    for f in frames:
        label = cv2.imread(str(labels_dir / f"{f['stem']}.png"), cv2.IMREAD_UNCHANGED)
        m = np.isin(label, sel).astype(np.uint8) * 255
        if kernel is not None:
            m = cv2.dilate(m, kernel)
        x0, y0, x1, y1 = f["box"]
        crop = m[y0:y1, x0:x1]
        cv2.imwrite(str(seg / f"{f['stem']}.png"), crop)
        area = float((crop > 0).mean())
        empty += area == 0
        # View inpaint mặc định: ảnh thấy vật rõ nhất mà vật không chạm mép ảnh
        ys, xs = np.nonzero(crop)
        touches_border = len(xs) > 0 and (xs.min() == 0 or ys.min() == 0 or xs.max() == crop.shape[1] - 1 or ys.max() == crop.shape[0] - 1)
        if area > best_area and not touches_border:
            best, best_area = f["stem"], area
    chosen = view or best or frames[len(frames) // 2]["stem"]
    (data / "selected_view.txt").write_text(chosen)
    (data / "selected_ids.txt").write_text(",".join(map(str, sel)))
    outputs_vol.commit()
    print(f"[✓] seg cho ID {sel}: {len(frames)} ảnh ({empty} ảnh không thấy vật) -> {seg}")
    print(f"[✓] View inpaint cho InFusion: {chosen}")
    return {"view": chosen, "empty": int(empty)}
