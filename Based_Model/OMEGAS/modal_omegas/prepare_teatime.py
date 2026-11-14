"""
Step 0: LERF teatime (nerfstudio transforms.json + SAM2 ID masks) -> dataset OMEGAS can read.

The OMEGAS loader (gaussian_splatting/scene_2D) only accepts COLMAP `sparse/0` with PINHOLE cameras and ignores
the principal point, and reads the ID masks from `object_mask/<image>.png`. This script:
  1. crops every image and its mask with the same box so the principal point is at the image centre
     (pure crop, no resampling: mask IDs stay exact)                       -> images/*.png, object_mask/*.png
  2. rotates the world so that "up" (mean camera up vector) becomes -y, like a COLMAP reconstruction
     (DreamGaussian's orbit cameras in Step 5 assume this; 2DGS itself is rotation invariant)
  3. writes the known poses as a COLMAP text model, extracts/matches SIFT and runs point_triangulator
     (no SfM, poses are kept)                                               -> sparse/0/{cameras,images,points3D}.bin
Same approach as F:/Safe-GS/modal_lerf_prep.py::lerf_colmap.

Usage: python prepare_teatime.py --raw /vol/raw/teatime --out /vol/data/teatime --report <json> [--threads 16]
"""
import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def run(cmd, log=None):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(p.stdout[-20000:])
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n$ {cmd}\n{p.stdout}\n")
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.stdout


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    return -q if q[0] < 0 else q


def rotation_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def frames_from_transforms(raw: Path):
    t = json.loads((raw / "transforms.json").read_text())
    # LERF has per-frame intrinsics -> one common crop size for all frames (the smallest centred box), so every
    # camera has the same W x H (DreamGaussian in Step 5 takes W, H, fx, fy of the first camera for its renders)
    half_w = min(math.floor(min(float(f["cx"]), int(f["w"]) - float(f["cx"]))) for f in t["frames"])
    half_h = min(math.floor(min(float(f["cy"]), int(f["h"]) - float(f["cy"]))) for f in t["frames"])
    frames = []
    for f in t["frames"]:
        w, h, cx, cy = int(f["w"]), int(f["h"]), float(f["cx"]), float(f["cy"])
        x0 = min(max(int(round(cx - half_w)), 0), w - 2 * half_w)
        y0 = min(max(int(round(cy - half_h)), 0), h - 2 * half_h)
        frames.append({
            "name": Path(f["file_path"]).name,
            "stem": Path(f["file_path"]).stem,
            "c2w": np.array(f["transform_matrix"], dtype=np.float64),
            "fx": float(f["fl_x"]), "fy": float(f["fl_y"]),
            "box": (x0, y0, x0 + 2 * half_w, y0 + 2 * half_h),
            "cx": cx - x0, "cy": cy - y0,
        })
    return sorted(frames, key=lambda x: x["name"]), {k: v for k, v in t.items() if k != "frames"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max_features", type=int, default=8192)
    args = ap.parse_args()
    t0 = time.time()
    raw, out = Path(args.raw), Path(args.out)
    log = str(Path(args.report).with_suffix(".log"))
    frames, meta = frames_from_transforms(raw)
    print(f"[step0] {len(frames)} frames, transforms meta: {meta}", flush=True)
    if "applied_transform" in meta or meta.get("camera_model") not in (None, "OPENCV", "PINHOLE"):
        print(f"[step0] note: transforms meta {meta}", flush=True)

    # ---- 1. crop images + masks with the same box
    img_out, msk_out = out / "images", out / "object_mask"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)
    sizes, missing_masks = set(), []
    for f in frames:
        im = Image.open(raw / "images" / f["name"]).convert("RGB")
        im.crop(f["box"]).save(img_out / f"{f['stem']}.png")
        mp = raw / "mask" / "labels" / f"{f['stem']}.png"
        if mp.exists():
            m = Image.open(mp)
            if m.size != im.size:
                raise RuntimeError(f"mask size {m.size} != image size {im.size} for {f['stem']}")
            ids = np.asarray(m.crop(f["box"]))
            if ids.ndim == 3:
                ids = ids[..., 0]
            # run_gaussian_2D.py reads the mask as H x W x 3 (gt_obj[:, :, 0], like DEVA's gray masks saved as RGB)
            Image.fromarray(np.repeat(ids[..., None].astype(np.uint8), 3, axis=2)).save(msk_out / f"{f['stem']}.png")
        else:
            missing_masks.append(f["stem"])
        sizes.add((f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]))
    print(f"[step0] cropped sizes: {sorted(sizes)}; frames without mask: {len(missing_masks)}", flush=True)

    # ---- 2. world alignment: mean camera up (OpenGL c2w column 1) -> -y
    ups = np.stack([f["c2w"][:3, 1] for f in frames])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    R_align = rotation_a_to_b(up, np.array([0.0, -1.0, 0.0]))
    spread = float(np.degrees(np.arccos(np.clip(ups @ up, -1, 1))).mean())
    print(f"[step0] mean camera up {up.round(4)} (mean deviation {spread:.1f} deg) -> rotate to -y", flush=True)
    A = np.eye(4)
    A[:3, :3] = R_align

    # ---- 3. COLMAP: features + matches (cached), known poses, triangulation
    work = Path("/tmp/colmap")
    shutil.rmtree(work, ignore_errors=True)
    (work / "known").mkdir(parents=True)
    db = work / "database.db"
    db_cache = out / "colmap_cache" / "database.db"
    if db_cache.exists():
        print(f"[step0] reuse cached features/matches {db_cache}", flush=True)
        shutil.copy(db_cache, db)
    else:
        run(f"colmap feature_extractor --database_path {db} --image_path {img_out} "
            f"--ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 0 "
            f"--SiftExtraction.max_num_features {args.max_features} --SiftExtraction.num_threads {args.threads}", log)
        run(f"colmap exhaustive_matcher --database_path {db} --SiftMatching.use_gpu 0 "
            f"--SiftMatching.num_threads {args.threads}", log)
        db_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(db, db_cache)

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
        c2w = A @ f["c2w"]
        c2w[:3, 1:3] *= -1                       # OpenGL -> OpenCV camera axes
        w2c = np.linalg.inv(c2w)
        q = rotmat2qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        img_lines.append(f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {image_id} {name}\n\n")
    con.commit()
    con.close()
    (work / "known" / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (work / "known" / "images.txt").write_text("".join(img_lines))
    (work / "known" / "points3D.txt").write_text("")
    (work / "sparse").mkdir()
    run(f"colmap point_triangulator --database_path {db} --image_path {img_out} "
        f"--input_path {work / 'known'} --output_path {work / 'sparse'}", log)
    analysis = run(f"colmap model_analyzer --path {work / 'sparse'}", log)

    dst = out / "sparse" / "0"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
        shutil.copy(work / "sparse" / fn, dst / fn)
    (work / "txt").mkdir()
    run(f"colmap model_converter --input_path {dst} --output_path {work / 'txt'} --output_type TXT", log)
    n_points = sum(1 for line in open(work / "txt" / "points3D.txt") if line.strip() and not line.startswith("#"))
    # images.txt: header line "<id> qw qx qy qz tx ty tz <cam> <name>" + POINTS2D line (empty if no keypoints)
    n_images = sum(1 for line in open(work / "txt" / "images.txt")
                   if not line.startswith("#") and line.strip().endswith(".png"))

    report = {
        "frames": len(frames), "registered_images": n_images, "points3D": n_points,
        "crop_sizes": sorted(sizes), "frames_without_mask": missing_masks,
        "mean_camera_up_original": up.tolist(), "up_deviation_deg": spread, "R_align": R_align.tolist(),
        "transforms_meta": meta, "colmap_model_analyzer": analysis[-2000:], "seconds": round(time.time() - t0, 1),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    ok = n_images == len(frames) and n_points > 10000 and len(sizes) == 1 and not missing_masks
    print(json.dumps({k: report[k] for k in ("frames", "registered_images", "points3D", "crop_sizes")}), flush=True)
    if not ok:
        raise SystemExit("[step0] checks failed (need all images registered, >10k points, one crop size, all masks)")
    print("[step0] OK", flush=True)


if __name__ == "__main__":
    main()
