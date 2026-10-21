"""
Modal runner cho Split&Splat giai đoạn 2+ (modal_split_splat.py)
================================================================
Dùng lại kết quả giai đoạn 1 (mask + lan truyền) đã có trong Volume 'safe-gs-outputs':
    Replica/split/split_and_splat/office0/{03_propagation,04_final}
    Replica/split/split_and_splat/_work/office0/data/office0   (images/depth/pose/intrinsic/ply)

KHÔNG sửa code trong Based_Model/Split_and_Splat. Code được copy vào container và chạy nguyên bản
(utils/scannet2blender.py, bash_dir_utils/prepare_folder.sh, run_all.sh, ...). File này chỉ làm:
chuẩn bị thư mục data/, ghi log tiến độ, lưu kết quả về Volume.

Kết quả ghi vào:  Replica/split/split_and_splat/<scene>/05_gaussians/
    raw/<id>/point_cloud/iteration_{500,1000}/point_cloud.ply   Gaussian của từng instance
    progress.csv                                                 thời gian từng instance
    logs/train_2a.log                                            log đầy đủ

Tự ngắt: hàm kết thúc là container tắt (không tính tiền thêm). Có timeout cứng làm trần chi phí.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import modal

app = modal.App("safe-gs-split-splat")
outputs_vol = modal.Volume.from_name("safe-gs-outputs", create_if_missing=True)

REPO = Path(__file__).resolve().parent
SS = REPO / "Based_Model" / "Split_and_Splat"
BUILD_IGNORE = ["**/__pycache__", "**/*.pyc", "**/*.so", "**/build", "**/*.egg-info"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libgomp1", "ffmpeg")
    .pip_install("setuptools", "wheel", "ninja")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "numpy==1.26.4", "scipy==1.15.2", "pandas==2.2.3", "scikit-learn==1.6.1", "scikit-image",
        "opencv-python-headless", "pillow", "matplotlib", "plyfile", "tqdm", "open3d==0.19.0",
        "pycolmap==3.11.1", "ipython", "ipywidgets", "hydra-core", "iopath", "omegaconf",
        "tensorboard", "imageio", "imageio-ffmpeg", "einops",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.6;8.9", "CUDA_HOME": "/usr/local/cuda", "FORCE_CUDA": "1",
        # python của Modal mặc định link bằng clang++ (không có trong image) -> ép dùng g++
        "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared",
    })
    .add_local_dir(SS / "submodules", "/opt/ss/submodules", copy=True, ignore=BUILD_IGNORE)
    .add_local_dir(SS / "point_projection", "/opt/ss/point_projection", copy=True, ignore=BUILD_IGNORE)
    .run_commands(
        "cd /opt/ss/submodules/diff-gaussian-rasterization && pip install . --no-build-isolation",
        "cd /opt/ss/submodules/simple-knn && pip install . --no-build-isolation",
        "cd /opt/ss/submodules/fused-ssim && pip install . --no-build-isolation",
        "cd /opt/ss/point_projection && pip install . --no-build-isolation",
    )
    .add_local_dir(
        SS, "/workspace/ss",
        ignore=BUILD_IGNORE + ["SIBR_viewers", "submodules", "point_projection", "split_&_splat_GA.png"],
    )
)

VOL = "/vol"
BASE = f"{VOL}/Replica/split/split_and_splat"
WORK = "/work/ss"


def _sh(cmd: str, cwd: str, on_line=None) -> int:
    """Chạy lệnh bash, stream log. Trả về exit code."""
    p = subprocess.Popen(
        cmd, shell=True, cwd=cwd, executable="/bin/bash",
        env={**os.environ, "PYTHONPATH": f"{WORK}:{os.environ.get('PYTHONPATH', '')}", "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in p.stdout:
        line = line.rstrip("\n")
        if on_line:
            on_line(line)
        else:
            print(line, flush=True)
    p.wait()
    return p.returncode


def _prepare(scene: str, ids: list[str]) -> list[str]:
    """Dựng /work/ss/data/<scene>/ từ kết quả giai đoạn 1, bằng chính script gốc của repo."""
    outputs_vol.reload()
    src_data = Path(BASE) / "_work" / scene / "data" / scene
    final_masks = Path(BASE) / scene / "04_final" / "masks"
    clusters = Path(BASE) / scene / "03_propagation" / "clusters"

    shutil.rmtree("/work", ignore_errors=True)
    shutil.copytree("/workspace/ss", WORK)
    # repo clone trên Windows -> .sh có CRLF, bash không chạy được. Chỉ sửa bản copy trong container.
    for sh in Path(WORK).rglob("*.sh"):
        sh.write_bytes(sh.read_bytes().replace(b"\r\n", b"\n"))
        sh.chmod(0o755)  # clone trên Windows làm mất quyền thực thi
    data = Path(WORK) / "data" / scene
    data.mkdir(parents=True)

    for sub in ("images", "depth", "pose", "intrinsic"):
        shutil.copytree(src_data / sub, data / sub)
    shutil.copy2(src_data / f"{scene}_vh_clean_2.ply", data)

    # scannet2blender.py đọc intrinsic/<frame>.txt theo từng frame; nếu thiếu nó dùng thông số ScanNet
    # (fx=577 ...) - sai với Replica (fx=600). Cấp file intrinsic cho từng frame thay vì sửa code.
    k = (data / "intrinsic" / "intrinsic_color.txt").read_text()
    for img in sorted((data / "images").glob("*.JPEG")):
        (data / "intrinsic" / f"{img.stem}.txt").write_text(k)

    print("[prep] scannet2blender.py (code gốc) -> transforms_train.json", flush=True)
    assert _sh("python utils/scannet2blender.py -dataset_path ./data > /dev/null", WORK) == 0
    import json
    n_fr = len(json.loads((data / "transforms_train.json").read_text())["frames"])
    print(f"[prep] transforms_train.json: {n_fr} frame", flush=True)

    all_ids = sorted(p.name for p in final_masks.iterdir() if p.is_dir())
    chosen = ids or all_ids
    chosen = sorted(chosen)  # cùng thứ tự glob của run_all.sh
    for i in chosen:
        dst = data / "masks" / i
        shutil.copytree(final_masks / i, dst)
        shutil.copy2(clusters / f"label_{i}.ply", dst / f"label_{i}.ply")
    print(f"[prep] {len(chosen)}/{len(all_ids)} instance: {chosen[:12]}{' ...' if len(chosen) > 12 else ''}", flush=True)

    print("[prep] prepare_folder.sh (code gốc)", flush=True)
    rc = _sh(f"bash ./bash_dir_utils/prepare_folder.sh {scene} > /work/prepare.log 2>&1", WORK)
    if rc != 0:
        print(f"[prep] prepare_folder.sh exit={rc}. Log:\n" + Path("/work/prepare.log").read_text()[-3000:], flush=True)
        raise RuntimeError("prepare_folder.sh failed")
    kept = sorted(p.name for p in (data / "masks").iterdir() if p.is_dir())
    dropped = sorted(p.name for p in (data / "discard").iterdir()) if (data / "discard").exists() else []
    print(f"[prep] sau prepare_folder: giữ {len(kept)}, discard {len(dropped)} {dropped}", flush=True)
    return kept


@app.function(image=image, timeout=900)
def check_env():
    """Kiểm tra import (CPU, vài cent) trước khi tốn GPU."""
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    for m in ("diff_gaussian_rasterization", "simple_knn._C", "fused_ssim", "point_projection_cuda",
              "open3d", "pycolmap", "plyfile", "hydra", "pandas", "skimage"):
        try:
            __import__(m)
            print(f"  [OK]  {m}")
        except Exception as e:
            print(f"  [ERR] {m}: {e}")
    sys.path.insert(0, "/workspace/ss")
    os.chdir("/workspace/ss")
    for m in ("scene", "gaussian_renderer", "arguments", "utils.loss_utils"):
        try:
            __import__(m)
            print(f"  [OK]  {m}")
        except Exception as e:
            print(f"  [ERR] {m}: {e}")


@app.function(image=image, volumes={VOL: outputs_vol}, timeout=1800, cpu=4, memory=16384)
def prep_only(scene: str = "office0", limit: int = 3):
    """Chỉ chạy bước chuẩn bị dữ liệu bằng CPU (không tốn GPU) để kiểm tra trước khi train."""
    all_ids = sorted(p.name for p in (Path(BASE) / scene / "04_final" / "masks").iterdir() if p.is_dir())
    kept = _prepare(scene, sorted(all_ids)[:limit] if limit else [])
    d = Path(WORK) / "data" / scene / "masks"
    for i in kept[:3]:
        print(f"  masks/{i}: {sorted(p.name for p in (d / i).iterdir())}  "
              f"n_images={len(list((d / i / 'images').glob('*')))} n_masks={len(list((d / i / 'masks').glob('*')))}")


@app.function(
    image=image, gpu="L4", volumes={VOL: outputs_vol},
    timeout=6 * 3600, cpu=4, memory=16384,
)
def train_instances(scene: str = "office0", limit: int = 3, ids: str = ""):
    """Giai đoạn 2a: run_all.sh (gốc) - train 1000 iter / instance. limit=0 -> tất cả."""
    t_start = time.time()
    id_list = [s for s in ids.split(",") if s] or []
    if not id_list and limit:
        all_ids = sorted(p.name for p in (Path(BASE) / scene / "04_final" / "masks").iterdir() if p.is_dir())
        id_list = sorted(all_ids)[:limit]
    kept = _prepare(scene, id_list)

    out_root = Path(BASE) / scene / "05_gaussians"
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    log_f = open("/work/train_2a.log", "w")
    rows: list[dict] = []
    cur = {"id": None, "t0": 0.0}
    n_done = [0]

    def save_instance():
        if cur["id"] is None:
            return
        i = cur["id"]
        src = Path(WORK) / "output" / scene / "raw" / i
        status = "ok" if (src / "point_cloud" / "iteration_1000" / "point_cloud.ply").exists() else "FAILED"
        sec = time.time() - cur["t0"]
        if src.exists():
            shutil.copytree(src, out_root / "raw" / i, dirs_exist_ok=True)
        rows.append({"id": i, "seconds": round(sec, 1), "status": status})
        with open(out_root / "progress.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "seconds", "status"])
            w.writeheader()
            w.writerows(rows)
        shutil.copy2("/work/train_2a.log", out_root / "logs" / "train_2a.log")
        n_done[0] += 1
        total_el = time.time() - t_start
        avg = sum(r["seconds"] for r in rows) / len(rows)
        print(f"==> [{n_done[0]}/{len(kept)}] instance {i}: {status}, {sec:.0f}s | "
              f"TB/instance {avg:.0f}s | còn lại ~{avg * (len(kept) - n_done[0]) / 60:.0f} phút | "
              f"đã chạy {total_el / 60:.1f} phút", flush=True)
        if n_done[0] % 5 == 0:
            outputs_vol.commit()

    def on_line(line: str):
        log_f.write(line + "\n")
        m = re.search(r"Running for folder: (\S+)", line)
        if m:
            save_instance()
            cur["id"], cur["t0"] = m.group(1), time.time()
            print(f"\n==> BẮT ĐẦU instance {cur['id']} ({n_done[0] + 1}/{len(kept)})", flush=True)
            return
        m = re.search(r"(\d+)/(\d+) \[", line)  # tqdm: chỉ in 4 mốc / thanh để log gọn
        if m:
            c, t = int(m.group(1)), int(m.group(2))
            if t > 1 and (c == t or c % max(t // 4, 1) == 0):
                print(f"    [{cur['id']}] {line.strip()[:150]}", flush=True)
            return
        print(line, flush=True)

    print(f"\n[2a] bash run_all.sh {scene}  (code gốc, {len(kept)} instance)\n", flush=True)
    rc = _sh(f"bash ./run_all.sh {scene}", WORK, on_line)
    save_instance()
    log_f.close()
    shutil.copy2("/work/train_2a.log", out_root / "logs" / "train_2a.log")
    outputs_vol.commit()

    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n[XONG] exit={rc} | {ok}/{len(rows)} instance OK | tổng {(time.time() - t_start) / 60:.1f} phút")
    print(f"Kết quả: safe-gs-outputs:/Replica/split/split_and_splat/{scene}/05_gaussians")
    print("Container tự tắt sau dòng này.")
    if rc != 0:
        raise RuntimeError(f"run_all.sh exit {rc}")


# =============================================================================
# FULL PIPELINE: 2a (train từng instance) -> 2b (tinh chỉnh mask) -> 2c (train lại) -> 3 (ghép)
# =============================================================================
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SED_MASK_LOSS = r"sed -i 's/Ll1_mask \* [0-9.]*/Ll1_mask * {w}/' train.py"


class Runner:
    """Chạy lần lượt các giai đoạn, ghi log/tiến độ/trạng thái ra Volume sau từng bước."""

    def __init__(self, scene: str):
        self.scene = scene
        self.out = Path(BASE) / scene / "05_gaussians"
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.stages: dict[str, dict] = {}
        self.log = open("/work/pipeline.log", "a")

    def status(self, name: str, state: str, note: str = "", seconds: float | None = None):
        self.stages[name] = {"state": state, "note": note,
                             **({"seconds": round(seconds, 1)} if seconds is not None else {})}
        (self.out / "status.json").write_text(json.dumps(
            {"elapsed_min": round((time.time() - self.t0) / 60, 1), "stages": self.stages},
            ensure_ascii=False, indent=2))
        self.log.flush()
        shutil.copy2("/work/pipeline.log", self.out / "logs" / "pipeline.log")
        outputs_vol.commit()

    def run(self, name: str, cmd: str, marker: str, total: int, save_sub: str | None = None,
            ok_rel: str | None = None):
        """Chạy 1 lệnh bash dài; theo dõi từng instance qua dòng `marker`. Trả về (exit_code, rows)."""
        rows: list[dict] = []
        cur = {"id": None, "t0": 0.0}
        t_stage = time.time()

        def fin():
            if cur["id"] is None:
                return
            i, sec = cur["id"], time.time() - cur["t0"]
            ok = True
            if save_sub:
                src = Path(WORK) / "output" / self.scene / save_sub / i
                ok = (src / ok_rel).exists() if ok_rel else src.exists()
                if src.exists():
                    shutil.copytree(src, self.out / save_sub / i, dirs_exist_ok=True)
            rows.append({"id": i, "seconds": round(sec, 1), "status": "ok" if ok else "FAILED"})
            with open(self.out / f"progress_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "seconds", "status"])
                w.writeheader()
                w.writerows(rows)
            avg = sum(r["seconds"] for r in rows) / len(rows)
            print(f"==> [{name}] {len(rows)}/{total} instance {i}: {'ok' if ok else 'FAILED'}, {sec:.0f}s | "
                  f"TB {avg:.0f}s | còn ~{avg * max(total - len(rows), 0) / 60:.0f} phút | "
                  f"stage {(time.time() - t_stage) / 60:.1f} phút | tổng {(time.time() - self.t0) / 60:.1f} phút",
                  flush=True)
            if len(rows) % 5 == 0:
                self.status(name, "running", f"{len(rows)}/{total}")

        def on_line(line: str):
            self.log.write(line + "\n")
            m = re.search(marker, line)
            if m:
                fin()
                cur["id"], cur["t0"] = m.group(1), time.time()
                print(f"\n==> [{name}] BẮT ĐẦU {cur['id']} ({len(rows) + 1}/{total})", flush=True)
                return
            m = re.search(r"(\d+)/(\d+) \[", line)  # tqdm: chỉ in 4 mốc / thanh cho gọn
            if m:
                c, t = int(m.group(1)), int(m.group(2))
                if t > 1 and (c == t or c % max(t // 4, 1) == 0):
                    print(f"    [{cur['id']}] {line.strip()[:150]}", flush=True)
                return
            print(line, flush=True)

        self.status(name, "running", f"0/{total}")
        print(f"\n######## STAGE {name}: {cmd}\n", flush=True)
        rc = _sh(cmd, WORK, on_line)
        fin()
        self.log.flush()
        return rc, rows


def _ids_with_ply(scene: str, sub: str, it: int = 1000) -> list[str]:
    root = Path(WORK) / "output" / scene / sub
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if (p / "point_cloud" / f"iteration_{it}" / "point_cloud.ply").exists())


def _ensure_sam_ckpt():
    p = Path(WORK) / "checkpoints" / "sam2.1_hiera_large.pt"
    p.parent.mkdir(exist_ok=True)
    for k in range(3):
        try:
            if not p.exists():
                print(f"[2b] tải SAM2 checkpoint (lần {k + 1}) ...", flush=True)
                urllib.request.urlretrieve(SAM_URL, p)
            return
        except Exception as e:
            print(f"[2b] tải lỗi: {e}", flush=True)
            p.unlink(missing_ok=True)
    raise RuntimeError("không tải được SAM2 checkpoint")


@app.function(
    image=image, gpu="L4", volumes={VOL: outputs_vol},
    timeout=20 * 3600, cpu=4, memory=16384,   # trần chi phí: 20h x $0.80 ~ $16
)
def full_pipeline(scene: str = "office0", ids: str = "", stages: str = "2a,2b,2c,3"):
    """Chạy toàn bộ giai đoạn 2-3 của Split&Splat. ids rỗng = tất cả instance. Xong là container tự tắt."""
    id_list = sorted(s for s in ids.split(",") if s)
    kept = _prepare(scene, id_list)
    R = Runner(scene)
    run_stages = [s.strip() for s in stages.split(",")]
    n = len(kept)
    ss = Path(WORK)

    def timed(name, fn):
        t = time.time()
        try:
            R.status(name, "running")
            note = fn() or ""
            R.status(name, "done", note, time.time() - t)
        except Exception as e:  # lỗi 1 giai đoạn không làm mất kết quả các giai đoạn trước
            print(f"[{name}] LỖI: {type(e).__name__}: {e}", flush=True)
            R.status(name, "FAILED", f"{type(e).__name__}: {e}", time.time() - t)

    # ---- 2a: train từng instance (run_all.sh gốc) ----
    def s2a():
        rc, rows = R.run("2a", f"bash ./run_all.sh {scene}", r"Running for folder: (\S+)", n,
                         "raw", "point_cloud/iteration_1000/point_cloud.ply")
        return f"exit={rc}, {sum(r['status'] == 'ok' for r in rows)}/{len(rows)} ok"

    # ---- 2b: tinh chỉnh mask bằng SAM2 (run_ref.sh gốc) + gộp masks_extra -> masks ----
    def s2b():
        _ensure_sam_ckpt()
        rc, rows = R.run("2b", f"bash ./run_ref.sh {scene}", r"Processing: (\S+)", n)
        # move_extra.sh gốc tìm thư mục 'mask_extra'/'mask' (README gọi ở sai cấp), trong khi
        # mask_optimizer_scannet.py ghi 'masks_extra' -> làm thủ công đúng ý đồ: masks_extra/* -> masks/
        moved = 0
        for inst in (ss / "data" / scene / "masks").iterdir():
            extra = inst / "masks_extra"
            if extra.is_dir():
                for f in extra.iterdir():
                    shutil.move(str(f), str(inst / "masks" / f.name))
                    moved += 1
                extra.rmdir()
        return f"exit={rc}, {len(rows)} instance, thêm/ghi đè {moved} mask"

    # ---- 2c: train lại với mask đã tinh chỉnh (run_all_ref.sh gốc) ----
    def s2c():
        rc, rows = R.run("2c", f"bash ./run_all_ref.sh {scene}", r"Running for folder: (\S+)", n,
                         "ref", "point_cloud/iteration_1000/point_cloud.ply")
        return f"exit={rc}, {sum(r['status'] == 'ok' for r in rows)}/{len(rows)} ok"

    # ---- 3: ghép (chuỗi lệnh README, sửa 3 chỗ README lệch với code) ----
    def s3():
        # dự phòng: instance nào chưa có bản 'ref' thì dùng bản 'raw' để giai đoạn 3 vẫn chạy
        ref_ok = set(_ids_with_ply(scene, "ref"))
        fb = 0
        for i in _ids_with_ply(scene, "raw"):
            if i not in ref_ok:
                shutil.copytree(ss / "output" / scene / "raw" / i, ss / "output" / scene / "ref" / i,
                                dirs_exist_ok=True)
                fb += 1
        if fb:
            print(f"[3] {fb} instance không có bản tinh chỉnh -> dùng bản raw", flush=True)
        notes = [f"fallback raw->ref: {fb}"]

        def step(cmd, label):
            def echo(line):
                R.log.write(line + "\n")
                print(line, flush=True)
            rc = _sh(cmd, WORK, echo)
            notes.append(f"{label}: exit={rc}")
            return rc

        step(SED_MASK_LOSS.format(w=0.05), "mask_loss=0.05")
        step(f"python move_PLY.py --scene {scene} --output=PLY_ref --init", "move_PLY PLY_ref")
        step(f"python move_PLY.py --scene {scene} --output=tmp --init", "move_PLY tmp")
        masks_dir = ss / "data" / scene / "masks"
        # README: cp images/* masks/ ; nhưng mask_combination.py liệt kê masks/images/ -> tạo cả hai
        step(f"cp ./data/{scene}/images/* ./data/{scene}/masks/ && cp -r ./data/{scene}/images ./data/{scene}/masks/images "
             f"&& cp ./data/{scene}/transforms_train.json ./data/{scene}/masks/", "cp images+json")
        combined = masks_dir / "combined"
        combo_cmd = f"bash ./combo.sh ./data/{scene}/masks/combined ./output/{scene}/comb"
        ok_ply = "point_cloud/iteration_1000/point_cloud.ply"

        def fix_case():
            # combine_ply ghi 'points3D.ply' nhưng train.py đọc 'points3d.ply' (chạy được trên Windows/macOS
            # không phân biệt hoa thường, lỗi trên Linux) -> đổi tên file kết quả, không sửa code gốc
            n_fix = 0
            for d in (combined.iterdir() if combined.exists() else []):
                a, b = d / "points3D.ply", d / "points3d.ply"
                if d.is_dir() and a.exists() and not b.exists():
                    a.rename(b)
                    n_fix += 1
            print(f"[3] đổi tên points3D.ply -> points3d.ply: {n_fix} thư mục", flush=True)

        def save_stage3(tag):
            # lưu sau MỖI lượt để lỡ hết giờ/lỗi vẫn còn kết quả
            for sub in ("PLY_ref", "tmp", "comb"):
                src = ss / "output" / scene / sub
                if src.exists():
                    shutil.copytree(src, R.out / "stage3" / sub, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("events.out*"))
            # Repo không có bước nào sinh final.ply (README nhắc nhưng không có code) -> nối thô các nhóm
            from plyfile import PlyData, PlyElement
            import numpy as np
            # nhóm = instance đơn (ref/<id>) hoặc nhóm đã ghép (comb/<id1>_<id2>...). Lấy các nhóm "cao nhất":
            # không nằm trọn trong nhóm lớn hơn -> tránh nối trùng Gaussian của cùng một vật.
            groups: dict[frozenset, Path] = {}
            for sub in ("ref", "comb"):
                root = ss / "output" / scene / sub
                for d in (root.iterdir() if root.exists() else []):
                    p = d / "point_cloud" / "iteration_1000" / "point_cloud.ply"
                    if p.exists():
                        groups[frozenset(d.name.split("_"))] = p
            top = [g for g in groups if not any(g < h for h in groups)]
            try:
                if top:
                    verts = np.concatenate([PlyData.read(str(groups[g]))["vertex"].data for g in top])
                    PlyData([PlyElement.describe(verts, "vertex")]).write(str(R.out / "stage3" / "merged_by_concat.ply"))
                    sizes = sorted((len(g) for g in top), reverse=True)
                    notes.append(f"[{tag}] merged_by_concat.ply: {len(top)} nhóm cao nhất (số instance/nhóm: {sizes[:10]}), "
                                 f"{len(verts)} Gaussian (wrapper nối)")
            except Exception as e:  # không để lỗi phần nối làm dừng các lượt train
                notes.append(f"[{tag}] nối thô LỖI: {type(e).__name__}: {e}")
                print(notes[-1], flush=True)
            outputs_vol.commit()

        step(f"python ./utils_mask/mask_combination.py --scene {scene}", "mask_combination")
        fix_case()
        n_pairs = len([p for p in combined.iterdir() if p.is_dir()]) if combined.exists() else 0
        notes.append(f"cặp ghép lượt đầu: {n_pairs}")
        if n_pairs:
            rc, rows = R.run("3_combo", combo_cmd, r"Running for folder: (\S+)", n_pairs, "comb", ok_ply)
            notes.append(f"combo: exit={rc}, {len(rows)} cặp")
            save_stage3("combo")
            # 3 lượt của run_composition_pipeline.sh (mask_loss 0.05 -> 0.1 -> 0.25). Script gốc gọi './combo.sh'
            # (mất quyền thực thi do clone Windows) và không cho chen bước đổi tên -> chạy lại đúng 3 lệnh của nó.
            for k, w in enumerate((0.05, 0.1, 0.25), 1):
                if k > 1:
                    step(SED_MASK_LOSS.format(w=w), f"mask_loss={w}")
                tmp_dir = ss / "output" / scene / "tmp"
                before = len(list(tmp_dir.glob("*.ply")))
                step(f"python move_PLY.py --scene {scene} --output tmp", f"pass{k} move_PLY")
                step(f"python ./utils_mask/mask_combination.py --scene {scene}", f"pass{k} mask_combination")
                fix_case()
                n_dirs = len([p for p in combined.iterdir() if p.is_dir()])
                rc, rows = R.run(f"3_pass{k}", combo_cmd, r"Running for folder: (\S+)", n_dirs, "comb", ok_ply)
                notes.append(f"pass{k}(w={w}): exit={rc}, {len(rows)} thư mục train, nhóm {before}->{len(list(tmp_dir.glob('*.ply')))}")
                save_stage3(f"pass{k}")
        else:
            save_stage3("no-pairs")
        return " | ".join(notes)

    plan = {"2a": s2a, "2b": s2b, "2c": s2c, "3": s3}
    for name in ("2a", "2b", "2c", "3"):
        if name in run_stages:
            timed(name, plan[name])

    total = (time.time() - R.t0) / 60
    lines = [f"SPLIT&SPLAT {scene}: {n} instance, tổng {total:.1f} phút (~${total / 60 * 0.80:.1f} GPU L4)"]
    for k, v in R.stages.items():
        lines.append(f"  [{k}] {v['state']} {v.get('seconds', 0) / 60:.1f} phút - {v['note']}")
    (R.out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    R.status("END", "done", f"{total:.1f} phút")
    print("\n".join(lines))
    print("Container tự tắt sau dòng này.", flush=True)
