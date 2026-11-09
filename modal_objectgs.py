"""
ObjectGS (ICCV 2025, github.com/RuijieZhu94/ObjectGS @ 0f6842b) chạy nguyên code tác giả trên LERF-Mask teatime.

    python -m modal run --detach modal_objectgs.py::train       # ply_preprocessing.py (corr) + train.py (3D, config lerf_mask)
    python -m modal run modal_objectgs.py::status

Dữ liệu = bộ LERF-Mask teatime của Gaussian Grouping (images, sparse/0, object_mask DEVA), đúng nguồn README chỉ.
Khác README (ghi rõ để báo cáo):
  1. Env conda tạo đúng environment.yml (python 3.8, torch 2.1.2, CUDA 11.8, gsplat-object) trong image CUDA 11.8 devel.
  2. ply_preprocessing.py chạy với --output_ply_name points3D.ply (cờ có sẵn) vì train.py đọc sparse/0/points3D.ply;
     file points3D.ply gốc (không nhãn) giữ lại thành points3D_orig.ply.
  3. train.py thêm cờ có sẵn --checkpoint_iterations để chạy tiếp khi Modal ngắt.
Output trên Volume safe-gs-come: teatime/objectgs/ (datasets, outputs, _done, log).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import modal

import modal_come as mc

app = modal.App("safe-gs-objectgs")
vol, VOL = mc.vol, mc.VOL
REPO = Path(__file__).resolve().parent / "Based_Model" / "ObjectGS"
R = "/opt/ObjectGS"
ROOT = Path(VOL) / "teatime" / "objectgs"
LOG = Path(VOL) / "teatime" / "logs"
CONDA = "/opt/conda/bin/conda"
PY = f"{CONDA} run --no-capture-output -n objectgs python"

img = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "build-essential", "ninja-build", "libglm-dev", "libgl1", "libglib2.0-0")
    .run_commands("wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O /tmp/mc.sh "
                  "&& bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh")
    .env({"CONDA_PLUGINS_AUTO_ACCEPT_TOS": "yes", "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.0;8.9",
          "MAX_JOBS": "8"})
    .add_local_file(REPO / "environment.yml", "/tmp/environment.yml", copy=True)
    .run_commands(f"{CONDA} env create --file /tmp/environment.yml")          # README: conda env create
    .apt_install("libtiff5")                                                    # pillow của env cần libtiff.so.5
    .add_local_dir(REPO, R, copy=True, ignore=[".git", "assets"])
    .add_local_file(Path(__file__).resolve().parent / "tools" / "patch_objectgs_ckpt.py", "/opt/patch_ckpt.py", copy=True)
    .run_commands(f"python /opt/patch_ckpt.py {R}/scene/base_model.py")   # capture/restore gốc không resume được
    .pip_install("pyyaml")                                                   # cho hàm Modal (sửa config test resume)
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("modal_come")
)

# Bộ nhớ: L4 24 GB hết ở bước đánh giá iter 10000, L40S 48 GB hết ở bước đánh giá iter 30000 (cần thêm 17 GB,
# 3M Gaussian); paper dùng A800 80 GB -> A100-80GB (gsplat biên dịch cho sm_80 + sm_89)
GPU = dict(gpu="A100-80GB", volumes={VOL: vol}, timeout=8 * 3600, cpu=8, memory=65536,
           retries=modal.Retries(max_retries=4, initial_delay=10.0, backoff_coefficient=1.0))


def _link(src: Path, dst: str):
    src.mkdir(parents=True, exist_ok=True)
    d = Path(dst)
    if d.is_symlink() or d.is_file():
        d.unlink()
    elif d.exists():
        shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.symlink_to(src, target_is_directory=True)


def _done(step):
    return (ROOT / "_done" / step).exists()


def _mark(step, info=None):
    f = ROOT / "_done" / step
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, ensure_ascii=False))
    vol.commit()


def _sh(step, cmd, cwd=R):
    if _done(step):
        mc._tlog(f"[objectgs] skip {step}")
        return
    LOG.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def loop():                                                   # commit Volume mỗi 60 s: checkpoint còn khi bị ngắt
        while not stop.wait(60):
            try:
                vol.commit()
            except Exception as e:
                print(f"[commit] {e}", flush=True)
    th = threading.Thread(target=loop, daemon=True); th.start()
    try:
        mc._run_train(f"set -e; {cmd}", cwd, LOG / f"objectgs_{step}.log")
    finally:
        stop.set(); th.join(timeout=10); vol.commit()
    _mark(step, {"cmd": cmd})


@app.function(image=img, **GPU)
def _train():
    vol.reload()
    _link(ROOT / "datasets", f"{R}/datasets")
    _link(ROOT / "outputs", f"{R}/outputs")
    d = ROOT / "datasets" / "lerf_mask" / "teatime"
    if not _done("prep"):
        src = Path(VOL) / "teatime" / "benchmarks" / "lerf_mask" / "teatime"
        for sub in ("images", "object_mask", "sparse"):
            shutil.copytree(src / sub, d / sub, dirs_exist_ok=True)
        if (d / "sparse/0/points3D.ply").exists():
            shutil.move(d / "sparse/0/points3D.ply", d / "sparse/0/points3D_orig.ply")
        _mark("prep", {"images": len(list((d / "images").iterdir())), "masks": len(list((d / "object_mask").iterdir()))})
    _sh("ply_corr", f"{PY} ply_preprocessing.py --dataset_path datasets/lerf_mask --algorithm corr "
                    f"--output_ply_name points3D.ply")
    if not _done("ckpt_test"):                                    # thử resume thật trước khi train 30k
        import yaml
        cfg = yaml.safe_load(Path(f"{R}/config/objectgs/3d/lerf_mask/config.yaml").read_text())
        cfg["model_params"]["exp_name"] = "ckpt_test"
        cfg["optim_params"]["iterations"] = 600
        Path(f"{R}/config/objectgs/3d/lerf_mask/ckpt_test.yaml").write_text(yaml.safe_dump(cfg))
        tdir = ROOT / "outputs" / "lerf_mask" / "ckpt_test"
        if tdir.exists():
            shutil.rmtree(tdir)
        base = f"{PY} train.py --config config/objectgs/3d/lerf_mask/ckpt_test.yaml --scene_name teatime"
        _sh("ckpt_test_a", f"{base} --checkpoint_iterations 500")
        ck = next(tdir.rglob("chkpnt500.pth"))
        _sh("ckpt_test_b", f"{base} --start_checkpoint {ck}")
        _mark("ckpt_test", {"resumed_from": str(ck)})
    if not _done("train"):
        ck = sorted((ROOT / "outputs" / "lerf_mask" / "rebuttal").rglob("chkpnt*.pth"), key=lambda p: int(p.stem[6:])) \
            if (ROOT / "outputs" / "lerf_mask" / "rebuttal").exists() else []
        resume = f"--start_checkpoint {ck[-1]}" if ck else ""
        _sh("train", f"{PY} train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name teatime "
                     f"--checkpoint_iterations {' '.join(str(i) for i in range(3000, 30000, 3000))} {resume}")
    runs = sorted((ROOT / "outputs" / "lerf_mask" / "rebuttal" / "teatime").iterdir())
    return {"run": str(runs[-1]), "files": sorted(p.name for p in runs[-1].iterdir())}


@app.function(image=img.add_local_file(Path(__file__).resolve().parent / "tools" / "objectgs_eval_lerf.py",
                                         "/opt/objectgs_eval_lerf.py"), **GPU)
def _eval_lerf():
    """LERF-Mask mIoU / mBIoU của model ObjectGS mới nhất (tools/objectgs_eval_lerf.py)."""
    vol.reload()
    _link(ROOT / "datasets", f"{R}/datasets")
    _link(ROOT / "outputs", f"{R}/outputs")
    gt = ROOT / "datasets" / "lerf_mask" / "teatime" / "test_mask"
    if not gt.exists():
        shutil.copytree(Path(VOL) / "teatime" / "benchmarks" / "lerf_mask" / "teatime" / "test_mask", gt)
    run = sorted((ROOT / "outputs" / "lerf_mask" / "rebuttal" / "teatime").iterdir())[-1]
    out = ROOT / "eval_lerf"
    (ROOT / "_done" / "eval_lerf").unlink(missing_ok=True)
    _sh("eval_lerf", f"{PY} /opt/objectgs_eval_lerf.py -m outputs/lerf_mask/rebuttal/teatime/{run.name} "
                     f"--gt datasets/lerf_mask/teatime/test_mask --out {out}")
    return json.loads((out / "lerf_mask_eval.json").read_text())


@app.local_entrypoint()
def eval_lerf():
    r = _eval_lerf.remote()
    print(json.dumps({"ours": r["ours"], "mapping": r["mapping"]}, indent=1, ensure_ascii=False))


# ============================================================================ ObjectGS trên COLMAP của mình + mask SAM 3
DS_SAM3 = "safegs_sam3"                                 # tên dataset không chứa "lerf" / "3dovs" / "scannet" (code tác giả rẽ nhánh theo chữ)


def _rename_images_bin(raw: bytes, rename) -> bytes:
    """Ghi lại images.bin của COLMAP (định dạng nhị phân) với tên ảnh đổi theo rename(name); phần còn lại giữ nguyên."""
    import struct
    n = struct.unpack("<Q", raw[:8])[0]
    pos, out = 8, [raw[:8]]
    for _ in range(n):
        head = raw[pos:pos + 64]                       # image_id (I) + qvec (4d) + tvec (3d) + camera_id (I)
        pos += 64
        end = raw.index(b"\x00", pos)
        name = raw[pos:end].decode()
        pos = end + 1
        n2d = struct.unpack("<Q", raw[pos:pos + 8])[0]
        pts = raw[pos + 8:pos + 8 + 24 * n2d]          # mỗi điểm 2D: x, y (double) + point3D_id (int64)
        pos += 8 + 24 * n2d
        out += [head, rename(name).encode() + b"\x00", struct.pack("<Q", n2d), pts]
    assert pos == len(raw), "images.bin có dữ liệu thừa / sai định dạng"
    return b"".join(out)


@app.function(image=modal.Image.debian_slim().pip_install("numpy").add_local_python_source("modal_come"),
              volumes={VOL: vol}, timeout=1800)
def _prep_sam3():
    """Dữ liệu ObjectGS: images + object_mask (SAM 3) + sparse/0 của mình. 23 view test (mỗi 8 ảnh theo tên, đúng cách
    CoMe / 3DGS --eval chia) đổi tên thành test_<tên>: code ObjectGS (eval: False) coi ảnh có chữ "test" là view test."""
    vol.reload()
    if _done("prep_sam3"):
        return "skip"
    V = Path(VOL) / "teatime"
    d = ROOT / "datasets" / DS_SAM3 / "teatime"
    (d / "images").mkdir(parents=True, exist_ok=True); (d / "object_mask").mkdir(exist_ok=True)
    (d / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    raw = (V / "colmap" / "sparse" / "0" / "images.bin").read_bytes()
    import struct
    names, pos = [], 8
    for _ in range(struct.unpack("<Q", raw[:8])[0]):
        pos += 64
        end = raw.index(b"\x00", pos)
        names.append(raw[pos:end].decode())
        pos = end + 1
        pos += 8 + 24 * struct.unpack("<Q", raw[pos:pos + 8])[0]
    stems = sorted(Path(n).stem for n in names)
    test = set(stems[::8])
    new = lambda name: ("test_" + name) if Path(name).stem in test else name
    for n in names:
        s = Path(n).stem
        shutil.copy2(V / "colmap" / "images" / n, d / "images" / new(n))
        shutil.copy2(V / "sam3_masks" / "labels" / f"{s}.png", d / "object_mask" / (Path(new(n)).stem + ".png"))
    (d / "sparse" / "0" / "images.bin").write_bytes(_rename_images_bin(raw, new))
    for f in ("cameras.bin", "points3D.bin"):
        shutil.copy2(V / "colmap" / "sparse" / "0" / f, d / "sparse" / "0" / f)
    (d / "test_views.json").write_text(json.dumps(sorted(test)))
    _mark("prep_sam3", {"images": len(names), "test_views": len(test)})
    return {"images": len(names), "test_views": len(test)}


@app.function(image=img, **GPU)
def _train_sam3(lam: float = 0.0, tag: str = ""):
    """ply_preprocessing (corr) -> train 30k (checkpoint mỗi 3000, tự resume) -> render view test -> metrics.
    lam > 0: đổi lambda_object_loss (config tác giả: 0.01), thư mục kết quả riêng theo tag, các bước _done riêng."""
    vol.reload()
    _link(ROOT / "datasets", f"{R}/datasets")
    _link(ROOT / "outputs", f"{R}/outputs")
    import yaml
    sfx = f"_{tag}" if tag else ""
    exp = f"sam3{sfx}"
    cfg = yaml.safe_load(Path(f"{R}/config/objectgs/3d/lerf_mask/config.yaml").read_text())
    cfg["model_params"]["dataset_name"] = DS_SAM3
    cfg["model_params"]["exp_name"] = exp
    cfg["model_params"]["source_path"] = f"datasets/{DS_SAM3}"
    if lam > 0:
        cfg["optim_params"]["lambda_object_loss"] = lam
    cp = Path(f"{R}/config/objectgs/3d/{DS_SAM3}/config{sfx}.yaml")
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(yaml.safe_dump(cfg))                                # mọi tham số khác giữ đúng config lerf_mask của tác giả
    _sh("ply_sam3", f"{PY} ply_preprocessing.py --dataset_path datasets/{DS_SAM3} --algorithm corr "
                    f"--output_ply_name points3D.ply")
    base = ROOT / "outputs" / DS_SAM3 / exp / "teatime"
    if not _done(f"train_sam3{sfx}"):
        ck = sorted(base.rglob("chkpnt*.pth"), key=lambda p: int(p.stem[6:])) if base.exists() else []
        resume = f"--start_checkpoint {ck[-1]}" if ck else ""
        _sh(f"train_sam3{sfx}", f"{PY} train.py --config config/objectgs/3d/{DS_SAM3}/config{sfx}.yaml "
                                f"--scene_name teatime --checkpoint_iterations "
                                f"{' '.join(str(i) for i in range(3000, 30000, 3000))} {resume}")
    run = sorted(base.iterdir())[-1]
    rel = f"outputs/{DS_SAM3}/{exp}/teatime/{run.name}"
    _sh(f"render_test_sam3{sfx}", f"{PY} render.py -m {rel} --skip_train")           # README: render.sh
    _sh(f"metrics_sam3{sfx}", f"{PY} metrics.py -m {rel}")
    return {"run": rel, "files": sorted(p.name for p in run.iterdir())}


EVAL_RM = "/opt/objectgs_eval_removal_v4.py"   # đổi hậu tố mỗi lần sửa script: Modal bỏ qua file gắn vào nếu định nghĩa hàm không đổi


@app.function(image=img.add_local_file(Path(__file__).resolve().parent / "tools" / "objectgs_eval_removal.py", EVAL_RM),
              **GPU)
def _eval_removal(exp: str = "sam3", remove_ids: str = "", tag: str = "", hull: bool = False, margin_cm: float = 1.5,
                  hmin_cm: float = 1.5, gauss_margin_cm: float = 0.5, gauss_hmin_cm: float = 0.3, only: str = "",
                  probe: bool = False, foot_hmin_cm: float = -2.0):
    """Đo ID render + độ sạch khi xoá vật của model ObjectGS (SAM 3) mới nhất của thí nghiệm `exp`:
    tools/objectgs_eval_removal.py. remove_ids: chỉ xoá tập vật này (vd tập của v1) để so công bằng; tag: hậu tố kết quả."""
    vol.reload()
    _link(ROOT / "datasets", f"{R}/datasets")
    _link(ROOT / "outputs", f"{R}/outputs")
    run = sorted((ROOT / "outputs" / DS_SAM3 / exp / "teatime").iterdir())[-1]
    out = ROOT / f"eval_removal_{exp}{tag}"
    (ROOT / "_done" / f"eval_removal_{exp}{tag}").unlink(missing_ok=True)
    sup = Path(VOL) / "teatime" / "pipeline" / "v1" / "S3_label" / "support.json"
    ids = f"--remove_ids {remove_ids}" if remove_ids else ""
    if hull:                                    # hull_<vật>.ply: tools/mask_hull.py --scheme sam3, đã put lên Volume
        ids += (f" --hull_dir {ROOT / 'hulls'} --hull_margin_cm {margin_cm} --hull_hmin_cm {hmin_cm}"
                f" --gauss_margin_cm {gauss_margin_cm} --gauss_hmin_cm {gauss_hmin_cm}")
    if only:
        ids += f" --only {only}"
    if probe:
        ids += " --probe"
    ids += f" --foot_hmin_cm {foot_hmin_cm}"
    _sh(f"eval_removal_{exp}{tag}", f"{PY} {EVAL_RM} -m outputs/{DS_SAM3}/{exp}/teatime/{run.name} "
                                   f"--support {sup} --out {out} {ids}")
    return json.loads((out / "removal_eval.json").read_text())


@app.function(image=modal.Image.debian_slim().add_local_python_source("modal_come"), timeout=12 * 3600)
def _pipeline_sam3():
    """Chuẩn bị dữ liệu rồi train, chạy được bằng spawn (không phụ thuộc máy local)."""
    return [_prep_sam3.remote(), _train_sam3.remote()]


@app.function(image=modal.Image.debian_slim().add_local_python_source("modal_come"), volumes={VOL: vol}, timeout=300)
def _status():
    vol.reload()
    d = ROOT / "_done"
    return "\n".join(f"{f.name:10s} {f.read_text()[:200]}" for f in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)) \
        if d.exists() else "chưa có gì"


@app.local_entrypoint()
def train():
    print(_train.remote())


@app.local_entrypoint()
def status():
    print(_status.remote())
