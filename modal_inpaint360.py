"""
Inpaint360GS (WACV 2026, github.com/dfki-av/Inpaint360GS @ d54c893) chạy nguyên pipeline của tác giả trên teatime.

    python -m modal run --detach modal_inpaint360.py::seg                    # run_seg.sh (3DGS + HQ-SAM + gán ID + distill)
    python -m modal run modal_inpaint360.py::ids                             # tải ảnh đánh số ID để chọn vật
    python -m modal run --detach modal_inpaint360.py::remove --target 1,2,3  # run_remove.sh (+ mask tự động)
    python -m modal run --detach modal_inpaint360.py::inpaint                # run_inpaint.sh (LaMa trên T4)
    python -m modal run modal_inpaint360.py::status

Khác README (bắt buộc vì chạy headless trên Modal, ghi rõ để báo cáo):
  1. Dữ liệu: dùng COLMAP sẵn có của mình (PINHOLE, 994x738) thay vì convert.py; --resolution 1 nên tạo images_1 = images
     (seg/raw_mask_sam.py và tools/add_label_num_hqsam.py đọc thư mục images_{resolution}).
  2. 3DGS: thêm cờ có sẵn --checkpoint_iterations / --start_checkpoint để chạy tiếp khi Modal ngắt.
  3. Cuối run_remove.sh tác giả mở app Segment-and-Track-Anything để CHỌN TAY vùng cần inpaint trên 30 view ảo.
     Thay bằng: mask = pixel mà model của chính họ dự đoán thuộc target_id (virtual/ours_<iter>/objects_pred,
     render cảnh đầy đủ) -> ghi vào đúng chỗ app ghi (tracking_results/images/images_masks). Các bước sau giữ nguyên
     (prepare_lama_data.py vẫn nới mask 10 px như tác giả).
  4. LaMa: env conda riêng của tác giả (python 3.6, cudatoolkit 10.2) không chạy được trên GPU đời mới -> chạy trên T4.
Mọi output nằm trên Volume safe-gs-come: teatime/i360gs/ (data, output, config, LaMa data/output, log).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import modal

import modal_come as mc

app = modal.App("safe-gs-inpaint360")
vol, VOL = mc.vol, mc.VOL
REPO = Path(__file__).resolve().parent / "Based_Model" / "Inpaint360GS"
R = "/opt/Inpaint360GS"
ROOT = Path(VOL) / "teatime" / "i360gs"          # không chứa chữ "inpaint360" (dataset_readers.py rẽ nhánh theo chữ này)
DS, SCENE, RES = "others", "teatime", 1
LOG = Path(VOL) / "teatime" / "logs"
CROPFORMER = ("https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/resolve/main/CropFormer_model/"
              "Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth")
BIG_LAMA = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
IGNORE = [".git", "media", "**/__pycache__"]

main_img = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "wget", "unzip", "libgl1", "libglib2.0-0", "libgomp1",
                 "libegl1", "libx11-6", "libusb-1.0-0")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.6", "FORCE_CUDA": "1", "MAX_JOBS": "8",
          "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared"})
    # docs/install.md: torch 2.0.0 + cu118
    .pip_install("torch==2.0.0+cu118", "torchvision==0.15.1+cu118", "torchaudio==2.0.1+cu118",
                 extra_index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("setuptools<70", "wheel", "ninja", "numpy<2")
    .pip_install("mmcv==2.1.0", find_links="https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html")
    .add_local_dir(REPO, R, copy=True, ignore=IGNORE)
    .run_commands(                                                      # install.sh, bước 2-5
        f"find {R} -name '*.sh' -exec sed -i 's/\\r$//' {{}} +",         # repo clone trên Windows -> CRLF
        f"cd {R} && pip install gaussian_splatting/submodules/diff-gaussian-rasterization --no-build-isolation",
        f"cd {R} && pip install gaussian_splatting/submodules/simple-knn --no-build-isolation",
        f"cd {R} && pip install submodules/diff-gaussian-rasterization --no-build-isolation",
        f"cd {R} && pip install .",
        "pip install git+https://github.com/facebookresearch/segment-anything.git",
        "pip install setuptools==75.8.0",
        f"cd {R}/seg && python -m pip install -e detectron2 --no-build-isolation",
        f"cd {R}/seg/detectron2/detectron2/projects/CropFormer/entity_api/PythonAPI && make",
        f"cd {R}/seg/detectron2/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops && sh make.sh",
        f"cd {R} && python seg/patch_fvcore.py",
    )
    .run_commands(                                                      # dataset HF có cổng: cần token
        f'mkdir -p {R}/seg/weight && wget -q --header="Authorization: Bearer $HF_TOKEN" '
        f"-O {R}/seg/weight/CropFormer_hornet_3x_03823a.pth {CROPFORMER}",
        secrets=[modal.Secret.from_name("huggingface")],
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("modal_come")
)

lama_img = (
    modal.Image.from_registry("continuumio/miniconda3:24.7.1-0")
    .apt_install("unzip", "curl", "libgl1", "libglib2.0-0")
    .env({"CONDA_PLUGINS_AUTO_ACCEPT_TOS": "yes"})
    .add_local_dir(REPO / "LaMa", f"{R}/LaMa", copy=True, ignore=IGNORE)
    .run_commands(                                                      # docs/install.md, mục 2
        f"cd {R}/LaMa && conda env create -f conda_env.yml",
        "conda install -n lama pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y",
        "conda run -n lama pip install pytorch-lightning==1.2.9",
        f"cd {R}/LaMa && curl -sLJO {BIG_LAMA} && unzip -q big-lama.zip && rm big-lama.zip",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("modal_come")
)

GPU = dict(volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=32768,
           retries=modal.Retries(max_retries=4, initial_delay=10.0, backoff_coefficient=1.0))


# ============================================================================================ tiện ích
def _link(src: Path, dst: str):
    """dst (trong repo) -> src (trên Volume), để mọi output của tác giả nằm trên Volume."""
    src.mkdir(parents=True, exist_ok=True)
    d = Path(dst)
    if d.is_symlink() or d.exists():
        if d.is_symlink() or d.is_file():
            d.unlink()
        else:
            shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.symlink_to(src, target_is_directory=True)


def _setup(lama=False):
    vol.reload()
    if lama:
        _link(ROOT / "LaMa_data", f"{R}/LaMa/data")
        _link(ROOT / "LaMa_output", f"{R}/LaMa/output")
        return
    _link(ROOT / "data", f"{R}/data")
    _link(ROOT / "output", f"{R}/output")
    _link(ROOT / "LaMa_data", f"{R}/LaMa/data")
    _link(ROOT / "LaMa_output", f"{R}/LaMa/output")
    _link(ROOT / "tracking_results", f"{R}/Segment-and-Track-Anything/tracking_results")
    for k in ("object_removal", "object_inpaint"):
        _link(ROOT / "config" / k / DS, f"{R}/config/{k}/{DS}")


def _done(step):
    return (ROOT / "_done" / step).exists()


def _mark(step, info=None):
    f = ROOT / "_done" / step
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, ensure_ascii=False))
    vol.commit()


def _sh(step, cmd, cwd=R, force=False):
    if _done(step) and not force:
        mc._tlog(f"[i360] skip {step}")
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
        mc._run_train(f"set -e; {cmd}", cwd, LOG / f"i360_{step}.log")
    finally:
        stop.set(); th.join(timeout=10); vol.commit()
    _mark(step, {"cmd": cmd})


S = f"data/{DS}/{SCENE}"
M = f"output/{DS}/{SCENE}"


# ============================================================================================ run_seg.sh
@app.function(image=main_img, gpu="L4", **GPU)
def _seg():
    _setup()
    d = ROOT / "data" / DS / SCENE
    if not _done("prep"):
        src = Path(VOL) / "teatime" / "colmap"
        for sub in ("images", "images_1"):
            shutil.copytree(src / "images", d / sub, dirs_exist_ok=True)
        shutil.copytree(src / "sparse" / "0", d / "sparse" / "0", dirs_exist_ok=True)
        _mark("prep", {"images": len(list((d / "images").iterdir()))})

    # 1. vanilla 3DGS (checkpoint mỗi 3000 iter; chạy tiếp từ checkpoint mới nhất)
    if not _done("seg1_3dgs"):
        out = ROOT / "output" / DS / SCENE / "3dgs_output"
        ck = sorted(out.glob("chkpnt*.pth"), key=lambda p: int(p.stem[6:])) if out.exists() else []
        resume = f"--start_checkpoint {M}/3dgs_output/{ck[-1].name}" if ck else ""
        _sh("seg1_3dgs", f"python gaussian_splatting/train.py -s {S} -m {M}/3dgs_output --init_mode sparse --eval "
                         f"--resolution {RES} --checkpoint_iterations {' '.join(str(i) for i in range(3000, 30000, 3000))} "
                         f"{resume}")
    pp = f"export PYTHONPATH={R}:{R}/seg/detectron2:$PYTHONPATH; "
    _sh("seg2_mask", pp + f"python seg/raw_mask_sam.py --dataset_path data/{DS}/ --scene_name {SCENE} "
                          f"--image_folder images_{RES} --method hqsam")
    _sh("seg3_assoc", pp + f"python seg/mask_associate.py --source_path {S} --model_path {M}/3dgs_output "
                           f"--resolution {RES} --mask_generator hqsam --eval")
    _sh("seg4_label", pp + f"python tools/add_label_num_hqsam.py --source_path {S} --resolution {RES} --mask_generator hqsam")
    _sh("seg5_distill", pp + f"WANDB_MODE=disabled python seg/distillation.py --source_path {S} --model_path {M} "
                             f"--vanilla_3dgs_path {M}/3dgs_output --resolution {RES} --object_path associated_hqsam --eval")
    _sh("seg6_render", f"python render.py -m {M} --render_video")
    return "seg xong"


# ============================================================================================ run_remove.sh
@app.function(image=main_img, gpu="L4", **GPU)
def _remove(target: str, surrounding: str = "None", force: bool = False):
    import numpy as np
    from PIL import Image
    _setup()
    tag = f"target={target}|sur={surrounding}"
    if force:
        for s in ("rm1_config", "rm2_removal", "rm3_virtual", "rm4_mask"):
            (ROOT / "_done" / s).unlink(missing_ok=True)
    cfg = f"config/object_removal/{DS}/{SCENE}.json"
    _sh("rm1_config", f"python tools/init_configs.py --dataset_name {DS} --scene {SCENE} --target_id {target} "
                      f"--target_surronding_id {surrounding}")
    _sh("rm2_removal", f"python edit_object_removal.py --source_path {S} -m {M} --config_file {cfg} --render_video")
    _sh("rm3_virtual", f"python tools/virtual_pose.py -s {S} -m {M} --config_file {cfg}")
    if not _done("rm4_mask"):                                     # thay bước chọn tay trong app Segment-and-Track
        it = json.loads(Path(f"{R}/config/object_distill/train_distill.json").read_text())["iterations"]
        pred = ROOT / "output" / DS / SCENE / "virtual" / f"ours_{it}" / "objects_pred"
        ids = [int(x) for x in target.split(",")]
        dst = ROOT / "tracking_results" / "images" / "images_masks"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        frac = []
        for f in sorted(pred.glob("*.png")):
            m = np.isin(np.array(Image.open(f)), ids)
            Image.fromarray((m * 255).astype(np.uint8)).save(dst / f.name)
            frac.append(float(m.mean()))
        assert frac, f"không thấy {pred}"
        _mark("rm4_mask", {"tag": tag, "views": len(frac), "mask_frac_mean": float(np.mean(frac)),
                           "mask_frac_min": float(np.min(frac))})
    return json.loads((ROOT / "_done" / "rm4_mask").read_text())


# ============================================================================================ run_inpaint.sh
@app.function(image=main_img, gpu="L4", **GPU)
def _inpaint_main(part: str):
    _setup()
    if part == "a":
        _sh("in1_tolama", f"python tools/prepare_lama_data.py -s {S} -m {M} -r {RES} --inpaint2lama")
        return "a"
    _sh("in3_fromlama", f"python tools/prepare_lama_data.py -s {S} -m {M} -r {RES}")
    _sh("in4_plyfusion", f"python edit_object_removal_plyfusion.py -s {S} -m {M} "
                         f"--config_file config/object_removal/{DS}/{SCENE}.json")
    _sh("in5_inpaint", f"python edit_object_inpaint.py -s {S} -m {M} --config_file config/object_inpaint/{DS}/{SCENE}.json "
                       f"--resolution {RES} --render_video")
    return "b"


@app.function(image=lama_img, gpu="T4", **GPU)
def _lama():
    _setup(lama=True)
    run = f"cd {R}/LaMa && export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd) && conda run --no-capture-output -n lama "
    _sh("in2_lama", f"{run} python bin/predict_color.py --data_name 360_{SCENE}_virtual && "
                    f"{run} python bin/predict_depth.py --data_name 360_{SCENE}_virtual", cwd=f"{R}/LaMa")
    return "lama"


@app.function(image=modal.Image.debian_slim().add_local_python_source("modal_come"), volumes={VOL: vol}, timeout=300)
def _status():
    vol.reload()
    d = ROOT / "_done"
    return "\n".join(f"{f.name:14s} {f.read_text()[:160]}" for f in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)) \
        if d.exists() else "chưa có gì"


# ============================================================================================ entrypoint
@app.local_entrypoint()
def seg():
    print(_seg.remote())


@app.local_entrypoint()
def remove(target: str, surrounding: str = "None", force: bool = False):
    print(_remove.remote(target, surrounding, force))


@app.local_entrypoint()
def inpaint():
    print(_inpaint_main.remote("a")); print(_lama.remote()); print(_inpaint_main.remote("b"))


@app.local_entrypoint()
def status():
    print(_status.remote())


@app.local_entrypoint()
def ids(n: int = 8):
    """Tải n ảnh images_{RES}_num (ảnh đánh số ID của tác giả) về Data/teatime/i360gs/ids/ để chọn target_id."""
    out = Path(__file__).resolve().parent / "Data" / "teatime" / "i360gs" / "ids"
    out.mkdir(parents=True, exist_ok=True)
    rem = f"teatime/i360gs/data/{DS}/{SCENE}/images_{RES}_num"
    files = sorted(e.path for e in vol.listdir(rem) if e.type == modal.volume.FileEntryType.FILE)
    for p in files[:: max(1, len(files) // n)][:n]:
        with open(out / Path(p).name, "wb") as f:
            for c in vol.read_file(p):
                f.write(c)
    print(f"{len(files)} ảnh có số ID; đã tải {n} -> {out}")
