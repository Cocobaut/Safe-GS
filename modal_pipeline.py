"""
Safe-GS pipeline end-to-end trên Modal (S0 -> S6), có checkpoint ở mọi stage.

    python -m modal run --detach modal_pipeline.py::run --config configs/pipeline/teatime_v1.json            # chạy tới S6
    python -m modal run --detach modal_pipeline.py::run --config configs/pipeline/teatime_v1.json --until S3
    python -m modal run modal_pipeline.py::run --config ... --force S4a       # chạy lại 1 stage (và các stage sau)
    python -m modal run modal_pipeline.py::status --config configs/pipeline/teatime_v1.json
    python -m modal run modal_pipeline.py::pull --config configs/pipeline/teatime_v1.json   # tải preview + json về
                                                                               # Data/<scene>/pipeline/<run>/ (không tải .ply)

Output trên Volume safe-gs-come: <scene>/pipeline/<run_name>/<stage>/ gồm:
    _done.json        trạng thái + hash config (đổi config của stage hoặc stage trước -> stage chạy lại)
    config_used.json  bản config đầy đủ lúc chạy (tái sử dụng / so sánh)
    preview/ ...      ảnh để xem nhanh stage đó ra gì
    (dữ liệu của stage)
Checkpoint trong stage: S0/S2 dùng checkpoint của modal_come.py; S3 lưu vote cộng dồn mỗi 20 view; S4a lưu từng đoạn
60 frame; S4b lưu Gaussian mỗi 500 iter; S5/S6 lưu từng vật. Hàm có retries; Volume được commit mỗi 30 giây.
Mỗi stage có log riêng: <scene>/logs/pipeline_<stage>.log
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import modal

import modal_come as mc

app = modal.App("safe-gs-pipeline")
vol = mc.vol
VOL = mc.VOL
REPO_DIR = Path(__file__).resolve().parent
STAGES = ["S0", "S1", "S2", "S3", "S4a", "S4b", "S5", "S6", "S7", "S8", "S9", "S10"]
DEPS = {"S0": [], "S1": [], "S2": ["S0"], "S3": ["S1", "S2"], "S4a": ["S3"], "S4b": ["S3", "S4a"],
        "S5": ["S3"], "S6": ["S3", "S5"], "S7": ["S6"], "S8": ["S7"], "S9": ["S4b", "S5", "S6", "S8"],
        "S10": ["S4b", "S5", "S6", "S8"]}
CFG_KEY = {"S0": "S0_colmap", "S1": "S1_masks", "S2": "S2_come", "S3": "S3_label", "S4a": "S4_background",
           "S4b": "S4_background", "S5": "S5_objects", "S6": "S6_mesh", "S7": "S7_physics", "S8": "S8_sim",
           "S9": "S9_render", "S10": "S10_eval"}
DIRNAME = {"S0": "S0_colmap", "S1": "S1_masks", "S2": "S2_come", "S3": "S3_label", "S4a": "S4a_inpaint",
           "S4b": "S4b_background", "S5": "S5_objects", "S6": "S6_mesh", "S7": "S7_physics", "S8": "S8_sim",
           "S9": "S9_render", "S10": "S10_eval"}

def _src(img):
    """Mọi container cần module modal_come (dùng chung Volume, image, hàm log/checkpoint)."""
    return img.add_local_python_source("modal_come")


def _pipe_hash() -> str:
    """Hash nội dung pipeline/*.py: đưa vào env của image để Modal buộc dùng bản script mới (nó có thể giữ thư mục gắn cũ)."""
    h = hashlib.sha1()
    for f in sorted((REPO_DIR / "pipeline").glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:12]


pipe_image = _src(
    mc.come_image
    .pip_install("coacd", "trimesh", "rtree", "shapely", "mapbox-earcut")   # shapely + earcut: slice_plane(cap=True)
    .env({"SAFE_GS_PIPE_HASH": _pipe_hash()})
    .add_local_dir(REPO_DIR / "pipeline", "/opt/pipeline")
)
sim_img = _src(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("mujoco", "trimesh", "numpy", "scipy", "matplotlib", "rtree")
    .add_local_dir(REPO_DIR / "pipeline", "/opt/pipeline")
)
colmap_img = _src(mc.colmap_image)
come_img = _src(mc.come_image)
light_img = _src(modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pillow"))

PP_URL = "https://github.com/sczhou/ProPainter.git"
PP_REL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0"
propainter_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "wget")
    .pip_install("torch==2.1.2", "torchvision==0.16.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2", "av", "addict", "einops", "future", "scipy", "opencv-python", "matplotlib",
                 "scikit-image", "imageio-ffmpeg", "pyyaml", "requests", "timm", "yapf", "pillow")
    .run_commands(
        f"git clone {PP_URL} /opt/ProPainter && git -C /opt/ProPainter rev-parse HEAD > /opt/ProPainter/COMMIT",
        "mkdir -p /opt/ProPainter/weights && cd /opt/ProPainter/weights && "
        f"wget -q {PP_REL}/ProPainter.pth && wget -q {PP_REL}/recurrent_flow_completion.pth && "
        f"wget -q {PP_REL}/raft-things.pth",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("modal_come")
)


# ============================================================================================ tiện ích
def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _stage_hash(cfg: dict, stage: str) -> str:
    """Hash = config của stage + hash các stage nó phụ thuộc (đổi upstream -> downstream chạy lại)."""
    return _hash({"cfg": cfg[CFG_KEY[stage]], "deps": [_stage_hash(cfg, s) for s in DEPS[stage]], "stage": stage})


def _root(cfg) -> Path:
    return Path(VOL) / cfg["scene"] / "pipeline" / cfg["run_name"]


def _sdir(cfg, stage) -> Path:
    return _root(cfg) / DIRNAME[stage]


def _done(cfg, stage) -> bool:
    f = _sdir(cfg, stage) / "_done.json"
    return f.exists() and json.loads(f.read_text()).get("hash") == _stage_hash(cfg, stage)


def _begin(cfg, stage, force=False) -> Path:
    d = _sdir(cfg, stage)
    if force and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config_used.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    old = d / "_done.json"
    if old.exists() and json.loads(old.read_text()).get("hash") != _stage_hash(cfg, stage):
        old.unlink()                                             # config đổi -> làm lại, giữ checkpoint con nếu còn hợp lệ
    vol.commit()
    return d


def _finish(cfg, stage, t0, extra=None):
    d = _sdir(cfg, stage)
    info = {"stage": stage, "hash": _stage_hash(cfg, stage), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes_this_run": round((time.time() - t0) / 60, 2), **(extra or {})}
    (d / "_done.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    vol.commit()
    mc._tlog(f"[{stage}] xong: {json.dumps(info, ensure_ascii=False)}")


class _Committer:
    def __enter__(self):
        self.stop = threading.Event()
        def loop():
            while not self.stop.wait(30):
                try:
                    vol.commit()
                except Exception as e:
                    print(f"[commit] {e}", flush=True)
        self.th = threading.Thread(target=loop, daemon=True); self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set(); self.th.join(timeout=10); vol.commit()


def _prep_come(cfg):
    """Chép COLMAP + model CoMe từ Volume ra local (cùng đường dẫn lúc train để cfg_args khớp)."""
    V = Path(VOL) / cfg["scene"]
    data = Path(f"/tmp/{cfg['scene']}_colmap")                  # cung duong dan luc train (modal_come.py) de cfg_args khop
    for sub in ("images", "sparse"):
        shutil.copytree(V / "colmap" / sub, data / sub, dirs_exist_ok=True)
    local = Path(f"/tmp/come_out/{cfg['scene']}"); local.mkdir(parents=True, exist_ok=True)
    for name in mc.SMALL_FILES:
        shutil.copy2(V / "come" / name, local / name)
    shutil.copytree(V / "come" / "point_cloud", local / "point_cloud", dirs_exist_ok=True)
    return data, local


def _run_script(cfg, stage, script, extra):
    data, local = _prep_come(cfg)
    cfg_path = _sdir(cfg, stage) / "config_used.json"
    cmd = (f"PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/{script} -m {local} -s {data} "
           f"--iteration {cfg['S2_come']['iterations']} --cfg {cfg_path} --out {_sdir(cfg, stage)} {extra}")
    with _Committer():
        mc._run_train(cmd, "/opt/CoMe", Path(VOL) / cfg["scene"] / "logs" / f"pipeline_{stage}.log")


GPU_KW = dict(gpu="L4", volumes={VOL: vol}, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=5, initial_delay=10.0, backoff_coefficient=1.0))


# ============================================================================================ stages
@app.function(image=colmap_img, timeout=2 * 3600, **GPU_KW)
def s0(cfg: dict, force: bool = False):
    if _done(cfg, "S0") and not force:
        return "skip"
    t0 = time.time(); d = _begin(cfg, "S0", force)
    V = Path(VOL) / cfg["scene"]
    if not (V / "colmap" / "sparse" / "0" / "points3D.bin").exists():
        mc.colmap_teatime.local()                                  # checkpoint của chính colmap_teatime
    vol.reload()
    chk = json.loads((V / "colmap_check.json").read_text())
    (d / "summary.json").write_text(json.dumps({"colmap_dir": f"{cfg['scene']}/colmap", **chk}, indent=2))
    _finish(cfg, "S0", t0, {"registered": chk["registered"], "input_images": chk["input_images"]})
    return "done"


@app.function(image=light_img, volumes={VOL: vol}, timeout=1800)
def s1(cfg: dict, force: bool = False):
    if _done(cfg, "S1") and not force:
        return "skip"
    import numpy as np
    from PIL import Image
    t0 = time.time(); d = _begin(cfg, "S1", force)
    c = cfg["S1_masks"]
    assert c["source"] in ("team_sam2", "sam3_text"), f"nguồn mask chưa hỗ trợ: {c['source']}"
    lab = Path(VOL) / c["labels_dir"]
    files = sorted(lab.glob("frame_*.png"))
    ids = set()
    for f in files[::10]:
        ids |= set(np.unique(np.array(Image.open(f))).tolist())
    (d / "summary.json").write_text(json.dumps({"labels_dir": c["labels_dir"], "frames": len(files),
                                                "ids_seen_every_10th_frame": len(ids - {0})}, indent=2))
    _finish(cfg, "S1", t0, {"frames": len(files)})
    return "done"


@app.function(image=come_img, timeout=4 * 3600, **GPU_KW)
def s2(cfg: dict, force: bool = False):
    if _done(cfg, "S2") and not force:
        return "skip"
    t0 = time.time(); d = _begin(cfg, "S2", force)
    V = Path(VOL) / cfg["scene"]
    final = V / "come" / "point_cloud" / f"iteration_{cfg['S2_come']['iterations']}" / "point_cloud.ply"
    if not final.exists():
        mc.train_teatime.local()                                   # checkpoint mỗi 1000 iter của train_teatime
    vol.reload()
    summ = {"model_dir": f"{cfg['scene']}/come", "ply_MB": round(final.stat().st_size / 1e6, 1)}
    if (V / "come" / "metrics.json").exists():
        summ["metrics_test"] = json.loads((V / "come" / "metrics.json").read_text())
    (d / "summary.json").write_text(json.dumps(summ, indent=2))
    _finish(cfg, "S2", t0, summ)
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s3(cfg: dict, force: bool = False):
    if _done(cfg, "S3") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S3", force)
    _run_script(cfg, "S3", "s3_label.py", f"--labels {Path(VOL) / cfg['S1_masks']['labels_dir']}")
    d = _sdir(cfg, "S3")
    _finish(cfg, "S3", t0, {"groups": json.loads((d / "groups.json").read_text())["objects"],
                            "support": json.loads((d / "support.json").read_text())})
    return "done"


@app.function(image=propainter_image, timeout=3 * 3600, **GPU_KW)
def s4a(cfg: dict, force: bool = False):
    """Inpaint video vùng các vật (ProPainter), từng đoạn 60 frame, đoạn nào xong thì lưu luôn."""
    if _done(cfg, "S4a") and not force:
        return "skip"
    import cv2
    import numpy as np
    t0 = time.time(); d = _begin(cfg, "S4a", force)
    c = cfg["S4_background"]["propainter"]
    V = Path(VOL) / cfg["scene"]
    groups = json.loads((_sdir(cfg, "S3") / "groups.json").read_text())
    ids = sorted({i for o in groups["objects"].values() for i in o["ids"]})
    names = sorted(p.stem for p in (V / "colmap" / "input").glob("frame_*.jpg"))
    (d / "inpainted").mkdir(exist_ok=True); (d / "masks").mkdir(exist_ok=True); (d / "preview").mkdir(exist_ok=True)
    chunk = 60
    log = V / "logs" / "pipeline_S4a.log"
    with _Committer():
        for k0 in range(0, len(names), chunk):
            part = names[k0:k0 + chunk]
            if all((d / "inpainted" / f"{n}.png").exists() for n in part):
                continue
            w = Path(f"/tmp/pp/chunk{k0}"); fr, ms = w / "frames", w / "masks"
            shutil.rmtree(w, ignore_errors=True); fr.mkdir(parents=True); ms.mkdir()
            for n in part:
                src = V / "colmap" / "images" / f"{n}.jpg"
                if not src.exists():
                    src = V / "colmap" / "input" / f"{n}.jpg"
                shutil.copy2(src, fr / f"{n}.jpg")
                lp = Path(VOL) / cfg["S1_masks"]["labels_dir"] / f"{n}.png"
                lab = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
                assert lab is not None, f"không đọc được label {lp}"
                m = (np.isin(lab, ids).astype(np.uint8) * 255)
                cv2.imwrite(str(ms / f"{n}.png"), m)
                cv2.imwrite(str(d / "masks" / f"{n}.png"), m)
            cmd = (f"python inference_propainter.py --video {fr} --mask {ms} --output {w / 'out'} --save_frames "
                   f"--mask_dilation {c['mask_dilation']} --subvideo_length {c['subvideo_length']} "
                   f"--neighbor_length {c['neighbor_length']} --ref_stride {c['ref_stride']}"
                   + (" --fp16" if c["fp16"] else ""))
            mc._run_train(cmd, "/opt/ProPainter", log)
            outs = sorted((w / "out" / "frames" / "frames").glob("*.png"))
            assert len(outs) == len(part), f"ProPainter trả {len(outs)} frame cho đoạn {len(part)} frame"
            for o, n in zip(outs, part):
                shutil.copy2(o, d / "inpainted" / f"{n}.png")
            vol.commit()
            mc._tlog(f"[S4a] xong đoạn {k0}-{k0 + len(part) - 1}")
    for n in cfg["S3_label"]["preview_frames"]:
        a = cv2.imread(str(V / "colmap" / "input" / f"{n}.jpg"))
        m = cv2.imread(str(d / "masks" / f"{n}.png"), 0)
        b = cv2.imread(str(d / "inpainted" / f"{n}.png"))
        ov = a.copy(); ov[m > 0] = (0.5 * ov[m > 0] + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
        cv2.imwrite(str(d / "preview" / f"{n}.jpg"), np.hstack([a, ov, b]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    _finish(cfg, "S4a", t0, {"frames": len(names), "mask_ids": ids,
                             "propainter_commit": Path("/opt/ProPainter/COMMIT").read_text().strip()})
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s4b(cfg: dict, force: bool = False):
    if _done(cfg, "S4b") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S4b", force)
    _run_script(cfg, "S4b", "s4_background.py",
                f"--s3 {_sdir(cfg, 'S3')} --inpainted {_sdir(cfg, 'S4a') / 'inpainted'}")
    d = _sdir(cfg, "S4b")
    _finish(cfg, "S4b", t0, {"fill": json.loads((d / "fill.json").read_text()) if (d / "fill.json").exists() else {}})
    return "done"


@app.function(image=pipe_image, timeout=2 * 3600, **GPU_KW)
def s5(cfg: dict, force: bool = False):
    if _done(cfg, "S5") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S5", force)
    _run_script(cfg, "S5", "s5_objects.py", f"--s3 {_sdir(cfg, 'S3')}")
    _finish(cfg, "S5", t0)
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s6(cfg: dict, force: bool = False):
    if _done(cfg, "S6") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S6", force)
    _run_script(cfg, "S6", "s6_mesh.py", f"--s3 {_sdir(cfg, 'S3')} --s5 {_sdir(cfg, 'S5')}")
    _finish(cfg, "S6", t0)
    return "done"


def _run_cpu(cfg, stage, script, extra):
    cmd = f"python /opt/pipeline/{script} --cfg {_sdir(cfg, stage) / 'config_used.json'} --out {_sdir(cfg, stage)} {extra}"
    with _Committer():
        mc._run_train(cmd, "/opt/pipeline", Path(VOL) / cfg["scene"] / "logs" / f"pipeline_{stage}.log")


CPU_KW = dict(volumes={VOL: vol}, cpu=4, memory=8192,
              retries=modal.Retries(max_retries=5, initial_delay=10.0, backoff_coefficient=1.0))


@app.function(image=sim_img, timeout=1800, **CPU_KW)
def s7(cfg: dict, force: bool = False):
    if _done(cfg, "S7") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S7", force)
    _run_cpu(cfg, "S7", "s7_physics.py", f"--s6 {_sdir(cfg, 'S6')}")
    import mujoco
    _finish(cfg, "S7", t0, {"mujoco": mujoco.__version__,
                            "check": json.loads((_sdir(cfg, "S7") / "physics.json").read_text())["check"]})
    return "done"


@app.function(image=sim_img, timeout=3600, **CPU_KW)
def s8(cfg: dict, force: bool = False):
    if _done(cfg, "S8") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S8", force)
    _run_cpu(cfg, "S8", "s8_sim.py", f"--s7 {_sdir(cfg, 'S7')}")
    _finish(cfg, "S8", t0, {"summary": json.loads((_sdir(cfg, "S8") / "summary.json").read_text())})
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s9(cfg: dict, force: bool = False):
    if _done(cfg, "S9") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S9", force)
    _run_script(cfg, "S9", "s9_render.py", f"--s4b {_sdir(cfg, 'S4b')} --s5 {_sdir(cfg, 'S5')} "
                                           f"--s6 {_sdir(cfg, 'S6')} --s8 {_sdir(cfg, 'S8')}")
    _finish(cfg, "S9", t0)
    return "done"


@app.function(image=sim_img, timeout=1800, **CPU_KW)
def _s10_cpu(cfg: dict):
    """Phần CPU của S10 (mesh S6 + mô phỏng S8) rồi gộp bảng; mỗi phần có json riêng làm checkpoint."""
    vol.reload()
    d = _sdir(cfg, "S10")
    if not (d / "eval_cpu.json").exists():
        _run_cpu(cfg, "S10", "s10_eval_cpu.py", f"--s5 {_sdir(cfg, 'S5')} --s6 {_sdir(cfg, 'S6')} "
                                                f"--s7 {_sdir(cfg, 'S7')} --s8 {_sdir(cfg, 'S8')}")
    mc._run_train(f"python /opt/pipeline/s10_report.py --gpu {d / 'eval_gpu.json'} --cpu {d / 'eval_cpu.json'} --out {d}",
                  "/opt/pipeline", Path(VOL) / cfg["scene"] / "logs" / "pipeline_S10.log")
    vol.commit()


@app.function(image=pipe_image, timeout=3600, **{**GPU_KW, "gpu": ["L4", "L40S"]})   # L40S cùng sm_89 với L4
def _s10_ghost(cfg: dict):
    """Đo 'ma' còn lại trong khoảng trống vật của nền S4b (pipeline/s10_ghost.py), để so với ObjectGS cùng định nghĩa."""
    vol.reload()
    _run_script(cfg, "S10", "s10_ghost.py",
                f"--s3 {_sdir(cfg, 'S3')} --s4b {_sdir(cfg, 'S4b')} --labels {Path(VOL) / cfg['S1_masks']['labels_dir']}")
    return json.loads((_sdir(cfg, "S10") / "ghost.json").read_text())


S4C_VER = "v2"     # đổi khi sửa pipeline/s4c_decouple.py hoặc s10_ghost.py (Modal bỏ qua thư mục gắn vào nếu định nghĩa hàm không đổi)


@app.function(image=pipe_image, timeout=4 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _s4c(cfg: dict, tag: str, iters: int = 8000, lam_sil: float = 0.1, lam_leak: float = 0.5, assign_ratio: float = 0.5):
    """Thử nghiệm S4c (pipeline/s4c_decouple.py): tinh chỉnh CoMe 2 nhóm với loss alpha theo nhóm, rồi đo ma bằng s10_ghost.py.
    iters = 0: không tinh chỉnh, chỉ gán nhãn SAM 3 và đo (đối chứng v1 với nhãn SAM 3)."""
    vol.reload()
    data, local = _prep_come(cfg)
    out = _root(cfg) / "S4c" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfgp = _sdir(cfg, "S10") / "config_used.json"
    labels = Path(VOL) / cfg["scene"] / "sam3_masks" / "labels"
    log = Path(VOL) / cfg["scene"] / "logs" / f"s4c_{tag}.log"
    env = f"PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline"
    common = f"-m {local} -s {data} --iteration {cfg['S2_come']['iterations']} --cfg {cfgp}"
    with _Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(f"{env}/s4c_decouple.py {common} --out {out} --labels {labels} "
                          f"--support {_sdir(cfg, 'S3') / 'support.json'} --iters {iters} --lam_sil {lam_sil} "
                          f"--lam_leak {lam_leak} --assign_ratio {assign_ratio}", "/opt/CoMe", log)
        if not (out / "ghost" / "ghost.json").exists():
            mc._run_train(f"{env}/s10_ghost.py {common} --out {out / 'ghost'} --s3 {out / 'S3like'} "
                          f"--s4b {out / 'S4like'} --labels {labels} --occluder_ids 20,30", "/opt/CoMe", log)
    vol.commit()
    return {"tag": tag, "metrics": json.loads((out / "metrics.json").read_text()),
            "ghost": json.loads((out / "ghost" / "ghost.json").read_text())}


@app.function(image=pipe_image, timeout=2 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _s4d(cfg: dict, tag: str, src_tag: str = "ctrl", cut_dilate: int = 2, reach: float = 2.0, low_cm: float = 3.0,
         fill_dilate: int = 4, protect: int = 1, ft_iters: int = 0, lr_scale: float = 0.5, ft_mode: str = "inpaint",
         freeze_region: int = 0, freeze_dilate: int = 6, fill_grid: float = 0.01, texture_fill: int = 0, tex_scale: float = 0.5,
         code_rel: str = ""):
    """Thử nghiệm S4d (pipeline/s4d_cutfill.py): cắt Gaussian nền trong vết chân + plane_fill, đo ma bằng s10_ghost.py."""
    vol.reload()
    data, local = _prep_come(cfg)
    src = _root(cfg) / "S4c" / src_tag
    out = _root(cfg) / "S4d" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfgp = _sdir(cfg, "S10") / "config_used.json"
    labels = Path(VOL) / cfg["scene"] / "sam3_masks" / "labels"
    log = Path(VOL) / cfg["scene"] / "logs" / f"s4d_{tag}.log"
    code = str(Path(VOL) / code_rel) if code_rel else "/opt/pipeline"
    env = f"PYTHONPATH=/opt/CoMe:{code} python {code}"
    common = f"-m {local} -s {data} --iteration {cfg['S2_come']['iterations']} --cfg {cfgp}"
    with _Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(f"{env}/s4d_cutfill.py --texture_fill {texture_fill} --tex_scale {tex_scale} {common} --out {out} --src {src} --labels {labels} "
                          f"--cut_dilate {cut_dilate} --reach {reach} --low_cm {low_cm} --fill_dilate {fill_dilate} "
                          f"--protect {protect} --ft_iters {ft_iters} --lr_scale {lr_scale} --ft_mode {ft_mode} --freeze_region {freeze_region} --freeze_dilate {freeze_dilate} --fill_grid {fill_grid} "
                          f"--inpainted {_sdir(cfg, 'S4a') / 'inpainted'}",
                          "/opt/CoMe", log)
        if not (out / "ghost" / "ghost.json").exists():
            mc._run_train(f"{env}/s10_ghost.py {common} --out {out / 'ghost'} --s3 {out / 'S3like'} "
                          f"--s4b {out / 'S4like'} --labels {labels} --occluder_ids 20,30", "/opt/CoMe", log)
    vol.commit()
    return {"tag": tag, "metrics": json.loads((out / "metrics.json").read_text()),
            "ghost": json.loads((out / "ghost" / "ghost.json").read_text())}


@app.function(image=pipe_image, timeout=3 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _s3i(cfg: dict, tag: str, base_tag: str, mold_tag: str, code_rel: str = "", trellis_tag: str = "", image_inpaint: int = 0,
         inpaint_iters: int = 600):
    """Bước 2 bản cơ sở: lấp khuôn bằng Gaussian dẹt (pipeline/s3i_mold_fill.py), đo ảnh test + camera ảo quanh vật.
    code_rel: thư mục code pipeline trên Volume (tools/push_code.py) để chắc chắn chạy đúng bản script; rỗng = /opt/pipeline."""
    vol.reload()
    data, local = _prep_come(cfg)
    out = _root(cfg) / "S3i" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Path(VOL) / cfg["scene"] / "logs" / f"s3i_{tag}.log"
    code = str(Path(VOL) / code_rel) if code_rel else "/opt/pipeline"
    with _Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(f"PYTHONPATH=/opt/CoMe:{code} python {code}/s3i_mold_fill.py -m {local} -s {data} "
                          f"--iteration {cfg['S2_come']['iterations']} --cfg {_sdir(cfg, 'S3') / 'config_used.json'} --out {out} "
                          f"--s3 {_sdir(cfg, 'S3')} --mold {_root(cfg) / 'S3m' / mold_tag} --base {_root(cfg) / 'S3c' / base_tag} "
                          f"--support_graph {Path(VOL) / cfg['scene'] / 'support_graph' / 'graph.json'} "
                          f"--image_inpaint {image_inpaint} --inpaint_iters {inpaint_iters}"
                          + (f" --trellis_dir {Path(VOL) / cfg['scene'] / 'trellis_out' / trellis_tag} "
                             f"--trellis_grid {Path(VOL) / cfg['scene'] / 'trellis_in'}" if trellis_tag else ""),
                          "/opt/CoMe", log)
    vol.commit()
    return json.loads((out / "metrics.json").read_text())


@app.function(image=pipe_image, timeout=4 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _s3c(cfg: dict, tag: str, lam_bound: float = 1.0, iters: int = 5000, mold_tag: str = "v1",
         lam_leak: float = 0.0, lam_cover: float = 0.0, aux_every: int = 2, leak_norm: str = "image"):
    """Tinh chỉnh Gaussian bị giới hạn trong khung chặt (pipeline/s3c_constrain.py)."""
    vol.reload()
    data, local = _prep_come(cfg)
    out = _root(cfg) / "S3c" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Path(VOL) / cfg["scene"] / "logs" / f"s3c_{tag}.log"
    with _Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(f"PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/s3c_constrain.py -m {local} -s {data} "
                          f"--iteration {cfg['S2_come']['iterations']} --cfg {_sdir(cfg, 'S3') / 'config_used.json'} --out {out} "
                          f"--s3 {_sdir(cfg, 'S3')} --mold {_root(cfg) / 'S3m' / mold_tag} "
                          f"--labels {Path(VOL) / cfg['S1_masks']['labels_dir']} --lam_bound {lam_bound} --iters {iters} "
                          f"--lam_leak {lam_leak} --lam_cover {lam_cover} --aux_every {aux_every} --leak_norm {leak_norm}",
                          "/opt/CoMe", log)
    vol.commit()
    return json.loads((out / "metrics.json").read_text())


@app.function(image=pipe_image, timeout=3 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _s3m(cfg: dict, tag: str = "v1", tol_cm: float = 1.0, min_vis: int = 3, hull_rel: str = "mask_hull_sam3", protect_vis: int = 3,
         da3_rel: str = "", protect_matter: int = 0):
    """Khung chặt + khuôn bằng bộ render (pipeline/s3m_mold.py): đóng góp Gaussian + khoét hull theo ảnh độ sâu."""
    vol.reload()
    data, local = _prep_come(cfg)
    out = _root(cfg) / "S3m" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Path(VOL) / cfg["scene"] / "logs" / f"s3m_{tag}.log"
    with _Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(f"PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/s3m_mold.py -m {local} -s {data} "
                          f"--iteration {cfg['S2_come']['iterations']} --cfg {_sdir(cfg, 'S3') / 'config_used.json'} --out {out} "
                          f"--s3 {_sdir(cfg, 'S3')} --hulls {Path(VOL) / cfg['scene'] / hull_rel} "
                          f"--labels {Path(VOL) / cfg['S1_masks']['labels_dir']} --tol_cm {tol_cm} --min_vis {min_vis} --protect_vis {protect_vis} "
                          f"--protect_matter {protect_matter}"
                          + (f" --da3_dir {Path(VOL) / cfg['scene'] / da3_rel}" if da3_rel else ""),
                          "/opt/CoMe", log)
    vol.commit()
    return json.loads((out / "metrics.json").read_text())


@app.function(image=pipe_image, timeout=2 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _unseen_check(cfg: dict, vis_rel: str = "visibility_mold", tag: str = "v1"):
    """Kiểm chứng tầm nhìn (pipeline/s3v_unseen_check.py): bỏ Gaussian vật không camera nào thấy, so render với đối chứng."""
    vol.reload()
    data, local = _prep_come(cfg)
    out = _root(cfg) / "S3v" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Path(VOL) / cfg["scene"] / "logs" / f"s3v_{tag}.log"
    with _Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(f"PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/s3v_unseen_check.py -m {local} -s {data} "
                          f"--iteration {cfg['S2_come']['iterations']} --cfg {_sdir(cfg, 'S3') / 'config_used.json'} --out {out} "
                          f"--vis_dir {Path(VOL) / cfg['scene'] / vis_rel} --labels {Path(VOL) / cfg['scene'] / 'sam3_masks' / 'labels'}",
                          "/opt/CoMe", log)
    vol.commit()
    return json.loads((out / "metrics.json").read_text())


@app.function(image=light_img, volumes={VOL: vol}, timeout=3600)
def _clone_stages(cfg: dict, src_run: str, stages: list):
    """Chép các stage đã xong (không phụ thuộc config mới) từ run src_run sang run của cfg, để khỏi chạy lại (vd S0, S2)."""
    vol.reload()
    src = Path(VOL) / cfg["scene"] / "pipeline" / src_run
    out = {}
    for st in stages:
        d = _sdir(cfg, st)
        if not d.exists():
            shutil.copytree(src / DIRNAME[st], d)
        out[st] = _done(cfg, st)
    vol.commit()
    return out


@app.function(image=light_img, volumes={VOL: vol}, timeout=3600)
def _mk_v2(cfg: dict, s4d_tag: str, new_run: str = "v2"):
    """Tạo run mới = bản sao run hiện tại, thay S4b/background_finetuned.ply bằng nền S4d/<tag>; xoá S9, S10 để chạy lại."""
    vol.reload()
    src, dst = _root(cfg), Path(VOL) / cfg["scene"] / "pipeline" / new_run
    dst.mkdir(parents=True, exist_ok=True)
    for sub in sorted(p.name for p in src.iterdir() if p.is_dir() and p.name not in ("S4c", "S4d", "S9_render", "S10_eval")):
        if not (dst / sub).exists():
            shutil.copytree(src / sub, dst / sub)
    b = dst / DIRNAME["S4b"]
    shutil.copy2(Path(VOL) / cfg["scene"] / "pipeline" / cfg["run_name"] / "S4d" / s4d_tag / "S4like" / "background_finetuned.ply",
                 b / "background_finetuned.ply")
    (b / "NOTE.txt").write_text(f"background_finetuned.ply thay bang S4d/{s4d_tag} cua run {cfg['run_name']}" + chr(10))
    for st in ("S9", "S10"):
        shutil.rmtree(dst / DIRNAME[st], ignore_errors=True)
    vol.commit()
    return sorted(p.name for p in dst.iterdir())


@app.function(image=pipe_image, timeout=3 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _s9v3(cfg: dict, tag: str, code_rel: str, filled_tag: str = "i_r1v3k", mold_tag: str = "v3",
          bg_rel: str = "pipeline/v1/S4d/e8/S4like/background_finetuned.ply", lift_rel: str = "pipeline/v3_sam3/lift_v4",
          eval_tag: str = ""):
    """Render Gaussian cả cảnh theo quỹ đạo MuJoCo (pipeline/s9v3_render.py), code lấy từ Volume (tools/push_code.py)."""
    vol.reload()
    data, local = _prep_come(cfg)
    out = _root(cfg) / "S9v3" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Path(VOL) / cfg["scene"] / "logs" / f"s9v3_{tag}.log"
    code = str(Path(VOL) / code_rel)
    V = Path(VOL) / cfg["scene"]
    with _Committer():
        mc._run_train(f"PYTHONPATH=/opt/CoMe:{code} python {code}/s9v3_render.py -m {local} -s {data} "
                      f"--iteration {cfg['S2_come']['iterations']} --cfg {_sdir(cfg, 'S3') / 'config_used.json'} --out {out} "
                      f"--bg {V / bg_rel} --filled {_root(cfg) / 'S3i' / filled_tag} --frames {_root(cfg) / 'S3m' / mold_tag / 'frame'} "
                      f"--s3 {_sdir(cfg, 'S3')}" + (f" --lift {V / lift_rel}" if lift_rel else "")
                      + (f" --eval_tag {eval_tag}" if eval_tag else ""), "/opt/CoMe", log)
    vol.commit()
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


@app.function(image=pipe_image, timeout=3 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _full_mesh(cfg: dict, tag: str = "vox010", voxel_size: float = 0.01):
    """Mesh bề mặt CẢ cảnh từ CoMe 30k (TSDF của tác giả: Based_Model/CoMe/extract_mesh_tsdf.py), đơn vị cảnh COLMAP."""
    vol.reload()
    data, local = _prep_come(cfg)
    out = _root(cfg) / "FullMesh" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Path(VOL) / cfg["scene"] / "logs" / f"fullmesh_{tag}.log"
    with _Committer():
        if not (out / "tsdf.ply").exists():
            mc._run_train(f"python extract_mesh_tsdf.py -m {local} -s {data} --iteration {cfg['S2_come']['iterations']} "
                          f"--voxel_size {voxel_size}", "/opt/CoMe", log)
            src = local / "test" / f"ours_{cfg['S2_come']['iterations']}" / "tsdf.ply"
            shutil.copy2(src, out / "tsdf.ply")
    vol.commit()
    return {"tag": tag, "voxel_size": voxel_size, "MB": round((out / "tsdf.ply").stat().st_size / 1e6, 1)}


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s10(cfg: dict, force: bool = False):
    """Đo lường mọi stage (pipeline/s10_*.py). Cần LERF-Mask teatime trên Volume tại S10_eval.lerf_dir."""
    if _done(cfg, "S10") and not force:
        return "skip"
    t0 = time.time(); d = _begin(cfg, "S10", force)
    V = Path(VOL) / cfg["scene"]
    if not (d / "eval_gpu.json").exists():
        _run_script(cfg, "S10", "s10_eval_gpu.py",
                    f"--lerf {Path(VOL) / cfg['S10_eval']['lerf_dir']} --s3 {_sdir(cfg, 'S3')} "
                    f"--s4b {_sdir(cfg, 'S4b')} --s5 {_sdir(cfg, 'S5')} --s4a {_sdir(cfg, 'S4a')} "
                    f"--labels {Path(VOL) / cfg['S1_masks']['labels_dir']} "
                    f"--n_input {len(list((V / 'colmap' / 'input').glob('*.jpg')))}")
    vol.commit()
    _s10_cpu.remote(cfg)
    vol.reload()
    _finish(cfg, "S10", t0)
    return "done"


@app.function(image=pipe_image, timeout=1800, gpu="L4", volumes={VOL: vol}, cpu=8, memory=32768)
def debug_tsdf(cfg: dict, obj: str = "coc_su", chunks: str = "50000,200000,1000000"):
    data, local = _prep_come(cfg)
    objs = list(cfg["S3_label"]["objects"]) if obj == "all" else [obj]
    for ob, ch in [(o, c) for o in objs for c in chunks.split(",")]:
        obj = ob
        cmd = (f"CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/debug_tsdf.py "
               f"-m {local} -s {data} --iteration {cfg['S2_come']['iterations']} "
               f"--cfg {_sdir(cfg, 'S6') / 'config_used.json'} --out /tmp/dbg "
               f"--ply {_sdir(cfg, 'S5') / 'objects' / obj / 'gaussians.ply'} --chunk {ch}")
        try:
            mc._run_train(cmd, "/opt/CoMe", Path(VOL) / cfg["scene"] / "logs" / "pipeline_debug.log")
        except Exception as e:
            print(f"chunk {ch}: lỗi {str(e)[-300:]}", flush=True)


@app.local_entrypoint()
def dbg(config: str = "configs/pipeline/teatime_v1.json", obj: str = "coc_su", chunks: str = "50000,200000,1000000"):
    debug_tsdf.remote(_load_cfg(config), obj, chunks)


FN = {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4a": s4a, "S4b": s4b, "S5": s5, "S6": s6, "S7": s7, "S8": s8,
      "S9": s9, "S10": s10}


# ============================================================================================ điều phối (máy local)
def _load_cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@app.local_entrypoint()
def run(config: str = "configs/pipeline/teatime_v1.json", until: str = "S6", only: str = "", force: str = ""):
    cfg = _load_cfg(config)
    stages = STAGES[:STAGES.index(until) + 1] if not only else [only]
    forced = set()
    if force:
        forced = set(STAGES[STAGES.index(force):])                # ép chạy lại stage này và mọi stage sau
    print(f"Pipeline {cfg['scene']}/{cfg['run_name']}: {stages}  (ép chạy lại: {sorted(forced) or 'không'})")
    done = _done_map.remote(cfg)                                   # kiểm tra bằng container CPU nhẹ, khỏi bật GPU
    for st in stages:
        if done.get(st) and st not in forced:
            print(f"<-- {st}: skip (đã xong, config không đổi)", flush=True)
            continue
        t = time.time()
        print(f"--> {st} ...", flush=True)
        r = FN[st].remote(cfg, st in forced)
        print(f"<-- {st}: {r} ({(time.time() - t) / 60:.1f} phút)", flush=True)
    print("XONG. Xem kết quả: python -m modal run modal_pipeline.py::pull --config", config)


@app.function(image=light_img, volumes={VOL: vol}, timeout=300)
def _done_map(cfg: dict) -> dict:
    vol.reload()
    return {st: _done(cfg, st) for st in STAGES}


@app.function(image=light_img, volumes={VOL: vol}, timeout=300)
def _status(cfg: dict):
    vol.reload()
    rows = []
    for st in STAGES:
        f = _sdir(cfg, st) / "_done.json"
        if f.exists():
            info = json.loads(f.read_text())
            ok = info.get("hash") == _stage_hash(cfg, st)
            rows.append(f"{st:4s} {'XONG' if ok else 'CŨ (config đã đổi)'}  {info.get('finished')}  "
                        f"{info.get('minutes_this_run')} phút")
        else:
            rows.append(f"{st:4s} chưa xong")
    return "\n".join(rows)


@app.local_entrypoint()
def status(config: str = "configs/pipeline/teatime_v1.json"):
    print(_status.remote(_load_cfg(config)))


@app.local_entrypoint()
def pull(config: str = "configs/pipeline/teatime_v1.json", with_ply: bool = False):
    """Tải mọi file trừ .ply/.pt (trừ khi --with-ply) về Data/<scene>/pipeline/<run>/."""
    cfg = _load_cfg(config)
    remote_root = f"/{cfg['scene']}/pipeline/{cfg['run_name']}"
    local_root = REPO_DIR / "Data" / cfg["scene"] / "pipeline" / cfg["run_name"]
    n = 0
    for e in vol.listdir(remote_root, recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        if not with_ply and (e.path.endswith(".ply") or e.path.endswith(".pt")):
            continue
        rel = e.path[len(remote_root.lstrip("/")):].lstrip("/") if e.path.startswith(remote_root.lstrip("/")) else e.path
        dst = local_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        n += 1
    print(f"Đã tải {n} file -> {local_root}")


@app.local_entrypoint()
def v5chain(config: str = "configs/pipeline/teatime_v3_sam3.json", mold_tag: str = "v5_da3", s3c_tag: str = "r1v5",
            s3i_tag: str = "i_r1v5", lam_bound: float = 0.1, iters: int = 5000, code_rel: str = ""):
    """S3c (ép khung + lọc khởi động) -> S3i (lấp khuôn) trên khuôn mold_tag. Mỗi hàm tự bỏ qua nếu metrics.json đã có (resume)."""
    cfg = _load_cfg(config)
    r = _s3c.remote(cfg, tag=s3c_tag, lam_bound=lam_bound, iters=iters, mold_tag=mold_tag)
    print("S3C_METRICS", json.dumps(r, ensure_ascii=False)[:6000], flush=True)
    r = _s3i.remote(cfg, tag=s3i_tag, base_tag=s3c_tag, mold_tag=mold_tag, code_rel=code_rel)
    print("S3I_METRICS", json.dumps(r, ensure_ascii=False)[:8000], flush=True)


@app.local_entrypoint()
def s4dtex(config: str = "configs/pipeline/teatime_v1.json", code_rel: str = "", variants: str = "e8r,e8g4,e8tex",
           gpu: str = "A100-80GB"):
    """Thí nghiệm lấp texture ảnh thật: cùng tham số e8 (cắt vết chân + bảo vệ cừu/gấu + lấp + tinh chỉnh ảnh thật masked,
    đóng băng Gaussian lấp), khác nhau ở cách lấp. Chạy song song, mỗi biến thể tự bỏ qua nếu đã có metrics.json."""
    cfg = _load_cfg(config)
    e8 = dict(src_tag="ctrl", cut_dilate=2, reach=2.0, low_cm=3.0, fill_dilate=4, protect=1, ft_iters=3000, lr_scale=0.5,
              ft_mode="masked", freeze_region=1, freeze_dilate=0, code_rel=code_rel)
    V = {"e8r": dict(fill_grid=0.01, texture_fill=0),                     # tái lập e8 (lấp màu vành, ô 1 cm)
         "e8g4": dict(fill_grid=0.004, texture_fill=0),                   # đối chứng: ô 4 mm, màu vành
         "e8tex": dict(fill_grid=0.004, texture_fill=1, tex_scale=0.5)}   # ô 4 mm, texture ảnh thật
    fn = _s4d.with_options(gpu=gpu) if gpu else _s4d        # bước vote bảo vệ cừu/gấu cần 51.9 GiB một lần cấp phát -> L40S (44 GiB) OOM
    calls = {t: fn.spawn(cfg, tag=t, **e8, **V[t]) for t in variants.split(",")}
    for t, c in calls.items():
        try:
            r = c.get()
            print("S4D_RESULT", t, json.dumps(r, ensure_ascii=False)[:5000], flush=True)
        except Exception as ex:
            print("S4D_FAIL", t, repr(ex)[:2000], flush=True)


@app.local_entrypoint()
def mold(config: str = "configs/pipeline/teatime_v3_sam3.json", stage: str = "s3m", tag: str = "v4",
         mold_tag: str = "v4", lam_bound: float = 1.0, iters: int = 5000, protect_vis: int = 3, da3_rel: str = ""):
    """Chạy khung + khuôn (stage s3m: pipeline/s3m_mold.py -> S3m/<tag>) hoặc tinh chỉnh khung (s3c: S3c/<tag> dùng S3m/<mold_tag>)."""
    cfg = _load_cfg(config)
    if stage == "s3m":
        r = _s3m.remote(cfg, tag=tag, protect_vis=protect_vis, da3_rel=da3_rel)
    elif stage == "s3c":
        r = _s3c.remote(cfg, tag=tag, lam_bound=lam_bound, iters=iters, mold_tag=mold_tag)
    else:
        raise SystemExit(f"stage không hợp lệ: {stage}")
    print(json.dumps(r, indent=1, ensure_ascii=False)[:4000])


@app.local_entrypoint()
def s9v5(config: str = "configs/pipeline/teatime_v3_sam3.json", code_rel: str = "", tag: str = "r_v5",
         filled_tag: str = "i_r1v5", mold_tag: str = "v5_da3", lift_rel: str = "pipeline/v3_sam3/lift_v5_com"):
    """S9v3 cho chuỗi v5_da3: nền e8 + vật S3i/<filled_tag> theo quỹ đạo MuJoCo của lift_rel."""
    r = _s9v3.remote(_load_cfg(config), tag=tag, code_rel=code_rel, filled_tag=filled_tag, mold_tag=mold_tag, lift_rel=lift_rel)
    print("S9V3_FILES", len(r), r[:60])


@app.local_entrypoint()
def s9final(config: str = "configs/pipeline/teatime_v3_sam3.json", code_rel: str = "", tag: str = "final_v3_tex",
            filled_tag: str = "i_r1v3k", mold_tag: str = "v3", lift_rel: str = "",
            bg_rel: str = "pipeline/v1/S4d/e8tex/S4like/background_finetuned.ply"):
    """Chot Winning Stack: nen S4d/e8tex (--texture_fill) + vat S3i/i_r1v3k (khung S3m v3); do PSNR / SSIM / LPIPS 23 view test; lift_rel khac rong thi render them quy dao MuJoCo."""
    r = _s9v3.remote(_load_cfg(config), tag=tag, code_rel=code_rel, filled_tag=filled_tag, mold_tag=mold_tag, lift_rel=lift_rel,
                     bg_rel=bg_rel, eval_tag=tag)
    print("S9V3_FILES", len(r), r[:80])


@app.function(image=light_img, volumes={VOL: vol}, timeout=1800)
def _copy_rel(src_rel: str, dst_rel: str) -> bool:
    """Chep 1 file tren Volume (vd contrib.npy: chi phu thuoc canh CoMe, khong phu thuoc nhan / khung). Bo qua neu dich da co."""
    vol.reload()
    src, dst = Path(VOL) / src_rel, Path(VOL) / dst_rel
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    vol.commit()
    return True


@app.local_entrypoint()
def v6chain(config: str = "configs/pipeline/teatime_v6_adap.json", base_config: str = "configs/pipeline/teatime_v3_sam3.json",
            mold_tag: str = "v6_adap", hull_rel: str = "mask_hull_sam3_adap", s3c_tag: str = "r1v6", lam_bound: float = 0.1,
            iters: int = 5000, base_mold: str = "v3", base_s3c: str = "r1v3", protect_vis: int = 3, protect_matter: int = 0,
            ablate: str = "", stop_after_s3m: int = 0):
    """v6: S3 (nhan Gaussian tu mask cong thich nghi) -> S3m/<mold_tag> (hull tu mask moi) -> S3c/<s3c_tag>; song song chay lai
    doi chung S3c/<base_s3c> tren khung v3 (cung code, cung lam_bound) de co final.ply so sanh. Moi buoc tu bo qua khi da xong."""
    cfg, base = _load_cfg(config), _load_cfg(base_config)
    print("S3", s3.remote(cfg), flush=True)
    contrib = f"{base['scene']}/pipeline/{base['run_name']}/S3m/v5_da3/contrib.npy"
    abl_tags = [a_.split(":")[0] for a_ in ablate.split(",") if a_]
    for c_, t_ in [(cfg, mold_tag), (base, base_mold)] + [(cfg, t_) for t_ in abl_tags]:
        print("contrib", t_, _copy_rel.remote(contrib, f"{c_['scene']}/pipeline/{c_['run_name']}/S3m/{t_}/contrib.npy"), flush=True)
    base_call = None if stop_after_s3m else _s3c.spawn(base, tag=base_s3c, lam_bound=lam_bound, iters=iters, mold_tag=base_mold)
    # ablate: "tag:protect_vis:protect_matter,..." extra S3m variants on the same S3 / hull (frames only, no S3c)
    abl = [(a_.split(":")[0], int(a_.split(":")[1]), int(a_.split(":")[2])) for a_ in ablate.split(",") if a_]
    abl_calls = [_s3m.spawn(cfg, tag=t_, hull_rel=hull_rel, protect_vis=pv_, protect_matter=pm_) for t_, pv_, pm_ in abl]
    r = _s3m.remote(cfg, tag=mold_tag, hull_rel=hull_rel, protect_vis=protect_vis, protect_matter=protect_matter)
    print("S3M_METRICS", json.dumps(r, ensure_ascii=False)[:6000], flush=True)
    for (t_, _, _), c_ in zip(abl, abl_calls):
        print("S3M_ABLATE", t_, json.dumps(c_.get().get("B_carve", {}).get("protect_matter")), flush=True)
    if stop_after_s3m:
        return
    r = _s3c.remote(cfg, tag=s3c_tag, lam_bound=lam_bound, iters=iters, mold_tag=mold_tag)
    print("S3C_METRICS", json.dumps({k: r[k] for k in ("before", "after", "frame_check") if k in r}, ensure_ascii=False)[:8000], flush=True)
    rb = base_call.get()
    print("S3C_BASE_METRICS", json.dumps({k: rb[k] for k in ("before", "after", "frame_check") if k in rb}, ensure_ascii=False)[:8000], flush=True)


@app.function(image=pipe_image, timeout=3 * 3600, **{**GPU_KW, "gpu": ["L40S", "L4"]})
def _inspect_render(cfg: dict, tag: str, runs: list, objects: str = "ly_tra,khan_giay,coc_su,dia,banh_quy,tao,tui_banh,goi_keo,thia,all",
                    n_frames: int = 72, size: int = 512):
    """Orbit Gaussian vat sau S3c (pipeline/s3c_inspect_render.py). runs: ["NHAN=<rel S3c>:<rel S3m>", ...] (rel theo Volume)."""
    vol.reload()
    out = _root(cfg) / "inspect" / tag
    if (out / "stats.json").exists():
        return json.loads((out / "stats.json").read_text())
    data, local = _prep_come(cfg)
    out.mkdir(parents=True, exist_ok=True)
    rr = " ".join(f"--run {r.split('=')[0]}={Path(VOL) / r.split('=')[1].split(':')[0]}:{Path(VOL) / r.split('=')[1].split(':')[1]}"
                  for r in runs)
    with _Committer():
        mc._run_train(f"PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/s3c_inspect_render.py -m {local} -s {data} "
                      f"--iteration {cfg['S2_come']['iterations']} --cfg {_sdir(cfg, 'S3') / 'config_used.json'} --out {out} "
                      f"--s3 {_sdir(cfg, 'S3')} {rr} --objects {objects} --n_frames {n_frames} --size {size}",
                      "/opt/CoMe", Path(VOL) / cfg["scene"] / "logs" / f"inspect_{tag}.log")
    vol.commit()
    return json.loads((out / "stats.json").read_text())


@app.local_entrypoint()
def inspect(config: str = "configs/pipeline/teatime_v6_adap.json", tag: str = "v6_vs_v3",
            runs: str = "v3_sam3=teatime/pipeline/v3_sam3/S3c/r1v3:teatime/pipeline/v3_sam3/S3m/v3,"
                        "v6_adap=teatime/pipeline/v6_adap/S3c/r1v6:teatime/pipeline/v6_adap/S3m/v6_adap",
            n_frames: int = 72):
    r = _inspect_render.remote(_load_cfg(config), tag, runs.split(","), n_frames=n_frames)
    print("INSPECT_STATS", json.dumps(r, ensure_ascii=False)[:6000], flush=True)


@app.local_entrypoint()
def leakrun(config: str = "configs/pipeline/teatime_v8_hullv4.json", tag: str = "r1v8_leak", mold_tag: str = "v8_hullv4",
            lam_bound: float = 0.1, lam_leak: float = 0.05, lam_cover: float = 0.02, aux_every: int = 2, iters: int = 5000,
            leak_norm: str = "image"):
    """Track A: S3c voi L_leak + L_cover (alpha rieng tung vat so voi mask SAM 3 tren view train) tren khung mold_tag."""
    r = _s3c.remote(_load_cfg(config), tag=tag, lam_bound=lam_bound, iters=iters, mold_tag=mold_tag,
                    lam_leak=lam_leak, lam_cover=lam_cover, aux_every=aux_every, leak_norm=leak_norm)
    print("S3C_METRICS", json.dumps({k: r[k] for k in r if k != "frame_check"}, ensure_ascii=False)[:9000], flush=True)


@app.local_entrypoint()
def imgfill(config: str = "configs/pipeline/teatime_v8_hullv4.json", tag: str = "i_r1v8_img", base_tag: str = "r1v8_leak",
            mold_tag: str = "v8_hullv4", image_inpaint: int = 1, inpaint_iters: int = 600):
    """Track B: S3i lap khuon + tinh chinh Gaussian lap theo anh camera ao da inpaint (pipeline/s3i_mold_fill.py --image_inpaint)."""
    r = _s3i.remote(_load_cfg(config), tag=tag, base_tag=base_tag, mold_tag=mold_tag, image_inpaint=image_inpaint,
                    inpaint_iters=inpaint_iters)
    print("S3I_METRICS", json.dumps(r, ensure_ascii=False)[:12000], flush=True)


@app.local_entrypoint()
def figchain(config: str = "configs/pipeline/figurines_v1.json", lam_bound: float = 0.1, lam_leak: float = 0.05,
             lam_cover: float = 0.02, iters: int = 5000, stop_after: str = "", s3i_tag: str = "", image_inpaint: int = 1):
    """Track C: canh thu hai chay dung chuoi chung: S3 (gan nhan Gaussian) -> S3m/<mold_tag> (hull S1_masks.hull_rel, protect_vis 3,
    protect_matter) -> S3c/<s3c_tag> (lam_bound + L_leak + L_cover). Can <scene>/come (CoMe 30k) + hull da put len Volume."""
    cfg = _load_cfg(config)
    c = cfg["S1_masks"]
    print("S3", s3.remote(cfg), flush=True)
    if stop_after == "S3":
        return
    r = _s3m.remote(cfg, tag=c["mold_tag"], hull_rel=c["hull_rel"], protect_vis=3, protect_matter=int(c.get("protect_matter", 1)))
    print("S3M_METRICS", json.dumps(r, ensure_ascii=False)[:8000], flush=True)
    if stop_after == "S3m":
        return
    r = _s3c.remote(cfg, tag=c["s3c_tag"], lam_bound=lam_bound, iters=iters, mold_tag=c["mold_tag"],
                    lam_leak=lam_leak, lam_cover=lam_cover)
    print("S3C_METRICS", json.dumps({k: r[k] for k in r}, ensure_ascii=False)[:12000], flush=True)
    if s3i_tag:     # S3i: lap khuon (+ inpaint anh camera ao) tren ket qua S3c
        fn = _s3i.with_options(gpu="L40S")     # inpaint 32 camera ao: L4 22 GB OOM (figurines v3)
        r = fn.remote(cfg, tag=s3i_tag, base_tag=c["s3c_tag"], mold_tag=c["mold_tag"], image_inpaint=image_inpaint)
        print("S3I_METRICS", json.dumps(r, ensure_ascii=False)[:12000], flush=True)
