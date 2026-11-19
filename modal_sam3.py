"""
S1 v2: mask có ID cho teatime bằng SAM 3 (facebookresearch/sam3 @ f1647e8, checkpoint HF facebook/sam3) theo prompt chữ.

    python -m modal deploy modal_sam3.py && python -c "import modal; modal.Function.from_name('safe-gs-sam3','_run').spawn()"
    python -m modal run modal_sam3.py::pull                 # tải labels + preview về Data/teatime/sam3_masks/

Dùng đúng API video của README / examples/sam3_video_predictor_example.ipynb: start_session (thư mục JPEG đặt tên
<số>.jpg), mỗi prompt: reset_session -> add_prompt(text, frame 0) -> propagate_in_video (mặc định cả 2 chiều).
Ảnh đầu vào = 177 ảnh đã undistort của COLMAP (cùng ảnh CoMe train) -> mask khớp camera.
Gộp: mỗi instance của prompt p nhận nhãn p*10 + k (k theo thứ tự xuất hiện); prompt sau đè prompt trước (vật nhỏ sau).
Checkpoint: mỗi prompt lưu masks_<p>.npz lên Volume ngay khi xong; chạy lại thì bỏ qua prompt đã có.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import modal

import modal_come as mc

app = modal.App("safe-gs-sam3")
vol, VOL = mc.vol, mc.VOL
SAM3_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
ROOT = Path(VOL) / "teatime" / "sam3_masks"

# thứ tự = độ ưu tiên tăng dần (prompt sau đè lên)
PROMPTS = ["table", "teddy bear", "stuffed sheep", "paper napkin", "paper bag", "plate", "glass of tea",
           "coffee mug", "apple", "cookie", "spoon", "creamer cup",
           # lần 2: 4 và 12 ra 0 instance; cà phê trong cốc không thuộc mask cốc
           "napkin", "coffee", "creamer",
           # lần 3 (thử bằng chế độ ảnh, xem probe/cand_*.jpg): vật nhỏ còn thiếu
           "small white cup", "puddle of liquid", "blue snack package"]
# instance bắt nhầm (xem preview lần 1): 21 = "teddy bear" trên con cừu, 31 = "stuffed sheep" trên con gấu,
# 61 / 91 = plate / apple chỉ 2 / 7 frame; 142 = "coffee" trên túi giấy
# + prompt "table": chỉ giữ 13 (bàn chính). 10-12, 14-19 bắt nhầm đầu gấu, cổ cừu, vũng trà lốm đốm, bàn xa
DROP = [21, 31, 61, 91, 142, 10, 11, 12, 14, 15, 16, 17, 18, 19]
# gộp: cà phê vào cốc, trà vào ly (mask cốc / ly bị thủng đúng chỗ đó), các miếng bánh quy thành 1 nhóm
MERGE = {140: 80, 141: 70, 101: 100, 102: 100, 103: 100, 104: 100}
# prompt lần 3 hay bắt cả vật đã có nhãn (vd "small white cup" bắt cả cốc lớn): bỏ instance trong frame đó nếu
# > 30% diện tích đã có nhãn vật khác (trừ bàn 13)
CHECK_OVERLAP = ["small white cup", "puddle of liquid", "blue snack package"]
MIN_PIECE = 40                      # xoá mảnh rời nhỏ hơn 40 px của mỗi nhãn trong mỗi frame

img = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")   # README
    .run_commands(f"git clone -q https://github.com/facebookresearch/sam3.git /opt/sam3 && "
                  f"git -C /opt/sam3 checkout -q {SAM3_COMMIT} && cd /opt/sam3 && pip install -e .")
    .pip_install("opencv-python-headless", "pillow", "einops", "pycocotools",
                 "psutil")                                    # sam3_video_predictor.py import, không có trong pyproject
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf_cache"})
    .add_local_python_source("modal_come")
)


@app.function(image=img, gpu="L4", volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=49152,
              secrets=[modal.Secret.from_name("huggingface")],
              retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0))
def _run(prompts: list = None):
    import numpy as np
    import cv2
    from sam3.model_builder import build_sam3_video_predictor

    vol.reload()
    prompts = prompts or PROMPTS
    src = sorted((Path(VOL) / "teatime" / "colmap" / "images").glob("frame_*.jpg"))
    frames = Path("/tmp/frames"); shutil.rmtree(frames, ignore_errors=True); frames.mkdir()
    for i, f in enumerate(src):
        shutil.copy2(f, frames / f"{i:05d}.jpg")                # loader cần tên <số>.jpg
    names = [f.stem for f in src]
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "frames.json").write_text(json.dumps(names))
    H, W = cv2.imread(str(src[0])).shape[:2]

    pred = build_sam3_video_predictor()
    sid = pred.handle_request(dict(type="start_session", resource_path=str(frames)))["session_id"]
    for p, text in enumerate(prompts, start=1):
        out_f = ROOT / f"masks_{p:02d}.npz"
        if out_f.exists():
            mc._tlog(f"[sam3] skip '{text}'"); continue
        t0 = time.time()
        pred.handle_request(dict(type="reset_session", session_id=sid))
        pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
        per = {}
        for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
            o = r["outputs"]
            per[r["frame_index"]] = {int(i): np.packbits(m) for i, m in zip(o["out_obj_ids"].tolist(), o["out_binary_masks"])
                                     if m.any()}
        ids = sorted({i for d in per.values() for i in d}, key=lambda i: min(f for f, d in per.items() if i in d))
        arr = {}
        for f, d in per.items():
            for i, m in d.items():
                arr[f"{f}_{i}"] = m
        np.savez_compressed(out_f, **arr)
        info = {"prompt": text, "instances": [{"obj_id": i, "label": p * 10 + k,
                                                "frames": sum(i in d for d in per.values())}
                                               for k, i in enumerate(ids[:10])],
                "minutes": round((time.time() - t0) / 60, 2)}
        (ROOT / f"masks_{p:02d}.json").write_text(json.dumps(info, ensure_ascii=False))
        vol.commit()
        mc._tlog(f"[sam3] '{text}': {len(ids)} instance, {info['minutes']} phút")

    # gộp thành ảnh nhãn (uint8) cho từng frame, prompt sau đè prompt trước
    lab_dir = ROOT / "labels"; lab_dir.mkdir(exist_ok=True)
    prev = ROOT / "preview"; prev.mkdir(exist_ok=True)
    labels = np.zeros((len(names), H, W), np.uint8)
    prompt_of = {}
    for p in range(1, len(prompts) + 1):
        info = json.loads((ROOT / f"masks_{p:02d}.json").read_text())
        z = np.load(ROOT / f"masks_{p:02d}.npz")
        lab_of = {d["obj_id"]: d["label"] for d in info["instances"] if d["label"] not in DROP}
        for d in info["instances"]:
            if d["label"] not in DROP:
                prompt_of[MERGE.get(d["label"], d["label"])] = info["prompt"] if d["label"] not in MERGE else \
                    prompt_of.get(MERGE[d["label"]], info["prompt"])
        for key in z.files:
            f, i = map(int, key.split("_"))
            if i in lab_of:
                m = np.unpackbits(z[key])[: H * W].reshape(H, W).astype(bool)
                if info["prompt"] in CHECK_OVERLAP:
                    occ = (labels[f] != 0) & (labels[f] != 13)
                    if (m & occ).sum() > 0.3 * m.sum():
                        continue
                labels[f][m] = MERGE.get(lab_of[i], lab_of[i])
    # xoá mảnh rời nhỏ
    for f in range(len(names)):
        for lb in np.unique(labels[f]):
            if lb == 0:
                continue
            n, cc, st, _ = cv2.connectedComponentsWithStats((labels[f] == lb).astype(np.uint8), connectivity=8)
            for c in range(1, n):
                if st[c, cv2.CC_STAT_AREA] < MIN_PIECE:
                    labels[f][cc == c] = 0
    legend = {}
    for lb in np.unique(labels):
        if lb:
            legend[int(lb)] = {"prompt": prompt_of.get(int(lb), "?"), "frames": int((labels == lb).any((1, 2)).sum())}
    merged_note = {str(k): v for k, v in MERGE.items()}
    (ROOT / "merge_rules.json").write_text(json.dumps({"merge": merged_note, "drop": DROP, "min_piece_px": MIN_PIECE,
                                                       "check_overlap": CHECK_OVERLAP}, indent=1))
    rng = np.random.default_rng(0); pal = rng.integers(40, 255, (256, 3)).astype(np.uint8); pal[0] = 0
    for k, n in enumerate(names):
        cv2.imwrite(str(lab_dir / f"{n}.png"), labels[k])
        if k % 20 == 0:
            im = cv2.imread(str(src[k]))
            ov = (0.45 * im + 0.55 * pal[labels[k]][..., ::-1]).astype(np.uint8)
            for lb in np.unique(labels[k]):
                if lb == 0:
                    continue
                ys, xs = np.nonzero(labels[k] == lb)
                cv2.putText(ov, f"{lb}:{legend[int(lb)]['prompt']}", (int(xs.mean()) - 30, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(prev / f"{n}.jpg"), np.hstack([im, ov]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (ROOT / "legend.json").write_text(json.dumps(legend, indent=1, ensure_ascii=False))
    vol.commit()
    return legend


# ---------------------------------------------------------------------------------------------------------------
# Opt-in: mask theo cổng Relative Top-1 (tools/sam3_gate_eval.py, LERF-Mask 92.0 mIoU ở floor 0.2, rho 0.5).
# Chế độ ảnh của SAM 3 cho TỪNG frame (giống bản đã chấm), cùng một quy tắc cho mọi prompt, KHÔNG có DROP / MERGE tay.
# Nhãn giữ đúng bảng ID của bản video (OBJECTS_SAM3 trong tools/mask_hull.py, config S3_label) để dùng lại toàn bộ pipeline.
# Prompt: dùng đúng câu LERF-Mask khi vật có câu đó (đã có điểm GT), còn lại giữ câu của bản video. Thứ tự = ưu tiên tăng dần.
ADAPTIVE_PROMPTS = [("table", 13), ("stuffed bear", 20), ("sheep", 30), ("paper napkin", 130), ("bag of cookies", 50),
                    ("plate", 60), ("tea in a glass", 70), ("coffee mug", 80), ("apple", 90), ("cookies on a plate", 100),
                    ("spoon", 110), ("small white cup", 160), ("puddle of liquid", 170), ("blue snack package", 180)]


def _gate(P, S, floor, rho):
    """P [N,H,W] xác suất (0..1), S [N] điểm. Top-1 nếu S >= floor; thêm instance có S >= rho * top1. Trả mask bool hoặc None."""
    import numpy as np
    if len(S) == 0:
        return None
    j0 = int(np.argmax(S))
    if S[j0] < floor:
        return None
    keep = S >= rho * S[j0]
    return (P[keep] >= 0.5).any(0)


@app.function(image=img, gpu="L4", volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=49152,
              secrets=[modal.Secret.from_name("huggingface")],
              retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0))
def _run_adaptive(tag: str = "sam3_masks_adaptive", floor: float = 0.2, rho: float = 0.5, low_floor: float = -1.0,
                  hyst_overlap: float = 0.5, hyst_radius: int = 1, scene: str = "teatime", prompts: list = None,
                  check_overlap: list = None, support_label: int = 13, overlap_mode: str = "area30", per_instance: int = 0):
    """Mask từng frame bằng Sam3Processor + cổng Relative Top-1. Checkpoint: mỗi frame lưu raw/<frame>.npz (mask đã gate
    theo prompt, packbits) + commit mỗi 10 frame; chạy lại bỏ qua frame đã có. Cuối cùng gộp thành labels/ như bản video.
    v7 (low_floor >= 0): raw lưu mask của mọi prompt có top1 >= low_floor; khi gộp, phát hiện yếu (low_floor <= top1 < floor)
    chỉ được nhận nếu một frame kề (|t - t'| <= hyst_radius) có phát hiện mạnh (top1 >= floor) của CÙNG prompt và
    >= hyst_overlap diện tích mask yếu nằm trong mask mạnh đó nới 2 % đường chéo ảnh (trễ theo thời gian, cùng quy tắc mọi prompt).
    low_floor < 0: như cũ (chỉ floor).
    per_instance = 1: raw luu RIENG tung instance da qua cong (khoa "<nhan>_<j>", j theo diem giam dan) thay vi hop; khi gop,
    loc chong lap (CHECK, swallow) xet TUNG instance -> instance bat nham vat lon bi bo nhung instance dung cung frame van giu."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    vol.reload()
    ADAPTIVE = [tuple(x) for x in prompts] if prompts else ADAPTIVE_PROMPTS     # [(prompt, nhan), ...] theo canh
    CHECK = CHECK_OVERLAP if check_overlap is None else check_overlap
    root = Path(VOL) / scene / tag
    raw = root / "raw"; raw.mkdir(parents=True, exist_ok=True)
    src = sorted((Path(VOL) / scene / "colmap" / "images").glob("frame_*.jpg"))
    names = [f.stem for f in src]
    (root / "frames.json").write_text(json.dumps(names))
    H, W = cv2.imread(str(src[0])).shape[:2]
    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.05)   # như _probe (bản đã chấm)
    store_floor = floor if low_floor < 0 else min(low_floor, floor)
    t0 = time.time()
    for k, (f, n) in enumerate(zip(src, names)):
        out_f = raw / f"{n}.npz"
        if out_f.exists():
            continue
        st0 = proc.set_image(Image.open(f).convert("RGB"))
        arr, meta = {}, {}
        for p, lb in ADAPTIVE:
            proc.reset_all_prompts(st0)
            st = proc.set_text_prompt(state=st0, prompt=p)
            S = st["scores"].float().cpu().numpy() if ("scores" in st and len(st["scores"])) else np.zeros(0, np.float32)
            P = st["masks_logits"][:, 0].float().cpu().numpy() if len(S) else None
            m = _gate(P, S, store_floor, rho) if len(S) else None
            meta[p] = {"n": int(len(S)), "top1": float(S.max()) if len(S) else 0.0,
                       "kept": int((S >= rho * S.max()).sum()) if (len(S) and S.max() >= store_floor) else 0}
            if m is not None and per_instance:
                order = np.argsort(-S)
                order = order[S[order] >= rho * S[order[0]]]
                meta[p]["scores"] = [float(S[j]) for j in order]
                for r_, j in enumerate(order):
                    mi = P[j] >= 0.5
                    if mi.shape != (H, W):
                        mi = cv2.resize(mi.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    if mi.any():
                        arr[f"{lb}_{r_}"] = np.packbits(mi)
            elif m is not None:
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                arr[str(lb)] = np.packbits(m)
        np.savez_compressed(out_f, meta=json.dumps(meta), **arr)
        if k % 10 == 0:
            vol.commit()
            mc._tlog(f"[sam3-adap] {k + 1}/{len(src)} frame, {(time.time() - t0) / 60:.1f} phút")
    vol.commit()

    # gộp: prompt sau đè prompt trước; vật nhỏ (CHECK_OVERLAP) bỏ nếu > 30 % diện tích đã có nhãn vật khác (trừ bàn 13)
    lab_dir = root / "labels"; lab_dir.mkdir(exist_ok=True)
    prev = root / "preview"; prev.mkdir(exist_ok=True)
    name_of = {lb: p for p, lb in ADAPTIVE}
    rng = np.random.default_rng(0); pal = rng.integers(40, 255, (256, 3)).astype(np.uint8); pal[0] = 0
    legend, per_frame = {}, {}
    metas = [json.loads(str(np.load(raw / f"{n}.npz")["meta"])) for n in names]
    rad = max(1, int(round(0.02 * float(np.hypot(H, W)))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    _mcache = {}

    def mask_of(k, lb):
        key = (k, lb)
        if key not in _mcache:
            z = np.load(raw / f"{names[k]}.npz")
            ks = inst_keys(z, lb)
            _mcache[key] = np.any([np.unpackbits(z[k_])[: H * W].reshape(H, W).astype(bool) for k_ in ks], 0) if ks else None
        return _mcache[key]

    def inst_keys(z, lb):
        if str(lb) in z.files:
            return [str(lb)]
        return sorted([k_ for k_ in z.files if k_.startswith(f"{lb}_")], key=lambda k_: int(k_.split("_")[1]))

    hyst = {p: {"strong": 0, "weak_accepted": 0, "weak_rejected": 0} for p, _ in ADAPTIVE}

    def accept(k, p, lb):
        t1 = metas[k][p]["top1"]
        if t1 >= floor:
            hyst[p]["strong"] += 1
            return True
        if low_floor < 0 or t1 < store_floor:
            return False
        m = mask_of(k, lb)
        if m is None or not m.any():
            return False
        for j in range(max(0, k - hyst_radius), min(len(names), k + hyst_radius + 1)):
            if j == k or metas[j][p]["top1"] < floor:
                continue
            mj = mask_of(j, lb)
            if mj is None:
                continue
            mjd = cv2.dilate(mj.astype(np.uint8), ker) > 0
            if (m & mjd).sum() >= hyst_overlap * m.sum():
                hyst[p]["weak_accepted"] += 1
                return True
        hyst[p]["weak_rejected"] += 1
        return False

    for k, n in enumerate(names):
        z = np.load(raw / f"{n}.npz")
        meta = metas[k]
        per_frame[n] = meta
        lab = np.zeros((H, W), np.uint8)
        for key in [kk for kk in _mcache if kk[0] < k - hyst_radius]:
            del _mcache[key]
        for p, lb in ADAPTIVE:
            if not inst_keys(z, lb) or not accept(k, p, lb):
                continue
            m = mask_of(k, lb)
            if p in CHECK:
                if overlap_mode == "swallow":
                    # bo instance neu no nuot > 50 % dien tich mot vat da co nhan (bat nham vat lon); vat nho NAM TRONG mask vat khac thi giu
                    def swallows(mm):
                        for L in np.unique(lab[mm]):
                            if L in (0, support_label):
                                continue
                            if (mm & (lab == L)).sum() > 0.5 * (lab == L).sum():
                                return True
                        return False
                    parts = [np.unpackbits(z[k_])[: H * W].reshape(H, W).astype(bool) for k_ in inst_keys(z, lb)]
                    parts = [mm for mm in parts if not swallows(mm)]
                    if not parts:
                        continue
                    m = np.any(parts, 0)
                else:
                    occ = (lab != 0) & (lab != support_label)
                    if (m & occ).sum() > 0.3 * m.sum():
                        continue
            lab[m] = lb
        for lb in np.unique(lab):
            if lb == 0:
                continue
            nc, cc, st, _ = cv2.connectedComponentsWithStats((lab == lb).astype(np.uint8), connectivity=8)
            for c in range(1, nc):
                if st[c, cv2.CC_STAT_AREA] < MIN_PIECE:
                    lab[cc == c] = 0
        for lb in np.unique(lab):
            if lb:
                legend.setdefault(int(lb), {"prompt": name_of[int(lb)], "frames": 0})["frames"] += 1
        cv2.imwrite(str(lab_dir / f"{n}.png"), lab)
        if k % 20 == 0:
            im = cv2.imread(str(src[k]))
            ov = (0.45 * im + 0.55 * pal[lab][..., ::-1]).astype(np.uint8)
            for lb in np.unique(lab):
                if lb == 0:
                    continue
                ys, xs = np.nonzero(lab == lb)
                cv2.putText(ov, f"{lb}:{name_of[int(lb)]}", (int(xs.mean()) - 30, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(prev / f"{n}.jpg"), np.hstack([im, ov]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (root / "legend.json").write_text(json.dumps(legend, indent=1, ensure_ascii=False))
    (root / "gate.json").write_text(json.dumps({"floor": floor, "rho": rho, "low_floor": low_floor, "hyst_overlap": hyst_overlap,
                                                "hyst_radius": hyst_radius, "hyst_dilate_px": rad, "hysteresis": hyst,
                                                "prompts": ADAPTIVE,
                                                "check_overlap": CHECK, "overlap_mode": overlap_mode, "min_piece_px": MIN_PIECE,
                                                "per_instance": per_instance,
                                                "per_frame": per_frame}, ensure_ascii=False))
    vol.commit()
    return {"legend": legend, "hysteresis": hyst}


@app.local_entrypoint()
def adaptive(tag: str = "sam3_masks_adaptive", floor: float = 0.2, rho: float = 0.5, low_floor: float = -1.0,
             hyst_overlap: float = 0.5, hyst_radius: int = 1):
    print(json.dumps(_run_adaptive.remote(tag, floor, rho, low_floor, hyst_overlap, hyst_radius), indent=1, ensure_ascii=False))


@app.local_entrypoint()
def pull_adaptive(tag: str = "sam3_masks_adaptive"):
    out = Path(__file__).resolve().parent / "Data" / "teatime" / tag
    for sub in ("labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for e in vol.listdir(f"teatime/{tag}/{sub}"):
            with open(out / sub / Path(e.path).name, "wb") as f:
                for c in vol.read_file(e.path):
                    f.write(c)
    for name in ("legend.json", "frames.json", "gate.json"):
        with open(out / name, "wb") as f:
            for c in vol.read_file(f"teatime/{tag}/{name}"):
                f.write(c)
    print(f"đã tải về {out}")


LERF_PROMPTS = ["apple", "bag of cookies", "coffee mug", "cookies on a plate", "paper napkin", "plate", "sheep",
                "spoon handle", "stuffed bear", "tea in a glass"]
CANDIDATES = ["sugar packet", "tea bag", "plastic creamer cup", "small white cup", "spilled tea", "puddle of liquid",
              "blue snack package", "candy wrapper", "coffee creamer", "cookie crumbs"]


@app.function(image=img, gpu="L4", volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768,
              secrets=[modal.Secret.from_name("huggingface")])
def _probe():
    """Chế độ ảnh của SAM 3 (README: Sam3Processor): (1) 10 prompt LERF-Mask trên ảnh test, lưu xác suất mask để chọn
    ngưỡng / độ nở bằng ground truth; (2) prompt ứng viên cho vật nhỏ còn thiếu trên vài frame của video."""
    import cv2
    import numpy as np
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    import torch
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()     # như examples/sam3_image_predictor_example.ipynb
    vol.reload()
    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.05)
    out = ROOT / "probe"; out.mkdir(parents=True, exist_ok=True)

    def run(path, prompt):
        st = proc.set_image(Image.open(path).convert("RGB"))
        st = proc.set_text_prompt(state=st, prompt=prompt)
        if "masks_logits" not in st or len(st["scores"]) == 0:
            return np.zeros((0, 1, 1), np.uint8), np.zeros(0, np.float32)
        p = (st["masks_logits"][:, 0].float().cpu().numpy() * 255).astype(np.uint8)
        return p, st["scores"].float().cpu().numpy()

    lerf = Path(VOL) / "teatime" / "benchmarks" / "lerf_mask" / "teatime" / "images"
    arr = {}
    for v in ("test_0", "test_2"):
        st0 = proc.set_image(Image.open(lerf / f"{v}.jpg").convert("RGB"))
        for pr in LERF_PROMPTS:
            proc.reset_all_prompts(st0)
            st = proc.set_text_prompt(state=st0, prompt=pr)
            if len(st["scores"]):
                arr[f"{v}|{pr}|p"] = (st["masks_logits"][:, 0].float().cpu().numpy() * 255).astype(np.uint8)
                arr[f"{v}|{pr}|s"] = st["scores"].float().cpu().numpy()
    np.savez_compressed(out / "lerf_probs.npz", **arr)
    vol.commit()
    mc._tlog(f"[sam3] probe LERF: {len(arr) // 2} (view, prompt) có mask")

    src = sorted((Path(VOL) / "teatime" / "colmap" / "images").glob("frame_*.jpg"))
    ks = [0, 20, 60, 90, 120, 150]
    rows = []
    rng = np.random.default_rng(1); pal = rng.integers(60, 255, (64, 3)).astype(np.uint8)
    for pr in CANDIDATES:
        tiles = []
        for k in ks:
            im = cv2.imread(str(src[k]))
            p, s = run(src[k], pr)
            ov = im.copy()
            n = 0
            for j in range(len(s)):
                if s[j] < 0.3:
                    continue
                m = cv2.resize(p[j], (im.shape[1], im.shape[0])) > 127
                ov[m] = (0.4 * ov[m] + 0.6 * pal[j % 64][::-1]).astype(np.uint8); n += 1
            cv2.putText(ov, f"{pr} #{k}: {n} inst, max {s.max() if len(s) else 0:.2f}", (6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tiles.append(cv2.resize(ov, (480, int(ov.shape[0] * 480 / ov.shape[1]))))
        rows.append(np.hstack(tiles[:3])); rows.append(np.hstack(tiles[3:]))
    for g in range(0, len(rows), 6):
        cv2.imwrite(str(out / f"cand_{g // 6}.jpg"), np.vstack(rows[g:g + 6]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    vol.commit()
    return "ok"


@app.local_entrypoint()
def pull_probe():
    out = Path(__file__).resolve().parent / "Data" / "teatime" / "sam3_masks" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    for e in vol.listdir("teatime/sam3_masks/probe"):
        with open(out / Path(e.path).name, "wb") as f:
            for c in vol.read_file(e.path):
                f.write(c)
    print(f"đã tải về {out}")


@app.local_entrypoint()
def pull():
    out = Path(__file__).resolve().parent / "Data" / "teatime" / "sam3_masks"
    for sub in ("labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for e in vol.listdir(f"teatime/sam3_masks/{sub}"):
            with open(out / sub / Path(e.path).name, "wb") as f:
                for c in vol.read_file(e.path):
                    f.write(c)
    for name in ("legend.json", "frames.json"):
        with open(out / name, "wb") as f:
            for c in vol.read_file(f"teatime/sam3_masks/{name}"):
                f.write(c)
    print(f"đã tải về {out}")


@app.local_entrypoint()
def adaptive_scene(prompts_json: str, tag: str = "sam3_masks_adaptive", floor: float = 0.2, rho: float = 0.5):
    """SAM 3 thich nghi cho canh bat ky: prompts_json = {"scene": .., "prompts": [[prompt, nhan], ...], "support_label": ..,
    "check_overlap": [...]} (nhan < 256, thu tu = uu tien tang dan). Can <scene>/colmap/images/frame_*.jpg tren Volume."""
    c = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
    r = _run_adaptive.remote(tag, floor, rho, -1.0, 0.5, 1, c["scene"], c["prompts"], c.get("check_overlap", []),
                             int(c.get("support_label", 0)), c.get("overlap_mode", "area30"), int(c.get("per_instance", 0)))
    print(json.dumps(r, indent=1, ensure_ascii=False))


@app.local_entrypoint()
def pull_scene(scene: str = "figurines", tag: str = "sam3_masks_adaptive"):
    out = Path(__file__).resolve().parent / "Data" / scene / tag
    for sub in ("labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for e in vol.listdir(f"{scene}/{tag}/{sub}"):
            with open(out / sub / Path(e.path).name, "wb") as f:
                for c in vol.read_file(e.path):
                    f.write(c)
    for name in ("legend.json", "frames.json", "gate.json"):
        with open(out / name, "wb") as f:
            for c in vol.read_file(f"{scene}/{tag}/{name}"):
                f.write(c)
    print(f"da tai ve {out}")


@app.function(image=img, gpu="L4", volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768,
              secrets=[modal.Secret.from_name("huggingface")])
def _probe_prompts(scene: str, prompts: list, n_frames: int = 12):
    """Cham prompt ung vien tren n_frames frame deu nhau cua canh: top1 SAM 3 (che do anh) moi frame. Dung de chon tu dong
    cau prompt cho vat ma cau LERF khong duoc SAM 3 nhan (cung quy tac moi vat: chon cau co trung vi top1 cao nhat)."""
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    vol.reload()
    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.05)
    src = sorted((Path(VOL) / scene / "colmap" / "images").glob("frame_*.jpg"))
    pick = [src[int(i)] for i in np.linspace(0, len(src) - 1, n_frames)]
    res = {p: [] for p in prompts}
    for f in pick:
        st0 = proc.set_image(Image.open(f).convert("RGB"))
        for p in prompts:
            proc.reset_all_prompts(st0)
            st = proc.set_text_prompt(state=st0, prompt=p)
            res[p].append(float(st["scores"].max()) if ("scores" in st and len(st["scores"])) else 0.0)
    return {p: {"median_top1": float(np.median(v)), "frac_ge_0.2": float(np.mean(np.array(v) >= 0.2))} for p, v in res.items()}


@app.local_entrypoint()
def probe_prompts(scene: str, prompts: str, n_frames: int = 12):
    r = _probe_prompts.remote(scene, prompts.split("|"), n_frames)
    for p, v in sorted(r.items(), key=lambda kv: -kv[1]["median_top1"]):
        print(f"PROBE {p!r}: median_top1 {v['median_top1']:.3f}, frac>=0.2 {v['frac_ge_0.2']:.2f}")
