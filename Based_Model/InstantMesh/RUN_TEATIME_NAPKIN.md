# InstantMesh – khăn giấy (napkin) của LERF *teatime* trên Modal GPU

Mục tiêu: từ ảnh thật của cảnh **teatime** (LERF), tạo ra **một mesh khăn giấy hoàn chỉnh** (có cả mặt dưới/mặt sau mà camera không nhìn thấy) bằng InstantMesh (single image → Zero123++ 6 view → LRM FlexiCubes mesh).

* Code gốc của tác giả (`run.py`, `src/`, `zero123plus/`, `configs/`) **không bị sửa**. Toàn bộ phần thêm vào nằm trong `modal_instantmesh/`.
* Tận dụng mask 2D có sẵn: `F:\LERF Datasets\teatime\teatime\mask\labels\frame_XXXXX.png` (ID map uint8 từ SAM2, `0` = nền). **Khăn giấy = ID 11** (giống OMEGAS / Inpaint360GS). Vì có mask nên không cần `rembg` → chạy với `--no_rembg`.
* GPU: Modal, profile `nguyenkhacd10`, `GPU = ["L4", "A10G"]` (24 GB), luôn chạy `--detach`.

---

## 1. Cấu trúc

```
InstantMesh/
├── RUN_TEATIME_NAPKIN.md                 ← file này
├── modal_instantmesh/
│   ├── prepare_teatime_napkin.py         ← Step 1 (local CPU): mask SAM2 → ảnh RGBA đầu vào
│   └── modal_instantmesh.py              ← Step 2 (Modal GPU): run.py cho từng frame + checkpoint + download
└── outputs/teatime_napkin/
    ├── step1_inputs/                     ← rgba/, preview/, contact_sheet.jpg, frames.json
    ├── step2_instantmesh/<frame>/<variant>/   ← kết quả run.py từng frame (tải từ volume)
    └── step3_final_mesh/                 ← mesh khăn giấy được chọn
```

Volume Modal `instantmesh`:

| đường dẫn trên volume | nội dung |
|---|---|
| `hf_cache/` | trọng số tải một lần: `sudo-ai/zero123plus-v1.2`, `TencentARC/InstantMesh` (`diffusion_pytorch_model.bin`, `instant_mesh_large.ckpt`), `facebook/dino-vitb16` |
| `torch_extensions/` | plugin nvdiffrast (JIT) build một lần, dùng lại |
| `outputs/teatime_napkin/step2_instantmesh/<frame>/<variant>/` | `instant-mesh-large/{images,meshes,videos}/`, `run.log`, `manifest.json` (sha256), `done.json` |

---

## 2. Step 1 – chuẩn bị ảnh đầu vào từ mask 2D (local)

```powershell
cd F:\Safe-GS\Based_Model\InstantMesh
F:\Safe-GS\.venv\Scripts\activate
python modal_instantmesh/prepare_teatime_napkin.py
```

Logic chọn frame (tự động, từ 106 frame có ID 11):

1. mask ID 11 **một mảnh**, **không chạm biên ảnh** (khăn không bị cắt) và diện tích ≥ 1.5 % ảnh;
2. lấy 8 frame lớn nhất: `00127, 00126, 00055, 00125, 00106, 00109, 00108, 00107`
   (các frame lớn hơn như `00140`, `00138`, `00128`, `00105` bị loại vì khăn bị cắt ở biên ảnh).

Xử lý mỗi frame: alpha = mask (erode 2 px để bỏ viền gỗ của bàn, fill holes, feather 1 px) → crop bbox → pad vuông với tỉ lệ foreground 0.85 (đúng như `resize_foreground(…, 0.85)` mà `run.py` bỏ qua khi có `--no_rembg`) → resize 640×640 → `step1_inputs/rgba/frame_XXXXX.png`.
`contact_sheet.jpg` (nền xám) để kiểm tra bằng mắt. Script không ghi đè nếu `step1_inputs/` đã tồn tại.

---

## 3. Step 2 – InstantMesh trên Modal GPU

```powershell
$env:PYTHONIOENCODING="utf-8"                       # Windows: tránh lỗi charmap của client Modal
python -m modal profile current                      # phải là nguyenkhacd10
python -m modal run --detach modal_instantmesh/modal_instantmesh.py::start
# tuỳ chọn: --frames frame_00127,frame_00109   --variants vertex_color
python -m modal run modal_instantmesh/modal_instantmesh.py::status
python -m modal app logs instantmesh-teatime-napkin  # xem log trực tiếp
```

Với mỗi frame, container gọi nguyên bản:

```bash
python run.py configs/instant-mesh-large.yaml /root/inputs/rgba/<frame>.png \
    --output_path /vol/outputs/teatime_napkin/step2_instantmesh/<frame>/<variant> \
    --no_rembg --seed 42 --diffusion_steps 75  [--save_video | --export_texmap]
```

Hai biến thể (`VARIANTS`): `vertex_color` (OBJ màu đỉnh + video xoay 360°) chạy hết các frame trước, sau đó `texmap` (OBJ + MTL + texture 1024², UV bằng xatlas).

### Môi trường (image Modal)

`nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` + Python 3.10, `torch==2.1.0` cu121, `xformers==0.0.22.post7` (đúng README của tác giả), các gói trong `requirements.txt` trừ `gradio` (chỉ `app.py` dùng). Pin thêm để tránh lỗi tương thích: `numpy==1.26.4`, `huggingface-hub==0.24.6` (diffusers 0.20.2 còn cần `cached_download`), `accelerate==0.30.1`, `rembg==2.0.56` + `onnxruntime==1.17.3` (run.py import rembg dù dùng `--no_rembg`), `nvdiffrast@v0.3.3` (JIT, chỉ dùng `RasterizeCudaContext` nên không cần EGL). `TORCH_CUDA_ARCH_LIST="8.6;8.9"` để plugin build một lần chạy được cả A10G lẫn L4.

### Chống mất GPU (preemption) – detach + checkpoint định kỳ

InstantMesh là **feed-forward** (không train), nên "checkpoint" ở đây là kết quả đã xong + trọng số đã cache:

| cơ chế | chi tiết |
|---|---|
| `modal run --detach` | job tiếp tục chạy khi đóng laptop / mất mạng |
| `retries=modal.Retries(max_retries=3)` | Modal tự khởi động lại container khi GPU bị thu hồi |
| commit định kỳ | thread `PeriodicCommit` gọi `vol.commit()` mỗi **120 s** + khi nhận `SIGTERM` (preemption) + cuối khối |
| checkpoint theo frame | xong mỗi `<frame>/<variant>` → ghi `manifest.json` (sha256) + `done.json` (thời gian, GPU, thống kê mesh) → commit ngay. Lần retry **bỏ qua** mọi frame đã có `done.json` |
| cache trọng số | ~10 GB weights tải 1 lần vào `hf_cache/` (file `.incomplete` của HF resume được), đánh dấu `instantmesh_weights.ok`; plugin nvdiffrast cache ở `torch_extensions/` |
| timeout | 3 h/container |

### Tải kết quả về

```powershell
python -m modal run modal_instantmesh/modal_instantmesh.py::download
```

Tải `step2_instantmesh/` về `outputs/teatime_napkin/` (không ghi đè: `_dl2`, `_dl3`…), sau đó **so sha256 với `manifest.json`** từng file và tự tải lại file lỗi (`modal volume get` từng làm hỏng file lớn mà vẫn đúng dung lượng).

---

## 4. Step 3 – chọn mesh cuối

Mỗi `done.json` chứa `stats`: số đỉnh/mặt, số thành phần liên thông, `watertight`, kích thước bbox. Chọn frame theo:
1. ảnh 6 view Zero123++ (`instant-mesh-large/images/<frame>.png`) nhất quán, giống khăn thật;
2. mesh 1 thành phần (hoặc thành phần lớn nhất chiếm gần 100 % mặt), kín (watertight);
3. hình dạng: tấm mỏng, gấp nếp như ảnh thật (xem video `videos/<frame>.mp4`).

Mesh được chọn (giữ thành phần lớn nhất) được copy vào `outputs/teatime_napkin/step3_final_mesh/` cùng `README.md`.

---

## 5. Kết quả

**Mesh cuối: `outputs/teatime_napkin/step3_final_mesh/vertex_color/napkin_frame_00126.obj`** (+ bản texture `texmap/napkin_frame_00126.obj`), 22 428 đỉnh, 44 868 mặt, 1 thành phần, watertight.

Kết quả Step 2 (bản `vertex_color`; bản `texmap` cùng hình học):

| frame | giây (A10G) | đỉnh | mặt | thành phần | watertight | bbox | nhận xét |
|---|---|---|---|---|---|---|---|
| **00126** | 38.0 | 22 428 | 44 868 | **1** | ✓ | 1.71 × 1.88 × **0.41** | **chọn**: tấm mỏng, gấp nếp |
| 00109 | 38.5 | 28 098 | 56 212 | 4 | ✓ | 1.69 × 1.79 × 0.54 | tốt, vài mảnh vụn |
| 00055 | 38.2 | 23 066 | 46 128 | 1 | ✓ | 1.42 × 1.92 × 0.73 | gấp cong, kém giống |
| 00125 | 38.5 | 29 350 | 58 692 | 2 | ✓ | 1.57 × 1.91 × 0.79 | ổn |
| 00127 | 109.1* | 23 618 | 47 228 | 2 | ✓ | 0.94 × 1.89 × 1.42 | dày |
| 00107 | 39.7 | 44 642 | 89 244 | 12 | ✓ | 1.49 × 1.86 × 1.46 | dày, vụn |
| 00106 | 38.1 | 44 474 | 89 016 | 34 | ✓ | 1.07 × 1.78 × 1.49 | dày, vụn |
| 00108 | 38.3 | 73 856 | 147 376 | 123 | ✓ | 1.58 × 1.86 × 1.32 | nhiễu |

\* gồm cả build plugin nvdiffrast lần đầu. `texmap`: 54–99 s/frame.

Thời gian thực tế: build image ~10 phút (chỉ lần đầu), cache weights 63 s, 16 lượt chạy ~15 phút, tải về ~1 phút → GPU ~17 phút (A10G, ≈ 0.3–0.4 USD).

**Lưu ý:** mesh ở hệ toạ độ chuẩn hoá của InstantMesh (không có scale/pose của cảnh teatime). Muốn đặt lại vào cảnh cần căn chỉnh (scale + pose) với point cloud/Gaussian của napkin.

## 6. Nhật ký chạy

| ngày | việc | ghi chú |
|---|---|---|
| 2026-11-05 | Step 1 local | 8 frame, `contact_sheet.jpg` OK |
| 2026-11-05 | Step 2 Modal, lần 1–2 | build image lỗi: `printf` với xuống dòng thật; xung đột `tokenizers 0.14` ↔ `huggingface-hub` → pin `huggingface-hub==0.17.3`, `accelerate==0.24.1` |
| 2026-11-05 | Step 2 Modal, lần 3 | A10G, 16/16 lượt xong, không bị preempt; `download` 105 file, sha256 OK |
| 2026-11-05 | Step 3 | chọn frame_00126 → `step3_final_mesh/` |

Mẹo Windows: đặt `PYTHONIOENCODING=utf-8` trước `modal run` (nếu không client Modal lỗi `'charmap' codec can't encode '✓'`).
