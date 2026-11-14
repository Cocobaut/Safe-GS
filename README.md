# Safe-GS — CoMe + Inpaint360GS + InstantMesh

Pipeline **16 bước (Step 0 → 15)**:
1. huấn luyện **CoMe 3DGS** từ ảnh;
2. xóa vật và inpainting 3DGS bằng **Inpaint360GS**, dùng rasterizer của CoMe;
3. trích xuất **mesh bàn** (CoMe Marching Tetrahedra) và **mesh vật** (**InstantMesh**);
4. đặt vật lên bàn;
5. **đánh giá** cho báo cáo hội đồng.

Chạy được trên ba bộ dữ liệu **LERF**, **Replica** và **Inpaint360GS**.

README này hướng dẫn từ đầu đến cuối cách tái tạo đầy đủ kết quả **LERF / teatime / khăn ăn (`teatime_napkin`)**.
Các dataset khác chạy cùng các lệnh, chỉ đổi `--dataset` và `--run` (xem [mục 9](#9-chạy-dataset--vật-khác)).

- Kế hoạch tái cấu trúc và trạng thái: [`PLAN_REBUILD_BASE_FRAMEWORK.md`](PLAN_REBUILD_BASE_FRAMEWORK.md).
- Tài liệu tham chiếu của bản mẫu (kế hoạch InstantMesh, hướng dẫn train CoMe teatime, sơ đồ framework drawio): [`docs/reference/`](docs/reference/).
- Pipeline cũ (GOF/MILo, 6 module, `modal_train.py`): `legacy/` (`legacy/README_v0.md`).

---

## Mục lục
1. [Tổng quan 16 bước](#1-tổng-quan-16-bước)
2. [Cài đặt môi trường local](#2-cài-đặt-môi-trường-local)
3. [Cài đặt Modal (GPU cloud)](#3-cài-đặt-modal-gpu-cloud)
4. [Chuẩn bị dữ liệu teatime](#4-chuẩn-bị-dữ-liệu-teatime)
5. [Chạy từng bước cho LERF / teatime_napkin](#5-chạy-từng-bước-cho-lerf--teatime_napkin)
6. [Chạy nhanh bằng `run_pipeline.py`](#6-chạy-nhanh-bằng-run_pipelinepy)
7. [Theo dõi, chạy lại, xử lý lỗi](#7-theo-dõi-chạy-lại-xử-lý-lỗi)
8. [Kết quả mong đợi](#8-kết-quả-mong-đợi)
9. [Chạy dataset / vật khác](#9-chạy-dataset--vật-khác)
10. [Cấu trúc thư mục](#10-cấu-trúc-thư-mục)

---

## 1. Tổng quan 16 bước

| Step | Nội dung | File thực thi | Nơi chạy | Thư mục kết quả (`outputs/LERF/ours/teatime_napkin/`) |
|---|---|---|---|---|
| 0 | COLMAP 3.11.1 (PINHOLE, giữ kích thước ảnh) | `scripts/modal_come_train.py::colmap` | Modal L4 | `step0_colmap/` |
| 1 | Train **CoMe 3DGS** 30k iter | `scripts/modal_come_train.py::train` (code `src/come/train.py`) | Modal L4/L40S | `step1_come_3dgs/` |
| 2 | Render CoMe (train + test) | `scripts/modal_come_train.py::render` | Modal L4/L40S | `step2_come_renders/` |
| prep | Đưa data và model CoMe sang Volume của Step 4–12 | `scripts/modal_come_pipeline.py::run --steps prep` | Modal | — |
| 4 | Kiểm tra renderer Inpaint360GS = renderer CoMe | `src/tools_come/render_check.py` | Modal | `step4_renderer_check/` |
| 5 | Mask 2D HQ-SAM (CropFormer) | `src/seg/raw_mask_sam.py` | Modal | `step5_mask2d_hqsam/` |
| 6 | Đồng bộ ID mask đa góc nhìn | `src/seg/mask_associate.py`, `src/tools/add_label_num_hqsam.py` | Modal | `step6_sync_mask/` |
| 3 + 6id | Chọn `target_id` của vật cần xóa | `src/tools_safe/select_target_id.py` | Local CPU | `step3_object_reference/`, `step6_sync_mask/object_id.json` |
| 7 | Chưng cất ID vào 3DGS (classifier) | `src/seg/distillation.py`, `src/render.py` | Modal | `step7_id_distill/` |
| 8 | Xóa vật + sinh 30 camera ảo | `src/tools/init_configs.py`, `src/edit_object_removal.py`, `src/tools/virtual_pose.py` | Modal | `step8_removal_virtual_cam/` |
| 9 | Mask vùng cần inpaint trên view ảo (SAM-Track) | `src/tools_safe/step9_samtrack_click.py` | Modal | `step9_inpaint_masks_samt/` |
| 10 | Inpaint 2D màu + độ sâu bằng LaMa | `src/tools/prepare_lama_data.py`, `src/LaMa/bin/predict_*.py` | Modal L4 + T4 | `step10_inpaint2d_lama/` |
| 11 | Inpaint 3DGS (CoMe loss) | `src/tools_come/select_init_view.py`, `src/edit_object_removal_plyfusion.py`, `src/edit_object_inpaint.py` | Modal | `step11_inpaint_3dgs_come_loss/` |
| 12 | Render cảnh đã xóa / đã inpaint + video | `src/tools_safe/step12_package.py` | Modal | `step12_render/` |
| 13 | Mesh CoMe của 3DGS đã inpaint | `scripts/modal_come_mesh.py` (code `src/come/extract_mesh_tets.py`) | Modal L40S | `step13_come_mesh/mesh_come_iter5000.ply` |
| 13p | Mặt phẳng bàn dưới vật | `src/tools_mesh/estimate_table_plane.py` | Local CPU | `table_plane.json` |
| 13b | Cắt mesh bàn | `src/tools_mesh/crop_table.py` | Local CPU | `step13_come_mesh/mesh_come_table.ply` |
| 14.1 | Tách Gaussian của vật | `src/tools_instantmesh/extract_object.py` | Local CPU | `step14_instantmesh/1_object_gaussians/` |
| 14.2–14.5 | Video 360° ảo → chọn view → InstantMesh → chọn mesh | `scripts/modal_instantmesh.py` (code `src/tools_instantmesh/step14_gpu.py`) | Modal L4 | `step14_instantmesh/2_…5_final_mesh/` |
| 15 | Đặt vật lên bàn | `src/tools_instantmesh/place_on_table.py` | Local CPU | `step15_object_mesh_merge/` |
| Đánh giá | PSNR/SSIM, độ phẳng bàn, IoU, xuyên thấu | `scripts/evaluate_all_metrics.py` | Local CPU | `evaluation_summary.json`, `outputs/council_report_summary.{md,csv}` |

Dữ liệu đi giữa các bước theo luồng sau:

```
Data/LERF/teatime/images ──upload──► Volume safe-gs-come ──(0,1,2)──► download ──► outputs/.../step0-2
                                          │ prep
                                          ▼
                                Volume inpaint360gs ──(4..12)──► download ──► outputs/.../step4-12
                                          │ (13)                  ──► download ──► outputs/.../step13
outputs/.../step13 ──(13p,13b,14.1 local)──► Volume instantmesh ──(14.2-14.5)──► download ──► (15 local) ──► đánh giá
```

---

## 2. Cài đặt môi trường local

Yêu cầu:
- Windows 10/11, Linux hoặc macOS;
- Python **3.10**;
- [`uv`](https://docs.astral.sh/uv/), hoặc `pip`;
- khoảng 20 GB ổ trống cho `outputs/` (riêng teatime_napkin khoảng 8 GB).

Máy local **không cần GPU**: các bước local (6id, 13p, 13b, 14.1, 15) và phần đánh giá đều chạy trên CPU.

```powershell
# 1. Lấy code
git clone <repo> Safe-GS
cd Safe-GS

# 2. Tạo môi trường (.venv) theo pyproject.toml
uv sync
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
# (không có uv: python -m venv .venv; .venv\Scripts\activate; pip install -e .)

# 3. Windows PowerShell: bật UTF-8 cho log có tiếng Việt / Unicode (đặt mỗi phiên, hoặc vào $PROFILE)
$env:PYTHONIOENCODING = "utf-8"

# 4. Kiểm tra
python scripts/run_pipeline.py --check-env
```

`--check-env` kiểm tra các gói Python (numpy, scipy, open3d, plyfile, scikit-learn, opencv, pillow, pyyaml, modal, torch), Modal profile và cấu trúc thư mục. Kết quả phải là `RESULT: OK`.

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `SAFE_GS_DATA_DIR` | `<repo>/Data` | gốc dữ liệu ảnh |
| `SAFE_GS_OUTPUTS_DIR` | `<repo>/outputs` | gốc kết quả (thư mục này **không có trong git**) |
| `MODAL_PROFILE` | (trống) | nếu đặt, các script dừng lại khi profile Modal đang dùng khác giá trị này |

Có thể ghi các biến này vào file `.env` ở gốc repo, ví dụ `SAFE_GS_DATA_DIR=E:/datasets`. File này được đọc bởi `src/common/paths.py`.
Mọi đường dẫn trong code đều là đường dẫn tương đối, nên **không cần sửa code** khi đổi máy hoặc đổi ổ đĩa.

---

## 3. Cài đặt Modal (GPU cloud)

```powershell
python -m modal token new                       # đăng nhập (mở trình duyệt), tạo profile
python -m modal profile current                 # xem profile đang dùng
# nhiều tài khoản: python -m modal profile activate <tên>
```

**Secret `huggingface`** (bắt buộc cho Step 4–12, dùng để tải trọng số CropFormer từ một dataset HF có giới hạn truy cập):
1. Vào https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg và chấp nhận điều khoản.
2. Tạo token (Read) tại https://huggingface.co/settings/tokens.
3. Tạo secret:
   ```powershell
   python -m modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxxxxx
   ```
   Có thể dùng tên secret khác bằng biến `SAFE_GS_HF_SECRET`.

**Volume** được tạo tự động ở lần chạy đầu tiên:

| Volume | Dùng cho |
|---|---|
| `safe-gs-come` | Step 0–2: `lerf/teatime/{input,colmap,come,logs}` |
| `inpaint360gs` | Step 4–13: `come360/{data,output,runs,export,state,mesh}` |
| `instantmesh` | Step 14: HF cache + `LERF/teatime_napkin/...` |

**GPU:**
- CoMe và Inpaint360GS dùng **L4 / L40S** (extension CUDA build cho sm_89, **không chạy trên A10G**);
- LaMa dùng **T4**;
- Step 13 dùng **L40S** (128 GB RAM);
- InstantMesh dùng L4 / A10G.

Kiểm tra build image (chỉ dùng CPU, rẻ). **Lần đầu mất 20–40 phút mỗi image**, các lần sau dùng cache:
```powershell
python -m modal run scripts/modal_come_train.py::check_come      # COLMAP + CoMe (Step 0-2)
python -m modal run scripts/modal_come_pipeline.py::check        # Inpaint360GS + HQ-SAM + SAM-Track (Step 4-12)
python -m modal run scripts/modal_come_mesh.py::check            # CoMe + tetra-triangulation (Step 13)
```
Với `modal_come_mesh.py::check`, kết quả mong đợi là `CoMe imports OK ... tetra OK`.

---

## 4. Chuẩn bị dữ liệu teatime

Cần **180 ảnh** `frame_00001.jpg … frame_00180.jpg` (994×738) của scene **teatime** trong LERF, đặt tại:

```
Data/LERF/teatime/images/frame_00001.jpg ... frame_00180.jpg
```

Đường dẫn này lấy từ `data_subdir: LERF` trong `configs/datasets/lerf.yaml`. Nếu ảnh nằm chỗ khác, có hai cách:
- đặt `SAFE_GS_DATA_DIR` (khi đó phải có `<SAFE_GS_DATA_DIR>/LERF/teatime/images`);
- hoặc truyền `--images <thư mục>` cho lệnh upload ở Step 0.

Run `teatime_napkin` đã được khai báo sẵn trong `configs/datasets/lerf.yaml`:
```yaml
teatime:
  objects:
    napkin: {run: teatime_napkin, target_id: 44, ref_mask_id: 11, samtrack_click: [482, 371]}
```
Xem các run đã khai báo:
```powershell
python scripts/run_pipeline.py --list
```

> **Nếu `outputs/LERF/ours/teatime_napkin/` đang chứa bản mẫu** (thư mục `step0_reuse_colmap`, …), hãy đổi tên nó trước khi chạy lại từ đầu.
> Cách này vừa giữ bản mẫu để đối chiếu, vừa tránh trộn kết quả cũ với kết quả mới. Các lệnh `download` không ghi đè file đã có.
> ```powershell
> New-Item -ItemType Directory -Force outputs\_reference | Out-Null
> Move-Item outputs\LERF\teatime_napkin outputs\_reference\teatime_napkin
> ```
> Mask tham chiếu của khăn trong bản mẫu (dùng ở Step 6id) nằm tại:
> `outputs/_reference/teatime_napkin/step3_reuse_napkin_reference/napkin_mask_id11/`

---

## 5. Chạy từng bước cho LERF / teatime_napkin

Tất cả lệnh chạy từ **thư mục gốc repo**, với `.venv` đã kích hoạt và `$env:PYTHONIOENCODING="utf-8"`.
- `--detach`: job tiếp tục chạy trên Modal kể cả khi bạn tắt máy hoặc mất mạng. Theo dõi bằng lệnh `status`, xong thì chạy `download`.
- Mọi bước trên Modal đều **chạy tiếp được** khi Modal thu hồi máy (có checkpoint và state trên Volume). Chạy lại cùng một lệnh là đủ.

### Step 0–2: COLMAP → train CoMe 3DGS → render (Modal)

```powershell
# 0a. Upload 180 ảnh lên Volume safe-gs-come:lerf/teatime/input
python -m modal run scripts/modal_come_train.py::upload_images --dataset LERF --scene teatime

# 0b. COLMAP (GPU L4, khoảng 20-40 phút)
python -m modal run --detach scripts/modal_come_train.py::colmap --dataset LERF --scene teatime

# 1. Train CoMe 30k iter (L4/L40S, khoảng 1-2 giờ; checkpoint mỗi 1000 iter, tự resume)
python -m modal run --detach scripts/modal_come_train.py::train --dataset LERF --scene teatime

# 2. Render train/test + metrics (khoảng 5-10 phút)
python -m modal run --detach scripts/modal_come_train.py::render --dataset LERF --scene teatime

# theo dõi
python -m modal run scripts/modal_come_train.py::status --dataset LERF --scene teatime

# tải về outputs/LERF/ours/teatime_napkin/{step0_colmap, step1_come_3dgs, step2_come_renders} (+ logs)
python -m modal run scripts/modal_come_train.py::download --dataset LERF --scene teatime --run teatime_napkin
```
Có thể chạy 0b → 1 → 2 liền một lần bằng `::all_steps --dataset LERF --scene teatime` (nên thêm `--detach`).

Cờ train CoMe dùng ở Step 1 (giống `scripts/run_tnt.py` của tác giả):
`--iterations 30000 --lambda_distortion 100 --eval --far_plane 100. --detach_alpha False --splatting_config configs/hierarchical.json --use_ssimdecoupled_appearance --color_confidence --color_confidence_max 0.075 --color_confidence_from_iter 500 --lambda_variance 0.5 --lambda_normal_variance 0.005`.

Kiểm tra sau khi download:
- `step0_colmap/summary.json`: `registered` = 177/180, camera `PINHOLE` 994×738.
- `step1_come_3dgs/summary.json`: khoảng 1.8 triệu Gaussian, có `confidence`, `filter_3D`.
- `step2_come_renders/summary.json`: 177 render.

> **Bỏ qua Step 0–2** (dùng lại model CoMe teatime đã train trên Volume `safe-gs-come`, key `teatime`): ở Step 4 thêm `--come-key teatime`.

### Step 4–7: kiểm tra renderer, mask HQ-SAM, đồng bộ ID, distill (Modal)

```powershell
# xem trước toàn bộ lệnh sẽ chạy trong container (không tốn GPU)
python -m modal run scripts/modal_come_pipeline.py::plan --dataset LERF --run teatime_napkin --steps prep,4-12

# prep + 4..7 (L4/L40S)
python -m modal run --detach scripts/modal_come_pipeline.py::run --dataset LERF --run teatime_napkin --steps prep,4-7

python -m modal run scripts/modal_come_pipeline.py::status --dataset LERF --run teatime_napkin
python -m modal run scripts/modal_come_pipeline.py::download --dataset LERF --run teatime_napkin
```
- `prep` lấy data và model CoMe từ Volume `safe-gs-come` (`--source volume`, mặc định).
- Muốn lấy từ các thư mục `step0-2` ở local thì chạy trước `::upload_local --dataset LERF --run teatime_napkin`, rồi thêm `--source local`.

Kiểm tra:
- `step4_renderer_check/summary.json`: `max_abs_diff_vs_come_render` = **0** (hai renderer cho kết quả giống hệt).
- `step6_sync_mask/3_ids_on_images/`: ảnh có ghi số ID của từng vật.

### Step 3 + 6id: xác nhận `target_id` của khăn (local)

ID ở Step 6 do HQ-SAM và bước đồng bộ sinh ra, nên **có thể khác 44 nếu chạy lại Step 5–6**. Bước này xác nhận lại ID:

```powershell
# Cách 1: so với mask tham chiếu của khăn (chọn ID có IoU cao nhất trên mọi view) - khuyến nghị
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 6id `
    --ref-masks outputs/_reference/teatime_napkin/step3_reuse_napkin_reference/napkin_mask_id11

# Cách 2: một điểm click trên ảnh gốc (tên frame, x, y theo pixel ảnh 994x738)
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 6id --click frame_00020 482 371
```
- Kết quả ghi vào `step6_sync_mask/object_id.json` (`napkin_id`, `iou_total`, `candidates_top5`). Với cách 1, mask tham chiếu được chép vào `step3_object_reference/`.
- **Nếu ID khác 44**: sửa `target_id` của `napkin` trong `configs/datasets/lerf.yaml` trước khi chạy Step 8.
- Cũng có thể đọc ID trực tiếp trên ảnh trong `step6_sync_mask/3_ids_on_images/`.

### Step 8–12: xóa vật, mask, LaMa, inpaint 3DGS, render (Modal)

```powershell
python -m modal run --detach scripts/modal_come_pipeline.py::run --dataset LERF --run teatime_napkin --steps 8-12
python -m modal run scripts/modal_come_pipeline.py::status --dataset LERF --run teatime_napkin
python -m modal run scripts/modal_come_pipeline.py::download --dataset LERF --run teatime_napkin
```
- Step 8 tạo bản sao data và model riêng cho run (`come360/runs/lerf/teatime/teatime_napkin/`), nên nhiều vật cùng một scene không ghi đè lên nhau.
- Step 9: SAM-Track headless, click `samtrack_click: [482, 371]` trên view ảo đầu tiên, rồi lan qua 30 view.
  Nếu YAML không có `samtrack_click`, Step 9 dùng mask mà chính model dự đoán cho `target_id`.
- Step 10 chạy LaMa trên T4 trong môi trường conda của tác giả. Step 11 chọn view khởi tạo tốt nhất từ kết quả LaMa.
- Các tham số có thể thêm vào `::run`:
  - `--force 9,11`: chạy lại các bước đã xong;
  - `--target-id 44`: ghi đè `target_id` trong YAML;
  - `--click 482,371`: ghi đè điểm click SAM-Track.

Kiểm tra:
- `step9_inpaint_masks_samt/summary.json`: 30 mask, `empty_masks` = [].
- `step11_inpaint_3dgs_come_loss/point_cloud_inpainted_iter5000.ply`: khoảng 1.9 triệu Gaussian.
- `step12_render/video_2_inpainted.mp4`: video cảnh đã inpaint.

### Step 13: mesh CoMe của cảnh đã inpaint (Modal L40S)

```powershell
python -m modal run --detach scripts/modal_come_mesh.py::run --dataset LERF --run teatime_napkin
python -m modal run scripts/modal_come_mesh.py::download --dataset LERF --run teatime_napkin
```
- Đầu vào là `point_cloud_inpainted_iter5000.ply` của Step 11, lấy thẳng trên Volume.
- Muốn dùng file local thì thêm `--source local`.
- Kết quả là `step13_come_mesh/mesh_come_iter5000.ply` (khoảng 16 triệu đỉnh, khoảng 5–10 phút) và `done.json`.

### Step 13p + 13b: mặt phẳng bàn và mesh bàn (local CPU)

```powershell
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 13p
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 13b
```
- **13p**: lấy vị trí khăn từ `step8_removal_virtual_cam/2_object_removal/removed_object_id44.ply`, chạy RANSAC trên mesh Step 13 để tìm mặt đỡ ngay dưới khăn, rồi ghi `table_plane.json` vào thư mục run.
  Trên teatime, mặt phẳng tìm được lệch bản mẫu 0.08°.
- **13b**: giữ các tam giác nằm trong lăng trụ của bàn, ghi `step13_come_mesh/mesh_come_table.ply` và `mesh_come_table_crop.json`.
- Có thể truyền `--plane <file.json>` để dùng mặt phẳng tự chọn.
- Cả hai bước đều **không ghi đè** kết quả đã có. Muốn làm lại thì xóa file đó, hoặc gọi trực tiếp tool kèm `--force`.

### Step 14: mesh khăn bằng InstantMesh

```powershell
# 14.1 tách Gaussian của khăn + quan sát của camera thật (local CPU, cần torch CPU)
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 14.1

# 14.2 video 360° ảo, 14.3 chọn view, 14.4 InstantMesh, 14.5 chọn mesh (Modal L4)
python -m modal run --detach scripts/modal_instantmesh.py::start --dataset LERF --run-name teatime_napkin
python -m modal run scripts/modal_instantmesh.py::status --dataset LERF --run-name teatime_napkin
python -m modal run scripts/modal_instantmesh.py::download --dataset LERF --run-name teatime_napkin
```
- Bước `download` kiểm tra sha256 của từng file.
- Nếu thư mục đích đã tồn tại, nó tải vào `<part>_dl2`, `<part>_dl3`, … thay vì ghi đè.
- Kết quả là `step14_instantmesh/5_final_mesh/mesh_object_instantmesh.ply` và `summary.json` (biến thể được chọn và IoU). Kết quả cũ tải lại từ Volume Modal có thể còn tên `mesh_napkin_instantmesh.ply`; Step 15 và `evaluate_all_metrics.py` vẫn đọc được tên này.

### Step 15: đặt khăn lên bàn (local CPU)

```powershell
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 15
```
Kết quả nằm trong `step15_object_mesh_merge/`:
- `mesh_object_placed.ply` (vật sau khi tịnh tiến);
- `scene_scene_demo_object.ply` (bàn + vật, hai khối riêng);
- `summary.json` với `after.vertices_inside_table` = 0 và khe hở nhỏ nhất = `gap` = 0.001.

### Đánh giá cho báo cáo hội đồng (local CPU)

```powershell
python scripts/evaluate_all_metrics.py --dataset LERF --run teatime_napkin
python scripts/evaluate_all_metrics.py --all                    # mọi outputs/<Dataset>/ours/<run>
```
Ghi ra `outputs/LERF/ours/teatime_napkin/evaluation_summary.json`, `outputs/council_report_summary.md` và `.csv` (dán thẳng vào slide):

| Nhóm | Chỉ số |
|---|---|
| S4 | sai khác renderer CoMe / Inpaint360GS; PSNR test của CoMe |
| S12 | PSNR / SSIM **ngoài vùng vật** (LERF và Replica không có ảnh thật của bàn khi đã bỏ khăn) |
| S12 (chỉ Inpaint360GS) | PSNR / SSIM trong vùng vật, LPIPS, FID (cần cài thêm `lpips`, `pytorch_fid`) |
| S13 | số đỉnh / mặt của mesh bàn; RMSE độ cao bàn dưới khăn; tỉ lệ phủ |
| S14 | IoU silhouette của mesh InstantMesh so với mask thật |
| S15 | số đỉnh xuyên bàn, khe hở nhỏ nhất, tỉ lệ đỉnh có bàn đỡ bên dưới, số cạnh biên / không đa tạp |

Các cờ có thể dùng:
- `--skip-geometry`: bỏ phân tích mesh lớn;
- `--skip-images`: bỏ chỉ số ảnh;
- `--dilate 10`: số pixel nới mask vật.

---

## 6. Chạy nhanh bằng `run_pipeline.py`

`scripts/run_pipeline.py` sinh đúng các lệnh ở mục 5 và tự chèn **upload / prep / download** giữa các bước:

```powershell
# xem toàn bộ lệnh (an toàn, không chạy gì)
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 0-15 --dry-run

# chạy thật, tuần tự (giữ terminal mở đến khi xong)
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 0-2  --modal
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 4-7  --modal
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 6id --ref-masks <dir>   # nếu cần
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 8-12 --modal
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 13-15 --modal
python scripts/evaluate_all_metrics.py --dataset LERF --run teatime_napkin
```

- **Không có `--modal`:** các bước Modal chỉ được **in ra**, còn các bước local (6id, 13p, 13b, 14.1, 15) **vẫn chạy thật**. Muốn chỉ xem lệnh thì dùng `--dry-run`.
- **`--detach`:** dừng sau bước Modal đầu tiên và in lệnh `download` để bạn chạy khi job xong.
- **Dùng lại model CoMe đã train:** `--steps 4-15 --modal --volume-key teatime`.

---

## 7. Theo dõi, chạy lại, xử lý lỗi

| Việc | Lệnh |
|---|---|
| Trạng thái Step 0–2 | `python -m modal run scripts/modal_come_train.py::status --dataset LERF --scene teatime` |
| Trạng thái Step 4–12 | `python -m modal run scripts/modal_come_pipeline.py::status --dataset LERF --run teatime_napkin` |
| Trạng thái Step 14 | `python -m modal run scripts/modal_instantmesh.py::status --dataset LERF --run-name teatime_napkin` |
| Xem app đang chạy / log | `python -m modal app list`, rồi `python -m modal app logs <app-id>` hoặc trang modal.com/apps |
| Chạy lại bước đã xong (4–12) | `::run ... --steps 9-12 --force 9,10,11,12` |
| Chạy lại Step 13 | `python -m modal run scripts/modal_come_mesh.py::run --dataset LERF --run teatime_napkin --force` |
| Chạy lại COLMAP từ một stage | `python -m modal run scripts/modal_come_train.py::reset_colmap_stage --key lerf/teatime --stage mapped` |
| Log từng bước | `outputs/LERF/ours/teatime_napkin/logs/stepN_*.log` (định dạng `===== [attempt k] HH:MM:SS $ cmd` … `===== exit rc`) |

Lỗi thường gặp:
- **`UnicodeEncodeError` trên Windows**: chưa đặt `$env:PYTHONIOENCODING="utf-8"`.
- **Image Step 4–12 lỗi khi tải CropFormer**: thiếu secret `huggingface`, hoặc chưa chấp nhận điều khoản dataset HF.
- **`no target_id`**: chưa điền `target_id` trong YAML. Chạy Step 6id rồi điền.
- **`... exists - not overwriting`**: bước local đã có kết quả. Xóa thư mục hoặc file đó để làm lại.
- **`scene` của CoMe báo `Found no NVIDIA driver`** trong `::check`: đây là hành vi bình thường. Module CoMe tạo tensor CUDA ngay lúc import, nên chỉ chạy được trên container có GPU.
- **Mất mạng khi đang chạy `modal run` (không `--detach`)**: job có thể dừng. Chạy lại cùng lệnh, job sẽ tiếp tục từ checkpoint.

---

## 8. Kết quả mong đợi

Số liệu của bản mẫu LERF / teatime_napkin, dùng để đối chiếu khi chạy lại. Kết quả có thể lệch nhẹ do tính ngẫu nhiên của train.

| Chỉ số | Bản mẫu |
|---|---|
| S0: ảnh đăng ký COLMAP | 177 / 180 |
| S1: số Gaussian CoMe / PSNR test | 1 804 024 / 23.34 |
| S4: sai khác renderer | 0 |
| S6: ID khăn (IoU với mask tham chiếu) | 44 (0.7275) |
| S11: số Gaussian sau inpaint | 1 896 824 |
| S12: PSNR / SSIM ngoài vùng khăn | 23.30 / 0.800 |
| S13: mesh đầy đủ / mesh bàn (đỉnh) | 16 381 340 / 3 435 540 |
| S13: RMSE bàn dưới khăn / tỉ lệ phủ | 0.0071 / 98.4 % |
| S14: biến thể được chọn, IoU sạch / tất cả | uniform6, 0.922 / 0.916 |
| S15: đỉnh xuyên bàn / khe hở nhỏ nhất / kín nước | 0 / 0.001 / có |

---

## 9. Chạy dataset / vật khác

1. Khai báo scene và vật trong YAML tương ứng (run = `<scene>_<object>`):
   - `configs/datasets/lerf.yaml`
   - `configs/datasets/replica.yaml`
   - `configs/datasets/inpaint360gs.yaml`

   Mỗi YAML có các trường `target_id` (điền sau 6id), `target_surrounding_id`, `resolution`, `samtrack_click`.
2. Đặt ảnh vào `Data/<data_subdir>/<scene>/images`.
3. Chạy như mục 5, thay `--dataset LERF --scene teatime --run teatime_napkin`.

Lưu ý theo từng dataset:

- **Inpaint360GS** (`data_subdir: Inpaint360GS/inpaint360`, ví dụ `doppelherz`, `resolution: 2`, `target_id: 26`, `target_surrounding_id: "24,10"`):
  - Ảnh có chữ `test` trong tên là **ảnh thật sau khi bỏ vật** (ground truth). Chúng được COLMAP đăng ký nhưng **không dùng để train** CoMe hay Inpaint360GS.
  - Các ảnh này dùng cho đánh giá S12, gồm PSNR/SSIM trong vùng vật, LPIPS và FID (cần `unseen_mask` của dataset).
  - Đường dẫn dữ liệu của dataset này phải chứa chữ `inpaint360`. Điều này được đảm bảo vì key trên Volume có dạng `inpaint360gs/<scene>`.
  - Tải dataset: `bash src/scripts/download_inpaint360gs_dataset.sh`.
- **Replica** (`office0..4`, `room0..2`):
  - COLMAP chạy trực tiếp từ ảnh như LERF.
  - Không có ground truth sau khi bỏ vật, nên S12 chỉ đo vùng ngoài vật.
- **Vật mới trong scene cũ** (ví dụ `teatime_bear`, đã có sẵn Step 0–12 trong `outputs/LERF/ours/teatime_bear`): Step 0–7 dùng chung theo scene. Chỉ cần chạy `--steps 8-15` với run mới.
- **Mặt đỡ không phải bàn** (sàn, kệ): Step 13p tự tìm mặt phẳng ngay dưới vật. Muốn tự chỉ định thì truyền `--plane`.
- **Vật cao hoặc nằm xa tâm cảnh** (ví dụ túi bánh `teatime_cafe_box`, ID 30), khai báo thêm trong YAML của vật:
  - `virtual_object_centered: true`: Step 8 đặt vòng 30 camera ảo quanh chính vật. Mặc định của Inpaint360GS đặt tâm tại điểm các camera thật cùng nhìn vào, nên sẽ không thấy vật nằm xa điểm đó.
  - `removal_thresh: 0.3`: Step 8 xóa cả các Gaussian mà classifier chưa chắc chắn. Mặc định 0.7 để sót lại đáy túi.
  - `object_max_height_frac: 3.0`: Step 14.1 giữ Gaussian tới độ cao 3 lần bề rộng chân vật. Mặc định 0.3 chỉ hợp với vật dẹt như khăn.
- **Dùng lại Step 0–6 của một run cùng scene** (giữ nguyên không gian ID):
  ```powershell
  $env:SAFE_GS_SKIP_CROPFORMER = "1"   # không chạy Step 5 nên không cần secret huggingface
  python -m modal run scripts/modal_come_pipeline.py::upload_local --dataset LERF --run teatime_cafe_box --scene-from teatime_napkin --with-masks
  python -m modal run --detach scripts/modal_come_pipeline.py::run --dataset LERF --run teatime_cafe_box --steps prep,4,7,8-12 --source local
  ```

---

## 10. Cấu trúc thư mục

```text
configs/
  paths.yaml                      Volume, mount, GPU
  datasets/{lerf,replica,inpaint360gs}.yaml   scene / vật / target_id / resolution / samtrack_click
src/                              Inpaint360GS + rasterizer CoMe (Step 4-12), chạy với cwd=src
  common/                         paths.py (PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR, step_dir), datasets.py (đọc YAML)
  come/                           CoMe (r4dl/CoMe @ caf62f6, đã vá resume checkpoint) - Step 1, 2, 13, cwd=src/come
    tools/                        convert_colmap.py, come_apply_appearance.py, come_sdf_vis.py, come_remove_object.py
  seg/                            HQ-SAM (raw_mask_sam), mask_associate, distillation, detectron2/CropFormer
  tools/                          init_configs, virtual_pose, prepare_lama_data, add_label_num_hqsam, metrics_fid_masked
  tools_come/                     render_check (S4), select_init_view (S11)
  tools_safe/                     select_target_id (S3/6id), step9_samtrack_click (S9), step12_package (S12)
  tools_mesh/                     mesh_io, estimate_table_plane (S13p), crop_table (S13b), table_plane.json (teatime)
  tools_instantmesh/              extract_object (S14.1), step14_gpu (S14.2-14.5), place_on_table (S15), gs_render
  edit_object_removal*.py, edit_object_inpaint.py, render.py      Step 8, 11, 12
  LaMa/, Segment-and-Track-Anything/, instantmesh/, gaussian_splatting/, submodules/   code bên thứ ba
scripts/
  run_pipeline.py                 entry-point: --check-env, --list, --step(s), --dry-run, --modal
  modal_come_train.py             Step 0-2 (Volume safe-gs-come)
  modal_come_pipeline.py          Step 4-12 (Volume inpaint360gs) + pipeline_steps.py (bảng lệnh)
  modal_come_mesh.py              Step 13
  modal_instantmesh.py            Step 14.2-14.5 (Volume instantmesh)
  modal_common.py                 helper chung + image CoMe
  evaluate_all_metrics.py         đánh giá -> council_report_summary.{md,csv}
outputs/                          (gitignored) <Dataset>/<scene>_<object>/step0..15, table_plane.json, evaluation_summary.json
docs/reference/                   tài liệu + sơ đồ framework của bản mẫu
Based_Model/                      repo gốc để đối chiếu (không sửa)
legacy/                           pipeline cũ (src_v0/, scripts_v0/, README_v0.md)
test_model/                       (gitignored, chỉ có ở máy local) các thử nghiệm ghép model
```

Các file ở gốc repo `modal_infusion.py`, `modal_lerf_prep.py`, `modal_sync.py`, `README_MODAL.md` thuộc pipeline InFusion / GOF cũ.
`modal_train.py` vẫn gọi `src/Module*`, mà thư mục này nay đã chuyển sang `legacy/src_v0/`.
