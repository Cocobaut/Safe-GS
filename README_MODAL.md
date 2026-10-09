# Hướng Dẫn Chạy Safe-GS Trên Modal Cloud GPU

Dự án được chia làm **2 file độc lập** với vai trò rõ ràng:
1. **`modal_sync.py`**: Quản lý dữ liệu và Volume (upload dataset, upload outputs, kiểm tra kết nối GPU/Volume, tải kết quả về máy tính).
2. **`modal_train.py`**: Chuyên biệt cho tác vụ **Huấn luyện trên GPU** (Train 3DGS mẫu, Train GOF và Trích xuất Mesh).

---

> [!IMPORTANT]
> **Lưu ý về đường dẫn tương đối (Relative Paths):**
> Tất cả các lệnh dưới đây được chạy trực tiếp từ thư mục gốc của dự án **`Safe-GS`** (ví dụ: `cd F:\Safe-GS`).
> - Thư mục dataset tự động trỏ theo đường dẫn tương đối: `../Replica 8 Scene`
> - Thư mục outputs tự động trỏ theo đường dẫn tương đối: `outputs` (hoặc `./outputs`)

---

## 1. Đăng nhập Modal (làm 1 lần duy nhất)

Mở PowerShell tại thư mục `Safe-GS` và xác thực tài khoản:

```bash
python -m modal setup
```

---

## 2. Upload Dữ Liệu Lên Cloud (`modal_sync.py`)

Chạy trực tiếp từ máy của bạn bằng Python:

### 2.1. Upload Dataset Replica 8 Scene (`../Replica 8 Scene`):
- **Tải toàn bộ 8 scenes (~11.7 GB - Khuyên dùng):**
  ```bash
  python modal_sync.py upload-data
  ```
- *(Tùy chọn)* Hoặc chỉ upload riêng Scene 1 (`office0` ~1.3 GB) để thử nghiệm nhanh:
  ```bash
  python modal_sync.py upload-data --scene office0
  ```

### 2.2. Upload Dataset SPIn-NeRF (`../SPIn-NeRF/spinnerf-dataset`):
- **Tải toàn bộ 10 scenes chuẩn SPIn-NeRF (`spinnerf-dataset` ~6 GB):**
  ```bash
  python modal_sync.py upload-spinnerf
  ```
- *(Tùy chọn)* Hoặc chỉ upload riêng 1 Scene cụ thể để thử nghiệm nhanh (ví dụ: `1`, `book`, `trash`, `2`...):
  ```bash
  python modal_sync.py upload-spinnerf --scene 1
  python modal_sync.py upload-spinnerf --scene book
  python modal_sync.py upload-spinnerf --scene trash
  ```
- *(Tùy chọn)* Dọn dẹp dữ liệu SPIn-NeRF cũ trên Cloud Volume:
  ```bash
  python modal_sync.py clean-spinnerf
  ```

### 2.3. Upload thư mục `outputs` hiện có lên Cloud:
Mặc định chỉ tải thư mục **`Replica`** bên trong `outputs/` lên Modal Volume (tự động bỏ qua `DTU`), đảm bảo cấu trúc cây thư mục trên cloud khớp chuẩn xác 1:1:
```bash
python modal_sync.py upload-outputs
```

*(Tùy chọn) Nếu chỉ muốn upload 1 thư mục con cụ thể trong Replica (ví dụ: `sfm` hoặc `bounding_box_3D`):*
```bash
python modal_sync.py upload-outputs --subfolder sfm
```

---

## 3. Kiểm Tra Kết Nối GPU & Volume (`modal_sync.py`)

Khởi động một worker trên Modal GPU (A10G, 24GB VRAM) để kiểm tra CUDA và trạng thái 2 Volume:

```bash
python -m modal run modal_sync.py::check_connection
```

*(Tùy chọn) Liệt kê các file kết quả hiện có trên Cloud Volume:*
```bash
python modal_sync.py list-outputs
```

*(Tùy chọn) Kiểm tra nhanh toàn bộ đường dẫn dữ liệu & CUDA extensions trên GPU worker trước khi train (chỉ mất ~15 giây):*
```bash
python -m modal run modal_train.py::verify_training_paths
```

---

## 4. Chạy Train 3DGS Mẫu Cho Scene 1 (`modal_train.py`)

Tự động chuẩn bị SfM (nếu chưa có), huấn luyện 3D Gaussian Splatting 30,000 iterations và render video 360° orbit:

```bash
python -m modal run modal_train.py::train_3dgs_sample
```

- **Đầu ra:** Checkpoint `outputs/Replica/3dgs/office0/scene_gs_30000.ply` và video `outputs/Replica/3dgs/office0/orbit_30000/orbit.mp4`.

---

## 5. Chạy Train + Trích Xuất Mesh Bằng GOF Cho Scene 1 (`modal_train.py`)

Tự động sinh SfM mật độ cao, huấn luyện mô hình Gaussian Opacity Fields (GOF 30,000 iterations) và trích xuất Mesh tam giác qua Marching Tetrahedra & Level Set Solver:

```bash
python -m modal run modal_train.py::run_gof_sample
```

- **Đầu ra:** File mesh tam giác tại `outputs/Replica/mesh/office0_gof/mesh_binary_search_*.ply` và checkpoint GOF tại `outputs/Replica/3dgs/office0_gof/`.

---

## 6. Chạy Train SplatFill Trên Modal GPU L4 (`modal_train.py`)

Huấn luyện SplatFill trên GPU **NVIDIA L4 (24GB VRAM)** sử dụng **CUDA C++ Rasterizer (`diff-gaussian-rasterization`)**:

```bash
# Huấn luyện mặc định (scene 1, 8000 base iters, 3 SGI rounds, kèm test evaluation):
python -m modal run modal_train.py::train_splatfill --scene 1

# Huấn luyện các scene khác trong 10 scenes (ví dụ: book, trash, 2, 3...):
python -m modal run modal_train.py::train_splatfill --scene book
python -m modal run modal_train.py::train_splatfill --scene trash

# Thử nghiệm nhanh 100 iterations cho scene 1:
python -m modal run modal_train.py::train_splatfill --scene 1 --iterations 100 --sgi-rounds 0
```

- **Backend:** `diff-gaussian-rasterization` C++ native CUDA kernel (tốc độ > 150 fps trên L4).
- **Đầu ra:** Checkpoint `outputs/splatfill/{scene}/checkpoint_final.pth`, file PLY `model_final.ply`, và kết quả đánh giá (PSNR, SSIM, LPIPS) tại `outputs/splatfill/{scene}/eval/`.

---

## 6b. Chạy InFusion (Inference) Trên Modal GPU L4 (`modal_infusion.py`)

InFusion chỉ chạy inference, dùng checkpoint pretrained `Johanan0528/Infusion`. Checkpoint được tự tải vào volume `safe-gs-hf-cache` ở lần chạy đầu. Dữ liệu đầu vào là scene SPIn-NeRF trên volume, nên cần upload trước: `python modal_sync.py upload-spinnerf --scene 1`.

```bash
# Kiểm tra image, CUDA extensions, checkpoint và dữ liệu scene:
python -m modal run modal_infusion.py::verify_infusion --scene 1

# Chạy toàn bộ pipeline (chuẩn bị dữ liệu -> train Gaussians thiếu -> inpaint depth -> compose + fine-tune):
python -m modal run modal_infusion.py::run_infusion --scene 1

# Hoặc chạy từng stage (stage sau dùng kết quả của stage trước trên volume):
python -m modal run modal_infusion.py::infusion_stage1 --scene 1
python -m modal run modal_infusion.py::infusion_stage2 --scene 1 --view 20220819_104351
python -m modal run modal_infusion.py::infusion_stage3 --scene 1 --threshold 0.5 --nb-points 50
```

- **Ảnh 2D đã inpaint:** mặc định dùng SDXL-inpainting. Nếu muốn dùng ảnh tự làm (Photoroom, ...), upload ảnh bằng `python -m modal volume put safe-gs-outputs my.png infusion/1/my.png`, rồi chạy stage2 với `--inpainted-rgb infusion/1/my.png`.
- **Floaters:** sau khi có kết quả, chạy lại riêng `infusion_stage3` và chỉnh `--nb-points` / `--threshold` (README gốc nhấn mạnh hai tham số này rất quan trọng).
- **Đầu ra:** `outputs/infusion/{scene}/gs/point_cloud/iteration_150/point_cloud.ply` và ảnh render tại `outputs/infusion/{scene}/gs/train/ours_150/renders`. Tải về máy (vào `Based_Model/Infusion/output/{scene}`): `python modal_sync.py download-outputs --remote infusion/1 --local Based_Model/Infusion/output/1`. Nên chỉ tải `gs/point_cloud/iteration_150`, `gs/train/ours_150` và `inpaint`, vì các checkpoint trung gian rất nặng.

---

## 6c. InFusion Trên LERF (figurines / teatime): Từ Chọn Vật Đến 3DGS Hoàn Chỉnh

LERF chỉ có ảnh + `transforms.json`, nên cần tạo thêm mask (SAM2) và COLMAP `sparse/0` trước khi chạy InFusion. Ảnh (`images` gốc 994×738) và `transforms.json` của cả 2 scene đã nằm trên volume `safe-gs-dataset/LERF/<scene>`. Các ví dụ dưới đây dùng `teatime`; thay bằng `figurines` nếu cần. Trong PowerShell, đặt trước `$env:PYTHONIOENCODING="utf-8"` để tránh lỗi in ký tự ✓.

**Cách khuyên dùng: chọn vật theo ID.** SAM2 tạo sẵn mask cho mọi vật kèm ảnh đánh số (đã chạy cho cả 2 scene; kết quả có trên máy ở `F:\LERF Datasets\<scene>\<scene>\mask`):

| # | Việc | Lệnh | Chạy ở |
|---|------|------|--------|
| 0 | (Đã chạy) COLMAP sparse/0 | `python -m modal run modal_lerf_prep.py::lerf_colmap --scene teatime` | Modal CPU |
| 0 | (Đã chạy) SAM2 mask mọi vật + ảnh đánh số ID | `python -m modal run modal_lerf_prep.py::lerf_sam2_all_objects --scene teatime` | Modal L4 |
| 1 | Chọn ID vật cần xóa | Xem `mask\vis\*.jpg`, `mask\overview.jpg`, `mask\objects.json` | Local |
| 2 | Tạo mask InFusion từ ID | `python -m modal run modal_lerf_prep.py::lerf_masks_from_ids --scene teatime --ids 11` | Modal CPU |
| 3 | Kiểm tra trước khi chạy | `python -m modal run modal_infusion.py::verify_infusion --scene teatime` | Modal L4 |
| 4 | Train 3DGS "thiếu" (bỏ vùng vật) + render depth | `python -m modal run modal_infusion.py::infusion_stage1 --scene teatime` | Modal L4 (~30-40 phút) |
| 5 | Inpaint ảnh 2D (SDXL) + depth inpainting | `python -m modal run modal_infusion.py::infusion_stage2 --scene teatime` | Modal L4 |
| 6 | Compose + fine-tune -> 3DGS hoàn chỉnh | `python -m modal run modal_infusion.py::infusion_stage3 --scene teatime --threshold 0.25 --radius 0.03 --nb-points 50` | Modal L4 |
| 7 | Tải kết quả về `Based_Model/Infusion/output/teatime` | `python -m modal volume get safe-gs-outputs infusion/teatime/gs/point_cloud/iteration_150 Based_Model/Infusion/output/teatime/gs/point_cloud` (tương tự cho `gs/train/ours_150` và `inpaint`) | Local |

**Xóa vật lớn (nhìn từ nhiều phía) – fine-tune chung nhiều view:** InFusion gốc chỉ inpaint từ một view nên các góc khác còn rác. Sau khi đã chạy InFusion một view (stage 1-3) cho `<scene>`:
```bash
python -m modal run modal_infusion.py::infusion_multiview_joint --src-scene teatime_v2 --scene teatime_v2_improve2 \
    --extra-views frame_00039,frame_00105,frame_00036 --mask-dilate 31
python -m modal run modal_infusion.py::infusion_extract_mesh --scene teatime_v2_improve2 --iteration 3000
```
- LaMa inpaint ảnh gốc ở mỗi view (không sinh vật mới); ngoài vùng vật giữ ảnh gốc.
- Train tiếp 3.000 bước trên cả 180 ảnh thật có mask, cộng các ảnh inpaint (lặp 20 lần, giám sát toàn ảnh).
- Mask nhất quán 3D khi SAM2 trôi ID: `scripts/lerf_masks_3d_consistent.py`. Cắt mesh quanh vị trí vật đã xóa: `Based_Model/Infusion/mesh/crop_region.py`.

**Trích mesh từ 3DGS cuối (logic GOF, code ở `Based_Model/Infusion/mesh`):**
```bash
python -m modal run modal_infusion.py::infusion_extract_mesh --scene teatime          # Modal L4, ~1 giờ
python -m modal volume get safe-gs-outputs infusion/teatime/mesh/mesh_gof.ply Based_Model/Infusion/output/teatime/mesh/mesh_gof.ply
```
Tùy chọn: `--filter-mesh` (bỏ mặt có cạnh dài hơn scale Gaussian, như GOF), `--view-stride 2` (dùng 1/2 camera để nhanh hơn), `--no-texture-mesh` (không gán màu đỉnh).

- **Bước 1:** do lỗi trôi ID, cùng một vật có thể mang nhiều ID ở các góc khác nhau. Hãy kiểm tra vài ảnh và truyền đủ, ví dụ `--ids 2,6,9`.
- **Bước 2:** view inpaint được chọn tự động là ảnh thấy vật rõ nhất mà vật không chạm mép ảnh. Có thể đổi bằng `--view frame_00127`.
- **Đổi vật cần xóa sau khi đã train:** xóa `infusion/teatime/gs` và `infusion/teatime/inpaint` (xem cuối mục này), rồi chạy lại từ bước 2.

**Cách cũ: click chọn vật** (dùng khi SAM2 tự động không tách được vật mong muốn):

| # | Việc | Lệnh | Chạy ở |
|---|------|------|--------|
| 1 | Chọn vật cần xóa (click) | `python scripts/lerf_select_object.py --scene teatime` | Local (cửa sổ GUI) |
| 2 | Upload điểm click | `python modal_sync.py upload-lerf --scene teatime --prompts-only` | Local |
| 3 | SAM2 tạo mask cho mọi ảnh | `python -m modal run modal_lerf_prep.py::lerf_sam2_masks --scene teatime` | Modal L4 |
| 4 | Kiểm tra mask | `python -m modal volume get --force safe-gs-outputs infusion/teatime/sam2_preview outputs/infusion/teatime` | Local |

Sau đó chạy tiếp từ bước 3 ở bảng trên.

**Phím tắt của công cụ click:**
- Chuột trái = điểm thuộc vật; chuột phải = điểm không thuộc vật (dùng để sửa khi mask lan sai).
- `A`/`D` = ảnh trước/sau; `W`/`S` = nhảy 10 ảnh.
- `N` = thêm vật mới; `U` = hoàn tác.
- `V` = chọn ảnh hiện tại làm view inpaint. Nên chọn ảnh thấy rõ toàn bộ vật và vùng nền phía sau.
- `Enter` = lưu.

Nên click vật trên 2-3 ảnh nhìn từ các góc khác nhau. Prompt được lưu ở `outputs/lerf_prompts/<scene>.json`.

**Bước 4 – kiểm tra:** xem ảnh trong `outputs/infusion/teatime/sam2_preview` (vùng mask phủ màu xanh lá). Mask phải phủ trọn vật ở mọi góc, kể cả bóng đổ nếu muốn xóa luôn bóng. Nếu thiếu hoặc lan sai: chạy lại bước 1 (prompt cũ được nạp lại để thêm điểm), rồi chạy lại bước 2 và 3.

**Bước 8:** view inpaint lấy từ phím `V` ở bước 1. Có thể đổi bằng `--view frame_00060`. Có thể dùng ảnh tự inpaint thay cho SDXL bằng `--inpainted-rgb` (xem mục 6b).

**Bước 9 – floaters:** scene LERF nhỏ (camera trong bán kính ~1.1-1.4 đơn vị), nên giá trị mặc định của tác giả (`--threshold 1.0 --radius 0.1`, chọn cho Mip-NeRF) quá lớn. Hãy bắt đầu từ các giá trị trong bảng, xem kết quả render, rồi chạy lại riêng bước 9 để tinh chỉnh.

**Làm lại mask sau khi đã train:** xóa kết quả cũ rồi chạy lại từ bước 7:
```bash
python -m modal volume rm -r safe-gs-outputs infusion/teatime/gs
python -m modal volume rm -r safe-gs-outputs infusion/teatime/inpaint
```

**Đầu ra:** 3DGS hoàn chỉnh tại `outputs/infusion/teatime/gs/point_cloud/iteration_150/point_cloud.ply`, ảnh render ở `gs/train/ours_150/renders`.

---

## 7. Tải Kết Quả Từ Cloud Về Máy Tính

Sau khi huấn luyện xong, bạn tải kết quả từ Cloud Volume về máy tính:

### Cách A: Qua lệnh Python (`modal_sync.py`) - Tự động & Nhanh chóng
```bash
# Tải toàn bộ kết quả SplatFill về trực tiếp thư mục Based_Model/SplatFill/output/:
python modal_sync.py download-outputs

# Hoặc chỉ tải riêng 1 scene cụ thể (ví dụ: scene 1):
python modal_sync.py download-outputs --scene 1

# Tải kết quả của Replica (về outputs/Replica):
python modal_sync.py download-outputs --target Replica
```

### Cách B: Qua lệnh trực tiếp của Modal CLI
```bash
# Tải kết quả SplatFill về Based_Model/SplatFill/output:
python -m modal volume get --force safe-gs-outputs splatfill Based_Model/SplatFill/output

# Tải kết quả Replica:
python -m modal volume get --force safe-gs-outputs Replica outputs/Replica
```

*(Tùy chọn) Tải riêng lẻ video orbit hoặc file mesh:*
```bash
# Tải video orbit xoay 360:
python -m modal volume get safe-gs-outputs Replica/3dgs/office0/orbit_30000/orbit.mp4 outputs/Replica/3dgs/office0/orbit.mp4

# Tải riêng mesh do GOF trích xuất:
python -m modal volume get safe-gs-outputs Replica/mesh/office0_gof outputs/Replica/mesh/office0_gof
```
