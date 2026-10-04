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

### 2.2. Upload thư mục `outputs` hiện có lên Cloud:
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

## 6. Tải Kết Quả Từ Cloud Về Máy Tính

Sau khi huấn luyện xong, bạn tải kết quả từ Cloud Volume về thư mục tương đối `outputs/Replica` trên máy tính bằng một trong hai cách:

### Cách A: Qua lệnh Python (`modal_sync.py`)
```bash
python modal_sync.py download-outputs
```

### Cách B: Qua lệnh trực tiếp của Modal CLI
```bash
python -m modal volume get safe-gs-outputs Replica outputs/Replica
```

*(Tùy chọn) Tải riêng lẻ video orbit hoặc file mesh:*
```bash
# Tải video orbit xoay 360:
python -m modal volume get safe-gs-outputs Replica/3dgs/office0/orbit_30000/orbit.mp4 outputs/Replica/3dgs/office0/orbit.mp4

# Tải riêng mesh do GOF trích xuất:
python -m modal volume get safe-gs-outputs Replica/mesh/office0_gof outputs/Replica/mesh/office0_gof
```
