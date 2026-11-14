# Datasets — Ảnh đa góc nhìn → 3D Gaussian Splatting → Mesh → Physics Engine

Tài liệu này tổng hợp toàn bộ dataset đã khảo sát cho pipeline **ảnh → 3DGS → mesh → physics engine**, phân theo 3 nhóm:

- **[A. Benchmark chuẩn](#a-benchmark-chuẩn---dùng-trong-7-paper-gốc)** — dùng trong 7 paper literature review gốc (3DGSR, CoACD, GASP, LERF, MILo, MeshGS, MeshSplatting)
- **[B. Có nhãn ROI/bộ phận sẵn](#b-dataset-có-nhãn-bộ-phậnroi-sẵn)** — cho thí nghiệm ROI-adaptive
- **[C. Ground-truth ảnh thật, quy mô lớn](#c-dataset-ground-truth-ảnh-thật-quy-mô-lớn)** — bổ sung, đối chứng

Mỗi mục có: mô tả, thông số, cách tải (script hoặc link bấm thẳng), và ghi chú rủi ro/giới hạn nếu có.

---

## A. Benchmark chuẩn — dùng trong 7 paper gốc

### A1. NeRF-Synthetic (Blender)

**Mô tả:** 8 vật thể tổng hợp dựng bằng Blender, path-traced, không có nhiễu — dùng để test nhanh, ít VRAM. Mỗi vật có 100 ảnh train góc nhìn quanh vật + camera pose chính xác tuyệt đối (không cần COLMAP).

| Thông số | Giá trị |
|---|---|
| Số vật | 8: `chair`, `drums`, `ficus`, `hotdog`, `lego`, `materials`, `mic`, `ship` |
| Độ phân giải | 800×800 |
| Loại camera pose | Chính xác tuyệt đối (dựng sẵn, không qua COLMAP) |
| Dùng trong paper | 3DGSR |

**Tải về:**
```
Link trực tiếp (Google Drive): 
https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4

File cần: nerf_synthetic.zip
```
Bấm vào link → chọn `nerf_synthetic.zip` → Download. Không cần đăng ký tài khoản.

---

### A2. DTU (MVS Dataset)

**Mô tả:** 124 scene vật thể thật, quét bằng máy **structured-light chuyên dụng** → ground-truth point cloud siêu chính xác. Đây là benchmark chuẩn cho Chamfer Distance (CD) trong toàn bộ literature review — 5/7 paper gốc dùng.

| Thông số | Giá trị |
|---|---|
| Số scene | 124 (thường dùng subset 15 scene chuẩn: 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122) |
| Ground-truth | ✅ Quét vật lý độc lập (structured-light) — **chuẩn nhất trong mọi dataset đã khảo sát** |
| Dùng trong paper | 3DGSR, MILo, MeshSplatting, GSurf |

**Tải về:**
```
Trang chủ: https://roboimagedata.compute.dtu.dk/?page_id=36
```
Chọn mục "MVS Data" trên trang → tải file ảnh (Rectified) + point cloud ground-truth (Points) tương ứng. Không cần đăng ký, nhưng nên đọc README vì có nhiều phiên bản dữ liệu (khác độ phân giải/số view).

---

### A3. Tanks & Temples

**Mô tả:** 14 scene thực tế (trong nhà + ngoài trời), ground-truth từ **laser scanner công nghiệp**. Có 2 tập con: `Intermediate` (dễ hơn) và `Advanced` (khó hơn). Benchmark chuẩn cho F1-score.

| Thông số | Giá trị |
|---|---|
| Scene phổ biến | Barn, Caterpillar, Ignatius, Courthouse, Meetingroom, Truck |
| Ground-truth | ✅ Laser scanner công nghiệp — nhưng **chỉ phủ được vùng foreground**, không quét được hết nền |
| Dùng trong paper | 3DGSR, MILo, MeshSplatting |

**Tải về:**
```
Trang tải chính thức: https://www.tanksandtemples.org/download/
```
Trang chia theo từng scene — bấm vào tên scene cần tải (ví dụ "Truck", "Barn") để lấy ảnh + camera pose + ground-truth point cloud riêng.

---

### A4. Mip-NeRF 360

**Mô tả:** 9 scene **unbounded** (không giới hạn, có cả tiền cảnh lẫn hậu cảnh 360°) — 4 outdoor (Bicycle, Garden, Stump, Flowers, Treehill) + 5 indoor (Room, Counter, Kitchen, Bonsai). **Không có ground-truth hình học** — đây là lý do MILo phải tự chế ra metric Mesh-Based NVS.

> ⚠️ **Sửa lỗi:** link Mip-NeRF 360 KHÔNG phải link Tanks & Temples — 2 dataset khác nhau hoàn toàn, dùng đúng link riêng bên dưới.

| Thông số | Giá trị |
|---|---|
| Số scene | 9 (4 outdoor + 5 indoor) |
| Ground-truth | ❌ Không có — chỉ có ảnh RGB |
| Dùng trong paper | MILo, MeshGS, MeshSplatting |

**Tải về:**
```
Trang chủ (đúng link): https://jonbarron.info/mipnerf360/
```
Trang có link tải trực tiếp file `.zip` cho từng phần (dataset gốc + 2 scene bổ sung công bố sau). Không cần đăng ký.

---

### A5. Deep Blending

**Mô tả:** 2 scene indoor thật (`Playroom`, `DrJohnson`) — Playroom có texture phức tạp, DrJohnson có nhiều vùng ít texture. Không có ground-truth hình học độc lập (mesh dựng bằng phần mềm thương mại RealityCapture, không phải máy quét).

| Thông số | Giá trị |
|---|---|
| Số scene | 2: Playroom, DrJohnson |
| Ground-truth | ❌ Không độc lập |
| Dùng trong paper | MILo, MeshGS |

**Tải về:**
```
Trang chủ: https://www-sop.inria.fr/reves/publis/2018/HPPFDB18/datasets.html
```
Trang chia 2 loại file: **"Reconstruction inputs & outputs"** (dùng cái này — ảnh gốc + COLMAP, phù hợp train 3DGS) và "IBR inputs & outputs" (dữ liệu rút gọn, không cần cho pipeline của mình).

---

### A6. Shiny Dataset (Ref-NeRF)

**Mô tả:** Vật thể tĩnh có bề mặt **phản chiếu phức tạp** (kim loại, gốm sứ bóng) — dùng để test độ bền của phương pháp với vật liệu khó, đúng kịch bản "tay cầm kim loại bóng" đã cảnh báo trước.

| Thông số | Giá trị |
|---|---|
| Dùng trong paper | GASP |
| Đặc điểm | Vật liệu phản chiếu mạnh — thách thức lớn cho 3DGS/SDF |

**Tải về:**
```
Trang chủ: https://dorverbin.github.io/refnerf/
```
Trang chứa link tải dataset "Shiny Blender" — kéo xuống phần Dataset trên trang để lấy link zip.

---

### A7. D-NeRF

**Mô tả:** 8 scene **động** (dynamic, monocular) — dùng cho GASP khi test mở rộng sang cảnh vật thể chuyển động (kết hợp D-MiSo).

| Thông số | Giá trị |
|---|---|
| Số scene | 8: Hell Warrior, Mutant, Hook, Bouncing Balls, Lego, T-Rex, Stand Up, Jumping Jacks |
| Dùng trong paper | GASP |

**Tải về:**
```
Repo GitHub (chứa link dataset): https://github.com/albertpumarola/D-NeRF
```
README của repo có link Google Drive tải file `data.zip` chứa toàn bộ 8 scene.

---

## B. Dataset có nhãn bộ phận/ROI sẵn

### B1. GAPartNet ⭐ (khuyến nghị hàng đầu cho thí nghiệm ROI)

**Mô tả:** 1.166 vật thể, 27 category, có sẵn nhãn **9 loại bộ phận thao tác được** (tay cầm, ngăn kéo, nắp bản lề, núm xoay...) trên **8.489 bộ phận**. Xây trên nền PartNet-Mobility + AKB-48.

> ⚠️ **Lưu ý bắt buộc:** đây là **mesh CAD + URDF**, KHÔNG có ảnh multi-view sẵn. Phải tự render ảnh qua SAPIEN trước khi train 3DGS (xem hướng dẫn bên dưới).

| Thông số | Giá trị |
|---|---|
| Số vật / category | 1.166 vật, 27 category |
| Số bộ phận có nhãn | 8.489, thuộc 9 GAPart class |
| License | CC BY-NC 4.0 (phi thương mại — đồ án OK, nhớ trích dẫn) |
| Cần bước bổ sung | ✅ Tự render ảnh qua SAPIEN renderer |

**Tải về:**
```
Trang chủ + demo: https://pku-epic.github.io/GAPartNet/
Code + hướng dẫn tải: https://github.com/PKU-EPIC/GAPartNet
Mirror trên Hugging Face: https://huggingface.co/datasets/yangyandan/GAPartNet_PhyScene
```
Làm theo README trong repo GitHub — có script Python để tải + load vào SAPIEN. Sau khi có mesh, cần tự viết script render multi-view (đặt N camera ảo quanh vật, dùng SAPIEN renderer xuất ảnh + pose).

---

### B2. PartNet-Mobility

**Mô tả:** 2.346 vật nội thất dạng khớp nối (tủ, kéo, cửa, máy in...). Dùng trực tiếp trong paper CoACD (làm test bed) và làm nền gốc cho GAPartNet.

> ⚠️ Cùng hạn chế B1: chỉ có mesh CAD, không có ảnh sẵn.

| Thông số | Giá trị |
|---|---|
| Số vật | 2.346, đa dạng category nội thất |
| Dùng trong paper | CoACD |
| Cần đăng ký | ✅ Bắt buộc (miễn phí) |

**Tải về:**
```
Trang tải (cần đăng ký tài khoản SAPIEN): https://sapien.ucsd.edu/downloads
Simulator dùng kèm: https://sapien.ucsd.edu/
```
Đăng ký tài khoản → lấy token tải → dùng script Python có sẵn trong SDK SAPIEN để tải theo category (không bắt tải hết 2.346 vật cùng lúc).

**Gợi ý chọn category để khớp với minh họa trong paper CoACD:** `StorageFurniture` (tủ có ngăn kéo), `Table`, `Kettle`, `Door` — các category này xuất hiện trực tiếp trong hình minh họa gốc của CoACD, giúp so sánh kết quả dễ thuyết phục hơn.

---

### B3. V-HACD Benchmark (phụ trợ cho CoACD)

**Mô tả:** 61 mesh chuẩn (động vật, người, vật cơ khí) — benchmark gốc cho bài toán phân rã lồi trước khi có CoACD.

| Thông số | Giá trị |
|---|---|
| Số mesh | 61 |
| Trạng thái | Tác giả tự ghi "đã ngừng phát triển, chuyển sang CoACD" |
| Dùng trong paper | CoACD (làm baseline so sánh) |

**Tải về:**
```
Repo (chứa dataset benchmark): https://github.com/kmammou/v-hacd
Code CoACD để chạy thuật toán trên bộ này: https://github.com/SarahWeiii/CoACD
```

---

## C. Dataset ground-truth ảnh thật, quy mô lớn

### C1. OmniObject3D ⭐ (khuyến nghị — có cả ảnh lẫn ground-truth)

**Mô tả:** 6.000 vật **quét thật** (không phải CAD), 190 category, mỗi vật có **cả 3 thứ cùng lúc**: mesh có texture độ chính xác cao (ground-truth), ảnh render đa góc nhìn có pose (dựng bằng Blender từ chính mesh), **và** video quay thật kèm pose COLMAP. Đã có tiền lệ dùng cho đúng bài toán này: paper **GSurf** (trong danh sách 27 paper nhóm) dùng 2 subset `OmniObject3D-d` (8 vật chi tiết) và `OO3D-SL` (24 vật ánh sáng mạnh).

| Thông số | Giá trị |
|---|---|
| Số vật / category | 6.000 vật, 190 category |
| Ground-truth | ✅ Quét thật, độ chính xác cao |
| Có ảnh sẵn | ✅ Cả render Blender lẫn video thật + COLMAP pose |
| Mỗi vật | ~100 ảnh, 800×800px |
| Subset gợi ý | `OmniObject3D-d` (8 vật) hoặc `OO3D-SL` (24 vật) — đúng subset GSurf đã dùng |

**Tải về — 2 cách:**

**Cách 1 — dùng Colab (đã có sẵn notebook, đơn giản nhất):**
```python
# Cell 1
!pip install openxlab

# Cell 2 — đăng nhập (cần tài khoản OpenXLab, miễn phí)
!openxlab login

# Cell 3 — tải dataset
!openxlab dataset get --dataset-repo omniobject3d/OmniObject3D-New
```
Notebook mẫu: https://colab.research.google.com/drive/1VbnCKI79y-W7-8wjrk87sI9yeYhKjYd6?usp=sharing

**Cách 2 — trang OpenXLab trực tiếp (bấm tải qua giao diện web):**
```
https://openxlab.org.cn/datasets/OpenXDLab/OmniObject3D-New/tree/main
```

**Trang chủ (mô tả đầy đủ + tài liệu):**
```
https://omniobject3d.github.io/
```

> 💡 **Lưu ý khi chạy Colab:** lệnh `openxlab login` sẽ yêu cầu nhập Access Key + Secret Key — lấy 2 thứ này tại trang cá nhân sau khi đăng ký tài khoản trên openxlab.org.cn (mục Account → Access Key trong dashboard).

---

### C2. CO3D (Common Objects in 3D)

**Mô tả:** 1.5 triệu khung hình từ ~19.000 video quay tay thật (50 category theo MS-COCO), có sẵn camera pose. **Không có ground-truth mesh** — chỉ có point cloud thưa. Phù hợp để test độ bền với nhiễu thật (ánh sáng đời thường, rung tay khi quay) — khác hẳn điều kiện phòng lab kiểm soát của DTU.

| Thông số | Giá trị |
|---|---|
| Số video / khung hình | ~19.000 video, 1.5 triệu khung hình |
| Category | 50 (theo MS-COCO), gợi ý chọn: `toaster`, `hydrant`, `bicycle` — có bộ phận rõ (tay cầm, nút, khung) |
| Ground-truth mesh | ❌ Không có — chỉ nên dùng metric gián tiếp kiểu Mesh-Based NVS |

**Tải về:**
```
Trang Meta AI: https://ai.meta.com/datasets/co3d-dataset/
Code + script tải: https://github.com/facebookresearch/co3d
```
Repo GitHub có script Python tải theo category cụ thể (không bắt buộc tải hết 50 category / 1.5 triệu khung hình).

---

### C3. ScanNet++

**Mô tả:** 1.006 cảnh **phòng đầy đủ** (không phải vật đơn lẻ) — quét bằng laser độ phân giải dưới-milimet + DSLR 33MP + iPhone LiDAR. Ground-truth chuẩn xác thật sự (độc lập, không vòng vo như BlendedMVS).

> ⚠️ **Cảnh báo quan trọng:** đây là quét CẢ CĂN PHÒNG, không phải 1 vật thể như tủ/ấm. Muốn có ROI kiểu "tay cầm/ngăn kéo" phải tự tìm trong scene rồi tự cắt ra — nặng công hơn nhiều so với B1/C1. Chỉ nên dùng nếu đề tài mở rộng sang cảnh phòng đầy đủ.

| Thông số | Giá trị |
|---|---|
| Số scene | 1.006 |
| Ground-truth | ✅ Laser dưới-milimet — độc lập, đáng tin |
| Dung lượng | 132 GB (chỉ mesh+semantics) → 9 TB (full DSLR gốc) |
| Cần đăng ký | ✅ Bắt buộc |

**Quy trình tải — 4 bước:**

**Bước 1 — Đăng ký + ký thỏa thuận sử dụng**
```
https://scannetpp.mlsg.cit.tum.de/scannetpp/register
```

**Bước 2 — Clone bộ công cụ chính thức**
```
Script tải + xử lý: https://github.com/scannetpp/scannetpp
Demo chạy 3DGS sẵn: https://github.com/scannetpp/3DGS-demo
```

**Bước 3 — Chọn đúng gói cần tải (đừng tải mặc định 1.5TB)**

| Gói | Dung lượng | Có nên tải? |
|---|---|---|
| Mặc định (DSLR 2MP + iPhone + mesh + semantics) | 1.5 TB | ❌ Quá nặng |
| Chỉ mesh + semantics | **132 GB** | ✅ Nhẹ nhất, đủ lấy ground-truth |
| Chỉ DSLR 2MP | 371 GB | 🔶 Nếu cần ảnh train 3DGS |
| DSLR 2MP + 33MP (gốc) | 9 TB | ❌ Không cần |
| Point cloud | 720 GB | ❌ Không cần |
| Panocam | 319 GB | ❌ Không cần |

**Bước 4 — Chỉ tải 1-2 scene, dùng file split có sẵn**
```
Danh sách 12 scene nhỏ dành riêng cho test NVS: nvs_test_small.txt
(nằm trong thư mục split/ sau khi clone repo)
```
Script hỗ trợ chỉ định scene ID cụ thể khi chạy lệnh tải, không bắt tải cả 1.006 scene.

---

### C4. BlendedMVS ⚠️ (dùng thận trọng)

**Mô tả:** 17.818 ảnh đa dạng cảnh (kiến trúc, tượng điêu khắc, vật nhỏ). 

> ⚠️ **Cảnh báo:** ground-truth ở đây **không phải quét vật lý độc lập** — mesh được dựng lại từ chính ảnh gốc qua pipeline photogrammetry thương mại (Altizure), rồi render ngược thành ảnh. So sánh CD với ground-truth kiểu này mang tính "vòng vo" (circular) — kém tin cậy hơn DTU/T&T/OmniObject3D/ScanNet++ rất nhiều. **Chỉ dùng bổ sung, không dùng làm bằng chứng chính cho độ chính xác.**

**Tải về:**
```
Trang GitHub chính thức (tìm "BlendedMVS" trên GitHub, tác giả YoYo000)
Paper: https://arxiv.org/abs/1911.10127
```

---

## Bảng tổng hợp nhanh — chọn dataset theo mục đích

| Mục đích | Dataset nên dùng |
|---|---|
| Test nhanh, ít VRAM, không lo ground-truth | **A1. NeRF-Synthetic** |
| Đối chứng trực tiếp với số liệu 7 paper gốc | **A2. DTU**, **A3. Tanks & Temples** |
| Có sẵn nhãn tay cầm/ngăn kéo, chấp nhận render ảnh giả | **B1. GAPartNet** |
| Vừa có ảnh thật vừa có ground-truth mesh chuẩn, đã có tiền lệ dùng | **C1. OmniObject3D** ⭐ |
| Test độ bền với nhiễu ảnh chụp tay thật | **C2. CO3D** |
| Mở rộng sang cảnh phòng đầy đủ (không chỉ 1 vật) | **C3. ScanNet++** |
| Test vật liệu phản chiếu khó (kim loại, gốm bóng) | **A6. Shiny Dataset** |
