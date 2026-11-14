# InstantMesh v2 – khăn giấy teatime dựng từ **nhiều ảnh thật** (không dùng Zero123++)

v1 ([RUN_TEATIME_NAPKIN.md](RUN_TEATIME_NAPKIN.md)): 1 ảnh → Zero123++ *tưởng tượng* 6 view → LRM → mesh. Mesh trông giống tờ giấy nhưng **không khớp hình dạng thật** trong ảnh 2D, và nằm trong hệ toạ độ chuẩn hoá (không đặt lại được vào cảnh).

v2: **6 ảnh thật + camera thật** của teatime → LRM (bỏ Zero123++) → **tinh chỉnh silhouette trên tất cả các view** → màu lấy từ ảnh thật → mesh **nằm đúng vị trí/tỉ lệ trong cảnh teatime** và đo được độ khớp (IoU) với mask 2D.

Code gốc (`src/`, `run.py`) không đổi. v2 = cặp **`src_v2/` + `output_v2/`**:

```
InstantMesh/
├── src_v2/                         ← bản copy src/ (import src.* → src_v2.*), không sửa model
│   ├── multiview/
│   │   ├── cameras.py              ← toán camera: scene ↔ LRM frame, homography xoay
│   │   ├── prepare_teatime.py      ← Step 1+2 (local CPU)
│   │   ├── render.py               ← nvdiffrast: silhouette với camera thật, IoU, overlay
│   │   └── gpu_steps.py            ← Step 3, 4, 5 (GPU)
│   └── modal_v2.py                 ← runner Modal (detach, commit định kỳ, checkpoint, download + sha256)
└── output_v2/teatime_napkin/
    ├── step1_view_selection/       ← views.json, inputs_k6/k12/k24.jpg, coverage_azim_elev.png
    ├── step2_inputs/               ← views/*.png (RGBA 320), masks/*.png (½ res), cameras.json
    ├── step3_lrm_multiview/{k6,k12,k24}/  ← mesh_lrm / mesh_scene, iou.json, overlay
    ├── step4_silhouette_refine/    ← mesh sau tinh chỉnh, log.jsonl, iou.json, overlay
    └── step5_final_mesh/           ← napkin_v2_scene.{obj,ply}, napkin_v2_lrm.obj
```

## Chạy

```powershell
cd F:\Safe-GS\Based_Model\InstantMesh
F:\Safe-GS\.venv\Scripts\activate
$env:PYTHONIOENCODING="utf-8"
python -m src_v2.multiview.prepare_teatime                      # Step 1+2, vài giây
python -m modal profile current                                 # nguyenkhacd10
python -m modal run --detach src_v2/modal_v2.py::start          # Step 3,4,5 trên GPU
python -m modal run src_v2/modal_v2.py::status
python -m modal run src_v2/modal_v2.py::download                # → output_v2/teatime_napkin (sha256)
```
Tuỳ chọn: `--steps 4,5` (chạy lại từ bước 4), `--refine-iters 3000`.

## Step 1 – chọn view (local)

1. **Tâm khăn C**: điểm gần nhất (least squares) với các tia camera đi qua tâm mask ID 11 của 106 frame (`transforms.json`, nerfstudio c2w OpenGL).
2. **Hướng lên** = trung bình trục y của các camera; với mỗi frame tính **phương vị / góc ngẩng / khoảng cách** quanh C.
3. **Lọc chất lượng**: không chạm biên ảnh, mask 1 mảnh, diện tích ≥ 0.6 %, `visible_ratio ≥ 0.8` (số pixel ÷ (fx/dist)² so với trung vị → phát hiện bị che), độ nét Laplacian ≥ 0.6 × trung vị → 37/106 view tốt.
4. **Greedy**: bắt đầu từ view chất lượng cao nhất, mỗi lần thêm view có (góc tới view đã chọn gần nhất) × (chất lượng) lớn nhất → bộ **k6** (chính), **k12**, **k24** (thí nghiệm "nhiều view hơn có tốt hơn?").

## Step 2 – đưa ảnh thật về dạng LRM hiểu (local)

* **Chuẩn hoá**: `X_lrm = s · Rw · (X − C)`; `Rw` đưa "up" của cảnh thành +z (như Zero123++/Objaverse); `s` sao cho bán kính bao của khăn (phân vị 90 của độ rộng mask chiếu ngược) = 0.85 (lưới FlexiCubes là ±1.05).
* **Ảnh vào**: mỗi view tốt được **resample bằng homography xoay thuần tuý** (chính xác, không méo hình học) sang camera ảo *ở đúng vị trí camera thật* nhưng nhìn thẳng vào C, FOV chọn sao cho vật trông như từ khoảng cách 4 / FOV 30 của LRM → 320×320 RGBA (alpha = mask SAM2 erode 2 px), nền trắng.
* **Camera vào LRM**: khoảng cách 4, FOV 30, hướng = hướng camera thật (đúng định dạng `get_zero123plus_input_cameras`: 12 số extrinsic + fx, fy, cx, cy chuẩn hoá). Xấp xỉ duy nhất: phối cảnh của camera thật (khoảng cách thật) được coi như ở khoảng cách 4.
* **Mask đánh giá**: mọi frame có khăn, ½ độ phân giải: 255 = khăn, 128 = vật khác (có thể che khăn → bỏ qua), 0 = nền/bàn (label bàn = 1, tự tìm từ vòng quanh mask).

## Step 3 – LRM nhiều view thật (GPU)

`InstantMesh.forward_planes(6 ảnh, 6 camera)` → `extract_mesh` (code model của tác giả, nạp từ `src_v2`). Chạy k6, k12, k24. Với mỗi mesh: chiếu lên **tất cả 106 frame** bằng nvdiffrast với camera thật → **IoU với mask** (bỏ pixel "128"). Bộ tốt nhất = IoU trung bình cao nhất trên các frame **held-out** (không là đầu vào của bộ nào) và không bị che.

## Step 4 – tinh chỉnh silhouette (GPU, có checkpoint)

Tối ưu offset từng đỉnh (Adam, lr 1e-3, 1500 vòng, 8 view ngẫu nhiên/vòng trong các view `visible_ratio ≥ 0.8`):
`loss = Σ (α_render − mask)² / Σ mask` (α khả vi nhờ `dr.antialias`) `+ 500 · ‖Δ − W Δ‖²` (Laplacian đều, giữ bề mặt trơn).
Bản đầu = thành phần lớn nhất của mesh Step 3 tốt nhất. **Checkpoint mỗi 100 vòng** (`ckpt/last.pt`: offset + trạng thái Adam + RNG, ghi tạm rồi `os.replace`) + commit volume → GPU bị thu hồi thì chạy tiếp từ checkpoint.

## Step 5 – màu thật + xuất (GPU)

Mỗi đỉnh lấy màu trung bình từ các ảnh thật nơi nó **nhìn thấy** (depth test bằng nvdiffrast, nằm trong mask khăn), trọng số cos(pháp tuyến, hướng nhìn); đỉnh không thấy (mặt dưới) giữ màu LRM. Xuất:
* `napkin_v2_scene.{obj,ply}` – **toạ độ cảnh teatime (transforms.json)**, đặt thẳng vào cảnh được;
* `napkin_v2_lrm.obj` – hệ chuẩn hoá của InstantMesh (để so với v1).

## Chống mất GPU

`modal run --detach`; `Retries(3)`; commit volume mỗi 120 s + khi SIGTERM; Step 3 `done.json` theo từng bộ view; Step 4 checkpoint 100 vòng; Step 5 chỉ chạy khi chưa có `done.json`. Volume `instantmesh` dùng lại cache trọng số + plugin nvdiffrast của v1.

## Kết quả (2026-11-05, A10G)

**Mesh cuối: `output_v2/teatime_napkin/step5_final_mesh/napkin_v2_scene.obj`** (+ `.ply`): 6 932 đỉnh, 13 952 mặt, **1 thành phần, watertight**, kích thước trong cảnh 0.18 × 0.15 × 0.07 (đơn vị transforms.json), màu từ 80 ảnh thật (74 % đỉnh; phần còn lại – chủ yếu mặt dưới – giữ màu LRM). So sánh ảnh thật ↔ mesh chiếu bằng camera thật: `step5_final_mesh/compare_photo_vs_mesh.jpg`.

IoU giữa hình chiếu mesh và mask SAM2 (bỏ pixel vật khác che):

| bước | IoU held-out không bị che (56 frame) | IoU trên view đầu vào | IoU tất cả 106 frame | ghi chú |
|---|---|---|---|---|
| Step 3 – LRM k6 | 0.531 | 0.557 | 0.496 | 2.8 s; 38 mảnh |
| Step 3 – LRM **k12** | **0.533** | 0.560 | 0.504 | chọn làm khởi tạo |
| Step 3 – LRM k24 | 0.253 | 0.274 | 0.238 | 65 mảnh – **nhiều view hơn làm hỏng** (mạng chỉ học với 6) |
| Step 4 – tinh chỉnh silhouette | 0.933* | – | **0.862** | 1 mảnh, 1 500 vòng ~3 phút |

\* Step 4 tối ưu trên chính 80 frame không bị che, nên 0.933 là **độ khớp trên dữ liệu huấn luyện**, không phải held-out. 106 frame gồm cả frame bị che/cắt biên (IoU thấp hơn vì mask chỉ có phần nhìn thấy).

Nhận xét:
* LRM trên ảnh thật cho **đúng đường viền tổng thể** nhưng tấm khăn **thủng/lỗ chỗ** (overlay Step 3: vùng xanh bên trong) → IoU ~0.53. Không phải lỗi scale.
* k6 ≈ k12, k24 tệ hẳn → khớp dự đoán: LRM chỉ được huấn luyện với 6 view.
* Tinh chỉnh silhouette lấp lỗ và khớp đường viền (IoU 0.53 → 0.93); hội tụ sau ~750 vòng.

Hạn chế còn lại:
* Silhouette chỉ ràng buộc **đường viền**; nếp gấp bên trong lấy từ LRM → bề mặt còn gồ ghề hơn khăn thật.
* Độ phân giải 6.9k đỉnh (v1: 22k) vì khăn chiếm phần nhỏ của lưới 128³ (kích thước LRM 0.94 × 1.12 × 0.40 trong khối 2.1). Có thể tăng `TARGET_R` (0.85 → ~1.0) hoặc chia nhỏ mesh trước Step 4.
* Mặt dưới khăn không có view nào nhìn thấy (góc ngẩng thấp nhất 11°) → vẫn là phỏng đoán.

### Lần chạy cải tiến `_v2` (2026-11-06): khăn to hơn trong lưới + bề mặt trơn

```powershell
python -m src_v2.multiview.prepare_teatime --suffix _v2 --center-from output_v2/teatime_napkin/step4_silhouette_refine/mesh_scene.obj --fill 0.92
python -m modal run --detach src_v2/modal_v2.py::start --suffix _v2 --lam-normals 0,0.1,0.3,1,5
python -m modal run src_v2/modal_v2.py::download --suffix _v2
```

* **Khung bootstrap**: tâm = tâm bbox của mesh Step 4 lần đầu, scale để khăn chiếm ±0.92 lưới (trước: ±0.71 và lệch tâm) → s ×1.60; khăn chiếm ~85 % ảnh vào như ảnh Zero123++.
* **Độ trơn**: Step 4 thêm `λ · normal_consistency` (trung bình 1 − cos giữa pháp tuyến 2 mặt kề nhau); quét λ = 0, 0.1, 0.3, 1, 5; chọn λ lớn nhất có IoU ≥ tốt nhất − 0.01. Thước đo độ gồ ghề: góc nhị diện trung bình giữa các mặt kề nhau (càng nhỏ càng trơn).

Step 3 (LRM, **held-out thật**, 55 frame không bị che, không dùng làm đầu vào):

| bộ view | lần đầu | `_v2` | số đỉnh `_v2` | mảnh |
|---|---|---|---|---|
| k6 | 0.531 | **0.803** | 27 738 | 9 |
| k12 | 0.533 | **≈0.81** | 27 230 | 4 |
| k24 | 0.253 | – | 25 822 | 22 |

Step 4 (từ k12, 79 frame không bị che dùng để tối ưu):

| λ normal | IoU 79 frame | IoU 106 frame | góc nhị diện TB / p90 |
|---|---|---|---|
| lần đầu (6 932 đỉnh) | 0.933 | 0.862 | 21.5° / 51.8° |
| 0 | 0.937 | 0.867 | 14.7° / 35.0° |
| **0.1 (chọn)** | **0.933** | **0.862** | **4.8° / 10.5°** |
| 0.3 | 0.923 | 0.853 | 3.4° / 7.3° |
| 1 | 0.864 | 0.795 | 2.5° / 5.7° |
| 5 | 0.845 | 0.775 | 3.7° / 8.1° |

**Mesh khuyến nghị: `output_v2/teatime_napkin/step5_final_mesh_v2/normal_0.1/napkin_v2_scene.obj`**: 27 198 đỉnh (×3.9 so với lần đầu), 54 408 mặt, 1 mảnh, watertight, IoU giữ nguyên 0.933, bề mặt trơn hơn ~4.5 lần (21.5° → 4.8°). λ ≥ 1 trơn quá: khăn phồng, mất góc nhọn, IoU giảm 0.07. Các λ khác nằm ở `normal_<λ>/` để so sánh; `step4_silhouette_refine_v2/selection.json` ghi lý do chọn.

### Thí nghiệm số view đầu vào `_v3` (2026-11-06): 6 / 12 / 24 / 48 / 96 view

```powershell
python -m src_v2.multiview.prepare_teatime --suffix _v3 --center-from output_v2/teatime_napkin/step4_silhouette_refine/mesh_scene.obj --fill 0.92 --sets 6,12,24,48,96 --heldout 10
python -m modal run --detach src_v2/modal_v2.py::start --suffix _v3 --steps 3
```

Cùng khung bootstrap như `_v2`. **10 frame held-out cố định** (sạch, không bị che, rải đều quanh khăn: 00005, 00019, 00031, 00036, 00047, 00108, 00111, 00125, 00127, 00180) bị loại khỏi mọi bộ → mọi bộ được chấm trên cùng 10 frame. Chỉ còn 28 view đạt chuẩn chất lượng → bộ 48 và 96 buộc phải dùng view kém (bị che / cắt biên / nhỏ / mờ). Chỉ Step 3 (LRM, chưa tinh chỉnh).

| số view | view dưới chuẩn | IoU 10 frame held-out | IoU trên chính đầu vào | đỉnh | mảnh | góc nhị diện TB |
|---|---|---|---|---|---|---|
| 6 | 0 | 0.768 | 0.786 | 25 204 | 17 | 13.2° |
| **12** | 0 | **0.852** | 0.861 | 27 328 | **6** | 11.5° |
| 24 | 0 | 0.826 | 0.827 | 26 642 | 14 | 12.7° |
| 48 | 20 | 0.540 | 0.528 | 19 168 | 47 | 18.4° |
| 96 | 68 | **0.016** | 0.018 | **606** | 2 | 23.3° |

* Đỉnh ở **12 view**; 24 hơi kém; 48 kém hẳn; **96 sụp đổ** (mesh gần như rỗng).
* Không hết bộ nhớ: GPU đỉnh ~19.7–19.8 GB ở mọi bộ (A10G) – bộ nhớ do lưới FlexiCubes, không phải số view.
* Từ 48 trở lên có hai nguyên nhân chồng nhau: (1) mạng chỉ được huấn luyện với 6 view, (2) phải dùng view kém. Nhưng 24 view (toàn view tốt) đã kém hơn 12 → số view tự nó đã có hại.
* IoU 0.852 của k12 ở đây không so trực tiếp được với 0.819 của `_v2` (bộ view và tập frame chấm khác nhau).

Thời gian `_v2`: GPU ~20 phút cho cả 5 λ (mỗi λ ~3 phút + tô màu < 1 phút).

Thời gian / chi phí (lần đầu): Step 1+2 local vài giây; GPU ~6 phút (image dùng cache của v1, LRM 3 bộ view ~10 s, tinh chỉnh ~3 phút, tô màu < 1 phút) ≈ 0.1 USD.
