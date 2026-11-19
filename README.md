# Safe-GS: Tách vật thể 3D, Chống Vỏ Ma & Mô phỏng Vật lý từ Ảnh RGB (3D Gaussian Splatting)

**Safe-GS** là hệ thống biến tập ảnh RGB đa góc nhìn thành **cảnh 3D Gaussian Splatting (3DGS)** có thể **tách rời từng vật thể**, **nhấc/xóa vật sạch sẽ** (không để lại "vỏ ma" lơ lửng hay làm thủng mặt bàn), **tự lấp phần đáy/mặt khuất bị che**, và **xuất lưới tam giác (mesh) kín** để đưa vào mô phỏng vật lý **MuJoCo**.

---

## 1. Giải thích các thuật ngữ & ký hiệu trong dự án

Để đọc code và các bảng kết quả trong [RESULTS.md](RESULTS.md) mà không bị rối bởi tên viết tắt nội bộ, đây là ý nghĩa cụ thể của từng phần:

- **CoMe (3DGS nền tảng):** Mô hình tái tạo cảnh 3D Gaussian Splatting tường minh (lấy từ `Based_Model/CoMe`), giữ từng hạt Gaussian độc lập kèm bộ lọc 3D và độ tự tin màu sắc để dễ gắn nhãn, ép vị trí và dựng mesh.
- **SAM 3 (Segment Anything Model 3):** Mô hình cắt ảnh 2D theo văn bản (ví dụ `"brown teddy bear"`, `"yellow bowl"`), tạo mặt nạ (mask) cho từng vật trên mọi góc ảnh.
- **Visual Hull (Khuôn 3D / `hull`):** Khối voxel 3D được dựng bằng cách chiếu ngược mặt nạ 2D từ nhiều góc nhìn vào không gian 3D ([tools/mask_hull.py](tools/mask_hull.py)). Vật thật chỉ được phép tồn tại bên trong khuôn này.
- **Các bước trong chuỗi xử lý (`S0` -> `S10`):**
  - `S0_colmap`: Chạy COLMAP (Structure-from-Motion) từ ảnh RGB để lấy vị trí camera và đám mây điểm thưa ban đầu.
  - `S1_masks`: Chạy SAM 3 lấy mask 2D và dựng khuôn Visual Hull 3D cho từng vật.
  - `S2_come`: Huấn luyện CoMe 30,000 bước để dựng cảnh 3DGS thô.
  - `S3_label` ([pipeline/s3_label.py](pipeline/s3_label.py)): Bỏ phiếu đa góc nhìn để gán mỗi hạt Gaussian cho đúng vật (`oid = 0..K-1`), mặt đỡ/bàn (`support`), hoặc nền (`-1`), đồng thời tính tỷ lệ kích thước mét thật (`meters_per_unit`).
  - `S3m_mold` ([pipeline/s3m_mold.py](pipeline/s3m_mold.py)): Khoét/gọt khuôn Visual Hull dựa trên độ che khuất và đóng góp render thực tế để tạo khung vật chính xác (`frame`).
  - `S3c_constrain` ([pipeline/s3c_constrain.py](pipeline/s3c_constrain.py)): Ép các hạt Gaussian nằm gọn trong khuôn (`L_bound`), chống tràn ra ngoài mask (`L_leak`) và giữ độ đặc bên trong mask (`L_cover`).
  - `S3i_mold_fill` ([pipeline/s3i_mold_fill.py](pipeline/s3i_mold_fill.py)): Đặt camera ảo quanh phần bị che (đáy vật, mặt sau) và dùng inpainting ảnh 2D (OpenCV Telea) để lấp lỗ khuyết bằng các hạt Gaussian mới.
  - `S4d_cutfill` ([pipeline/s4d_cutfill.py](pipeline/s4d_cutfill.py)): Xóa vết chân vật trên mặt bàn và dán lại vân gỗ/texture từ ảnh thật đa góc nhìn.
  - `S6_mesh` / `S7_physics` / `S8_sim` / `S9v3_render` / `S10`: Xuất mesh kín -> chạy mô phỏng nhấc/thả trong MuJoCo -> render video và đo chất lượng toàn cảnh.
- **Ý nghĩa các chỉ số đánh giá:**
  - **PSNR / SSIM / LPIPS:** Độ trung thực ảnh render trên tập góc nhìn kiểm tra (`test views`); PSNR càng cao càng nét.
  - **Ngoài khung (`outside_pct`):** Tỷ lệ % hạt Gaussian của vật bị văng ra ngoài khuôn 3D quá `0.5 cm` (càng thấp càng gọn, ít floaters).
  - **`leak` / `cover`:** Độ tràn alpha của vật ra ngoài mask 2D (càng nhỏ càng sạch) và độ phủ alpha bên trong mask (càng gần `1.0` vật càng đặc, không bị rỗng).
  - **LERF-Mask mIoU:** Độ khớp của khung 3D chiếu xuống 2D so với nhãn chuẩn của bộ benchmark Gaussian Grouping / LERF-Mask (càng cao càng chuẩn).
  - **Vỏ ma (*Ghost shell*):** Lớp hạt mờ bám lơ lửng còn sót lại khi nhấc hoặc ẩn một vật khỏi cảnh.

---

## 2. Cấu trúc thư mục thực tế của Repository

```text
Safe-GS/
├── configs/
│   ├── pipeline/                  # Cấu hình chạy từng cảnh & phiên bản (teatime_v8_hullv4.json, figurines_v7.json, ramen_v4.json...)
│   ├── sam3/                      # Danh sách prompt & ánh xạ nhãn chuẩn LERF-Mask cho SAM 3
│   └── support_graph/             # Cấu hình đồ thị chống đỡ vật thể (vật nào nằm trên vật nào)
├── pipeline/                      # Mã nguồn lõi từng bước xử lý (S3 -> S10)
│   ├── common.py                  # Hàm dùng chung (nạp scene CoMe, đọc/ghi PLY, RANSAC mặt phẳng)
│   ├── s3_label.py                # Gán nhãn Gaussian đa góc nhìn (count, present, fg_present, fg_present_hull) & tính tỷ lệ mét
│   ├── s3m_mold.py                # Tinh chỉnh khuôn 3D từ Visual Hull
│   ├── s3c_constrain.py           # Tối ưu ép hạt vào khuôn (L_bound + L_leak + L_cover)
│   ├── s3i_mold_fill.py           # Lấp mặt khuất/đáy vật bằng camera ảo + OpenCV Telea
│   ├── s4d_cutfill.py             # Lấp mặt bàn bằng ảnh thật (texture_fill)
│   ├── s6_mesh.py / s7_physics.py # Xuất mesh tam giác kín & dựng mô hình vật lý MuJoCo
│   └── s9v3_render.py / s10_*.py  # Render quỹ đạo nhấc vật & báo cáo định lượng
├── tools/                         # Công cụ độc lập hỗ trợ dựng khuôn, kiểm tra và huấn luyện chung
│   ├── mask_hull.py               # Bộ dựng Visual Hull tổng quát từ ảnh RGB + SfM (không phụ thuộc mặt bàn phẳng)
│   ├── come_joint_train.py        # Huấn luyện chung CoMe + Safe-GS (nhúng L_bound + L_leak + L_cover ngay lúc train CoMe)
│   ├── inspect_joint_lift.py      # Kiểm tra nhấc/ẩn vật (free_vote + peel) để phát hiện và triệt tiêu vỏ ma
│   ├── frame_lerf_iou.py          # Chấm điểm LERF-Mask IoU cho khung 3D trên view test
│   ├── frame_mesh.py / scene_lift.py # Dựng mesh từ khung & chạy kiểm thử nhấc/thả MuJoCo
│   └── prep_scene_colmap.py       # Chuẩn hóa dữ liệu COLMAP (loại trừ view test với --exclude "^test_")
├── modal_*.py                     # Script chạy GPU từ xa trên Modal (modal_come.py, modal_pipeline.py, modal_sam3.py...)
├── Based_Model/                   # Mã nguồn gốc các mô hình tham chiếu (CoMe, SAM 3, Depth Anything 3, TRELLIS, Amodal3R...)
├── logs/                          # File metrics JSON (*.json) & nhật ký chạy (nhiều file log lớn không commit lên Git)
├── RESULTS.md                     # Nhật ký thực nghiệm chi tiết toàn bộ dự án
├── SUMMARY_APPROACHES.md          # So sánh ưu/nhược điểm các hướng tiếp cận lớn trên teatime
└── SUMMARY_SESSION_20261118_19.md # Tổng kết chi tiết đợt chạy cải tiến đa cảnh & Joint CoMe (18–19/11/2026)
```

*(Lưu ý: thư mục `Data/` chứa ảnh, COLMAP, model `.ply` vài chục GB nên đã được `.gitignore` bỏ qua, chỉ giữ tài liệu `.md` trên Git).*

---

## 3. Chúng tôi đã nghiên cứu và thử nghiệm những gì?

Dự án trải qua **5 hướng nghiên cứu chính**, kiểm chứng trên **3 cảnh chuẩn LERF-Mask** (`teatime`, `figurines`, `ramen`):

### A. Tách vật 2D -> 3D (Từ SAM 2 đến `fg_present_hull`)
- **SAM 2 + vote (v1):** Bị lỗi nhảy ID tới `1,010` lần, lẫn thìa vào cốc (`68.0` mIoU) -> **Loại**.
- **ObjectGS (ICCV 2025):** Tách 3D khá tốt (`79.3` mIoU), nhưng dùng neural anchor khó xuất mesh trực tiếp sang MuJoCo và bị tụt `0.7 - 1.0 dB` PSNR -> **Dùng làm đối chiếu**.
- **SAM 3 + Adaptive Gate (`Relative Top-1`, `floor=0.2, rho=0.5`, `per_instance 1`):** Bỏ hoàn toàn luật chỉnh tay, đạt **`92.0` mIoU** trên probe 2D `teatime` và tăng LERF mIoU trên `figurines` từ `65.8 -> 81.2`.
- **Cải tiến gán nhãn 3D (`pipeline/s3_label.py`):** Giải quyết xung đột khi hai vật chồng chéo hoặc vật nhỏ bị bắn nhầm frame (ví dụ `yellow_rubber_duck` nuốt `rubber_duck_with_red_hat`):
  - `fg_present` (chỉ chuẩn hóa theo view có nhãn khi Gaussian thực sự thuộc tiền cảnh) và **`fg_present_hull`** (kiểm tra thêm tính nhất quán 3D bằng `cKDTree` với Visual Hull) giúp cảnh `figurines` giảm tỷ lệ hạt ngoài khung xuống **`5.32%`** (`0.95%` trên 16 vật có ràng buộc) và PSNR đạt **`22.6914 dB`**.

### B. Dựng khuôn 3D không bị phụ thuộc mặt bàn phẳng ([tools/mask_hull.py](tools/mask_hull.py))
- **Thất bại với Depth Anything 3 (`v5_da3`):** Depth đơn ảnh bị nhiễu mép mỏng, khoét mất thìa (`0.0 cm3`), tạo lỗ giả (`genus` gấu tăng `4 -> 53`) và đẩy `35.12%` hạt ra ngoài khung -> **Loại**.
- **Thất bại với Image-to-3D (`TRELLIS`, `Amodal3R`):** Mesh kín đẹp nhưng `TRELLIS` làm phình thể tích `10 - 40%` gây va chạm; `Amodal3R` bị lệch vị trí (`ly_tra` IoU chỉ `0.085`) -> **Loại**.
- **Thành công với Visual Hull tổng quát (`tools/mask_hull.py`):** Kết hợp `--adaptive_min_views`, `--soft_empty_weight`, `--clean_3d`, `--sfm_anchor`, `--vox_rel`, `--assign_mode ratio`. Không giả định mặt bàn phẳng, chạy tổng quát trên cả `teatime` (`85.9` LERF mIoU), `figurines` (`18/18` vật có hull), và `ramen` (`11/11` vật có hull).

### C. Ép hậu kỳ (`S3c`) vs Huấn luyện chung (`Joint CoMe + Safe-GS`)
- **Ép hậu kỳ (`S3c`):** Sau khi CoMe train xong 30k bước, `S3c` chạy thêm 5k bước ép hạt vào khuôn. Trên `teatime`, `r1v8_leak` đạt `23.406 dB`, ngoài khung `0.28%`.
- **Huấn luyện chung ([tools/come_joint_train.py](tools/come_joint_train.py)):** Nhúng trực tiếp `fg_present` + `L_bound` + `L_leak` + `L_cover` vào quá trình train CoMe từ bước `10,000` (khi còn densify) đến `30,000`.
  - Kết quả (`come_joint` trên `teatime`): Đạt **PSNR test `23.5520 dB`** (vượt CoMe 30k gốc `+0.207 dB` và vượt `S3c` `+0.146 dB`), ngoài khung ngay tại iter 30k chỉ **`0.88%`**, `12/12` vật đều đáng tin.
  - *Thử nghiệm thất bại (`come_joint_clean`, `lam_ghost=0.05`):* Phạt opacity hạt rìa trong lúc train làm tụt `-0.456 dB` PSNR (`23.0956 dB`) -> **Không dùng**.

### D. Triệt tiêu "Vỏ Ma" khi nhấc vật ([tools/inspect_joint_lift.py](tools/inspect_joint_lift.py))
- Khi nhấc `gau` hoặc `cuu` trực tiếp bằng nhãn `oid == k`, vẫn còn lớp vỏ mờ và dây đeo cổ xanh của `cuu` do lớp lông sâu bên trong có độ truyền qua `T_i ~ 0` bị gán nhầm vào `oid = -1` (nền), `ban` (`oid=13`), hoặc `goi_keo` (`oid=9`).
- Kết hợp **`free_vote`** (chiếu tâm 3D bỏ qua che khuất) + **`peel`** (tích phân rasterizer 2 lượt giới hạn vỏ `2.0 cm` trên mặt bàn và khóa bảo vệ toàn bộ vật trên bàn) đã xóa sạch **100% vỏ ma & dây đeo cổ xanh của `cuu`** khi ẩn/nhấc vật mà vẫn giữ nguyên vẹn các vật lân cận trên bàn.

### E. Lấp mặt khuất vật (`S3i`) & Lấp nền mặt bàn (`S4d`)
- **`S3i --image_inpaint 1`:** Dùng camera ảo + OpenCV Telea lấp đáy/mặt khuất, giảm tỷ lệ lỗ camera ảo từ `6.66% -> 4.06%` (`teatime`), `1.50% -> 1.22%` (`figurines v7`), `9.96% -> 7.12%` (`ramen v4`) mà không làm giảm PSNR test.
- **`S4d s4d_cutfill --texture_fill 1` (`e8tex`):** Vượt trội `ProPainter` và `Inpaint360GS` (WACV 2026) trên mặt bàn phẳng, đạt PSNR vết chân **`23.835 dB`**, **`91.78%`** điểm phẳng trong `1 cm`.

---

## 4. Tóm tắt kết quả tốt nhất trên 3 cảnh

| Cảnh | Bộ cấu hình tốt nhất | LERF mIoU | PSNR test | Ngoài khung tổng / vật ràng buộc | S3i lỗ camera ảo |
|---|---|---|---|---|---|
| **`teatime`** | `come_joint` (Joint CoMe) / `r1v8_leak` + `e8tex` | `85.9` (9 prompt) | **`23.5520`** (`come_joint`) / `23.406` (`S3c`) | `0.88%` (`come_joint`) / **`0.28%`** (`S3c`) | `6.66% -> 4.06%` |
| **`figurines`** | `v7` (`fg_present_hull` + `per_instance` + scale Rubik) | **`81.2`** (7 prompt) | **`22.6914`** | **`5.32%`** / **`0.95%`** (`16` vật) | `1.50% -> 1.22%` |
| **`ramen`** | `v3` (`fg_present`) / `v4` (`fg_present_hull`) | `69.0` (6 prompt) | **`24.7465`** (`v3`) / `24.7403` (`v4`) | **`17.80%`** / **`0.67%`** (`7` vật) | `9.96% -> 7.12%` |

- **Mô phỏng vật lý MuJoCo trên `teatime` (`mesh_final`):** `12/13` mesh kín (`0` cạnh hở, `0` non-manifold), nhấc `9/9` vật ổn định, thả cốc `20 cm` lệch chỉ `0.0005 mm` / `0.0007 độ`.

---

## 5. Hướng dẫn chạy nhanh trên Modal

Các script chạy từ xa nằm ở thư mục gốc và sử dụng Modal Volume `safe-gs-come`:

```bash
# 1. Chạy chuỗi S3 -> S3m -> S3c -> S3i cho figurines (hoặc ramen)
python -m modal run modal_pipeline.py::figchain --config configs/pipeline/figurines_v7.json --s3i-tag i_r1_fig_v7 --image-inpaint 1

# 2. Huấn luyện chung Joint CoMe + Safe-GS (từ iter 10,000 -> 30,000)
python -m modal run modal_come.py::joint_train --scene teatime --config-name teatime_v8_hullv4.json --labels-rel sam3_masks_adaptive/labels --hull-rel mask_hull_sam3_adap_v4 --out-rel come_joint --from-iter 10000

# 3. Kiểm tra nhấc gấu & cừu (free_vote + peel) trên mô hình Joint CoMe
python -m modal run modal_come.py::joint_lift_inspect --scene teatime --joint-rel come_joint --out-rel come_joint_lift_exact
```

Để xem chi tiết từng lần chạy, hãy mở [SUMMARY_APPROACHES.md](SUMMARY_APPROACHES.md), [SUMMARY_SESSION_20261118_19.md](SUMMARY_SESSION_20261118_19.md) và [RESULTS.md](RESULTS.md).
