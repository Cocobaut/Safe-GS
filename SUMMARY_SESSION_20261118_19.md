# Tổng kết thực nghiệm Safe-GS (2026-11-18 -> 2026-11-19)

Tài liệu tổng hợp toàn bộ các thử nghiệm đã chạy từ hôm qua (`2026-11-18`) đến sáng nay (`2026-11-19`) trên Modal (`quangkhaisaigonhcm`), phân rõ **hướng tốt**, **hướng kém/không đạt**, và **thời gian chạy thực tế**. Mọi số liệu chi tiết hơn nằm trong [RESULTS.md](RESULTS.md).

---

## 1. Hướng 1: Cải thiện gán nhãn 3D (`S3_label`) trên `figurines` và `ramen`

Đã thử 3 cơ chế bỏ phiếu mới (opt-in trong `pipeline/s3_label.py`):
- `fg_present`: chỉ Gaussian có tổng đóng góp tiền cảnh `>= assign_ratio` mới được chuẩn hóa theo số view có nhãn (`present`).
- `all_then_fg_present`: giữ `ratio_all` nếu đã vượt ngưỡng, chỉ dùng `fg_present` khi tranh chấp.
- `fg_present_hull`: dùng `cKDTree` kiểm tra nhất quán 3D với Visual Hull (`d_vox <= 2.5`) trước khi kích điểm `fg_present`, đồng thời hạ điểm nếu Gaussian nằm ngoài hull của vật `k` nhưng nằm trong hull của vật ứng viên `j` khác.

### Kết quả trên `figurines` (38 view test)

| Phiên bản (`vote_norm`) | PSNR sau S3c / SSIM / LPIPS | Ngoài khung tổng | Ngoài khung (vật ràng buộc) | `leak` / `cover` | S3i PSNR | S3i hole % TB | Đánh giá |
|---|---|---|---|---|---|---|---|
| `v2` (`all`, nền cũ) | 22.670 / 0.8297 / 0.2050 | 8.26% | 1.05% (14 vật) | 0.0228 / 0.906 | 22.669 | 3.51 -> 2.75% (11 vật) | Vịt mũ đỏ lệch 83.9% |
| `v3` (`present`) | 22.684 / 0.8299 / 0.2047 | 10.77% | 1.11% (13 vật) | 0.0301 / 0.957 | 22.683 | 9.50 -> 8.61% (12 vật) | Cover cao nhưng cat/rabbit/waldo nhiễu |
| `v5` (`fg_present`) | 22.685 / 0.8298 / 0.2047 | 10.36% | 1.10% (13 vật) | 0.0256 / 0.939 | 22.684 | - | Cải thiện nhẹ so với v3 |
| `v6` (`all_then_fg_present`) | 22.676 / 0.8299 / 0.2049 | 11.33% | 1.12% (12 vật) | 0.0256 / 0.940 | 22.676 | - | **Không đạt**: vịt mũ đỏ tụt lại 63.5% |
| **`v7` (`fg_present_hull`)** | **22.6914 / 0.8302 / 0.2046** | **5.32%** | **0.95% (16 vật)** | **0.0229 / 0.9092** | **22.6906** | **1.50 -> 1.22% (13 vật)** | **Tốt nhất**: vịt vàng, vịt mũ đỏ, cat, waldo đều về **0.00%** |

### Kết quả trên `ramen` (17 view test)

| Phiên bản (`vote_norm`) | PSNR sau S3c / SSIM / LPIPS | Ngoài khung tổng | Ngoài khung (vật ràng buộc) | `leak` / `cover` | S3i PSNR | S3i hole % TB (7 vật) | Đánh giá |
|---|---|---|---|---|---|---|---|
| `v2` (`all`, nền cũ) | 24.739 / 0.8679 / 0.1792 | 18.17% | 0.68% (7 vật) | 0.0078 / 0.937 | 24.726 | 10.16 -> 7.11% | Nền đối chiếu |
| **`v3` (`fg_present`)** | **24.7465 / 0.8681 / 0.1792** | 18.14% | 0.68% (7 vật) | 0.0081 / **0.9433** | **24.7371** | - | **Tốt nhất về PSNR & cover** |
| **`v4` (`fg_present_hull`)** | 24.7403 / 0.8681 / 0.1792 | **17.80%** | **0.67% (7 vật)** | 0.0078 / 0.9394 | 24.7300 | 9.96 -> **7.12%** | **Tốt nhất về ngoài khung tổng** |

- Ghi chú điểm còn tồn tại trên `ramen`: 4 vật trong suốt/lồng nhau (`glass_of_water 56.3%`, `metal_plate 42.1%`, `paper_napkin 23.1%`, `wavy_noodles_in_bowl 21.8%`) vẫn vượt ngưỡng `20%` ngoài khung nên cơ chế an toàn không ép.

---

## 2. Hướng 2: Huấn luyện chung CoMe + Safe-GS (`Joint CoMe + Safe-GS`) trên `teatime`

Tích hợp `fg_present` + `L_bound` + `L_leak` + `L_cover` trực tiếp vào vòng train CoMe từ iter `10,000 -> 30,000` (`tools/come_joint_train.py`, `modal_come.py::joint_train`):

| Phiên bản | PSNR test (23 view) | Tổng Gaussian | Gaussian tiền cảnh | Ngoài khung tổng | Đánh giá |
|---|---|---|---|---|---|
| CoMe 30k gốc | 23.345 | 1,732,750 | - | - | Baseline gốc |
| S3c `r1v8_leak` (ép hậu kỳ) | 23.406 | - | 139,144 | 0.28% (sau cắt S3c) | Tốt nhất ở nhánh ép hậu kỳ |
| **`come_joint` (`lam_ghost=0`)** | **23.5520** | 1,651,693 | 140,117 | **0.88%** (ngay tại 30k) | **Tốt nhất**: vượt CoMe gốc +0.207 dB, 12/12 vật đáng tin |
| `come_joint_clean` (`lam_ghost=0.05`) | 23.0956 | 1,601,165 | 119,372 | 0.87% | **Không đạt**: tụt -0.456 dB PSNR, mất chi tiết mà vẫn còn vỏ ma |

---

## 3. Hướng 3: Kiểm tra nhấc `gau` & `cuu` và triệt tiêu vỏ ma (`tools/inspect_joint_lift.py`)

Đã chạy 5 lượt kiểm tra tách/nhấc/ẩn `gau` và `cuu` trên `teatime`:
1. `come_joint_lift` (chỉ ẩn `oid == k`): còn rõ lớp vỏ mờ + dây đeo cổ xanh của `cuu` và chân dưới bàn `gau`; vector nhấc bị ngược dấu.
2. `come_joint_clean_lift` (chạy trên `come_joint_clean`, đã sửa vector `n_up`): hướng nhấc lên trên đúng nhưng vỏ ma vẫn còn.
3. `come_joint_lift_hull` (lọc thêm `oid == -1` trong `hull + chiều cao`): gom thêm `12,440` hạt (`gau`) và `4,133` hạt (`cuu`), nhưng vẫn sót hạt vỏ ma mang nhãn `ban` (`oid=13`) và `goi_keo` (`oid=9`).
4. `come_joint_lift_free` (thêm `free_vote` bỏ qua che khuất + hạt mờ `opa < 0.5`): gom thêm `16,407` hạt (`gau`) và `6,701` hạt (`cuu`), dọn sạch 99% bàn tay lông `gau` sát cốc sứ.
5. `come_joint_lift_peel` & `come_joint_lift_exact` (tích phân rasterizer 2 lượt / vỏ 2.0 cm khóa vật trên bàn): xóa sạch **100% vỏ ma và dây đeo cổ xanh của `cuu`** ở cột 3; bản `exact` giữ nguyên toàn bộ vật trên bàn khi nhấc `cuu` lên +25 cm ở cột 2.

---

## 4. Thời gian chạy thực tế (từ log trên Modal)

- **Chuỗi `figurines` (`S3 -> S3m -> S3c -> S3i`):**
  - `figurines_v5`: ~18 phút (riêng `S3`: `6.05` phút)
  - `figurines_v6`: ~20 phút (riêng `S3`: `6.12` phút)
  - `figurines_v7`: ~23 phút (riêng `S3`: `6.40` phút)
- **Chuỗi `ramen` (`S3 -> S3m -> S3c -> S3i`):**
  - `ramen_v3`: ~19 phút (riêng `S3`: `4.22` phút)
  - `ramen_v4`: ~23 phút (riêng `S3`: `4.38` phút)
- **Joint CoMe + Safe-GS (`teatime`, 30k bước trên L40S):**
  - `come_joint`: **`33.6` phút** cho chặng chính `9k -> 30k` (+ ~13 phút chặng `0 -> 9k` trước đó; log `logs/teatime_joint_come.log`)
  - `come_joint_clean`: **`105.3` phút** (do bật thêm truy vấn `ghost_mask` + loss phạt; log `logs/teatime_joint_clean.log`)
- **Các lượt kiểm tra nhấc `gau` & `cuu` (`joint_lift_inspect`, 5 lượt):** ~1–2 phút/lượt (tổng ~7–8 phút).
- **Tổng thời gian GPU thực tế từ tối 18/11 đến sáng 19/11:** khoảng **4.1 - 4.3 giờ** GPU Modal (nhiều job chạy song song theo cặp nên tổng thời gian chờ thực tế ngắn hơn).
