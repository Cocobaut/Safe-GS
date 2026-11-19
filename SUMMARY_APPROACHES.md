# Tổng hợp các hướng tiếp cận Safe-GS trên `teatime`

Tài liệu tổng hợp toàn bộ các phương pháp đã thử nghiệm, kết quả đo thực nghiệm, điểm mạnh/điểm yếu từng hướng và kiến trúc tốt nhất tính đến **2026-11-18**. Mọi số liệu được trích trực tiếp từ [RESULTS.md](RESULTS.md) và các file kết quả trong `Data/teatime/`.

---

## 1. Bức tranh tổng thể bài toán

Để biến một cảnh **3D Gaussian Splatting (3DGS)** tĩnh thành các **vật thể 3D tách rời + mặt bàn sạch + có thể tương tác/mô phỏng vật lý** mà không bị rách đáy hay xuất hiện Gaussian nhiễu ("ma") bay lơ lửng, pipeline cần giải quyết 4 khâu cốt lõi:

1. **Tách vật (Segmentation):** nhận diện đúng từng vật trên 177 ảnh đa góc nhìn.
2. **Dựng khung / khuôn 3D (Mold / Frame):** tạo vùng bao 3D chính xác quanh vật để làm ranh giới ép Gaussian và xuất mesh.
3. **Ép Gaussian & lấp mặt khuất (`S3c` + `S3i`):** kéo các hạt Gaussian thừa vào trong khung, đồng thời lấp phần đáy/mặt khuất.
4. **Lấp nền mặt bàn (`S4d`):** xóa sạch vết chân vật trên bàn mà không làm nhoè hoặc thủng bề mặt bàn.

---

## 2. So sánh chi tiết theo từng khâu

### 2.1. Khâu Tách vật (2D -> 3D)

| Hướng đã thử | Kết quả chính (LERF-Mask mIoU / mBIoU) | Điểm mạnh | Điểm yếu | Nguồn dữ liệu |
|---|---|---|---|---|
| **Safe-GS v1 (SAM 2 + vote)** | `68.0 / 65.0` (10 prompt) | Pipeline đầu tiên chạy thông suốt từ S0 đến S10. | Nhảy ID `1,010` lần (`106` ID), gộp thìa vào cốc, gấu đổi 3 ID, làm khung 3D bị nát. | `Data/teatime/pipeline/v1/S10_eval/eval_report.md` |
| **ObjectGS (ICCV 2025)** | `79.3 / 76.8` (bỏ thìa: `88.2`, paper `88.9`) | Tách vật 3D tốt; `lambda=1.0` giúp mặt bàn dưới vật phẳng hơn (`51.5%` trong 1 cm so với `16.5%` ở `lambda=0.01`). | Dùng neural anchor nên khó xuất mesh trực tiếp sang MuJoCo; tăng `lambda` làm tụt `0.7 - 1.0 dB` PSNR (`23.19 -> 22.46`); vẫn sót "vỏ" vật dẹt bám mặt bàn. | `Data/teatime/objectgs/eval_lerf/`, `eval_removal_*` |
| **SAM 3 bản 3 (ngưỡng cứng `0.5` + luật tay)** | `77.6 / 75.9` (`17` ID ổn định) | Khử được lỗi nhảy ID (`gau` giữ 1 ID, `thia` tách riêng ID 110). | Ngưỡng cứng `0.5` làm rớt `ly_tra` (`48.3`) và `paper napkin` (`0.0`); hạ xuống `0.3` thì `ly_tra` lên `95.5` nhưng `gau` tụt từ `98.9` xuống `65.7`. | `Data/teatime/sam3_masks/` |
| **SAM 3 + Adaptive Gate (`Relative Top-1`, `floor=0.2, rho=0.5`)** | **`92.0 / 90.2`** (`ly_tra 95.5`, `gau 98.9`, `khan_giay 96.5`) | **Tốt nhất hiện tại.** Không cần train lại, bỏ toàn bộ luật tay, giải quyết triệt để mâu thuẫn giữa vật khó (`ly_tra`, `khan_giay`) và vật dễ nhiễu (`gau`). | Số `92.0` đo trên 15 cặp view-prompt LERF-Mask 2D; hiệu quả 3D trên 177 frames (`v6_adap`) đang được chạy kiểm chứng. | `Data/teatime/sam3_masks/probe/gate_eval.json`, `tools/sam3_gate_eval.py` |

---

### 2.2. Khâu Dựng Khung 3D & Mesh (cho ép Gaussian + MuJoCo)

| Hướng đã thử | Kết quả chính | Điểm mạnh | Điểm yếu | Nguồn dữ liệu |
|---|---|---|---|---|
| **v1 (SAM 2 hull + CoACD)** | `0/7` mesh kín; Chamfer `2.5 - 3.9 mm`; Hausdorff `14 - 40 mm` | Dựng nhanh ở giai đoạn đầu. | Mesh hở hoàn toàn, ly trà bị CoACD chia `210` mảnh, phải ép dùng bao lồi. | `Data/teatime/pipeline/v1/S10_eval/eval_report.md` |
| **v3_sam3 (`S3m v3` + `frame_mesh` `mesh_final`)** | **`12/13` mesh kín** (0 biên hở / 0 non-manifold); khung LERF mIoU `87.3`; nhấc `9/9` vật (`9.986 - 9.989 cm`); thả cốc 20 cm lệch **`0.0005 mm` / `0.0007 độ`** | **Ổn định và chính xác nhất.** Vật đứng yên trên bàn (`0.00 mm` trượt, xoay `<= 0.01 độ`), không méo hình học. | Vật dẹt bị phình nhẹ do camera nhìn từ trên xuống; `thia` vẫn rơi khỏi `coc_su` (do phần mũi thìa nằm khuất trong cốc không quan sát được). | `Data/teatime/pipeline/v3_sam3/mesh_final/`, `lift_final_com/`, `gaussian_inspect_v6/frame_lerf_iou_v3.json` |
| **v4 (`protect_vis=3`)** | Mở rộng khung để bảo vệ điểm thấy được | Ít cắt nhầm Gaussian bề mặt thấy được. | Làm phình thể tích khung, không gọn bằng `v3`. | `RESULTS.md` (mục so sánh v3/v4) |
| **v5_da3 (khoét thêm bằng Depth Anything 3)** | `35.12%` Gaussian ngoài khung; `gau` genus `4 -> 53`, `cuu` `1 -> 16`; xóa `thia` (`0.0 cm3`) | Giảm nhẹ thể tích phần bị che. | **Thất bại nặng:** tạo lỗ/genus giả, xóa mất thìa, làm `44.82%` (`gau`) và `45.68%` (`cuu`) văng ngoài khung khiến kiểm định khung từ chối ép. | `Data/teatime/pipeline/v3_sam3/S3c/r1v5/metrics.json`, `mesh_v5/` |
| **TRELLIS (Image-to-3D)** | `13/13` mesh kín; làm sạch topo (`dia` genus `3 -> 1`, `khan_giay` `1 -> 0`) | Lấp mặt khuất tạo hình học liền mạch, giảm đường hầm giả. | **Phình thể tích `10 - 40%`**, làm tăng chồng lấn ban đầu (`banh_quy\|dia 29 mm`, `coc_su\|gau 10 mm`). | `Data/teatime/trellis_out/`, `mesh_trellis/` |
| **Amodal3R (Amodal 3D)** | `5/5` mesh kín | Sinh hình học 3D đầy đủ phần bị che. | **Lỗi căn chỉnh (alignment) vào cảnh thật:** `ly_tra` chỉ phủ `51.5%` (IoU `0.085`), `dia` lệch `3.82 cm`, không thể dùng làm khuôn ép Gaussian. | `Data/teatime/amodal3r_out/a1_aligned/metrics.json` |

---

### 2.3. Khâu Lấp nền Mặt bàn (`S4d`)

| Hướng đã thử | PSNR vết chân / PSNR không-vật | Phẳng trong 1 cm / Độ nét | Điểm mạnh & Điểm yếu | Nguồn dữ liệu |
|---|---|---|---|---|
| **v1 (ProPainter + tinh chỉnh)** | `23.74 dB` (ngoài vật) | `48.2%` / `0.31` (ảnh thật `1.82`) | Chạy được sớm nhất nhưng **bề mặt chỗ nhấc vật bị nhoè**, lệch độ cao trung vị `1.10 cm`. | `Data/teatime/pipeline/v1/S10_eval/eval_report.md` |
| **Inpaint360GS (WACV 2026)** | `23.30 dB` (toàn cảnh 3DGS) | Chưa đo S10 | Lấp khá sạch dưới cốc, đĩa, táo, ly; nhưng còn mảng nhoè lớn chỗ túi bánh + ly. | `Data/teatime/i360gs/result/` |
| **TRELLIS lấp đáy (`i_trellis_d0`)** | `23.0540 dB` (23 view test) | Kém hơn lấp phẳng | Gaussian sinh ra quá nhỏ (`~0.17 mm`), không khớp kernel lọc của CoMe nên render bị mờ. | `RESULTS.md` (mục TRELLIS `i_trellis_d0`) |
| **`s4d_cutfill --texture_fill 1` (`e8tex`)** | **`23.835 dB` / `23.3911 dB`** | **`91.78%` / `0.986`** (lệch trung vị `0.042 cm`) | **Tốt nhất tuyệt đối:** chiếu ngược lấy màu ảnh thật từ 154 view train, vùng đáy che hoàn toàn thì chép texture tịnh tiến từ bàn thật (`20/20` vùng tìm được), tăng `+2.0 dB` và độ nét gấp ~1.6x so với lấp phẳng (`e8r`: `21.829 dB`, `0.627`). | `Data/teatime/pipeline/v1/S4d/e8tex/metrics.json` |

---

### 2.4. Khâu Ép Gaussian (`S3c`) & Ghép toàn cảnh (`S9v3`)

- **Ép Gaussian trong khung (`S3c`, `lam_bound=0.1`, `r1v3`):**
  - Giảm lượng Gaussian vật thể văng ngoài khung từ **`5.66%` (`9,318` hạt) xuống `2.35%` (`3,734` hạt)** trong khi vẫn giữ nguyên chất lượng render 23 view test: **`PSNR 23.3426 / SSIM 0.7991 / LPIPS 0.2648`** (`Data/teatime/pipeline/v3_sam3/S3c/r1v3_rerun/metrics.json`).
- **Toàn cảnh Winning Stack (`final_v3_tex` = CoMe + SAM 3 v3 + `S3m v3` + `S3i i_r1v3k` + `mesh_final` + nền `e8tex`):**
  - Đạt **`PSNR 22.937 / SSIM 0.7849 / LPIPS 0.2886`** trên 23 view test (`Data/teatime/pipeline/v3_sam3/S9v3/final_v3_tex/test_eval/final_v3_tex/metrics.json`), chỉ thấp hơn CoMe 30k gốc (`23.345 / 0.8053 / 0.2629`) khoảng `-0.41 dB` dù toàn bộ vật đã được tách rời và lấp nền thật.

---

## 3. Kết luận: Phương án tốt nhất để báo cáo

1. **Kiến trúc tốt nhất đã kiểm chứng toàn chuỗi (`Winning Stack`):**
   **CoMe 3DGS tường minh + SAM 3 + Khung visual hull (`S3m` / `frame_mesh`) + Ép Gaussian (`S3c lam_bound=0.1`) + Lấp nền texture ảnh thật (`s4d_cutfill --texture_fill`).**
   - Lý do: giữ được tính tường minh của Gaussian để ép khung và dựng mesh kín (`12/13`), không bị phình/lệch tâm như TRELLIS/Amodal3R, không bị thủng/mất chi tiết nhỏ như DA3, mặt bàn sắc nét nhất (`91.78%` trong 1 cm), và mô phỏng MuJoCo ổn định (`9/9` nhấc, thả cốc 20 cm lệch `0.0005 mm`).

2. **Nâng cấp sáng giá nhất đang chạy kiểm chứng (`v6_adap`):**
   - Thay bộ mask SAM 3 bản 3 (nhiều luật tay, ngưỡng cứng `0.5`) bằng **Adaptive Gate (`Relative Top-1`, `floor=0.2, rho=0.5`)** — đã chứng minh tăng LERF-Mask từ `77.6 -> 92.0 mIoU` ở 2D — nhằm tạo khung 3D sạch hơn (`v6_adap`) và giúp `S3c r1v6` ép hạt Gaussian gọn hơn mức `2.35%` của `v3`.

---

## 4. Giải thích kỹ thuật chuyên sâu các module cốt lõi & Giới hạn mặt phẳng

### 4.1. `v3_sam3` (`S3m v3` + `frame_mesh` `mesh_final`)
- **`S3m v3` (Visual Hull từ mask SAM 3):** Chiếu ngược 177 mask 2D của SAM 3 (bản 3) vào lưới voxel 3D. Một voxel được giữ lại làm **khung/khuôn 3D (`frame`)** khi nằm trong vùng nhìn thấy của đủ nhiều góc camera (kèm bảo vệ các vùng bị vật khác che khuất để không khoét nhầm).
- **`frame_mesh` (`mesh_final`):** Chuyển lưới voxel của khung `S3m v3` thành mesh tam giác kín. Nhờ khung đặc và không bị nhiễu lỗ, `mesh_final` đạt **`12/13` mesh kín** (`0` cạnh hở, `0` non-manifold), chạy MuJoCo ổn định (`9/9` nhấc vật, thả cốc 20 cm lệch `0.0005 mm`).

### 4.2. `v5_da3` (khoét thêm bằng Depth Anything 3)
- **Ý tưởng:** Dùng mô hình ước lượng độ sâu đơn ảnh **Depth Anything 3 (DA3)** dự đoán depth map cho các góc nhìn, rồi dùng depth này **khoét bớt** phần voxel bị phình ở mặt khuất của khung `S3m`.
- **Tại sao bị loại:** Độ sâu đơn ảnh bị nhiễu và lệch tỷ lệ ở mép mỏng nên khoét lủng luôn cấu trúc thật:
  - Xóa mất hoàn toàn `thia` (`0.0 cm3`).
  - Tạo lỗ/genus giả (`coc_su` `1 -> 4`, `gau` `4 -> 53`, `cuu` `1 -> 16`).
  - Làm `44.82%` Gaussian `gau` và `45.68%` `cuu` bị đẩy ra ngoài khung.

### 4.3. `s4d_cutfill --texture_fill 1` (`e8tex`) và giới hạn khi nền không phải mặt bàn
Trong `pipeline/s4d_cutfill.py`, `e8tex` thực hiện 3 bước:
1. **Cắt (`cut`):** Xóa các Gaussian nền nằm trong cột vết chân vật (trừ vật được bảo vệ).
2. **Dựng lưới phẳng (`plane_fill`):** Tạo lưới Gaussian phẳng (ô `4 mm`) nằm đúng trên **mặt phẳng bàn** dựa vào pháp tuyến `sup["table"]["normal"]`.
3. **Dán ảnh thật (`texture_fill`):**
   - Chiếu từng Gaussian lấp vào 154 ảnh train, chỉ lấy pixel có nhãn SAM 3 = bàn (ID 13, co 3 px). Điểm nào có `>= 2` view thấy thì tô bằng **trung vị màu ảnh thật**.
   - Với vùng bị đáy vật che kín 100% (không view nào thấy), thuật toán **tịnh tiến cả vùng đó trên mặt phẳng bàn `(u, v)`** (16 hướng x `1 / 1.5 / 2 / 3` lần đường kính) đến chỗ bàn trống lân cận (`>= 90%` điểm thấy được) rồi chép texture sang.

#### Giới hạn khi bề mặt bị che không phải mặt bàn phẳng (ví dụ nhấc con gấu khỏi cái ghế)
- **Với code `s4d_cutfill` hiện tại: KHÔNG xử lý đúng được**, vì thuật toán giả định cứng bề mặt đỡ là **một mặt phẳng duy nhất (`sup["table"]["normal"]`)** và lọc pixel theo nhãn bàn (ID 13).
- **Lý do hỏng trên bề mặt phi phẳng như ghế:** Ghế gồm nhiều mặt không đồng phẳng (mặt ngồi ngang, tựa lưng đứng, tay vịn cong, khe nối). Nếu ép `plane_fill` + tịnh tiến trên `(u, v)`:
  - Mặt phẳng lấp chỉ khớp được một mảnh (ví dụ mặt ngồi), sẽ bị hở hoặc xuyên qua tựa lưng/tay vịn.
  - Phép tịnh tiến 2D có thể trượt ra khỏi ghế, chép nhầm màu tường/sàn vào vùng ghế.
- **Hướng khắc phục (Limitations & Future Work):** Bỏ giả định 1 mặt phẳng bằng cách (a) dựng mặt cong/đa mặt phẳng cục bộ từ các Gaussian lân cận của chính vật đỡ (`support_id` trong đồ thị chống đỡ) thay vì chỉ `table`, hoặc (b) dùng inpainting 2D/3D cục bộ có điều kiện hình học trên các vùng bị che của vật đỡ đó.

---

## 5. Tổng hợp SOTA 2024–2026 cho bài toán xóa vật & lấp nền phi phẳng (vượt khỏi mặt bàn)

Để xử lý vật thể đặt trên **ghế, sofa, sàn gồ ghề hoặc cảnh 360° bất kỳ**, các công trình SOTA mới nhất tập trung vào 3 hướng:

### 5.1. Reference-Guided + Depth Alignment (Khớp hình học & ảnh chiếu chuẩn đa góc nhìn)
- **3DGIC — *3D Gaussian Inpainting with Depth-Guided Cross-View Consistency*** (**CVPR 2025**, `arXiv:2502.11801`, Sheng-Yu Huang et al.): Dùng depth render từ từng góc nhìn train để lần dấu các pixel nền thật còn thấy ở góc nhìn khác, tự động cập nhật mask vùng bị che và tối ưu 3DGS mà không cần giả định mặt phẳng.
- **AuraFusion360 — *Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting*** (**CVPR 2025**): Sinh mask vùng không thấy dựa trên depth, kết hợp *Adaptive Guided Depth Diffusion* và *SDEdit* để căn chỉnh 2D-sang-3D trong cảnh 360° không ràng buộc; công bố bộ dữ liệu **360-USID**.
- **RePaintGS — *Reference-Guided Gaussian Splatting for Realistic and View-Consistent 3D Scene Inpainting*** (`arXiv:2507.08434`, Ji Hyun Seo et al.): Chọn một view tham chiếu đã inpaint, ước lượng độ tương đồng sang các view còn lại và warp nội dung tham chiếu làm pseudo-GT tối ưu trường 3DGS.

### 5.2. Object-Aware + Virtual Camera Synthesis (Tách vật & lấp nền bằng camera ảo)
- **Inpaint360GS — *Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes*** (**WACV 2026**, `arXiv:2511.06457`, Shaoxiang Wang et al.): Chưng cất nhãn phân tách 2D vào 3D, xóa vật theo ngữ nghĩa và dùng **camera ảo** nhìn vào vùng bị che để inpaint (màu + depth) rồi tối ưu lại 3DGS; kèm bộ benchmark **Inpaint360GS**.
- **GScream — *Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal*** (**ECCV 2024**): Kết hợp ước lượng độ sâu đơn ảnh với *online alignment* và *bidirectional cross-attention feature propagation* để giữ cấu trúc 3D ở vùng nền phi phẳng mới lộ ra.

### 5.3. Hướng mới 2026 (Bi-directional 2D-3D, Intrinsic & Texture Space)
- **CoIn — *Comprehensive 2D-3D Inpainting with Gaussian Splatting Guidance*** (`arXiv`, 06/2026): Khung hai chiều (2D-sang-3D dùng *Reference Adaptive GS with Feature Attention*, 3D-sang-2D dùng *Reference Feature Warping*) giúp diffusion 2D giữ đúng hình học 3D bề mặt phức tạp.
- **GOR-IS — *3D Gaussian Object Removal in the Intrinsic Space*** (`arXiv`, 05/2026): Phân rã cảnh sang không gian nội tại (*intrinsic space*: vật liệu và ánh sáng) để tránh lệch bóng/phản xạ và hạt lơ lửng trên bề mặt phi phẳng.
- **3DTexMOR** (`arXiv`, 10/2026): Xóa nhiều vật và inpaint trực tiếp trong không gian texture để điền hình học bị che.

### 5.4. Đề xuất tích hợp vào Safe-GS
1. **Tích hợp thực tế vào codebase hiện tại:** Kế thừa **3DGIC (`arXiv:2502.11801`)** + **Inpaint360GS (`arXiv:2511.06457`)**: thay vì dùng 1 mặt phẳng `sup["table"]["normal"]` trong `s4d_cutfill.py`, sử dụng `support_graph` để xác định vật đỡ `support_id` (ví dụ cái ghế), dựng bề mặt/depth cục bộ từ các Gaussian của chính vật đỡ đó, chiếu ảnh thật từ các góc nhìn thấy được rồi mới dùng 2D inpaint + depth alignment cho vùng che khuất hoàn toàn.
2. **Benchmark mở rộng:** Đánh giá thêm trên **360-USID** (của `AuraFusion360`) hoặc **Inpaint360GS dataset** vì cả hai có ground-truth cảnh thực sau khi nhấc vật khỏi các bề mặt phi phẳng.

---

## 6. Bảng chú giải các lớp (checkbox) trong `mold_inspector_360.html`

Khi mở trình xem 3D `Data/teatime/mold_inspector_360.html`, từng ô tích tương ứng với các phương pháp/bước xử lý sau:

1. **`Khuon du doan - TRUOC = v3_sam3 (khung day)`**: Khung voxel 3D gốc (`S3m/frame`, dựng bằng Visual Hull đa góc nhìn từ mask SAM 3 bản 3, có bảo vệ vùng bị che khuất) trước khi áp dụng khoét độ sâu DA3.
2. **`Khuon du doan - SAU = v5_da3 (khung day)`**: Khung voxel 3D sau khi **khoét thêm bằng Depth Anything 3 (DA3)** dựa trên depth map dự đoán đơn ảnh.
3. **`Voxel bi khoet (TRUOC -> SAU, do)`**: Tập các voxel (tô màu đỏ) bị DA3 cắt bỏ khi chuyển từ `TRUOC` sang `SAU` — dùng để kiểm tra trực quan những chỗ DA3 khoét đúng phần phình hay khoét lủng nhầm cấu trúc thật (như `thia`, `gau`, `cuu`).
4. **`Mat tiep xuc ban (z = 0)`**: Mặt phẳng tham chiếu mặt bàn (`z = 0`), dùng để kiểm tra vật thể có đặt sát bàn, bị lơ lửng hay lún xuyên bàn không.
5. **`Amodal3R_a1 (dat vao canh)`**: Kết quả tái tạo 3D amodal từ mô hình **Amodal3R** (`a1_aligned`) được căn chỉnh và đặt vào cảnh thật (`5/5` mesh kín nhưng lệch căn chỉnh: `ly_tra` chỉ phủ `51.5%`, `dia` lệch `3.82 cm`).
6. **`TRELLIS_t06_rp (dat vao canh)`**: Kết quả sinh 3D từ mô hình **TRELLIS** (`t06_rp_aligned`) đặt vào cảnh thật (`13/13` mesh kín, làm sạch topo nhưng phình thể tích `10 - 40%`).
7. **`mesh_v4 (dat vao canh)`**: Mesh tam giác xuất bằng `tools/frame_mesh.py` từ khung **`S3m_v3`** (cùng họ với `mesh_final` của Winning Stack — bản ổn định nhất dùng cho mô phỏng MuJoCo).
8. **`mesh_v5 (dat vao canh)`**: Mesh tam giác xuất bằng `tools/frame_mesh.py` từ khung **`v5_da3`** (bị tăng genus giả ở `coc_su`, `gau`, `cuu` và xóa mất `thia`).
9. **`mesh_v5g (dat vao canh)`**: Mesh của `v5_da3` có bật thêm cờ **`--min_thick_vox 2`** (ép độ dày tối thiểu 2 voxel để giữ lại `thia` `0.0 -> 5.2 cm3`, nhưng vẫn không sửa được lỗi `thia` rơi khỏi cốc trong MuJoCo nên không dùng mặc định).

---

## 7. Hướng đi tương lai: Huấn luyện phân tách cùng CoMe (`L_leak`) & Cải thiện Visual Hull

### 7.1. Kết hợp Cross-Object Alpha Disentanglement Loss (`L_leak`) cùng CoMe
- **Hiện trạng (`pipeline/s3c_constrain.py`):** Hàm loss khi train mới phạt khoảng cách tâm Gaussian ra khỏi khung 3D (`lb = opacity * d_out_cm`), chưa đưa chỉ số `leak` (alpha của vật `i` rò ra ngoài mask 2D của vật `i`) vào vòng lặp tối ưu.
- **Hướng phát triển tiếp theo:** Đưa `L_leak = (a_i * (~M_i_dilated)).mean()` (phạt alpha của vật `i` đóng góp vào pixel bàn hoặc vật khác) vào quá trình **train kết hợp trực tiếp với CoMe**, giúp tách rạch ròi hạt của từng vật với mặt bàn ngay từ giai đoạn tối ưu 3DGS mà vẫn giữ được cấu trúc bề mặt của CoMe.

### 7.2. Trọng tâm trước mắt: Cải thiện Visual Hull (`tools/mask_hull.py`)
- Hull hiện tại được dựng thuần từ **ảnh RGB -> COLMAP SfM -> Mask 2D (SAM 3)** bằng Space Carving (`min_views=15, ratio=0.85`).
- Các điểm cần cải thiện trước tiên ở bước dựng Hull: giảm phình ở vật phẳng/nhìn từ trên xuống (`dia`, `banh_quy`), cứu các vật mỏng/bị thiếu góc nhìn (`khan_giay`, `coc_kem_sua`, `vung_tra`), và giữ chi tiết lõm/hốc của vật (`coc_su`, `ly_tra`).

## 8. Ket qua 2026-11-18: L_leak / L_cover, inpaint camera ao, canh thu hai (chi tiet trong RESULTS.md)

- **Track A (S3c + L_leak / L_cover, teatime v8_hullv4):** L_leak gan nhu khong tac dung (leak tung vat da ~1e-5); L_cover co tac dung: cover 0.897 -> 0.903
  (lam 0.02) / 0.922 (lam 0.2), PSNR giu 23.35 - 23.41, outside 0.28 / 0.31 %, nhung leak tong tang 0.023 -> 0.025 / 0.030 (danh doi).
- **Track B (S3i --image_inpaint, OpenCV Telea, khong LaMa):** PSNR test 23.358 (i_r1v3k 23.286); hole % camera ao 4.43 -> 4.06 so voi doi chung
  cung base. Lo con lai o gau / khan_giay nam ngoai voxel khuon nen inpaint khong lap duoc.
- **Track C (LERF figurines):** chuoi chung CoMe -> SAM 3 thich nghi -> hull -> S3 -> S3m -> S3c chay het; frame mIoU 76.8 (6 prompt) / 65.8 (7);
  S3c PSNR 22.45 -> 22.66, outside 8.96 -> 4.42 %. Ty le met / don vi sai ~5x (mat san sai) va 2 vat nho khong co hull: viec can lam tiep.
