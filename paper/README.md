# Paper liên quan: tách vật trong 3D Gaussian Splatting cho physics

**Xem trước: [PIPELINE_SOTA.md](PIPELINE_SOTA.md)** (sơ đồ pipeline ghép SOTA 2025-2026 cho từng khâu + gap nghiên cứu).

Kiểm tra ngày 2026-10-20. Mỗi paper có 1 note riêng trong thư mục này. "Code" = đã mở trang GitHub và thấy file
nguồn (chưa clone, chưa chạy thử).

## Bài toán của mình (nhắc lại)

Từ cảnh CoMe (teatime, 1.8M Gaussian) + mask SAM2 có ID (bị tách vụn, đổi ID ở vật phức tạp), cần:
1. **Đồng bộ ID** giữa các view (gấu 2 / 9 / 6 là một vật).
2. **Phân vùng đầy đủ**: mỗi Gaussian thuộc đúng 1 vật hoặc nền. Nhấc vật thì cả vật đi, xoá thì sạch,
   không còn "vỏ ma" (Gaussian bên trong không view nào thấy: 21% cảnh teatime).
3. **Lấp chỗ trống**: mặt bàn dưới vật + đáy vật, để đưa vào physics.
4. Chạy được trên **Modal L4 (24 GB)**, nhanh.

## Bảng tổng hợp

| # | Paper | Năm / venue | Code | Bước giải quyết | Train lại 3DGS? | Chạy L4 | Ghi chú nhanh |
|---|---|---|---|---|---|---|---|
| 01 | [FlashSplat](01_FlashSplat.md) | ECCV 2024 | Có | 1 phần, 2 (vote) | Không | Dễ (8 GB, ~30 s trên A6000) | Gán nhãn tối ưu 1 bước, có bias nền |
| 02 | [Gradient-Driven Seg.](02_GradientDrivenSegmentation.md) | arXiv 2024 | Có | 2 (vote) | Không | Rất dễ (giây) | Đúng ý "vote bằng gradient" của mình |
| 03 | [Trace3D](03_Trace3D.md) | ICCV 2025 | Có | **1 + 2** | Có (2DGS) | Được (paper dùng 4090) | Sửa mask không nhất quán + tách Gaussian mập mờ |
| 04 | [COB-GS](04_COB-GS.md) | CVPR 2025 | Có | 2 (mép vật sạch) | Tinh chỉnh | Dễ (2.7 GB, ~30 s / 3090) | Tách Gaussian nằm vắt qua mép vật |
| 05 | [Unified-Lift](05_Unified-Lift.md) | CVPR 2025 | Có (doc chưa xong) | **1** | Có (~1 h / 3090) | Được | Codebook vật, lọc nhãn nhiễu |
| 06 | [InstaScene](06_InstaScene.md) | ICCV 2025 | Có (phần generation chưa rõ) | **1 + 2 + 3 (vật)** | Có (2DGS) | Khó (A100 40GB, nhiều phụ thuộc) | Tách + hoàn thiện vật bằng 3D diffusion |
| 07 | [DecoupledGaussian](07_DecoupledGaussian.md) | CVPR 2025 | Có | **3 + physics** | Có (PGSR) | Được (A40 trong paper) | Gần bài toán của mình nhất, lấp cả vật và mặt tiếp xúc |
| 08 | [Seed2GS](08_Seed2GS.md) | arXiv 08/2026 | **Không** | 2 | Không | ? (9.3 s) | Mới nhất, không cần camera gốc |
| 09 | [Multi-Cue Mask Refinement](09_MultiCueMaskRefinement.md) | ICPR 2026 | **Không** | **1** | Có | ? | Gộp mask vụn bằng semantic + depth + edge |
| 10 | [Occlusion-Robust Multi-Object Decoupling](10_OcclusionRobustDecoupling.md) | arXiv 06/2026 | **Không** | 2 + 3 + physics | Có | Khó (SDS) | Nhiều vật bị che, ra MPM |
| 11 | [GASE](11_GASE.md) | arXiv 06/2026 | **Chưa** ("will be released") | 1 + 3 + sim | Có | ? | Hệ thống real-to-sim hoàn chỉnh |
| 12 | [Semantic-Guided Progressive Removal](12_ProgressiveObjectRemoval.md) | arXiv 07/2026 | **Không** | 3 (nền) | Có | ? | Inpaint nền sau khi xoá vật |
| 13 | [Baseline: Gaussian Grouping, SAGA](13_Baselines_GaussianGrouping_SAGA.md) | ECCV 2024 / AAAI 2025 | Có | 1, 2 | Có | Gaussian Grouping nặng (34 GB) | Baseline hay được so sánh |

Đã xem nhưng không làm note riêng (không tìm thấy code): Lifting by Gaussians (WACV 2025, arXiv 2502.00173),
GaussianCut (NeurIPS 2024, arXiv 2411.07555), PanopticSplatting (arXiv 2503.18073).

## Nhận xét chung

- **Chưa có paper nào giải trọn cả 4 yêu cầu** với code chạy được. Paper gần nhất là DecoupledGaussian (07), nhưng
  chỉ thử 1 vật trên 1 mặt tiếp xúc, và tự train lại cảnh bằng PGSR.
- **"Vỏ ma" (Gaussian bên trong không view nào thấy)**: các phương pháp vote theo render (01, 02, 03) đều không gán
  được nhóm này, vì đóng góp của chúng vào ảnh bằng 0. Trace3D (03) còn ghi rõ giới hạn: tracing dừng khi opacity tích
  luỹ đạt 1, nên Gaussian bị mặt đặc che bị bỏ sót. Chỉ InstaScene (06) và DecoupledGaussian (07) có bước hoàn thiện
  vật trong 3D. => Bước "lan nhãn vào bên trong" trong thiết kế của mình là chỗ còn trống, có thể là đóng góp.
- **Đồng bộ ID**: Trace3D (03), Unified-Lift (05), Multi-Cue (09) đều xử lý, nhưng đều phải train lại đặc trưng
  hoặc cả 3DGS. Trace3D gần ý "đồ thị đồng xuất hiện" của mình nhất (gộp mask theo đa số phiếu của Gaussian).

## Đề xuất thử trên L4 (theo thứ tự)

1. **02 Gradient-Driven** + **01 FlashSplat**: không train lại, chạy trên chính cảnh CoMe đã có. 02 chỉ cần
   rasterizer trả gradient theo màu nên có thể làm ngay với rasterizer của CoMe (cần kiểm tra). Dùng làm bước 1
   và 3 trong thiết kế (vote theo đóng góp thay cho vote theo tâm).
2. **03 Trace3D**: lấy ý tưởng gộp mask theo phiếu Gaussian cho bước 2 (gom ID gấu 2/9/6). Code dùng torch 2.0.1
   + CUDA 11.8, train trên 2DGS, nên chạy riêng một môi trường.
3. **07 DecoupledGaussian**: tham khảo cách lấp mặt tiếp xúc (Poisson field) và pipeline MPM cho giai đoạn B, C, D.
4. **04 COB-GS** nếu mép vật còn bẩn sau bước 1 đến 3.
