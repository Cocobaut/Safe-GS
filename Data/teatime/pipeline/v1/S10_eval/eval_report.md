# Bảng đo lường pipeline Safe-GS v1 (teatime)

Số của mình = đo thật trên kết quả pipeline. Mũi tên: ↑ càng cao càng tốt, ↓ càng thấp càng tốt.

## 1. Tổng hợp theo stage

| Stage | Chỉ số | Giá trị | Ghi chú |
|---|---|---|---|
| S0 COLMAP | Ảnh đăng ký ↑ | 177/180 | |
| S0 COLMAP | Sai số chiếu lại trung bình ↓ | 0.66 px | |
| S0 COLMAP | Độ dài track trung bình ↑ | 4.74 | số ảnh thấy 1 điểm 3D |
| S0 COLMAP | Quan sát / ảnh ↑ | 661 | |
| S0 COLMAP | Số điểm 3D | 24685 | |
| S2 CoMe | PSNR ↑ / SSIM ↑ / LPIPS(vgg) ↓, 23 ảnh test của mình | 23.34 / 0.805 / 0.263 | --eval, 1/8 ảnh |
| S2 CoMe | PSNR / SSIM / LPIPS, ảnh test_0 của LERF-Mask (góc nhìn mới) | 25.40 / 0.818 / 0.227 | |
| S2 CoMe | PSNR / SSIM / LPIPS, ảnh test_1 của LERF-Mask (góc nhìn mới) | 24.45 / 0.745 / 0.229 | |
| S2 CoMe | PSNR / SSIM / LPIPS, ảnh test_2 của LERF-Mask (góc nhìn mới) | 24.63 / 0.821 / 0.242 | |
| (căn hệ) | Căn COLMAP LERF -> của mình: sai số tâm camera RMS / max | 0.49 / 2.64 cm | 177 frame chung; PSNR render ở frame LERF 23.31 dB |
| S3 tách vật | **LERF-Mask mIoU ↑ / mBIoU ↑ (10 prompt)** | **68.0 / 65.0** | benchmark chuẩn, so được với paper (mục 2) |
| S3 tách vật | LERF-Mask mIoU / mBIoU (bỏ prompt không có nhóm: spoon handle) | 75.6 / 72.2 | |
| S4 nền | PSNR ngoài vùng vật: nền đã lấp / cảnh gốc ↑ | 23.74 / 23.73 dB | chênh = phần nền bị ảnh hưởng do bỏ vật + tinh chỉnh |
| S4 nền | LPIPS ngoài vùng vật: nền đã lấp / cảnh gốc ↓ | 0.230 / 0.230 | |
| S4 nền | Chỗ vật vừa đứng: % pixel nền nằm trong 1 cm / 2 cm quanh mặt bàn ↑ | 48.2% / 65.4% | nền sạch về hình học |
| S4 nền | Chỗ vật vừa đứng: khoảng cách trung vị tới mặt bàn ↓ | 1.10 cm | |
| S4 nền | Độ nét chỗ lấp / vành bàn quanh đó (render) vs cùng tỉ số trên ảnh thật | 0.31 vs 1.82 | < tỉ số ảnh thật = bị mờ hơn bàn thật |
| S9 ghép lại | PSNR / SSIM / LPIPS: nền S4b + vật S5 ở vị trí nghỉ, 23 ảnh test | 22.62 / 0.794 / 0.271 | so với S2 = mất mát do tách + ghép |
| S8 mô phỏng | Để yên 2 s: dời lớn nhất ↓ / xoay lớn nhất ↓ | 1.40 mm / 0.01° | |
| S8 mô phỏng | Lún sâu nhất lúc nghỉ ↓ | 0.00 mm | |
| S8 mô phỏng | Vận tốc lớn nhất cuối ↓ / NaN | 0.0021 m/s / không | |
| S8 mô phỏng | Tốc độ so thời gian thực ↑ | 8.4x | CPU |
| S8 mô phỏng | lift_coc_su: vật rơi về lệch chỗ cũ ↓ / vật khác bị dời ↓ | 0.00 cm / 0.14 cm | |

## 2. LERF-Mask (teatime) so với paper

| Phương pháp | mIoU ↑ | mBIoU ↑ | Nguồn |
|---|---|---|---|
| SA3D* | 42.5 | 39.20 | Gaga Table 1 |
| LEGaussians | 42.8 | 35.30 | Gaga Table 1 |
| LERF* | 49.7 | 42.60 | Gaga Table 1 |
| GOI | 52.6 | - | Gaga Table 1 (mBIoU không báo) |
| LangSplat | 59.8 | 52.70 | Gaga Table 1 |
| OmniSeg3D | 69.8 | 63.80 | Gaga Table 1 |
| Gaga | 71.2 | 68.40 | Gaga Table 1 |
| Gaussian Grouping* | 71.7 | 66.10 | Gaga Table 1 / Gaussian Grouping (ECCV 2024) |
| ILGS | 84.3 | 75.50 | Gaga Table 1 |
| **Safe-GS v1 (của mình)** | **68.0** | **65.0** | đo thật, cùng code tính IoU / Boundary IoU |

Lưu ý: mình nối prompt -> ID mask bằng tay (config S10_eval.lerf_prompts), còn các paper dùng Grounded-SAM theo chữ;
mask của mình lấy từ SAM 2 lan truyền video (team), không phải từ prompt chữ.

| Prompt | IoU | Boundary IoU |
|---|---|---|
| apple | 93.4 | 93.2 |
| bag of cookies | 96.0 | 95.9 |
| coffee mug | 83.7 | 83.7 |
| cookies on a plate | 56.2 | 56.2 |
| paper napkin | 84.3 | 84.2 |
| plate | 89.1 | 89.1 |
| sheep | 61.6 | 43.9 |
| spoon handle | 0.0 | 0.0 |
| stuffed bear | 90.9 | 77.2 |
| tea in a glass | 25.2 | 26.5 |

## 3. Mesh từng vật (S6)

| Vật | Kín | Số mặt | Thể tích (cm3) | Kích thước (cm) | Chamfer TB / p95 (mm) ↓ | Mảnh CoACD | Thể tích mảnh / mesh | Hausdorff mesh->mảnh (mm) ↓ |
|---|---|---|---|---|---|---|---|---|
| banh_quy_1 | không | 45805 | 84.6 | 7.4 x 6.7 x 3.0 | 3.1 / 7.3 | 1 | 1.00 | 14.5 |
| banh_quy_2 | không | 30465 | 82.6 | 7.1 x 6.8 x 2.9 | 3.9 / 9.6 | 1 | 1.05 | 14.2 |
| coc_su | không | 61722 | 818.1 | 10.6 x 11.6 x 15.7 | 2.5 / 7.3 | 1 | 1.00 | 38.8 |
| dia | không | 48657 | 1220.0 | 25.4 x 25.5 x 3.0 | 3.4 / 12.1 | 1 | 1.12 | 16.2 |
| hop_kem_sua | bỏ qua: trường opacity không vượt 0.5 | | | | | | | |
| ly_tra | không | 88508 | 417.2 | 9.2 x 8.7 x 10.9 | 3.8 / 12.6 | 210 | 0.42 | 3.5 |
| tao | không | 184995 | 216.0 | 7.3 x 7.4 x 7.8 | 2.7 / 7.9 | 1 | 1.04 | 27.2 |
| tui_banh | không | 108664 | 1492.5 | 11.8 x 13.0 x 20.7 | 3.3 / 10.3 | 1 | 1.00 | 40.4 |

## 4. Định nghĩa chỉ số

- **mIoU / mBIoU (LERF-Mask)**: IoU giữa mask dự đoán (Gaussian của nhóm render riêng, opacity tích luỹ > 0.5) và mask GT; Boundary IoU chỉ tính dải biên dày round(0.02 x đường chéo ảnh) px; trung bình theo view rồi theo prompt (đúng script/eval_lerf_mask.py của Gaussian Grouping).
- **PSNR / SSIM / LPIPS(vgg)**: như metrics.py của 3DGS / CoMe.
- **Ngoài vùng vật**: mask vật (SAM2) nới 15 px, chỉ tính phần còn lại -> đo mức nền bị hỏng khi bỏ vật.
- **Chỗ vật vừa đứng**: pixel có tia nhìn cắt mặt bàn trong vết chân vật (chiếu Gaussian vật xuống mặt bàn, lưới 1 cm); đo khoảng cách điểm depth của nền tới mặt bàn.
- **Độ nét**: phương sai Laplacian (ảnh xám) trong vùng lấp / ở vành bàn quanh đó.
- **Chamfer mesh <-> Gaussian**: 20k điểm trên mesh vs tâm Gaussian (opacity >= 0.5) của vật, trung bình 2 chiều.
- **Hausdorff mesh -> mảnh lồi**: khoảng cách xa nhất từ điểm trên mesh tới hợp các mảnh CoACD (độ hở / độ phình của va chạm).
- **Lún**: khoảng cách tiếp xúc âm lớn nhất của MuJoCo khi để yên 2 s.