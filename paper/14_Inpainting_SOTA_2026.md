# 3D inpainting / xoá vật trên 3DGS: khảo sát 2026-10-21

Mục đích: thay S4 (ProPainter + tinh chỉnh) vì chỗ vật vừa đứng bị nhoè (S10: độ nét 0.31 vs 1.82 ảnh thật,
48% pixel trong 1 cm mặt bàn). Số dưới đây chép từ bảng của từng paper (đọc bản html arXiv); số mỗi paper đo trên
dataset khác nhau nên KHÔNG so chéo trực tiếp được.

## A. Inpainting 3D (cảnh Gaussian)

| Paper | Nơi / ngày | 2D inpainter | Vùng không view nào thấy | Code |
|---|---|---|---|---|
| **3DTexMOR** (2610.11198) | arXiv 10/2026 | LaMa trên texture của MẶT PHẲNG đỡ (gom mọi view vào 1 texture rồi chiếu lại) | lấp trên mặt phẳng (giả định vùng cần lấp nằm trên 1 mặt phẳng) | Không |
| **GOR-IS** (2605.00498) | arXiv 05/2026 | LaMa trên ảnh vật liệu (diffuse/roughness), xoá cả phản chiếu | không nói rõ | Chỉ project page |
| **CoIn** (2606.27584) | arXiv 06/2026 | SD 2.0 inpaint + guidance nhất quán (DINO warp) + discriminator | không nói rõ; hỏng khi đổi góc > 180° | "sẽ công bố" |
| **Progressive Removal** (2607.04144) | arXiv 07/2026 | SD 2.1 + ghép block DINOv2 giữa view, inpaint lại block kém | không nói | Không |
| **Inpaint360GS** (2511.06457) | **WACV 2026** | LaMa (màu + depth), camera ảo vòng quanh vật, frame trước làm tham chiếu | **Có: vùng "never-before-seen"** lộ ra ở view ảo, inpaint rồi khởi tạo Gaussian từ depth | **Có**, github.com/dfki-av/Inpaint360GS, Apache-2.0, 01/2026 |
| **AuraFusion360** (2502.05176) | CVPR 2025 | LeftRefill (view tham chiếu) + depth diffusion AGDD + SDEdit | chiếu ngược view tham chiếu theo depth | **Có**, github.com/kkennethwu/AuraFusion360_official, Apache-2.0 |
| 3DGIC (2502.11801) | CVPR 2025 | depth lọc mask theo view khác | dùng nền thấy từ view khác | không thấy |
| VEIGAR (2506.15821) | arXiv 2025 | 1 view neo, chiếu sang view khác bằng depth stereo | | project page |
| Benchmark single-step (2605.30987) | CVPR 2026 WS | LaMa / PowerPaint / Nano Banana: **LaMa + khởi tạo lại tốt nhất trong 3D** | | Không |

Số chính:
- 3DTexMOR, Syn-Diffuse (PSNR / mPSNR / LPIPS): Ours 30.85 / 28.02 / 0.181; GOR-IS 25.04 / 20.94 / 0.232;
  3DGIC 22.49; AuraFusion360 20.44; Inpaint360GS 20.27. (Chỉ dataset tổng hợp; cảnh thật chỉ định tính.)
- Inpaint360GS, dataset của họ (PSNR / LPIPS / FID): Ours 24.40 / 0.130 / 35.9; GauGroup 23.20 / 0.177 / 65.9;
  AuraFusion360 23.15 / 0.192 / 47.7; GScream 20.95; SPIn-NeRF 19.71. 15-24 phút / cảnh trên H100.
- CoIn, SPIn-NeRF (LPIPS / FID): Ours 0.23 / 28.9; GScream 0.26 / 31.2; 3DGIC 0.26 / 36.4. 2.5 h / cảnh, RTX 4090.
- GOR-IS: thắng trên dataset của họ, nhưng trên SPIn-NeRF GScream hơn về PSNR / LPIPS / FID.
- Remove360 (benchmark "còn sót"): 3DGIC+i RQS 0.693 cao nhất, AuraFusion360 0.630; code đo
  github.com/simonakocour/anything_left.

## B. Xoá vật trong video (thay ProPainter ở S4a)

VOR-Bench (BMVC 2026, chấm theo cảm nhận người, điểm TB 1-5): ROSE 4.33, EffectErase 4.27, VOID 4.22,
Casper 4.15, MiniMax-Remover 4.09, DiffuEraser 3.89, ... **ProPainter 3.38 (hạng 9/10)**.
- ROSE (NeurIPS 2025): Wan2.1-Fun-1.3B, weights HF Kunbyte/ROSE, Apache-2.0. Xoá cả bóng / phản chiếu.
- VOID (Netflix, 04/2026): CogVideoX, github.com/Netflix/void-model, cần GPU >= 40 GB.
- EffectErase (CVPR 2026): code có (theo nguồn thứ cấp), giấy phép có thể CC BY-NC.
- MiniMax-Remover: github.com/zibojia/MiniMax-Remover, 480x832, 81 frame.
Giới hạn chung: video inpainting chỉ "bịa" theo từng đoạn video, không bảo đảm nhất quán 3D cho vùng chưa từng thấy.

## C. Kết luận cho Safe-GS
- Vùng dưới vật trên bàn = vùng không view nào thấy, nằm trên mặt phẳng bàn (đã có từ S3 RANSAC).
- Ý của **3DTexMOR** (inpaint 1 texture trên mặt phẳng rồi chiếu ra mọi view) khớp đúng trường hợp này và nhất quán
  đa view theo cấu trúc; số mới nhất và cao nhất nhưng không có code -> tự cài (LaMa sẵn có, phần còn lại là chiếu).
- **Inpaint360GS** là SOTA CÓ CODE gần nhất, xử lý rõ vùng chưa từng thấy -> chạy nguyên pipeline của tác giả trên
  teatime làm baseline so sánh (đo bằng S10).
- **ROSE** thay ProPainter ở S4a nếu vẫn giữ đường video.
