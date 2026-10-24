# Train Gaussian có ID vật để xoá sạch: khảo sát 2026-10-22

Câu hỏi: có phương pháp nào train sẵn Gaussian theo vật (mỗi Gaussian thuộc đúng 1 vật / nền) để xoá vật là sạch,
trước khi tự làm "tinh chỉnh tách lớp"? Số chép từ bảng của paper (bản html arXiv).

| Paper | Nơi / ngày | Cách tách | LERF-Mask teatime mIoU | Code |
|---|---|---|---|---|
| **ObjectGS** (2507.15454) | **ICCV 2025** | Scaffold-GS: mỗi anchor thuộc 1 vật, sinh Gaussian mang ID one-hot cố định, loss phân loại; 1 model cả cảnh | **88.9** (figurines 88.2, ramen 88.0) | **Có**, github.com/RuijieZhu94/ObjectGS, MIT; py3.8, torch 2.1.2, CUDA 11.8, gsplat bản sửa; nhận COLMAP sparse/0 + object_mask/ |
| OP2GS (2605.20044) | arXiv 05/2026 | ID nguyên + opacity thứ 2 (Gaussian gán nhầm trong suốt ở nhánh mask) | 87.0 (3DOVS 97.1, cao nhất) | Không |
| Gaga (2404.07977) | TMLR 2026 | gom mask 2D bằng memory bank (Gaussian Grouping) | 71.2 | Có, MIT; xoá vật: "tham khảo Gaussian Grouping" |
| ILGS | 2025 | (theo bảng Gaga) | 84.3 | chưa kiểm |
| ObjectSplat (2608.30423) | arXiv 08/2026 | nền và từng vật dựng RIÊNG (mesh splatting), nền inpaint trước | không đo | "sẽ công bố" |
| DecoupledGaussian (2503.05484) | CVPR 2025 | chọn vật bằng tay trong viewer, LaMa, Poisson, PGSR | | Có (không thấy license) |
| Clean-GS (2601.00913) | arXiv 01/2026 | ngược lại: giữ 1 vật, xoá phần còn lại | | Có, MIT |
| Prob. object-level GS (2603.14316) | arXiv 03/2026 | xác suất thuộc vật cho từng Gaussian, 1 vật | NVOS 90.1 | Không |

Remove360 (benchmark "còn sót sau khi xoá", 2025-2026): "No method dominates all evaluation axes";
3DGIC+i RQS 0.693 cao nhất (không thấy code); ObjectClearer (xoá 2D trước rồi mới dựng) xoá ngữ nghĩa sạch nhất
(acc_seg 0.976) nhưng hình học kém (acc_Δdepth 0.320).

## Kết luận
- **ObjectGS** là lựa chọn: SOTA có code, ID gắn cứng theo anchor nên MỌI Gaussian đều có chủ (không còn
  "vỏ ma" chưa ai vote như v1), xoá vật = bỏ anchor của ID đó. teatime 88.9 so với v1 của mình 68.0.
- Không xử lý riêng vùng nền bị che dưới vật -> sau khi xoá vẫn phải lấp (bước D, kiểu 3DTexMOR / Inpaint360GS).
- Đầu vào cần mask ID nhất quán giữa view: dùng mask SAM2 của team đã gộp ID theo merge graph S3 (gấu 2+9, bánh 61+99,...).
