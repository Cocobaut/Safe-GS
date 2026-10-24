# Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks

- **Tác giả:** Joji Joseph, Bharadwaj Amrutur, Shalabh Bhatnagar
- **Venue:** arXiv 2409.11681 (2024, nộp ICRA 2025, chưa thấy ghi đã được nhận)
- **Project:** https://jojijoseph.github.io/3dgs-segmentation
- **Code:** Có. https://github.com/JojiJoseph/3dgs-gradient-segmentation (`demo.py`, `viewer.py`,
  `interactive_segmenter.py`, `requirements.txt`; rasterizer gsplat)

## Ý tưởng chính

- Mỗi view: render ảnh, nhân với mask 2D, lấy trung bình, backprop về **màu (hệ số SH DC) của từng Gaussian**.
  Gradient của Gaussian k bằng alpha_k x T_k, tức đúng mức đóng góp của nó vào vùng mask.
- Cộng gradient của mask vật, trừ gradient của phần ngoài mask; Gaussian có tổng dương thì thuộc vật.
- Cùng cơ chế dùng để tỉa Gaussian vô dụng (gradient luôn 0) và chuyển nhãn affordance từ ảnh sang 3D.

## Số liệu (theo project / paper)

- A6000, mỗi cảnh: 3.35 s dùng toàn bộ mask, 0.35 s dùng 10% mask (27.7 s nếu tính cả sinh mask SAM2).
- Mip-NeRF 360: mIoU 84.72 (baseline tốt nhất 77.68). Cảnh bicycle yếu nhất (59.46) vì chi tiết mảnh.

## Liên quan bài toán của mình

- **Đúng ý tưởng "vote bằng gradient"** mình đề xuất cho bước 1. Không cần kernel CUDA riêng: chỉ cần rasterizer
  trả gradient theo màu.
- Có thể chạy thẳng trên cảnh CoMe bằng `render(..., override_color=...)` nếu backward của rasterizer CoMe trả
  gradient cho `colors_precomp` (cần kiểm tra trước).

## Chạy trên L4

- Rất dễ: không train, vài giây mỗi cảnh trên A6000. 177 view trên L4 mình ước tính dưới 1 phút.

## Ưu điểm

- Đơn giản nhất, gần như không phụ thuộc thư viện riêng, dễ port sang rasterizer CoMe.
- Tự động xử lý bị che (T gần 0 thì gradient gần 0).

## Nhược điểm

- Như FlashSplat: không sửa ID không nhất quán, không gán được Gaussian bên trong vật.
- Paper tự nhận: phụ thuộc mask chính xác và góc nhìn được chọn; vật mảnh / có lỗ kém; tính gradient trên nhiều
  frame chậm hơn phương pháp dùng feature field.
- Chỉ đánh giá trên Mip-NeRF 360 (5 cảnh), chưa có bản hội nghị chính thức.
