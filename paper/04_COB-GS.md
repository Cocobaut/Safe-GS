# COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting

- **Tác giả:** Jiaxin Zhang, Junjun Jiang, Youyu Chen, Kui Jiang, Xianming Liu (HIT)
- **Venue:** CVPR 2025
- **Paper:** https://arxiv.org/abs/2503.19443
- **Code:** Có. https://github.com/ZestfulJX/COB-GS (env `cobgs`, rasterizer riêng có nhánh train mask,
  Grounded-SAM-2; README có 2 TODO chưa xong: multi-object hiệu quả, texture hiệu quả)

## Ý tưởng chính

- Gaussian 3DGS hay **vắt ngang mép vật** (vì có thể tích và không có tín hiệu ngữ nghĩa lúc train).
- Hậu xử lý cảnh 3DGS đã train: học nhãn mask cho Gaussian, dùng **thống kê gradient ngữ nghĩa** để tìm Gaussian mập mờ
  ở mép rồi **tách** chúng cho khớp mép; xen kẽ tối ưu hình học / texture để giữ chất lượng render.
- Mask lấy từ SAM2 + Grounding DINO (prompt bằng chữ), 2 giai đoạn thô / tinh.

## Số liệu (theo paper)

- 1 x RTX 3090, cảnh Fortress 42 view: tổng ~0.46 phút (SAGA 22.4 phút). VRAM đỉnh 2.7 GB (FlashSplat 2.4 GB).
- NVOS 92.1 mIoU.

## Liên quan bài toán của mình

- Giải **mép vật bẩn / Gaussian dính 2 vật** (bước 2) để khi nhấc vật lên không kéo theo mảnh nền và ngược lại.

## Chạy trên L4

- Dễ về tài nguyên (2.7 GB, dưới 1 phút trên 3090).
- Vướng: dùng rasterizer riêng và tinh chỉnh lại Gaussian, nên không chạy thẳng trên cảnh CoMe; phải train lại bằng
  3DGS gốc của họ hoặc port.

## Ưu điểm

- Mép vật sạch, bền với mask không chính xác, nhẹ và nhanh.

## Nhược điểm (paper tự nêu)

- Floater của reconstruction bị **phóng đại sau khi tách**.
- Tham số delta đánh đổi: tăng thì bỏ nhiều Gaussian mập mờ hơn nhưng sinh thêm nhiều Gaussian ở mép.
- Gaussian mập mờ nhỏ do mask sai vẫn có thể còn; SAM2 mất liên tục vật ở chuỗi dài.
- Chủ yếu từng vật một (multi-object hiệu quả còn trong TODO). Không xử lý ID không nhất quán, không xử lý phần bên trong.
