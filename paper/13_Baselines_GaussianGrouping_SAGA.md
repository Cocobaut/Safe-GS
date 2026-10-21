# Baseline hay được so sánh: Gaussian Grouping, SAGA

## Gaussian Grouping: Segment and Edit Anything in 3D Scenes

- **Tác giả:** Mingqiao Ye, Martin Danelljan, Fisher Yu, Lei Ke
- **Venue:** ECCV 2024
- **Code:** Có. https://github.com/lkeab/gaussian-grouping
- **Ý tưởng:** thêm "identity encoding" cho mỗi Gaussian, render ra và giám sát bằng mask SAM đã được liên kết ID giữa
  các frame (video tracker), train cùng lúc với 3DGS. Hỗ trợ xoá vật, inpaint, đổi màu.
- **Ưu:** phổ biến, đầy đủ tính năng sửa cảnh, nhiều paper so sánh.
- **Nhược:** nặng (FlashSplat đo 34 GB VRAM đỉnh, vượt L4 24 GB), train lại cả cảnh; kết quả trên LERF-Mask thấp
  (InstaScene đo mIoU 58.1); phụ thuộc chất lượng tracker 2D.

## SAGA: Segment Any 3D Gaussians

- **Tác giả:** Jiazhong Cen và cộng sự
- **Venue:** AAAI 2025 (arXiv 2312.00860)
- **Code:** Có. https://github.com/Jumpat/SegAnyGAussians , project https://jumpat.github.io/SAGA
- **Ý tưởng:** học đặc trưng affinity có cổng theo scale cho mỗi Gaussian từ mask SAM nhiều mức; phân đoạn theo prompt
  (click) rất nhanh (~4 ms).
- **Ưu:** tương tác nhanh, chọn được vật ở nhiều mức (bộ phận / cả vật).
- **Nhược:** cần train đặc trưng (COB-GS đo ~22 phút cảnh Fortress, 7.6 GB); cho ra phân đoạn theo prompt từng vật chứ
  không phải phân vùng toàn cảnh; không xử lý phần bên trong vật.

## Vai trò với mình

- Dùng làm **baseline** trong báo cáo (đa số paper mới đều so với 2 phương pháp này).
- Trên L4: SAGA chạy được; Gaussian Grouping có thể hết bộ nhớ nếu không giảm độ phân giải / số Gaussian.
