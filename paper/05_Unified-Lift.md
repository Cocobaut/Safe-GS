# Unified-Lift: Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting

- **Tác giả:** Runsong Zhu, Shi Qiu, Zhengzhe Liu, Ka-Hei Hui, Qianyi Wu, Pheng-Ann Heng, Chi-Wing Fu
- **Venue:** CVPR 2025
- **Paper:** https://arxiv.org/abs/2503.14029
- **Code:** Có. https://github.com/Runsong123/Unified-Lift (Apache-2.0; `train_unified_lift.py`, `render.py`,
  có kèm DEVA, LaMa; README ghi hướng dẫn sử dụng "đang xây dựng")

## Ý tưởng chính

- Học end-to-end: mỗi Gaussian có vector đặc trưng instance (16 chiều, contrastive loss) + **codebook cấp vật**
  (tối đa 256 vật) nối đặc trưng Gaussian với vật.
- Module **association learning** và **lọc nhãn nhiễu** để codebook học ổn định (mask SAM giữa các view không khớp).
- Không cần hậu xử lý phân cụm kiểu HDBSCAN (dễ phân mảnh quá / thiếu).

## Số liệu (theo paper)

- Train 30,000 iter trên code 3DGS gốc; ~1 giờ trên 1 x RTX 3090 (Panoptic Lifting, Contrastive Lift >= 20 giờ).
- Dữ liệu: **LERF-Mask (có teatime)**, Replica, Messy Rooms (25-500 vật mỗi cảnh).

## Liên quan bài toán của mình

- Giải **bước 1 (đồng bộ ID)** ở cấp vật, chạy được cảnh nhiều vật. Có đánh giá đúng cảnh teatime nên so sánh được.

## Chạy trên L4

- Được về VRAM. Mình ước tính 1.5-2 giờ trên L4 (3090 ~1 giờ). Phải train lại 3DGS + đặc trưng, không dùng cảnh CoMe.

## Ưu điểm

- Đồng bộ ID tốt, nhiều vật, có code, license Apache-2.0, có kết quả trên teatime.

## Nhược điểm

- Train lại toàn cảnh; hướng dẫn dùng chưa đầy đủ; phụ thuộc nhiều thành phần (DEVA, LaMa).
- Kết quả là nhãn trên Gaussian đã train: vẫn không có bước xử lý Gaussian bên trong vật / lấp chỗ trống.
- Paper không báo bộ nhớ và không có mục giới hạn.
