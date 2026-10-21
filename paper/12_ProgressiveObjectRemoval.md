# Semantic-Guided Progressive Object Removal with Gaussian Splatting

- **Tác giả:** Xianliang Huang, Chen Xiao, Yuanxiang Ni, Guanming Liu, Mingkai Liu, Dikai Fan, Xiao Liu, Hao Zhang
- **Venue:** arXiv 2607.04144 (05/07/2026), mục cs.RO
- **Code:** **Không**

## Ý tưởng chính

- Xoá vật khỏi cảnh 3DGS và lấp vùng trống cho nhất quán giữa các view.
- **Semantic-guided Block Matching:** dùng đặc trưng DINOv2 từ nhiều view tìm token tương ứng để lấp vùng mask ở view đích.
- **Region-wise Progressive Refinement:** chia mask thành vùng con, chỉ làm lại vùng chất lượng kém.

## Liên quan bài toán của mình

- **Giai đoạn B** (lấp mặt bàn sau khi nhấc vật): đúng vấn đề inpaint từng ảnh không nhất quán giữa view.

## Chạy trên L4

- Không có code, không có số GPU / thời gian.

## Ưu điểm

- Mới, nhắm đúng độ nhất quán đa view của inpainting.

## Nhược điểm

- Không code, chưa bình duyệt; chỉ lo phần nền, không lo hình học vật / physics.
