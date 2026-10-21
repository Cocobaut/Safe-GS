# Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing

- **Tác giả:** Hongyu Shen, Junfeng Ni, Yixin Chen, Weishuo Li, Mingtao Pei, Siyuan Huang (BIGAI, BIT, Tsinghua)
- **Venue:** ICCV 2025
- **Paper:** https://arxiv.org/abs/2508.03227
- **Project:** https://trace-3d.github.io/
- **Code:** Có. https://github.com/trace-3d/Trace3D (Apache-2.0; `train.py`, `train_gaus.py`, `merge_patches.py`,
  `remove_ab_gaus.py`; torch 2.0.1 + CUDA 11.8, thử trên RTX 4090; cần SAM vit_h)

## Ý tưởng chính

- **Gaussian Instance Tracing (GIT):** thêm cho mỗi Gaussian một ma trận trọng số theo instance qua các view. Dùng
  "rasterize ngược" để biết Gaussian nào góp vào pixel nào, rồi gán xác suất theo nhãn của pixel. Nhanh ngang render.
- Dùng ma trận này để **phát hiện và sửa mask 2D không nhất quán**: gộp mask theo đa số phiếu của các Gaussian
  thuộc mask đó.
- **Density control theo GIT:** Gaussian mập mờ (thuộc nhiều vật) thì tách đôi (giảm nửa scale) hoặc bỏ, cho mép vật
  sắc trong cả 2D lẫn 3D.

## Số liệu (theo paper)

- 1 x RTX 4090. Không báo thời gian train / bộ nhớ.
- Dữ liệu: Replica (8 cảnh), NVOS; định tính BlendedMVS, ScanNet++, DL3DV-10K.

## Liên quan bài toán của mình

- Giải **bước 1 (đồng bộ ID)** gần với ý "đồ thị đồng xuất hiện": gộp mask theo phiếu Gaussian.
- Giải một phần **bước 2 (Gaussian vắt qua 2 vật)** bằng tách / bỏ Gaussian mập mờ.

## Chạy trên L4

- Được: paper dùng 4090 (24 GB, cùng cỡ VRAM với L4). L4 có băng thông bộ nhớ thấp hơn nên mình ước tính chậm
  hơn khoảng 1.5-2 lần.
- Phải train lại cảnh bằng 2DGS (không dùng trực tiếp cảnh CoMe), môi trường riêng torch 2.0.1 / CUDA 11.8.

## Ưu điểm

- Xử lý đúng điểm yếu của mình: ID mask đổi giữa các view.
- Có code, license thoáng (Apache-2.0).

## Nhược điểm (paper tự nêu ở Appendix A)

- Gán Gaussian theo render nên **lệch với bề mặt 3D thật**; chi tiết nhỏ (mô hình Gundam, máy ảnh) còn lỗi.
- **Tracing dừng khi opacity tích luỹ đạt 1**, nên Gaussian bị mặt đặc che bị bỏ sót: đúng chỗ "vỏ ma" của mình.
- Gaussian rác nhỏ thuộc vật A ở view này, vật B ở view khác thì không bị coi là mập mờ.
- Bước gộp mask không nhất quán "khá tốn thời gian".
