# InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes

- **Tác giả:** Zesong Yang và cộng sự (Zhejiang University, ByteDance)
- **Venue:** ICCV 2025
- **Paper:** https://arxiv.org/abs/2507.08416
- **Project:** https://zju3dv.github.io/instascene
- **Code:** Có phần tách vật. https://github.com/zju3dv/instascene (`train.py`, `train_semantic.py`,
  `semantic_gui.py`). Mục cài "in-situ generation" (phần hoàn thiện vật) trong README **trống**, chưa rõ đã phát hành.
  Không thấy file license.

## Ý tưởng chính

- **Tách:** cảnh 2DGS, mask từ EntitySeg (CropFormer); theo dõi Gaussian qua các view ("spatial tracker") và học
  contrastive để gom Gaussian thành vật; gộp 2 mask là cùng vật khi đồng thuận giữa các view > 0.9.
- **Hoàn thiện (in-situ generation):** dùng mô hình 3D diffusion (MVDFusion) có điều kiện từ phần đã quan sát để
  **sinh ra phần vật không nhìn thấy**, cho vật đầy đủ hình học.

## Số liệu (theo paper / repo)

- LERF-Mask (figurines, waldo-kitchen, **teatime**): mIoU trung bình 85.6 (Gaussian Grouping 58.1, LangSplat 60.6).
- Hoàn thiện trên Replica-CAD: F1 0.716, cao nhất.
- Repo: train trên A100 40 GB; bước semantic 10,000 iter ~20 phút, dưới 8 GB.

## Liên quan bài toán của mình

- Paper hiếm hoi xử lý cả **đồng bộ ID + tách + hoàn thiện phần không thấy của vật** (đáy vật, mặt sau vật).
- Có kết quả trên teatime.

## Chạy trên L4

- Khó: nhiều phụ thuộc (cudf/cuml, CropFormer build riêng, detectron2, mmcv, StableNormal), python 3.9, torch 2.1 +
  CUDA 11.8. Bước semantic vừa L4; phần 2DGS / CropFormer / diffusion chưa có số bộ nhớ.

## Ưu điểm

- Bao phủ nhiều bước nhất trong các paper có code; độ chính xác tách vật cao trên chính teatime.

## Nhược điểm

- Paper tự nêu: không tách được vật động, trong suốt, phản chiếu mạnh (ly thuỷ tinh trà ở teatime có thể bị ảnh hưởng).
- Phần sinh hoàn thiện dựa diffusion: hình dạng sinh ra **không đảm bảo đúng thật**, và phần này chưa chắc có code.
- Không lấp mặt nền dưới vật; không có pipeline physics.
