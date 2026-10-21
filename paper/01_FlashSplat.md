# FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally

- **Tác giả:** Shen, Yang, Wang (National University of Singapore)
- **Venue:** ECCV 2024
- **Paper:** https://arxiv.org/abs/2409.08270
- **Code:** Có. https://github.com/florinshen/FlashSplat (có `train.py`, `render.py`, `gaussian_renderer/`,
  `submodules/`; phần Installation trong README để trống)

## Ý tưởng chính

- Với cảnh 3DGS đã train, ảnh mask render ra là **hàm tuyến tính theo nhãn của từng Gaussian**. Vì vậy bài toán gán
  nhãn cho Gaussian từ mask 2D giải được **tối ưu toàn cục trong 1 bước** (quy hoạch tuyến tính dạng đóng), không cần
  lặp gradient.
- Bước chính: duyệt qua mọi view, cộng dồn **đóng góp (alpha x transmittance) của mỗi Gaussian vào từng nhãn**, rồi
  lấy argmax. Viết bằng CUDA dùng lại pipeline rasterize theo tile.
- Có **background bias gamma** trong [-1, 1] để chống mask nhiễu. Paper dùng gamma = -0.4 khi xoá vật.
- Nhiều vật: tách thành chuỗi bài toán 2 lớp (vật này với phần còn lại), cộng dồn một lần rồi argmax.

## Số liệu (theo paper)

- A6000: xây tập đóng góp ~26 s, gán nhãn ~1 ms. Bộ nhớ đỉnh 8 GB (SAGA 15 GB, Gaussian Grouping 34 GB).
- NVOS: 91.8% mIoU.

## Liên quan bài toán của mình

- Thay đúng **bước 1 + 3** trong thiết kế: vote theo đóng góp khi render thay cho vote theo tâm Gaussian. Tự xử lý
  chuyện bị che (Gaussian phía sau có transmittance gần 0).
- Background bias giúp với mask SAM2 bị hở mép.

## Chạy trên L4

- Dễ: không train lại 3DGS, 8 GB VRAM. Mình ước tính vài chục giây đến 1-2 phút cho 177 view.
- **Vướng:** cần kernel CUDA của FlashSplat chạy trên rasterizer của nó. Cảnh của mình dùng rasterizer CoMe (13 kênh,
  sort hierarchical) nên phải port kernel, hoặc dùng mẹo gradient của paper 02 để lấy cùng đại lượng.

## Ưu điểm

- Nhanh, không train, tối ưu toàn cục cho mask đã cho, ít bộ nhớ.
- Có cơ chế chống mask nhiễu.

## Nhược điểm

- Không sửa được lỗi **ID không nhất quán** giữa các view: nhận mask "như đã cho". Gấu đổi ID 2 sang 9 thì vẫn ra 2 vật.
- Gaussian **không view nào thấy** (vỏ ma) có đóng góp 0 nên không gán được nhãn.
- Gaussian có thể bị chia cho nhiều vật (paper: ~20% Gaussian ở cảnh Counter thuộc hơn 2 vật).
- Paper tự nhận: phải duyệt mọi pixel mask, khó mở rộng cho cảnh lớn / độ phân giải cao.
