# Seed2GS: Camera-Free, Training-Free Object Extraction from 3D Gaussian Scenes via a Single Reference-View Grounding

- **Tác giả:** Zongjian Ding và cộng sự (10 tác giả)
- **Venue:** arXiv 2608.11928 (12/08/2026), chưa có hội nghị
- **Code:** **Không** (trang arXiv không có link)

## Ý tưởng chính

- Tách 1 vật từ cảnh 3DGS **đã đóng băng**, không cần camera gốc, không train theo cảnh.
- QD-SAM3 chọn 1 mask tham chiếu tin cậy từ các ứng viên open-vocabulary (prompt bằng chữ).
- "Seed lifting" + vòng camera ảo thích ứng theo độ nhìn thấy để quan sát vật từ nhiều phía; tracking lan seed.
- Chỉ học 1 logit foreground tạm thời cho mỗi Gaussian.

## Số liệu (theo abstract)

- LERF-MASK 92.1% mIoU, 3D-OVS 95.7%. Độ trễ tính toán 9.3 s trên LERF-MASK (không ghi GPU).

## Liên quan bài toán của mình

- Bước 2 (tách vật) theo kiểu prompt chữ ("cup", "teddy bear"): tránh luôn chuyện ID mask không nhất quán.
- Camera ảo quanh vật có thể giúp gán nhãn Gaussian ở góc mà ảnh gốc không có.

## Chạy trên L4

- Nhanh theo paper (9.3 s) nhưng **không có code**, không kiểm chứng được.

## Ưu điểm

- Mới nhất, nhanh, không train, chạy trên cảnh có sẵn (kể cả cảnh CoMe).

## Nhược điểm

- Không code. Mỗi lần 1 vật; Gaussian bên trong vật vẫn là vấn đề mở; chưa qua bình duyệt.
