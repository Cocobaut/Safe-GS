# Consistent Scene Understanding in 3D Gaussian Splatting via Multi-Cue Mask Refinement

- **Tác giả:** Hyunjoon Park, Donghyeon Cho
- **Venue:** ICPR 2026 (arXiv 2607.01708, 02/07/2026)
- **Code:** **Không** (trang arXiv không có link)

## Ý tưởng chính

- Sửa mask 2D **trước** khi đưa lên 3D, 3 bước:
  1. Trích cue ngữ nghĩa, hình học (depth), cấu trúc (edge) từ ảnh.
  2. **Gộp mask vụn** theo điểm gộp tính từ 3 cue đó.
  3. **Ghép mask giữa các view** để gán ID toàn cục nhất quán, rồi mới học feature field 3DGS.

## Số liệu

- Abstract không ghi dataset, GPU, thời gian. Chỉ nói cải thiện độ nhất quán giữa view so với baseline.

## Liên quan bài toán của mình

- Nhắm **đúng lỗi gấu bị tách 2 / 9 / 6 và bàn 1 + 65**: gộp mask vụn + đồng bộ ID giữa view (bước 1).

## Chạy trên L4

- Không có thông tin và không có code.

## Ưu điểm

- Mới (07/2026), đã được nhận ở ICPR, xử lý ở 2D nên có thể dùng chung với bất kỳ phương pháp lift nào.

## Nhược điểm

- Không code, abstract ít chi tiết; một trang tóm tắt bên thứ ba nói ghép mask dựa ngưỡng độ tương đồng đặc trưng và
  phân cụm đồ thị (chưa kiểm chứng trong paper). Venue nhỏ hơn CVPR / ICCV.
