# Bố trí workspace và tiếp nhận batch2

Gốc máy hiện tại: `D:\bachkhoa\ai_challenge`. Git repository: `my-repo`, remote `https://github.com/leantri06/AIC.git`.

| Đường dẫn từ workspace | Mục đích |
| --- | --- |
| `my-repo/` | Mã nguồn, test, tài liệu, MEMORY và cấu hình inventory |
| `data/raw_data/` | ZIP dữ liệu cũ (batch1 theo cách gọi trong workspace) |
| `data/extracted/` | Keyframes, Videos, map-keyframes, Metadata, Objects, ASR, CLIP cũ |
| `data/processed/` | Manifest, vector và BM25 hiện tại; giữ nguyên đường dẫn |
| `data/.cache/` | Model và cache dùng chung; giữ để không phải tải lại |
| `data/batches/batch2/raw/` | Đặt ZIP batch2 mới tại đây |
| `data/batches/batch2/extracted/` | Giải nén batch2 riêng, không ghi đè batch1 |
| `data/batches/batch2/processed/` | Index riêng của batch2 |
| `data/combined/` | Dành cho index ghép đã kiểm chứng; hiện chưa có |
| `archive/submissions/2026-09-21/` | Các bản nộp cũ ngoài repo, giữ nguyên nội dung |
| `docs/reference/` | Tài liệu cuộc thi ở máy local |

`config/datasets.json` là danh mục phục vụ quản lý, **chưa phải cấu hình runtime**. App hiện vẫn dùng `data/processed` và `data/extracted`. Không đổi Master Root sang batch2 hoặc combined khi chưa có index hợp lệ.

## Những thứ cần giữ

Giữ `.venv`, `uv.lock`, source/tests, model cache và tất cả dữ liệu gốc/đã xử lý. Chưa xác minh index SigLIP cũ có thể thay thế nhau nên không xóa theo tên hay kích thước giống nhau. Chưa đối chiếu đầy đủ nội dung ZIP với file đã giải nén nên không xóa ZIP. `my-repo/submission/` đang được Git quản lý, giữ nguyên.

Script `data/unzip_script.py` cũ dùng đường dẫn tương đối theo thư mục chạy. Không chạy nguyên mặc định với batch2. Khi nhập batch mới cần kiểm tra đường dẫn trong ZIP, dung lượng và va chạm trước khi giải nén vào thư mục riêng.

## Trình tự khi batch2 đến

1. Kiểm kê ZIP, checksum và cấu trúc; xác nhận batch2 thực tế và các video ID. Không suy ra batch dữ liệu từ tên query p1/p2.
2. Giải nén vào thư mục batch2 riêng; giữ nguyên batch1 và ghi nguồn gốc dữ liệu.
3. Tạo manifest và embedding batch2 bằng cùng model/checkpoint, dimension và normalization với batch1. Không trộn CLIP, SigLIP và SigLIP2 dù dimension bằng nhau.
4. Phát hiện trùng `(video_id, frame_id)` và video ID giữa các batch. Nếu có, cần quy tắc namespace và tương thích định dạng nộp bài trước khi ghép.
5. Tạo index mới trong `data/combined/`: cập nhật `vector_id`, giữ đúng thứ tự manifest và các ma trận visual/ASR; kiểm tra đường dẫn ảnh/video của cả hai batch. Không nối `.npy` ở mức byte.
6. Tạo lại BM25 từ manifest ghép. Kiểm tra số dòng, dimension, giá trị hữu hạn, ID, truy vấn mẫu trên cả hai batch và thời gian/RAM.
7. Chỉ chuyển app sang index ghép khi các kiểm tra đạt; giữ index cũ để quay lại. Cần triển khai/kiểm chứng hỗ trợ đường dẫn runtime ở bước đó.

Việc chuẩn bị thư mục không có nghĩa pipeline ghép batch đã được triển khai hay kiểm thử.
