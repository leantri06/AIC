# Bố trí workspace và tiếp nhận batch2

Gốc máy hiện tại: `D:\bachkhoa\ai_challenge`. Git repository: `my-repo`, remote `https://github.com/leantri06/AIC.git`.

Nguồn danh sách batch2: `https://docs.google.com/spreadsheets/d/1jXR0oUDc0zq1BFQLrcHrS1Dhl7seJgIMFL-1dlTxiXQ/edit?gid=0#gid=0`.

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

## Giải nén batch2

Script `scripts/extract_batch2.py` chỉ đọc các file `.zip` đã tải xong và bỏ qua
`.part`. Nó chuẩn hóa `videos/` thành `Videos/`, `media-info/` thành `Metadata/`,
chặn path traversal, kiểm tra đủ file/kích thước và lưu tiến độ vào
`data/batches/batch2/extracted/.extraction_state.json`.

Do ổ D không đủ giữ đồng thời toàn bộ ZIP và bản giải nén, chạy chế độ xóa ZIP
chỉ sau khi archive tương ứng đã được xác minh:

```powershell
& 'C:\Users\ADMIN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  .\my-repo\scripts\extract_batch2.py --delete-archives-after-verify
```

Lệnh có thể chạy lại sau khi ngắt: archive đã xác minh được bỏ qua; file đích
đang ghi dở sẽ được tạo lại từ ZIP nguồn. Không chạy hai tiến trình giải nén cùng lúc.

## Trạng thái bố trí Batch 2 (2026-09-23)

Batch 2 đã nằm gọn tại `data/batches/batch2/extracted/`, tách biệt hoàn toàn
với Batch 1. Tất cả video nằm phẳng trong `Videos/`; tên và container nguồn
được giữ nguyên để tránh phá tham chiếu: `Mxx`/`S01` là `.mp4`, `Nxxx` là
`.mov`, và nguồn dùng cả dấu gạch dưới lẫn gạch nối. Audit quét
`Videos/*.{mp4,mov}`; `Metadata/` cũng giữ nguyên cấu trúc nguồn.

Kiểm kê đã xác nhận 495 video, 228.473.998.977 byte và 605 JSON metadata,
không có ID video trùng hay tên sai định dạng. Chạy lại sau mỗi đợt bổ sung:

```powershell
python .\my-repo\scripts\audit_batch2.py --write
```

Lệnh này chỉ tạo/cập nhật `inventory.json` cạnh dữ liệu, không sửa video hay
metadata. Batch 2 vẫn chưa có keyframe, ASR, embedding hoặc index nên chưa
được ghép vào `data/combined/` và app vẫn dùng Batch 1.

Hướng dẫn tự trích keyframe bằng CPU (BTC không cấp keyframe Batch 2):
[CPU_KEYFRAMES.md](CPU_KEYFRAMES.md). Script đã chuẩn bị, chưa chạy toàn bộ video.
