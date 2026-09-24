# Thử embedding keyframe CPU batch 2

Pipeline thử riêng cho output của `scripts/extract_keyframes_cpu.py`.
Không dùng reader này cho keyframe BTC: ảnh CPU có tên là `frame_idx`,
trong khi ảnh BTC có thể dùng số thứ tự `n`.

`ingestion/cpu_keyframes.py` đối chiếu owner/report, kích thước và thời gian
sửa video, số ảnh, CSV, thứ tự frame và timestamp trước khi tạo record.
Nó hỗ trợ `.mov` và `.mp4`, giữ nguyên video ID, lưu cả `pts_time` tương đối
và `source_pts_time` gốc. Không tự suy timestamp từ FPS hoặc đổi sang định
dạng nộp bài. Quy ước timestamp nộp cho video VFR cần đối chiếu BTC riêng.
Reader này không thay thế lệnh build-manifest tổng quát hiện có.

Chạy từ repo, dùng một thư mục output mới cho mỗi lần thử:

```powershell
& .\.venv\Scripts\python.exe scripts/benchmark_batch2_embeddings.py `
  --output ..\data\batches\batch2\benchmarks\siglip2-trial-new `
  --per-video 16 --batch-size 2
```

Script lấy 16 ảnh phân bố đều theo thứ tự keyframe của mỗi video M05_V001,
N001-V001, S01-V001; tính lại 6 vector rải đều trong index batch1 để đối chiếu.
Nó chỉ dùng checkpoint SigLIP2 cache local, chạy FP32, chuẩn hóa L2; loại
text tower khỏi model của tiến trình benchmark để dành GPU cho image encoder.
Preprocess lấy mean/std/interpolation/resize_mode từ cấu hình OpenCLIP webli.
Không tải dữ liệu hay model. Output gồm manifest.jsonl, features.npy và
report.json với SHA256 checkpoint, phiên bản thư viện, tốc độ và VRAM.
Script từ chối output đã tồn tại; đây là thử nghiệm có giới hạn, chưa phải
pipeline chạy toàn bộ có resume. Không ghi đè index hoặc đổi app runtime.

## Ý nghĩa “transformer = slow”

Không có tham số chung tên `transformer = slow`. Trong ngữ cảnh embedding
ảnh, có thể người nói muốn dùng image processor PIL/NumPy (`use_fast=False`
ở API cũ). Transformers 5.14.1 đang cài chuyển cờ này thành `backend="pil"`
và cảnh báo deprecation. Đây là lựa chọn tiền xử lý ảnh để giữ tương thích
số học, không phải chuyển model sang CPU hoặc bảo đảm độ chính xác cao hơn.
`use_fast` cũng có thể nói về tokenizer; cần xem câu lệnh gốc để phân biệt.

Thử nghiệm này dùng OpenCLIP với PIL RGB, resize bicubic 378x378, mean/std
0.5 và chuẩn hóa L2, không dùng AutoImageProcessor nên không gắn thêm cờ
`use_fast=False`. So vector với batch1 là phép kiểm chứng trực tiếp hơn.

Tham khảo: https://huggingface.co/docs/transformers/main/en/main_classes/image_processor

## Kết quả 2026-09-24

- Máy: GTX 1070 8 GiB; torch 2.5.1+cu121, OpenCLIP 3.3.0,
  Transformers 5.14.1.
- Checkpoint: timm/ViT-SO400M-14-SigLIP2-378, snapshot
  `93e0e6ad1063507a7777d0b570292a29a3d94c5f`.
- SHA256: `7ed1f8fb32da2d18fd71028ca2c6d61ea5fec9a6a8f99d1d9159b62c0c1b7a9f`.
- Batch size 2: 48x1152 vector hữu hạn, norm xấp xỉ 1; 16,579 giây,
  2,895 ảnh/giây; GPU peak allocated 1,700 GiB, reserved 1,844 GiB.
- Batch size 8: 16,289 giây, 2,947 ảnh/giây; GPU peak allocated 1,924 GiB,
  reserved 2,297 GiB. Chênh lệch tốc độ nhỏ trên mẫu này, chưa đủ kết luận
  batch lớn hơn tốt hơn. Ngoại suy 670.151 ảnh là khoảng 63–64 giờ, chưa
  gồm overhead của pipeline toàn bộ và không phải cam kết thời gian.
- Audit reader mới trên toàn bộ 495 video/670.151 ảnh đạt. Kiểm tra mapping,
  sự tồn tại/kích thước ảnh và danh tính video; không giải mã lại toàn bộ JPEG.
- 24 test đạt cho CPU manifest, manifest cũ, submission, official index và
  video progress. Không thay đổi cách app chọn index.
- Sáu vector batch1 có cosine 0,999948–0,999999 với vector tính lại. Đạt
  ngưỡng mẫu 0,999, nhưng không chứng minh toàn bộ index đồng nhất hay
  chất lượng tìm kiếm. Thời gian đo đã gồm mở ảnh, preprocess và inference,
  không gồm load model, audit và ghi file cuối.

Trước khi chạy toàn bộ cần chọn bộ keyframe CPU hay bộ BTC mới phát hành,
chuẩn bị resume theo video và metadata provenance, kiểm tra thêm retrieval,
rồi mới ghép batch1+batch2. Giữ nguyên bộ cũ để rollback.
