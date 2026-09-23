# Trích keyframe Batch 2 bằng CPU trên Windows

Script: `scripts/extract_keyframes_cpu.py`. Chạy PowerShell tại
`D:\bachkhoa\ai_challenge\my-repo`. Dùng Python trong `.venv` hiện có,
đã kiểm tra có PyAV, NumPy, Pillow; không cần CUDA hay tải model.

```powershell
Set-Location D:\bachkhoa\ai_challenge\my-repo
& .\.venv\Scripts\python.exe -c "import av, numpy, PIL; print('Ready')"
```

Nếu dựng máy mới thiếu thư viện, cài vào môi trường riêng bằng
`uv pip install --python .\.venv\Scripts\python.exe av numpy pillow`.

## Chạy thử một video

```powershell
& .\.venv\Scripts\python.exe .\scripts\extract_keyframes_cpu.py --video-id M05_V001 --threads 4
```

Các đường dẫn mặc định được tính từ vị trí script. Kết quả tại
`D:\bachkhoa\ai_challenge\data\batches\batch2\extracted`:

- `Keyframes/M05_V001/000000000.jpg`: tên ảnh là chỉ số frame gốc, bắt đầu 0.
- `map-keyframes/M05_V001.csv`: n, pts_time, fps, frame_idx, source_pts_time.
- `keyframe-reports/M05_V001.json`: số frame, thời lượng, thời gian chạy, bytes,
  cấu hình và nhận dạng video nguồn.

`pts_time` lấy từ PTS decoder, trừ PTS của frame đầu để bắt đầu ở 0;
`source_pts_time` giữ timestamp nguồn. `fps` chỉ là average rate tham khảo.
Không lấy tên ảnh hay ordinal `n` chia FPS để thay thế timestamp.

Mở thư mục ảnh để xem chất lượng. Mặc định khảo sát 2 frame/giây, giữ ảnh khi
chênh lệch RGB trung bình trên thumbnail 160x90 >=12 hoặc khoảng cách >=4 giây;
luôn giữ frame đầu/cuối. Mốc 4 giây được kiểm tra ở các điểm lấy mẫu nên có thể
vượt nhẹ. Ảnh JPEG quality 90, rộng tối đa 1280, giữ tỷ lệ hình.
Đây là heuristic ứng viên, không phải bộ phát hiện shot chính xác: có thể bỏ
sót cảnh ngắn dưới 0,5 giây hoặc giữ nhiều ảnh khi camera rung/chữ chạy.
Video vẫn phải được giải mã liên tục; 2 FPS giảm tính toán và ảnh ghi, không
có nghĩa CPU chỉ giải mã 2 frame/giây.

## Đo trên ba loại video

```powershell
& .\.venv\Scripts\python.exe .\scripts\extract_keyframes_cpu.py --video-id M05_V001 --video-id N001-V001 --video-id S01-V001 --threads 4
```

Video M đã hoàn tất sẽ được kiểm tra và bỏ qua. Xem `elapsed_seconds`,
`video_seconds`, `image_bytes`, `kept_frames` trong từng report. Chưa có benchmark
toàn bộ 495 video; các ước lượng ngày/giờ trước đây không phải kết quả đo.
Không ngoại suy theo GB vì codec, độ phân giải và bitrate khác nhau; dùng tỷ lệ
giây xử lý / giây video riêng cho từng nhóm sau khi đo đủ mẫu.

## Chạy toàn bộ

```powershell
& .\.venv\Scripts\python.exe .\scripts\extract_keyframes_cpu.py --threads 4
```

Script xử lý tuần tự, mỗi decoder tối đa 4 luồng. Nếu máy dùng để làm việc,
giảm `--threads 2`. Tham số này không làm thay đổi kết quả lựa chọn ảnh.
Không mở nhiều bản script trên cùng output (script có khóa hệ điều hành).

Giữ máy cắm điện và tắt Sleep khi đang chạy; có thể tắt màn hình. Trong Task
Manager, theo dõi CPU và Disk; nếu máy phản hồi chậm giảm số luồng ở lần chạy
tiếp theo. Script dừng ghi khi dung lượng trống dưới 30 GiB.

## Dừng và tiếp tục

Nhấn Ctrl+C một lần trong PowerShell, đợi script thoát, sau đó chạy lại cùng
lệnh. Video hoàn tất được bỏ qua; video đang dở chạy lại từ đầu và ghi lại các
ảnh do chính script tạo với cùng nguồn/cấu hình. Resume không lưu vị trí frame
giữa video. Marker hoàn tất chỉ được ghi sau decode, kiểm tra JPEG và lưu CSV.
Nếu mất điện, video dở cũng được xử lý lại. Trước khi làm lại, script xác minh
`.owner.json` khớp chính xác nguồn/cấu hình rồi xóa riêng `.jpg` dở do chính
script tạo trong thư mục video đó; thư mục thiếu marker hoặc marker lệch sẽ bị
từ chối thay vì bị sửa.

`ERROR` nghĩa video đó chưa hoàn tất; xem `keyframe-last-run.json`. Sau khi xử
lý nguyên nhân, chạy lại. Không coi chỉ có thư mục ảnh là đã xong.

Một số `.mov` Batch 2 khai báo số frame trong container nhiều hơn số frame
decoder trả về khi đến EOS; đây được ghi thành `declared_frame_delta` và cảnh
báo, không còn làm video thất bại. Nếu PTS nguồn đi lùi, script giữ PTS thô ở
`source_pts_time` và hiệu chỉnh timeline tương đối tăng đều; số lần hiệu chỉnh
nằm trong `timeline_corrections`.

## Thử cấu hình khác

Phải dùng output khác để không trộn ảnh từ nhiều cấu hình:

```powershell
& .\.venv\Scripts\python.exe .\scripts\extract_keyframes_cpu.py --video-id M05_V001 --sample-fps 3 --max-gap 2 --pixel-threshold 9 --output-root ..\data\batches\batch2\keyframes-trial
```

Giảm pixel-threshold/tăng sample-fps/giảm max-gap thường giữ nhiều ảnh hơn.
Có thể đặt `--pixel-threshold 0` để giữ mọi mẫu (ít nhất 2 FPS mặc định), đổi lại
tốn đĩa và thời gian embedding. Chỉ chọn cấu hình sau khi xem ảnh thử.

## Bước tiếp theo

Ảnh và CSV chưa là index tìm kiếm. Encode bằng cùng SigLIP2 checkpoint và
preprocessing với Batch 1 rồi xây manifest/embedding riêng. Kiểm tra map theo
frame_idx, thứ tự ảnh/vector, khả năng resolve .mov và ID có dấu '-' trước khi
ghép; CSV cùng schema không chứng minh app hỗ trợ Batch 2 hoàn chỉnh.
