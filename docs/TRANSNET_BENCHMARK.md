# Benchmark TransNet V2 local

`scripts/benchmark_transnetv2.py` chỉ benchmark một video và giữ các frame
48x27 RGB trong RAM. Đây chưa phải runner streaming cho toàn bộ Batch 2.

Môi trường hiện có: Python `.venv`, torch 2.5.1+cu121, PyAV 18.1.0,
GTX 1070 8 GiB, i5-10400F 6 nhân/12 luồng. FP32, CUDA, batch 1; cửa sổ
100 frame, stride 50, lấy 50 dự đoán ở giữa theo cách inference upstream.

Nguồn cache (ngoài Git, `data/.cache/transnetv2/`):

- Architecture chính thức: https://github.com/soCzech/TransNetV2/blob/85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed/inference-pytorch/transnetv2_pytorch.py
  SHA256 `f7c1d437465579a8ec28a5add19853d2cb2755248ea4a4207678210a609428e1`.
- Weights PyTorch chuyển đổi do cộng đồng cung cấp (không phải file phát hành
  trực tiếp từ tác giả): https://huggingface.co/magnusdtd/TransNetV2/blob/c90da43a211d390a5fe2603f78f82419b7e6cd9c/transnetv2-pytorch-weights.pth
  SHA256 `eed5336d5d6a013c67f5863505a26e7e835053e64a9ce413d6b089ccba07bb53`.
  Nạp `weights_only=True` và `load_state_dict(strict=True)` thành công;
  chưa đối chiếu numerical parity với bản TensorFlow gốc.

Chạy lại với đường dẫn output mới:

```powershell
cd D:\bachkhoa\ai_challenge\my-repo
& .\.venv\Scripts\python.exe -u scripts/benchmark_transnetv2.py --output ..\data\batches\batch2\benchmarks\transnetv2-M05_V001-repeat
```

Script ghi `benchmark.json`, `shots.json`, `transition_probabilities.npy`, ảnh
và CSV trong output thử. Stable shot được xác định bằng các đoạn xác suất
transition <=0.5 liên tiếp; không chọn ảnh trong vùng transition. Mỗi shot
được chia thành các đoạn dài không quá 3 giây, lấy ảnh gần giữa mỗi đoạn.
Đây là chính sách chọn ảnh của benchmark, không phải chức năng mặc định của
model. Chưa đánh giá độ chính xác ranh giới/recall truy xuất.

Đo riêng setup/warmup, decode+resize CPU, inference GPU và decode lần hai+xuất
JPEG rộng tối đa 1280 quality90. `processing_seconds` là tổng ba bước xử lý,
không gồm tải model, setup và overhead ghi báo cáo. Video giữ nguyên FPS gốc;
không lấy mẫu 2 FPS trước model. CSV dùng chỉ số frame và PTS tương đối đầu video.

Trong lần thử 2026-09-23, máy đồng thời chạy `extract_keyframes_cpu.py
--threads 4` (PID 3096, do người dùng khởi chạy trước). Benchmark chia sẻ CPU
và ổ đĩa với tác vụ này; không coi đây là phép đo hiệu năng máy khi rảnh.

## Kết quả M05_V001 (2026-09-23)

25.645 frame, 25 FPS, 1.025,76 giây (17 phút 05,76 giây):

| Công đoạn | Giây |
| --- | ---: |
| Nạp model + CUDA warmup (ngoài tổng xử lý) | 6,21 |
| CPU decode và resize tất cả frame | 258,84 |
| GPU inference FP32 | 19,20 |
| CPU decode lần hai và xuất JPEG | 66,49 |
| Tổng ba bước xử lý | 344,53 |

264 stable shot; 458 JPEG, 71.941.098 byte. GPU peak allocated bởi PyTorch
167,48 MiB (không phải tổng VRAM của process, CUDA context hay desktop).
Tốc độ end-to-end khoảng 2,98 lần thời gian thực; inference riêng khoảng
1.335 frame nguồn/giây. Cả hai đã gồm tác động cạnh tranh tài nguyên lúc đo.

Report local: `data/batches/batch2/benchmarks/transnetv2-M05_V001-20260923/benchmark.json`.
Script CPU cũ trên cùng video: 68,594 giây, 1.005 ảnh, 159.510.418 byte;
khác chính sách chọn ảnh và điều kiện tải máy nên không dùng để kết luận
TransNet chậm hơn chính xác 5 lần hoặc cho chất lượng tốt hơn. Chưa benchmark
N/S, chưa ngoại suy toàn bộ 495 video.

## Chạy batch có resume

`scripts/run_transnetv2_batch.py` gọi pipeline đã benchmark cho từng video,
giữ output TransNet riêng tại `data/batches/batch2/transnetv2/`. Mỗi video
được ghi trong `partial/`, chỉ chuyển nguyên thư mục sang `completed/` sau khi
có report thành công. Chạy lại tự kiểm tra và bỏ qua video đã hoàn tất. Output
lỗi được giữ trong `failed/` để điều tra; script không tự xóa dữ liệu.

Chạy thử ba nhóm trước:

```powershell
cd D:\bachkhoa\ai_challenge\my-repo
& .\.venv\Scripts\python.exe -u .\scripts\run_transnetv2_batch.py `
  --video-id M05_V001 --video-id N001-V001 --video-id S01-V001
```

Sau khi xem ảnh và dung lượng, chạy toàn bộ:

```powershell
& .\.venv\Scripts\python.exe -u .\scripts\run_transnetv2_batch.py
```

Nhấn Ctrl+C một lần để dừng. Video đang chạy giữ trong `partial/`; lần sau tạo
partial mới và xử lý lại video đó, còn video trong `completed/` được bỏ qua.
Các partial cũ không được tự dọn. Không chạy hai batch cùng output; có OS lock.
Runner nạp model lại cho mỗi video (khoảng 6 giây trong mẫu), đơn giản và cô
lập lỗi nhưng thêm overhead. Pipeline hiện buffer toàn bộ frame 48x27 của một
video trong RAM và decode video hai lượt; theo dõi RAM với video S01 dài.

Trước khi chạy, xác nhận cache tồn tại:

```powershell
Get-FileHash ..\data\.cache\transnetv2\transnetv2_pytorch.py
Get-FileHash ..\data\.cache\transnetv2\transnetv2-pytorch-weights.pth
```

Hash mong đợi được ghi ở phần đầu tài liệu này. Nếu cache mất, không chạy với
file ngẫu nhiên cùng tên; tải lại đúng revision đã pin và đối chiếu SHA256.
