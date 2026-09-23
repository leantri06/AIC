# Memory dự án — đọc đầu mỗi phiên

Cập nhật: 2026-09-23 (Asia/Saigon).

## Thỏa thuận với người dùng

- Giao tiếp tiếng Việt; đọc và hiểu dự án trước khi sửa.
- Ghi lại việc đã hoàn thành sau mỗi đầu việc để phiên sau tiếp tục được.
- Commit và push thay đổi mã nguồn/tài liệu lên GitHub; không đưa data/model/cache lên Git.
- Ưu tiên hiện tại: dọn workspace và chuẩn bị batch2, luôn giữ dữ liệu cũ để dùng kết hợp.
- Memory hỗ trợ bàn giao khi mở phiên tiếp theo; chưa có lịch tự chạy vào ngày hôm sau.

## Bản đồ dự án đã đọc

- Workspace: `D:\bachkhoa\ai_challenge`; repo thực tế: `my-repo/`.
- Remote: `https://github.com/leantri06/AIC.git`, branch `main`; commit ban đầu `5e7efbf`. Working tree sạch trước khi làm.
- Python >=3.11, package `aic2026`, quản lý dependencies bằng `pyproject.toml`/`uv.lock`; `.venv` hiện có Python 3.11.15, NumPy 2.4.6.
- PEGASUS: tìm kiếm video cho KIS, Q&A, TRAKE. Streamlit `src/aic2026/app/ui.py`; API `app/api.py`; CLI `cli.py`; ingestion tạo manifest/index; retrieval kết hợp visual/ASR/BM25; reranking, QA, temporal và xuất submission có module/test riêng.
- Entry point local: `run_app.bat`. Master Root cũ `D:\bachkhoa\ai_challenge\data`; app tự dò `processed` và `extracted`.
- Đã đọc README, launcher, dependency config, các phần ingestion/CLI/UI liên quan đường dẫn/index. Chưa audit toàn bộ thuật toán hay chạy toàn bộ app/model.
- Manifest có đường dẫn cũ `D:\ai_challenge\...`; UI có fallback sang workspace hiện tại. Không tự sửa toàn bộ manifest khi dọn.

## Hoàn thành 2026-09-21

1. Khảo sát mã nguồn, data, Git và đường dẫn runtime. Giữ nguyên source/runtime, `.venv`, data cũ và cache model.
2. Gom `submission/` (33 file), `submission_2/` (30 file) và `submission.zip` ngoài repo vào `archive/submissions/2026-09-21/`; giữ tên và cấu trúc riêng vì nội dung khác nhau. `my-repo/submission` vẫn giữ nguyên.
3. Chuyển PDF cuộc thi vào `docs/reference/` ngoài repo. Xác minh SHA-256 toàn bộ 65 file trước/sau các thao tác di chuyển, không mất/thay đổi nội dung. Nhật ký: `docs/cleanup-2026-09-21.json` trong repo.
4. Tạo `data/batches/batch2/{raw,extracted,processed}` và `data/combined` ngoài Git; hiện trống. Data cũ không di chuyển để không phá đường dẫn.
5. Tạo `AGENTS.md`, memory này, `docs/WORKSPACE.md` và `config/datasets.json`. AGENTS ở gốc workspace trỏ về repo; bản hướng dẫn chính trong repo được version control. Thư mục trống và di chuyển local không được Git tái hiện; dùng tài liệu bố trí khi dựng máy mới.
6. Kiểm tra dữ liệu cũ bằng NumPy mmap + đọc manifest: 177.321 records, 873 video, vector_id tuần tự, không thiếu ảnh nào khi resolve về `data/extracted/Keyframes`.

| Ma trận | Shape | dtype |
| --- | --- | --- |
| siglip2_features | 177321 × 1152 | float32 |
| siglip_features | 177321 × 1152 | float32 |
| derived_features | 177321 × 768 | float32 |
| asr_dense_features | 177321 × 1024 | float16 |

## Phát hiện cần xử lý trước khi ghép batch

- Có **614 dòng dư do trùng khóa `(video_id, frame_id)`** trong manifest SigLIP2 hiện tại. Chưa xác định nguyên nhân, chưa loại bỏ. Cần đối chiếu map-keyframes/ordinal/frame_idx trước khi dedup; không xóa theo khóa vì có thể mất ảnh và lệch vector.
- 32 ZIP gốc tổng 114.605.488.279 byte (~106,7 GiB). Giữ vì chưa kiểm chứng từng entry với dữ liệu giải nén. Đợt này tổ chức lại thư mục, chưa giải phóng dung lượng đáng kể.
- Hai ma trận SigLIP có cùng shape không chứng minh chúng trùng nội dung/model; giữ cả hai.
- `datasets.json` chỉ là inventory, chưa được runtime đọc. Batch2 chưa đến; chưa triển khai/kiểm thử pipeline ghép batch1+batch2.
- Script `data/unzip_script.py` cũ lấy đường dẫn theo cwd và giải nén trực tiếp; không chạy mặc định để nhập batch2.

## Việc tiếp theo

1. Khi batch2 đến: đặt ZIP vào `data/batches/batch2/raw/`, kiểm kê cấu trúc/checksum/dung lượng và video ID.
2. Điều tra 614 khóa trùng batch1; xác nhận ý nghĩa frame_id theo map-keyframes và định dạng nộp bài.
3. Theo quy trình `docs/WORKSPACE.md`: xử lý batch2 riêng, đồng nhất model, kiểm tra va chạm, tạo index kết hợp vào `data/combined`, tái tạo BM25 và xác minh visual/ASR cùng thứ tự.
4. Kiểm tra truy vấn mẫu cả hai batch rồi mới chuyển app; giữ bộ cũ để rollback.
5. Nếu cần thu hồi dung lượng ZIP/cache, đối chiếu dữ liệu trước khi xóa.

## Tiếp nhận và tải Batch 2 (2026-09-21)

1. Khảo sát liên kết Google Sheet và server BTC (`https://aic-data.ledo.io.vn/`):
   - Tìm thấy 11 file ZIP video (tổng 145.97 GB) và 1 file `media-info-aic25-b2.zip` (~753 KB).
   - Phát hiện Google Sheet bị trùng dòng N011-N020 và sót `Video_N031-N040.zip`, đã probe server và xác nhận file này tồn tại (10.17 GB).
   - 9 phần Video tin tức vẫn đang ở trạng thái "Đang upload" trên sheet, chưa có link trên server.
2. Viết script tải resumable `scripts/download_batch2.py` hỗ trợ:
   - Tiếp tục tải khi bị ngắt qua HTTP Range header (`bytes=...-`).
   - Cập nhật tiến độ live vào `data/batches/batch2/raw/progress.txt` và `download_status.json`.
   - Kiểm tra tính toàn vẹn ZIP và đối chiếu kích thước sau khi tải xong từng file.
   - Kiểm tra ngưỡng an toàn dung lượng ổ đĩa (tối thiểu 30 GB trống).
   - Hỗ trợ cờ `--status` để xem dashboard trực tiếp.
3. Khởi chạy tiến trình tải ngầm 12 file vào `data/batches/batch2/raw/`. File `media-info-aic25-b2.zip` đã hoàn thành và kiểm tra hợp lệ. File `Video_N001-N010.zip` đang tải với tốc độ ~18-19 MB/s.
4. Cập nhật `config/datasets.json` phản ánh trạng thái tải của Batch 2.

## Việc tiếp theo

1. Giám sát quá trình tải toàn bộ 12 file Batch 2 cho đến khi hoàn tất.
2. Theo dõi khi BTC cập nhật 9 phần video tin tức còn lại.
3. Khi tải xong ZIP, kiểm tra cấu trúc bên trong trước khi giải nén vào `data/batches/batch2/extracted/`.
4. Tạo index riêng `data/batches/batch2/processed/` và ghép vào `data/combined/` theo quy trình trong `docs/WORKSPACE.md`.

## Cập nhật Batch 2 ngày 2026-09-22

1. Người dùng cung cấp Google Sheet chính thức của batch2:
   `https://docs.google.com/spreadsheets/d/1jXR0oUDc0zq1BFQLrcHrS1Dhl7seJgIMFL-1dlTxiXQ/edit?gid=0#gid=0`.
2. Đã đọc tab `batch2` và đối chiếu với thư mục raw. 12 file cũ đã tải đủ và hợp lệ. Sheet mới bổ sung 6 file video tin tức `Videos_M05.zip` đến `Videos_M10.zip`; các dòng tương ứng M01-M04 vẫn chưa có filename/link.
3. Kích thước được xác nhận bằng HTTP Range trực tiếp từ server:
   - M05: 10.161.147.478 byte
   - M06: 10.889.052.849 byte
   - M07: 11.840.337.561 byte
   - M08: 12.632.959.226 byte
   - M09: 13.294.614.361 byte
   - M10: 12.918.192.698 byte
   - Tổng mới: 71.736.304.173 byte (~66,8 GiB); tổng inventory hiện tại: 18 file, 228.474.855.072 byte.
4. Đã cập nhật `scripts/download_batch2.py` và `config/datasets.json`; sửa bộ đếm để các file có sẵn vẫn được tính là hoàn thành và bỏ hard-code thông báo 12 file.
5. Đã chạy kiểm tra cú pháp Python. Downloader nền PID 9556 bắt đầu lúc 19:46:35, đang tải `Videos_M05.zip` vào `data/batches/batch2/raw/`; cơ chế resume và kiểm tra ZIP vẫn hoạt động. Dữ liệu batch1 không thay đổi.

### Bước tiếp theo sau phiên này

1. Kiểm tra `python scripts/download_batch2.py --status`. Nếu tiến trình bị ngắt, chạy lại script để resume; không xóa file `.part`.
2. Sau khi đủ 18/18, chạy lại downloader bản mới một lần để xác nhận trạng thái và ZIP, rồi cập nhật memory thành hoàn tất.
3. Theo dõi sheet cho M01-M04 và các gói Keyframes/map/objects/CLIP; chưa thể tạo index batch2 chỉ từ video và metadata hiện có.

## Giải nén và sắp xếp Batch 2 ngày 2026-09-22

1. Đã thêm `scripts/extract_batch2.py`: chỉ nhận ZIP hoàn chỉnh, chặn đường dẫn
   không an toàn, chuẩn hóa thư mục thành `Videos/` và `Metadata/`, xác minh số
   file/kích thước, lưu state để tiếp tục, và chỉ xóa ZIP sau xác minh khi dùng
   `--delete-archives-after-verify`.
2. Người dùng cho phép xóa ZIP sau khi giải nén. Đã giải nén, xác minh và xóa:
   `media-info-aic25-b2.zip`, `Video_N001-N010.zip`, `Video_N011-N020.zip`,
   `Video_N021-N030.zip`, `Video_N031-N040.zip`, `Video_N041-N050.zip`.
3. Kết quả đã xác minh: 605 JSON trong `Metadata/`; 153 video từ 5 ZIP video
   (49,02 GiB). Khi người dùng yêu cầu dừng, archive
   `Video_N051-N060.zip` đang xử lý và có thêm 15 file đích, nâng tổng tạm thời
   lên 168 video / 53,59 GiB. Tiến trình đã được dừng; ZIP nguồn vẫn còn nên
   phần đang dở có thể làm lại an toàn.
4. Sáu ZIP cũ còn nguyên: N051-N060, N061-N070, N071-N080, N081-N090,
   N091-N100 và S01. Trong lúc giải nén, downloader đã hoàn tất thêm M05;
   tại lần kiểm tra cuối có 7 ZIP hoàn chỉnh sẵn sàng giải nén và M06 đang tải.
5. Đã sửa extractor để khi gặp file đích đang ghi dở (kích thước khác metadata
   ZIP), nó chỉ xóa file đích chưa hoàn chỉnh rồi tạo lại từ ZIP nguồn.

## Git và kiểm chứng cuối phiên

- Dữ liệu cũ Batch 1 giữ nguyên 100%, không bị ảnh hưởng.
- File dữ liệu đang tải nằm ngoài Git (`data/batches/batch2/raw/`).
- File mới trong Git: `scripts/download_batch2.py`, cập nhật `config/datasets.json` và `MEMORY.md`.

## Hoàn tất bố trí Batch 2 ngày 2026-09-23

1. Toàn bộ dữ liệu đã tải nằm trong vùng riêng `data/batches/batch2/`; không
   đụng tới Batch 1. `raw/` trống vì các ZIP đã được xóa sau khi giải nén/xác
   minh theo yêu cầu trước đó; `processed/` và `combined/` vẫn để trống.
2. Cấu trúc dùng tiếp: `extracted/Videos/` (495 video: N là `.mov`, M/S là
   `.mp4`) và `extracted/Metadata/` (605 JSON). Giữ tên nguồn, kể cả hai kiểu
   phân cách `_`/`-`, để không phá mapping tương lai.
3. Thêm `scripts/audit_batch2.py`, chạy thành công với `--write`. Nó tạo
   `data/batches/batch2/extracted/inventory.json` (bị Git ignore) và xác nhận
   495 video, 228.473.998.977 byte, 605 JSON, ID duy nhất và tên hợp lệ.
4. `config/datasets.json` đã chuyển Batch 2 sang `extracted_and_audited`; tài
   liệu layout được cập nhật. Kiểm tra cú pháp cho downloader/extractor/auditor
   đều đạt.

### Việc tiếp theo

1. Chờ BTC cung cấp keyframe, map-keyframes, objects/ASR/feature tương ứng
   hoặc sinh chúng bằng pipeline cùng model Batch 1.
2. Tạo index Batch 2 riêng rồi kiểm chứng trước khi ghép sang `data/combined/`.

## Hướng dẫn CPU keyframe 2026-09-23

- Người dùng xác nhận BTC không cấp keyframe Batch 2, yêu cầu hướng dẫn tự
  trích bằng CPU desktop. Thêm `scripts/extract_keyframes_cpu.py` và tài liệu
  `docs/CPU_KEYFRAMES.md`; chưa chạy toàn bộ dữ liệu hay benchmark video thật.
- PyAV, NumPy, Pillow có sẵn trong `.venv`. Trong sandbox Python trampoline
  bị access denied; kiểm tra/thử nghiệm chạy được sau escalation.
- Script dùng CPU decode tuần tự với 4 luồng, khảo sát 2 FPS, lọc chênh lệch RGB,
  giữ ảnh định kỳ 4 giây, frame đầu/cuối, ảnh rộng tối đa 1280/JPEG90. Có lock,
  reserve 30 GiB, kiểm tra JPEG, CSV và marker nguyên tử, resume cấp video.
  Video dở được chạy lại từ đầu. Đổi nguồn/cấu hình phải đổi output root.
- CSV ghi frame_idx 0-based từ thứ tự decode, pts_time từ PTS tương đối frame
  đầu; source_pts_time giữ PTS nguồn. Không suy timestamp từ average FPS.
- Test synthetic VFR/resume/config mismatch đạt: 1 passed. Chế độ --list
  xác nhận M05_V001, N001-V001, S01-V001. Chưa kiểm chứng retrieval Batch 2.
- Ước tính 24–48 giờ/20–60 GB trước đây chưa đo; không dùng làm cam kết.
  Bước tiếp theo: người dùng chạy thử 3 nhóm, xem chất lượng và thời gian rồi
  chạy toàn bộ; sau đó embedding/index riêng bằng model đồng nhất Batch 1.
- Hai commit `f6190f4` và `afb169e` đã push thành công lên `origin/main`
  ngày 2026-09-23 (`9b8666c..afb169e`).
- Đã đo thử `M05_V001`: 25.645 frame giải mã, giữ 1.005 keyframe cho
  1.025,76 giây video; chạy 68,594 giây và tạo 159.510.418 byte ảnh. Chưa đo
  mẫu nhóm N/S và chưa chạy toàn bộ 495 video.
