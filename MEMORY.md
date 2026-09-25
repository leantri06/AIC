# Memory dự án — đọc đầu mỗi phiên

Cập nhật: 2026-09-25 (Asia/Saigon).


## Quyết định mới nhất — 2026-09-25 (ưu tiên hơn lịch sử bên dưới)

- Người dùng xác nhận: **Ứng dụng đã chạy thành công thực tế** trên keyframe chính thức BTC + SigLIP2 SO400M-14-378 / webli.
- Tinh giản toàn diện repository: gỡ bỏ toàn bộ code dư thừa không còn sử dụng:
  + Gỡ các script giải nén (`extract_batch2.py`, `extract_keyframes_cpu.py`).
  + Gỡ bộ Kaggle pipeline (`kaggle/`, `prepare_kaggle_batch2.py`, `merge_kaggle_batch2.py`, `prepare_kaggle_full.bat`, `docs/KAGGLE_BATCH2.md`).
  + Gỡ các script benchmark/tải batch2 cũ và toàn bộ thư mục unit tests (`tests/`).
  + Giữ lại script kiểm tra index/dữ liệu: `scripts/audit_siglip2.py`.
- Toàn bộ backend, CLI, UI và paths chỉ tập trung vào visual-only SigLIP2 + keyframe BTC.
- Dọn dẹp cấu hình repository: `.gitignore`, `pyproject.toml`, `README.md`.
- Sẵn sàng đồng bộ toàn bộ codebase lên GitHub repository.


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

## Benchmark TransNet V2 ngày 2026-09-23

- Người dùng yêu cầu chạy thử một video. Xác nhận máy i5-10400F (6C/12T),
  GTX 1070 8 GiB, torch 2.5.1+cu121/CUDA hoạt động, av 18.1.0.
- Thêm `scripts/benchmark_transnetv2.py`, tài liệu `docs/TRANSNET_BENCHMARK.md`.
  Architecture upstream pin 85cef72, weights chuyển đổi PyTorch từ cộng đồng
  magnusdtd/TransNetV2 pin c90da43; ghi SHA256, nạp weights_only + strict.
  Model/cache nằm ngoài Git; video xử lý hoàn toàn local.
- M05_V001 (1.025,76s, 25.645 frame): setup 6,21s; decode/resize 258,84s;
  GPU inference FP32 19,20s; xuất JPEG 66,49s; tổng xử lý 344,53s.
  264 shot, 458 keyframe, 71.941.098 byte. Output riêng trong
  `data/batches/batch2/benchmarks/transnetv2-M05_V001-20260923/`.
- Máy đồng thời đang chạy extractor CPU toàn bộ do người dùng khởi chạy
  trước (PID 3096, --threads 4); không dừng/sửa tiến trình đó. Benchmark bị
  chia sẻ CPU/disk nên không so trực tiếp với mẫu CPU 68,594s lúc trước.
- Không chạy toàn bộ bằng TransNet. Chưa đánh giá chất lượng ranh giới shot,
  recall tìm kiếm hoặc numerical parity của weights với TensorFlow gốc.

## Audit sau lần chạy CPU toàn bộ 2026-09-23

- Người dùng báo đã trích xong nhưng audit thấy 495 thư mục ảnh, chỉ 222 CSV
  và 222 report hoàn tất, tổng 667.859 JPG; `keyframe-last-run.json` có 273
  failures. Không được dùng các thư mục dở để embedding.
- Phần lớn `.mov` decode tới EOS nhưng container khai báo dư frame (thường 1,
  đôi khi nhiều); M08_V001 có PTS đi lùi. Sửa extractor: EOS decoder là số
  frame thực tế, ghi delta/cảnh báo; PTS thô giữ ở source_pts_time và timeline
  tương đối được làm đơn điệu, có bộ đếm corrections. Video dở có owner marker
  được dọn riêng JPG sinh bởi script trước khi chạy lại.
- Test VFR/resume đạt. Chạy bù đại diện đạt: M08_V001 1.653 ảnh/130,9s;
  N001-V001 325 ảnh/45,3s, decoded 14.649 so với declared 14.650 và được ghi
  cảnh báo. Sau hai video có 224/495 report+CSV hoàn chỉnh; cần chạy lại lệnh
  toàn bộ để skip video hoàn tất và xử lý 271 video còn lại. Chưa embedding.

## Kiểm tra và đề xuất sau trích keyframe 2026-09-24

- Đọc lại hai PDF trong `docs/reference/` ngoài repo. `AIC2026.pdf` hiện là
  thông báo BTC ngày 21/9 về batch2 và tập huấn chung kết: dữ liệu chung kết
  gồm cả hai batch. Thông báo nói dữ liệu bổ trợ sẽ gửi sau; chưa có thể lệ
  chấm/nộp chung kết chi tiết trong hai file này. Không mặc định áp dụng quy
  tắc top-100 của tài liệu sơ tuyển cho chung kết. Camera giao thông được
  BTC giới hạn sử dụng trong cuộc thi, không phân phối lại.
- `keyframe-last-run.json`: selected=495, failures=[]; đối chiếu từng video
  có 495 report, 495 CSV và tổng 670.151 JPG. Số dòng CSV, JPG và kept_frames
  khớp ở mọi video; không gặp frame_idx âm. Đây là kiểm tra cấu trúc/số lượng,
  chưa giải mã lại toàn bộ JPEG hay xác minh chính xác từng mapping với video.
- `batch2/processed/` và `data/combined/` vẫn trống. Chưa chạy embedding.
- Phát hiện `ingestion/manifest.py::build_manifest` chưa đọc map-keyframes
  CSV; lấy frame từ Metadata hoặc ordinal, và chỉ dò video `.mp4`. Phải bổ
  sung mapping n -> frame_idx/pts_time và hỗ trợ `.mov` trước khi dùng cho
  batch2. Giữ nguyên video ID nguồn và xác nhận quy ước frame khi nộp.
- Thứ tự đề xuất: audit mapping/ảnh mẫu N/M/S; sửa và kiểm thử manifest;
  xác nhận checkpoint/preprocess/normalization thực sự của vector batch1;
  benchmark embedding GPU mẫu rồi chạy batch2 có resume; điều tra 614 khóa
  trùng batch1 và va chạm liên batch; ghép index vào combined, kiểm tra app
  và submission cả hai batch. Sau đó bổ sung ASR/OCR, BM25 và tinh chỉnh
  khung hình quanh kết quả truy xuất để phục vụ Q&A/TRAKE.
- Không đổi runtime hay dữ liệu trong phiên tư vấn này. Chưa kiểm tra online
  xem BTC có phát hành thêm video/dữ liệu bổ trợ sau thông báo hay không.

## Hoàn tất keyframe CPU và nguồn chính thức mới ngày 2026-09-24

- Lần chạy tiếp tục đã hoàn tất đủ 495/495 video: 495 report, 495 CSV map,
  670.151 JPG (112.653.200.707 byte) và `keyframe-last-run.json` không có lỗi.
  Không chạy lại extractor; chưa dùng output này để embedding/index.
- Google Sheet Batch 2 đã bổ sung các gói chính thức: keyframe cho S01, N001-N100
  và M01-M10; `aic26-b2-clip-features.zip`, `aic26-b2-map-keyframes.zip`, cùng
  Objects cho M01-M10, N001-N100 và S01. Đây là thay đổi quan trọng so với lần
  kiểm tra trước.
- Ghi chú chính thức: video N001-N100 dùng VFR nên `frame_idx` trong map chỉ
  tương đối; task KIS/QA có thể nộp millisecond (`pts_time * 1000`), TRAKE không
  dùng dữ liệu này.
- Việc tiếp theo cần người dùng quyết định: ưu tiên tải và kiểm chứng keyframe,
  map, CLIP và Objects chính thức, hay giữ bộ keyframe CPU local làm phương án
  bổ sung. Không tự tải các gói mới hoặc chạy embedding trước quyết định đó.

## Thử embedding CPU keyframe theo yêu cầu 2026-09-24

- Người dùng yêu cầu làm thử bước tiếp theo và hỏi “transformer = slow”.
  Đã chạy thử giới hạn trên keyframe CPU local, không tải gói BTC mới,
  không khởi chạy toàn bộ hay đổi app/index. Quyết định bộ dữ liệu dùng cho
  lần chạy toàn bộ vẫn chưa chốt.
- Làm rõ nhận định mapping trước: extractor CPU đặt tên JPEG bằng frame_idx
  thật (không phải n). Thêm reader riêng `ingestion/cpu_keyframes.py` để tránh
  ánh xạ hai lần; kiểm tra owner/report, video source stat, map/ảnh, thứ tự
  frame, timestamp. Hỗ trợ .mov/.mp4, giữ nguyên ID. Không sửa reader BTC cũ.
- FrameRecord có thêm pts_time và source_pts_time tùy chọn, giữ tương thích
  manifest cũ. Quy ước timestamp VFR khi nộp chưa được xác minh trong đợt này.
- Audit reader mới đạt toàn bộ 495 video, 670.151 ảnh. Chỉ kiểm tra cấu trúc,
  mapping và stat, không giải mã lại toàn bộ JPEG hoặc video.
- Thêm `scripts/benchmark_batch2_embeddings.py`, tài liệu
  `docs/BATCH2_EMBEDDING_TRIAL.md` và test. 24 test liên quan đạt.
- Thử 48 ảnh (16 ảnh rải đều/video M05_V001, N001-V001, S01-V001), FP32,
  L2, OpenCLIP SigLIP2 SO400M-14-378/webli từ cache, snapshot 93e0e6a.
  Batch size 2: 16,579s, 2,895 ảnh/s, peak allocated 1,700 GiB.
  Batch size 8: 16,289s, 2,947 ảnh/s, peak allocated 1,924 GiB.
  Cả hai tạo ma trận 48x1152 hữu hạn, norm xấp xỉ 1. Report đầy đủ cùng
  SHA256/preprocess nằm trong data/batches/batch2/benchmarks/
  siglip2-cpu-keyframes-20260924 và siglip2-cpu-keyframes-b8-20260924.
- Tính lại 6 vector batch1 tại hàng 0,35464,70928,106392,141856,177320:
  cosine 0,999948–0,999999, phù hợp trên mẫu; chưa chứng minh toàn bộ index
  hoặc đánh giá retrieval. Ngoại suy tốc độ mẫu cho toàn bộ là ~63–64 giờ,
  chưa tính overhead, không cam kết.
- “Slow” có thể là use_fast=False của image processor/tokenizer HF. Bản
  Transformers 5.14.1 ở máy chuyển image flag này thành backend="pil".
  Pipeline thử dùng OpenCLIP/PIL bicubic 378x378 nên không gắn cờ HF này.
- Việc còn lại: chọn CPU/BTC cho lần chạy đầy đủ, thêm pipeline resume theo
  video, kiểm thử tìm kiếm và timestamp nộp VFR; chưa ghép batch1+batch2.

## Chốt keyframe CPU, xem xét Kaggle 2026-09-24

- Người dùng đã CHỐT dùng keyframe CPU cho batch2, thay thế mục còn chờ
  chọn CPU/BTC ở trên. Hỏi khả năng tăng tốc với Kaggle T4 x2; chưa yêu cầu
  upload hoặc khởi chạy cloud trong lượt này.
- T4 có Tensor Cores FP16; phương án đề xuất là hai process độc lập, mỗi
  GPU giữ một image encoder, chia video cân theo số ảnh, dùng FP16 autocast
  nhưng normalize/lưu float32 và kiểm chứng lại với vector batch1 FP32.
  Script benchmark hiện tại chỉ dùng một GPU và FP32, không tự tận dụng x2.
- Chưa đo T4 nên không cam kết số giờ. Với 670.151 ảnh, tổng throughput
  giả định 20/40/60 ảnh/s tương ứng khoảng 9,31/4,65/3,10 giờ, chưa gồm I/O,
  upload và setup. Cần benchmark vài nghìn ảnh trên cả hai GPU rồi ngoại suy.
- Keyframe hiện ~112,65 GB; cân nhắc shard, chỉ chuyển ảnh/map và metadata
  cần thiết, không cần video gốc cho embedding. Reader audit local hiện
  yêu cầu video source stat, cần tách bước kiểm chứng/manifest portable khi
  chuẩn bị pipeline Kaggle. Không tự resize/re-JPEG làm thay đổi preprocess.
- Kaggle docs nêu phiên GPU tối đa 12h và quota GPU theo tài khoản; cần lưu
  output bền vững theo shard và resume qua phiên, không chỉ ghi file tạm.
- Dữ liệu camera có giới hạn phân phối từ BTC; không công khai Dataset hoặc
  Notebook chứa dữ liệu. Cần xác minh quyền dùng cloud theo quy định BTC
  trước khi chuyển dữ liệu. Chưa upload bất kỳ dữ liệu nào.

## Triển khai bộ Kaggle T4 x2 ngày 2026-09-24

- Người dùng yêu cầu thực hiện và chọn tự import notebook, thêm private
  dataset bằng tài khoản Kaggle có GPU. Không có cấu hình Kaggle API local;
  chưa upload/chạy Kaggle, không có benchmark T4 thực tế trong phiên này.
- Đã tạo `kaggle/batch2_siglip2_t4x2.ipynb`, runner độc lập
  `kaggle/siglip2_runner.py`, packer `scripts/prepare_kaggle_batch2.py`,
  merger `scripts/merge_kaggle_batch2.py`, launcher .bat và hướng dẫn
  `docs/KAGGLE_BATCH2.md`. Inventory ghi nhận keyframe_source=cpu và sample
  local đã kiểm chứng, Kaggle chưa chạy.
- Notebook mặc định benchmark; mỗi GPU một process/image tower, chia shard
  theo số ảnh, FP16 inference/L2 float32, pin checkpoint SHA256/revision.
  Kiểm chứng 6 vector batch1 trên mỗi GPU, cosine >=0.999 rồi mới chạy.
  Dataset ảnh đọc trực tiếp từ ZIP_STORED đuôi .aicshard, không giải nén
  toàn bộ. JPEG giữ nguyên byte. Record portable giữ frame_idx và hai PTS.
- Resume qua output các phiên: xác minh input/mapping hash, cấu hình, norm
  và shape; ghi npz nguyên tử. Thời gian full mặc định 9,5h; người dùng
  Save Version/Run All rồi Add Input toàn bộ output cũ cùng catalog. Không
  tự upload output; file trong phiên interactive chưa lưu không bền vững.
- Merger từ chối catalog sample/thiếu shard/config khác/khóa trùng; tạo
  index batch2 mới với đường dẫn local, chưa ghép batch1 hay chuyển app.
- 35 test đạt: giữ JPEG/mapping, corruption, chia tải, resume khác config,
  merge đường dẫn/thứ tự, notebook compile và test ingestion/submission cũ.
- Smoke GPU local thực: 4 ảnh + 6 parity, FP16 GTX1070; cosine min
  0,9999899. Spawn worker, ghi/đọc output thành công. Resume sang thư mục
  output mới với --resume-root bỏ qua đúng shard, frames=0. Chưa thể test
  hai GPU thật trên máy chỉ có một GPU; notebook assert đủ 2 GPU khi chạy.
- Gói benchmark HOÀN TẤT tại data/batches/batch2/kaggle-sample/: 2.354 ảnh,
  3 shard, 373.005.447 byte; control có catalog/parity/runner. User tạo hai
  private datasets từ nội dung control/ và images-000/, import ipynb.
- Bộ FULL ĐANG ĐÓNG GÓI tại data/batches/batch2/kaggle-full/. Đã khởi chạy
  Python nền qua exec session 76162, process Python thực PID 4388, trampoline
  PID 5916 (19:43:08 local). Đổi sang ghi ZIP tuần tự có buffer 64MiB; giữ
  shard cũ và resume sau khi dừng phiên pack trước PID10788. Lần kiểm tra
  cuối: 39 shard, 46.924 ảnh, 6,721 GiB, catalog_ready=false; KHÔNG coi full
  là hoàn tất. Không khởi chạy thêm packer khi PID4388 còn chạy.
- Xem tiến độ: `.venv/Scripts/python.exe scripts/prepare_kaggle_batch2.py
  --output ../data/batches/batch2/kaggle-full --status`. Catalog full chỉ được
  ghi khi đóng gói xong. Nếu process bị ngắt, chạy lại cùng lệnh không có
  --status (hoặc scripts/prepare_kaggle_full.bat); source không bị thay đổi.
- Việc tiếp theo: người dùng benchmark gói sample trên Kaggle; kiểm tra
  pack full hoàn tất; thay sample/control/output sample bằng bộ full, chạy
  theo quota và persist từng version. Giữ toàn bộ output để tải/merge local.

## Kiểm tra lỗi upload Kaggle 2026-09-25

- Người dùng gửi ảnh lỗi UI `Failed to execute 'json' on 'Response':
  Unexpected end of JSON input`, danh sách có shard S01-V012 và quota
  hiển thị 119.19 GB / 214.75 GB. Chưa có HTTP status/response nên chưa xác
  định lỗi mạng, phiên đăng nhập, backend hoặc giới hạn upload; không kết
  luận JSON bên trong shard hỏng hay đuôi .aicshard bị từ chối.
- Bộ full đã đóng gói XONG: catalog ngày 24/9 21:39; 495 video, 670.151 ảnh,
  628 shard, 14 nhóm images-000..013, 112.880.513.115 byte. Đối chiếu mọi
  file với catalog: đủ, kích thước khớp. Chưa rehash toàn bộ 112,88 GB.
- Kiểm tra mẫu S01-V012-0016.aicshard trong ảnh: SHA256 khớp catalog và
  ZipFile.testzip() không phát hiện CRC lỗi. Không sửa dữ liệu.
- Đề xuất kiểm tra My Datasets ở tab khác trước khi upload lại để tránh
  trùng khi create đã thành công; dùng từng nhóm ~8 GiB thay vì gom cả bộ
  trong một lần, nếu vẫn lỗi lấy status/response request lỗi trong Network.
  Không đề nghị xóa/reset upload lớn trước khi kiểm tra trạng thái máy chủ.

## Tinh giản codebase và cập nhật Git 2026-09-25

- Người dùng xác nhận ứng dụng đã chạy thành công trên máy local.
- Tiến hành dọn dẹp các thành phần không còn cần thiết:
  + Xóa toàn bộ thư mục `kaggle/` và các scripts liên quan (`prepare_kaggle_batch2.py`, `merge_kaggle_batch2.py`, `prepare_kaggle_full.bat`, `docs/KAGGLE_BATCH2.md`).
  + Xóa các script giải nén và trích keyframe CPU (`extract_batch2.py`, `extract_keyframes_cpu.py`, `docs/CPU_KEYFRAMES.md`).
  + Xóa các script tải/benchmark cũ (`download_batch2.py`, `benchmark_batch2_embeddings.py`, `benchmark_transnetv2.py`, `run_transnetv2_batch.py`, `audit_batch2.py`, `docs/BATCH2_EMBEDDING_TRIAL.md`, `docs/TRANSNET_BENCHMARK.md`).
  + Xóa toàn bộ thư mục `tests/` và cấu hình testpaths trong `pyproject.toml`.
  + Giữ lại script kiểm tra tính hợp lệ của index BTC: `scripts/audit_siglip2.py`.
  + Dọn dẹp các script scratch tạm ở thư mục gốc ngoài repo.
- Thêm `.ruff_cache/` vào `.gitignore`.
- Cập nhật `README.md` và `MEMORY.md`.
- Sẵn sàng stage, commit và push lên nhánh `main` trên GitHub.

## Thích ứng Thể thức Chung kết AIC 2026-09-25

### Thể thức thi đấu mới (Finals):
- Thời gian: **5 phút / câu**, điểm giảm dần theo thời gian nộp (decay score).
- Đề bài chiếu trên màn hình, **không thể Ctrl+C/V** đề.
- BTC có App riêng để nộp bài (không còn nộp CSV trực tiếp):
  + **KIS** (Textual & Video KIS): Nhập `Video ID` + `Frame Time (ms)`.
  + **Q&A**: Nhập `Answer` + `Video ID` + `Frame Time (ms)` → App tự sinh `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>`.
  + **TRAKE**: Nhập `Video ID` + `danh sách Frame IDs` → App tự sinh `TR-<VIDEO_ID>-<F1>,<F2>,...`.
  + **Raw JSON**: Payload thô.

### Cải tiến `src/aic2026/app/ui.py` đêm trước Chung kết:
- Thêm hàm `get_frame_time_ms(video_id, frame_id)` → trả về `int` mili-giây tính từ `pts_time × 1000` trong `map_keyframes/*.csv`.
- Thêm `_render_finals_submission_dock(query, candidates, selected, map_dir)`:
  + Xác định mục tiêu nộp là ứng viên đầu tiên trong `selected`, hoặc **Top #1** nếu chưa chọn.
  + 4 Tab tương ứng 4 định dạng của App BTC:
    * **Tab KIS**: Hiển thị `Video ID` + `Frame Time ms` (lấy từ BTC map_keyframes thực) + nút 1-Click Copy riêng từng trường + nút Copy Cặp (Tab-separated `VideoID\t{ms}`).
    * **Tab Q&A**: Ô nhập Answer → Tự sinh chuỗi `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>` + nút Copy.
    * **Tab TRAKE**: Sinh tự động chuỗi `TR-<VIDEO_ID>-<F1>,<F2>,...` từ `event_frames` hoặc frames đã chọn + nút Copy.
    * **Tab Raw JSON**: Payload JSON đầy đủ + nút Copy.
- Cập nhật `_render_frame_gallery`:
  + Badge trên ảnh hiển thị `{ms}ms ({MM:SS.ss})`.
  + Card-meta hiển thị `video_id · frame_id · {ms}ms` (màu xanh nổi bật).
  + Action bar đơn giản hóa thành 2 cột: `[🎬 Xem]` + `[Chọn]`.
  + Thêm 2 nút mini bên dưới mỗi ảnh: `[📋 ID]` copy Video ID và `[⏱️ {ms}ms]` copy Frame Time ms.
- Cập nhật `_render_trake_gallery`: Mỗi kết quả có nút **Copy Chuỗi TRAKE** ngay trên header.
- Dock được gọi cho **cả hai chế độ KIS/QA và TRAKE** trước khi render gallery.
- `get_single_video_pts_map`: Hỗ trợ thêm `map_keyframes/` (dấu gạch dưới) bên cạnh `map-keyframes/`.
- Tích hợp **Auto Warm-up (`warmup_engine`)**: Tự động nạp sẵn SigLIP2 SO400M weights, FAISS index và 177.000 dòng manifest vào RAM/VRAM ngay khi app vừa mở. Hiển thị banner trạng thái `🟢 Model & Index đã nằm sẵn trong VRAM (0.1s/query)` và nút `[🔄 Nạp lại]` hỗ trợ rewarm khi cần. Loại bỏ hoàn toàn độ trễ 90s ở câu hỏi đầu tiên.

### Chiến thuật thi đấu (khuyến nghị):
- Gõ **3-5 từ khóa tiếng Anh** (không cần câu đầy đủ) → Search trong 0.2s → Quét Top 5 → Bấm **Copy ms** trên ảnh đúng → Dán vào App BTC → Nộp trong **< 45 giây** → Ăn điểm tối đa.
- Bật **English Direct Mode** (bỏ tick "Dịch VI→EN") khi nhập tiếng Anh trực tiếp.

## Rà soát tốc độ truy vấn 2026-09-25

- Người dùng yêu cầu giảm ~30s xuống ~5s, sau đó yêu cầu bỏ dịch tiếng Việt.
- Tách cache corpus/pipeline khỏi cache orchestrator: đổi task/NMS/coarse không đọc lại manifest + tính lại centroid. Benchmark xác nhận 1 cache miss, 3 hits khi đổi KIS/QA/TRAKE.
- Bỏ dịch khỏi UI, CLI và đường chạy agent/API; dùng trực tiếp câu nhập. Module translator cũ chưa xóa vật lý.
- Sửa warmup nuốt exception/báo thành công giả; bỏ cam kết VRAM/0.1s chưa đo. Thêm timing load/search/total vào trace và UI; bỏ forced gc mỗi query.
- Sửa run_task bỏ qua coarse_top_k; sửa empty video filter thành không có kết quả thay vì tìm toàn corpus, xử lý k<=0 cho filtered search.
- Máy đo thực: RTX 3050 Laptop 4 GiB, torch 2.11.0+cu128; matrix 1.832.099.456 bytes. Baseline cold 61,56s (cProfile), warm KIS 2,36s. Sau sửa warm KIS 1,27–1,61s; QA retrieval 1,26s; TRAKE 2 events/top200 1,67s. Cold khoảng 47–51s. Chưa đo UI end-to-end, p95 hoặc recall trên ground truth; không khẳng định đã tái hiện chính xác 30s người dùng gặp.
- Kiểm chứng: compileall, git diff --check, regression trực tiếp NumPy cho filter rỗng/k=0/filter khớp đạt; benchmark corpus thật đạt. Ruff báo 2 lỗi tồn tại sẵn ở API (BLE001, TRY401); không coi toàn repo lint sạch.
- Thêm scripts/benchmark_queries.py để tái đo. Cần khởi động lại app/API để dùng code mới.
- Vấn đề audit còn lại: get_frame_timestamp fallback 25 FPS có thể sai ms khi thiếu map; /images đọc đường dẫn tùy ý nếu API bị expose; encode_images không dùng được khi visual tower đã bị xóa. Chưa sửa trong đợt tối ưu này.

## Thông báo BTC tên media-info S01 ngày 2026-09-25

- Người dùng báo BTC đính chính S01_V0*.json phải tương ứng video S01-V0*.mp4.
- Kiểm tra local: 12 JSON S01_V001..S01_V012 trong data/extracted/media_info; keyframe và CSV đều dùng S01-V001..S01-V012. JSON chứa title/description/keywords và thông tin nguồn video, không phải CSV timestamp.
- Đối chiếu toàn bộ 37.140 dòng manifest S01 với map_keyframes: không có frame nào thiếu giá trị pts_time (kiểm tra hiện diện, chưa đối chiếu video gốc).
- Sửa ingestion.manifest._metadata: ưu tiên tên đúng S01-V0*.json, fallback tên cũ S01_V0*.json chỉ cho nhóm BTC nêu. Đọc thành công 12/12; kiểm tra ưu tiên tên đúng và không ảnh hưởng nhóm khác đạt.
- Không đổi tên/xóa dữ liệu, không sửa manifest/index/embedding. Manifest hiện có metadata_path=None ở S01; sửa loader có hiệu lực khi đọc/import metadata lần sau, không tự backfill manifest hiện tại. Luồng tìm kiếm visual hiện tại không cần rebuild vì thông báo đổi tên JSON này.

## Rà soát mức sẵn sàng hệ thống 2026-09-25

- Kiểm tra read-only toàn bộ manifest với CSV hiện tại: 397.591 dòng, 1.487 video; 0 frame thiếu timestamp, 0 dòng CSV timestamp âm/không hữu hạn/không parse được. Có 637 dòng dư trùng (video_id, frame_id), chưa tự xóa. TRAKE hiện đã dedup frame khi dựng timeline.
- Phát hiện chưa sửa: UI giữ finals_qa_answer/finals_qa_answer_input giữa các truy vấn, có thể ghép đáp án câu trước vào video câu mới. Raw JSON copy lồng json.dumps trong thuộc tính onclick có dấu nháy kép chưa escape; tái hiện HTMLParser chỉ đọc được onclick=navigator.clipboard.writeText(. Nút QA cũng dùng HTML escaping thay vì JS string escaping; cần thay cơ chế copy và kiểm tra trình duyệt thật.
- Tái hiện lỗi TRAKE với corpus synthetic 2 video: coarse top1 chọn A trước rồi giao bộ lọc {B}, trả 0 dù B có frame hợp lệ; coarse=0 trả B. Cần áp dụng video filter trước coarse top-k. Luồng UI mặc định không bật metadata filter nên không khẳng định các kết quả mặc định bị lỗi này.
- Đề xuất ưu tiên sửa thao tác nộp/copy, bộ lọc TRAKE; sau đó đo end-to-end/p95 trên nhiều câu và recall top5/top10 có nhãn. _record_lookup hiện dựng dict toàn corpus mỗi lần gọi kể cả các rerank weight=0; có thể tránh việc thừa này để giảm latency/allocations.
- Đợt này chỉ audit, không đổi hành vi ứng dụng; chưa kiểm chứng clipboard bằng trình duyệt và chưa đánh giá recall.

## Giải thích 637 khóa frame trùng — 2026-09-25

- Đối chiếu chi tiết: 637 nhóm, mỗi nhóm 2 dòng manifest có cùng (video_id, frame_id) nhưng KHÁC keyframe_path. Phân bố: L=614, M=16, S=7, N=0. Không phải bằng chứng một file ảnh được nhập hai lần.
- Cả 637 nhóm đều tồn tại nguyên trong CSV BTC với cùng số dòng. Có 628 nhóm khác pts_time, chênh tối đa 40ms; 9 nhóm cùng pts_time. Chưa hash/so sánh pixels để kết luận nội dung ảnh có trùng.
- Ví dụ L21_V006: n=1/001.jpg -> frame_idx=0, pts_time=0; n=2/002.jpg -> frame_idx=0, pts_time=0.0333333. Có dấu hiệu quy đổi/làm tròn frame_idx ở nguồn nhưng chưa có code tạo CSV BTC nên không kết luận nguyên nhân sinh mapping.
- Đính chính mức kiểm chứng timestamp: tất cả frame có timestamp KHÔNG đồng nghĩa mapping frame_idx -> pts_time duy nhất. UI dict hiện lấy dòng cuối khi frame_idx trùng, có thể sai đến 40ms so với ảnh được chọn trong các nhóm đã kiểm tra. Cách sửa phù hợp: tra timestamp theo ordinal tên ảnh/n trước, không tự xóa dòng hay đổi frame_idx BTC. Chưa thực hiện sửa ở lượt giải thích này.

## Sửa timestamp theo đúng ảnh — 2026-09-25

- Thêm data_platform/timestamps.py: CSV tra theo ordinal tên ảnh -> cột n -> (frame_idx, pts_time), kiểm tra frame_id khớp trước khi dùng. Giữ fallback frame-id cho caller không có ảnh; không sửa CSV/frame_id/vector.
- UI truyền keyframe_path vào timestamp ở gallery, dock nộp KIS/QA/Raw JSON, TRAKE gallery và video dialog. Dialog giữ đúng ảnh đã chọn khi frame_id trùng, state phân biệt theo ảnh ban đầu. Cache CSV theo absolute path + mtime_ns + size, tránh dùng nhầm khi đổi data root.
- Đối chiếu toàn bộ 397.591 dòng manifest với CSV theo n: tất cả timestamp khớp; 628 dòng lấy thời gian đúng khác cách tra cũ. Ví dụ L21_V006 frame0: 001.jpg=0ms, 002.jpg=33ms.
- Kiểm chứng: 2 pytest regression đạt (frame trùng, Windows/POSIX path, ordinal sai, cache khác root và refresh); AST-extracted UI wrapper trả 0/33ms đúng; compileall, lint module/test và diff check đạt. Chưa thao tác trình duyệt/clipboard. Dữ liệu gốc giữ nguyên. Cần restart app để dùng code mới.
