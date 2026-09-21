# Memory dự án — đọc đầu mỗi phiên

Cập nhật: 2026-09-21 (Asia/Saigon).

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

## Git và kiểm chứng cuối phiên

- Các kiểm tra dữ liệu/move ở trên đã đạt, ngoài phát hiện khóa trùng được ghi rõ.
- Không thay đổi code runtime nên không chạy model/app hay test suite trong đợt dọn này.
- Commit `e3599a0` đã push thành công lên `origin/main` ngày 2026-09-21 (Git báo `5e7efbf..e3599a0 main -> main`). Bản cập nhật nhật ký này được lưu bằng commit tiếp theo; dùng `git status -sb` và `git log -2 --oneline` để xác nhận trạng thái mới nhất.
- Git sandbox báo ownership khác người dùng; dùng `git -c safe.directory=D:/bachkhoa/ai_challenge/my-repo ...` theo từng lệnh, không đổi global config.
- Python `.venv` cần quyền ngoài sandbox (uv trampoline bị permission denied trong sandbox).
