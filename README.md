# PEGASUS — SigLIP2 Video Retrieval System
> **Hệ thống tìm kiếm video tối ưu cho Vòng Chung kết AI Challenge (AIC)**  
> Hỗ trợ tìm kiếm hình ảnh/khoảnh khắc theo mô tả ngữ nghĩa cho các thể thức: **KIS (Known-Item Search)**, **Visual Q&A**, và **TRAKE (Event Tracking)**.

---

## 🌟 Tính năng nổi bật

### 1. Kiến trúc Dual-Model (SigLIP2 Giant & SO400M)
- **SigLIP2 Giant (1536D)** (`google/siglip2-giant-opt-patch16-384`): Model thị giác - ngôn ngữ mạnh nhất của Google, độ nhạy ngữ nghĩa cực cao, nhận diện vượt trội các chi tiết nhỏ, hành động phức tạp, màu sắc và bối cảnh fine-grained.
- **SigLIP2 SO400M (1152D)** (`ViT-SO400M-14-SigLIP2-378 / webli`): Model cân bằng, hiệu năng cao và nhẹ bộ nhớ.
- **Strict Paired Index Integrity**: Cơ chế tự động khóa cặp ma trận vector và manifest tương ứng (`giant_features.npy` ↔ `giant_manifest.jsonl`), ngăn ngừa hoàn toàn rủi ro sai lệch vector.
- **Chuyển đổi 1-click trên UI**: Dễ dàng đổi qua lại giữa Giant và SO400M ngay trong lúc tìm kiếm.

### 2. Tối ưu hóa tốc độ & Auto-Warmup Engine
- **Tốc độ truy vấn ~0.9s – 1.0s / câu** trên toàn bộ kho dữ liệu gần 400.000 keyframes (đo thực tế trên GPU RTX 3050 Laptop).
- **Auto-Warmup khi khởi động**: Tự động nạp sẵn model weights và ma trận vector vào bộ nhớ ngay khi mở ứng dụng. Loại bỏ hoàn toàn tình trạng đơ/chậm ở câu hỏi đầu tiên.

### 3. Finals Submission Dock (Dock nộp bài chuẩn thể thức BTC)
Tích hợp ngay trên đầu trang kết quả, hỗ trợ copy thần tốc vào ứng dụng nộp bài của Ban tổ chức:
- **Tab KIS**: Tự động trích xuất `Video ID` và `Frame Time (ms)` của kết quả Top 1 (hoặc ảnh được click chọn). Cung cấp nút 1-click copy riêng từng trường hoặc copy cả cặp (Tab-separated `VideoID\t{ms}`).
- **Tab Visual Q&A**: Nhập câu trả lời ngắn → Hệ thống tự động sinh chuỗi định dạng chuẩn `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>` đi kèm nút 1-click copy.
- **Tab TRAKE**: Nhập/chọn chuỗi các khung hình sự kiện → Tự động tạo chuỗi `TR-<VIDEO_ID>-<F1>,<F2>,...`.
- **Tab Raw JSON**: Xuất payload JSON đầy đủ cho hệ thống chấm tự động.

### 4. Xử lý chuẩn xác Timestamp & Khung hình
- **Khắc phục triệt để lỗi trùng ID frame**: Tra cứu timestamp chính xác đến từng mili-giây (`pts_time × 1000`) dựa theo thứ tự ảnh (`ordinal`/`n`) trong file CSV `map_keyframes/` chính thức của BTC.
- **Temporal NMS (Non-Maximum Suppression)**: Tự động khử các khung hình quá gần nhau trong cùng một phân cảnh, giúp danh sách Top kết quả đa dạng và bao quát nhiều góc quay khác nhau.
- **Gallery trực quan**: Hiển thị ảnh kèm badge thời gian `ms (MM:SS.ss)`, xem video tại mốc thời gian tương ứng, phóng to và duyệt các frame lân cận.

---

## 🚀 Khởi chạy ứng dụng

### Cách 1: Khởi chạy 1-Click (Khuyên dùng)
Nhấp đúp chuột vào file:
```cmd
run_app.bat
```

### Cách 2: Chạy qua dòng lệnh (PowerShell / Command Prompt)
Tại thư mục gốc của repository `AIC`:
```powershell
.venv\Scripts\python.exe -m streamlit run src/aic2026/app/ui.py --server.port 8501
```

Sau khi khởi chạy, mở trình duyệt tại: **`http://localhost:8501`** (hoặc `http://127.0.0.1:8501`).  
Hệ thống sẽ chạy auto-warmup khoảng 30–40 giây. Khi xuất hiện banner xanh báo sẵn sàng, mọi truy vấn sau đó sẽ phản hồi tức thì trong ~1 giây.

---

## 📁 Cấu trúc thư mục dữ liệu

Ứng dụng tự động nhận diện thư mục `data/` nằm ngang hàng với repository `AIC/`:

```text
ai_challenge/
├── AIC/                                     # Thư mục mã nguồn repository
│   ├── run_app.bat                          # File khởi chạy ứng dụng
│   ├── src/aic2026/                         # Backend, embeddings, pipeline, UI
│   ├── scripts/                             # Script kiểm tra, benchmark, downloader
│   └── tests/                               # Bộ unit test
└── data/                                    # Thư mục dữ liệu (ngoài Git)
    ├── processed/
    │   ├── siglip2_giant_features.npy       # Ma trận 397.591 × 1536 float32 (Giant)
    │   ├── siglip2_giant_manifest.jsonl     # Manifest tương ứng của Giant
    │   ├── siglip2_features.npy             # Ma trận 397.591 × 1152 float32 (SO400M)
    │   └── siglip2_manifest.jsonl           # Manifest tương ứng của SO400M
    ├── extracted/
    │   ├── keyframes/<VIDEO_ID>/<001>.jpg   # Thư mục ảnh keyframe chính thức BTC
    │   ├── map_keyframes/<VIDEO_ID>.csv     # File CSV ánh xạ frame_idx và pts_time
    │   └── Videos/                          # (Tùy chọn) Video gốc .mp4 / .mov
    └── .cache/                              # Cache lưu trọng lượng model
        ├── huggingface/
        └── torch/
```

> **Lưu ý về đường dẫn:**  
> Bạn có thể tùy biến đường dẫn thư mục dữ liệu bằng cách đặt biến môi trường `AIC_DATA_ROOT` hoặc chỉnh trực tiếp trong mục **Cấu hình Thư mục Dữ liệu Gốc** trên thanh Sidebar của ứng dụng.

---

## 🎯 Chiến thuật thi đấu hiệu quả (AIC Finals)

Trong phòng thi, thời gian mỗi câu hỏi chỉ khoảng **5 phút** và điểm số giảm dần theo thời gian (Decay Score):

1. **Lấy câu hỏi từ màn hình chiếu:**
   - **Gõ từ khóa cốt lõi (Nhanh nhất - khuyên dùng):** Không cần gõ cả câu mô tả dài. Chỉ cần nhìn lướt và gõ **3–5 từ khóa tiếng Anh quan trọng nhất** (Ví dụ: `girl pink dress bicycle park`). Mô hình SigLIP2 rất nhạy với semantic keywords, gõ mất 3–5 giây và cho kết quả Top 1–5 cực chuẩn.
   - **Phím tắt OCR màn hình:** Nhấn `Win + Shift + S` (Windows 11 Snipping Tool) quét vùng đề bài rồi chọn **Text Actions** (hoặc `Win + Shift + T` của PowerToys) để copy nhanh toàn bộ văn bản.
2. **Quy trình nộp bài 30 giây:**
   - Nhập từ khóa → Bấm **Tìm kiếm** (phản hồi trong ~1s).
   - Xem nhanh các ảnh trong Top 10 → Click chọn ảnh đúng nhất (nếu không chọn, hệ thống mặc định lấy Top 1).
   - Ngay tại **Finals Submission Dock** ở đầu trang:
     - Thể thức KIS: Bấm **Copy ms** (hoặc Copy Cặp).
     - Thể thức Q&A: Nhập nhanh đáp án ngắn vào ô Answer → Bấm **Copy Chuỗi Q&A**.
     - Thể thức TRAKE: Chọn các frame sự kiện → Bấm **Copy Chuỗi TRAKE**.
   - Chuyển sang App BTC, dán (`Ctrl + V`) và bấm **Nộp bài**.

---

## 🛠️ Cài đặt môi trường mới (Nếu dựng máy mới)

Yêu cầu: Python >= 3.11, CUDA hỗ trợ PyTorch (khuyến nghị GPU NVIDIA 4GB VRAM trở lên).

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường và cài đặt dependencies
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[models,dev]"

# Cài đặt PyTorch với CUDA (ví dụ CUDA 12.4 / 12.1)
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 🧪 Công cụ kiểm tra & Đo lường (CLI Scripts)

### 1. Đo lường tốc độ truy vấn (Benchmark Queries)
Kiểm tra thời gian nạp và độ trễ truy vấn thực tế trên index:
```powershell
# Đo với model SigLIP2 Giant (1536D)
.venv\Scripts\python.exe scripts/benchmark_queries.py --model siglip2-giant

# Đo với model SigLIP2 SO400M (1152D)
.venv\Scripts\python.exe scripts/benchmark_queries.py --model siglip2
```

### 2. Kiểm toán dữ liệu (Audit Integrity)
Kiểm tra cấu trúc file, tính toàn vẹn của ma trận vector, manifest và CSV mapping:
```powershell
.venv\Scripts\python.exe scripts/audit_siglip2.py
```

### 3. Chạy Unit Tests
Kiểm tra tính tương thích và cơ chế chống xung đột giữa các model:
```powershell
.venv\Scripts\python.exe -m pytest tests/test_giant.py -v --basetemp=scratch/pytest -p no:cacheprovider
```

---

## 📝 Tài liệu bàn giao & Quy ước
- Mọi lịch sử nghiên cứu, benchmark, thay đổi kiến trúc và nhật ký bàn giao giữa các phiên được ghi nhận chi tiết tại [MEMORY.md](MEMORY.md).
- Quy tắc làm việc và bảo vệ mã nguồn được quy định tại [AGENTS.md](AGENTS.md).
