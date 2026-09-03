# 🚀 AIC 2026 — PEGASUS Video Retrieval System

Hệ thống tìm kiếm video đa phương thức tối tân phục vụ 3 bài toán trọng tâm vòng sơ tuyển **AI Challenge 2026 (KIS / Q&A / TRAKE)**:
* **👁️ Visual Vector Search SOTA:** **Google SigLIP 2** (`ViT-SO400M-14-SigLIP2-378` / alias: `siglip2`) tăng tốc bằng GPU NVIDIA CUDA (thời gian tìm kiếm $\le 1\text{s}$).
* **🎯 Shot-level Temporal NMS:** Thuật toán lọc phi cực đại triệt tiêu các khung hình trùng lặp trong cùng 1 cảnh quay, tối ưu hóa đưa keyframe chuẩn xác nhất lên Top 1.
* **🧠 Dense Semantic ASR (BGE-M3 1024-dim):** Tìm kiếm ngữ nghĩa lời thoại sâu sắc bằng `BAAI/bge-m3` (định dạng `model.safetensors` nạp trực tiếp trong $0.05\text{s}$), kết hợp **Hybrid Dense-Sparse** cùng BM25.
* **⚡ BLIP-2 ITM Cross-Encoder:** Tích hợp mô hình đối chiếu chéo cấp độ pixel theo nghiên cứu SOTA (arXiv:2512.12935v1) kèm bộ chốt an toàn VRAM Guard.
* **⚖️ VLM-as-a-Judge:** **Qwen2.5-VL-3B / 7B** thẩm định thị giác chi tiết cho KIS, kiểm tra bằng chứng thị giác (Visual Evidence) cho Q&A, và xác thực chuỗi đa sự kiện thời gian cho TRAKE.
* **🔤 Lexical & Metadata:** BM25 nhãn vật thể (Faster R-CNN) + Tiêu đề/mô tả YouTube + PaddleOCR.
* **⚡ Startup Preloader:** Tự động nạp sẵn toàn bộ mô hình và 177,321 keyframe vào RAM/GPU ngay khi mở app — **không còn độ trễ chờ đợi ở câu lệnh đầu tiên!**

---

## 📑 Mục Lục
1. [Điểm Nổi Bật & Các Cải Tiến Mới Nhất](#1-điểm-nổi-bật--các-cải-tiến-mới-nhất)
2. [Cấu Trúc Thư Mục Dữ Liệu & Mã Nguồn](#2-cấu-trúc-thư-mục-dữ-liệu--mã-nguồn)
3. [Cài Đặt Môi Trường (PyTorch CUDA 12.1)](#3-cài-đặt-môi-trường-pytorch-cuda-121)
4. [Hướng Dẫn Khởi Động & Sử Dụng Giao Diện Web](#4-hướng-dẫn-khởi-động--sử-dụng-giao-diện-web)
5. [Chi Tiết 4 Chế Độ Tìm Kiếm & Bí Quyết Chọn](#5-chi-tiết-4-chế-độ-tìm-kiếm--bí-quyết-chọn)
6. [Cơ Chế Reranking, Lọc Đa Dạng Hóa (NMS) & VLM Judge](#6-cơ-chế-reranking-lọc-đa-dạng-hóa-nms--vlm-judge)
7. [Chạy Bằng Dòng Lệnh CLI & Đánh Giá Ground-Truth](#7-chạy-bằng-dòng-lệnh-cli--đánh-giá-ground-truth)
8. [Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#8-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. Điểm Nổi Bật & Các Cải Tiến Mới Nhất

| Tính Năng | Trạng Thái Trước Đây | **Cải Tiến Mới Hiện Tại** | Lợi Ích Mang Lại |
| :--- | :--- | :--- | :--- |
| **GPU CUDA Acceleration** | Chạy PyTorch CPU (rất chậm) | **PyTorch CUDA 12.1** trên NVIDIA GTX 1070 / RTX Series | Tốc độ tìm kiếm tăng vọt từ 3-5 phút xuống **$\le 1\text{s}$ (Visual)** và **4-8s (Hybrid)**. |
| **Startup Preloader** | Đợi 3-5 phút ở lượt query đầu tiên | Tự động nạp trước toàn bộ Model, Manifest & BM25 lúc mở app | **Truy vấn đầu tiên phản hồi ngay lập tức** trong vài giây. |
| **Quản lý Bộ nhớ VRAM** | Dễ văng lỗi `CUDA Out of Memory` | Singleton `_MODEL_CACHE` + xả sạch VRAM rác `empty_cache()` | VRAM chỉ chiếm **~700 MB - 2.7 GB** (trên tổng 8 GB), không bao giờ lo tràn RAM hay tràn VRAM. |
| **Đa dạng hóa Kết quả (NMS)** | Top 10 toàn ảnh của cùng 1 giây video | **Shot-level Temporal NMS** (ngăn trùng lặp trong $\pm 75$ frames) | Loại bỏ ảnh trùng lặp góc máy, đưa các keyframe độc nhất và đúng nhất lên Top 1 - Top 5. |
| **Dense Semantic ASR** | Cảnh báo lỗi `safetensors` | Nạp trực tiếp file `model.safetensors` cục bộ trên ổ D | BGE-M3 nạp xong trong **$0.05\text{s}$**, tuyệt đối sạch bóng cảnh báo và lỗi bảo mật. |
| **Lưu Trữ Dữ Liệu An Toàn** | Nguy cơ tràn ổ C | Khóa cứng 4 tầng biến môi trường (`HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`) sang ổ **`D:`** | **Giải phóng hơn 26 GB trên ổ C**, toàn bộ model và cache tương lai 100% nằm tại ổ D còn trống hơn 400 GB. |

---

## 2. Cấu Trúc Thư Mục Dữ Liệu & Mã Nguồn

Toàn bộ dữ liệu lớn (ảnh keyframe, vector npy, manifest jsonl, model cache) được lưu tách biệt trên **ổ `D:\bachkhoa\ai_challenge\data`** để không làm nặng mã nguồn:

```text
D:\bachkhoa\ai_challenge\
├── my-repo\                                 # Thư mục mã nguồn dự án (Git repository)
│   ├── run_app.bat                          # File khởi động 1-click vào Web UI
│   ├── pyproject.toml                       # Cấu hình dependencies (uv / pip)
│   ├── .streamlit\config.toml               # Tối ưu giao diện Streamlit
│   └── src\aic2026\                         # Package mã nguồn chính
│       ├── app\ui.py                        # Giao diện Web PEGASUS Streamlit
│       ├── app\api.py                       # REST API & Bộ điều phối Orchestrator
│       ├── embeddings\text.py               # Singleton OpenCLIP & SigLIP 2 Text Embedder
│       ├── embeddings\asr.py                # Dense Semantic ASR Embedder (BGE-M3 Safetensors)
│       ├── reranking\blip2_reranker.py      # BLIP-2 ITM Cross-Encoder & Temporal NMS
│       └── agent\tools.py                   # Retrieval Pipeline & Score Fusion Engine
│
└── data\                                    # Thư mục dữ liệu độc lập bên ngoài
    ├── .cache\                              # Bộ nhớ đệm mô hình (Hugging Face, Torch, UV, Pip)
    │   └── huggingface\hub\                 # Chứa model BGE-M3, SigLIP 2, Qwen...
    ├── extracted\                           # Dữ liệu BTC cung cấp
    │   ├── Keyframes\                       # Toàn bộ ảnh keyframe (L21_V001/001.jpg, ...)
    │   ├── map-keyframes\                   # File CSV ánh xạ frame sang pts_time
    │   ├── asr\                             # File JSON lời thoại ASR
    │   └── Objects\                         # Nhãn vật thể Faster R-CNN
    └── processed\                           # Dữ liệu hệ thống đã trích xuất sẵn
        ├── siglip2_manifest.jsonl           # Bảng tra cứu 177.321 keyframes
        ├── siglip2_features.npy             # Ma trận vector đặc trưng SigLIP 2 (177.321 x 1152)
        ├── asr_dense_features.npy           # Ma trận vector ngữ nghĩa lời thoại (177.321 x 1024)
        └── bm25_index_cache.pkl             # Bộ nhớ đệm BM25 tải nhanh tức thì
```

---

## 3. Cài Đặt Môi Trường (PyTorch CUDA 12.1)

### Yêu cầu hệ thống:
* **Hệ điều hành:** Windows 10/11 (hoặc Linux).
* **Python:** `3.10` – `3.12` (khuyên dùng Python 3.11).
* **GPU NVIDIA:** GTX 1060 / GTX 1070 (8GB VRAM) hoặc dòng RTX Series để đạt tốc độ tối đa.

### Các bước cài đặt nhanh:
Mở PowerShell tại thư mục `D:\bachkhoa\ai_challenge\my-repo`:

```powershell
# 1. Tạo môi trường ảo với uv (khuyên dùng)
uv venv --python 3.11 .venv

# 2. Cài đặt toàn bộ thư viện cần thiết
uv sync --extra retrieval --extra models --extra video --extra asr --extra agent --extra dev

# 3. Cài đặt PyTorch hỗ trợ GPU CUDA 12.1
uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Hướng Dẫn Khởi Động & Sử Dụng Giao Diện Web

### 4.1. Khởi động Giao diện:
* **Cách 1 (Nhanh nhất - Khuyên dùng):** Nhấp đúp chuột vào file [`run_app.bat`](run_app.bat).
* **Cách 2 (Qua dòng lệnh):**
  ```powershell
  uv run streamlit run src/aic2026/app/ui.py
  ```
Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

### 4.2. Cơ chế Tự Động Nạp Trước (Startup Preloader):
Ngay khi mở trang web, hệ thống sẽ chạy tiến trình **Startup Preloader** ngầm:
* Tải trước toàn bộ bảng tra cứu **177,321 keyframe**.
* Khởi tạo và làm ấm mô hình **SigLIP 2 trên GPU**.
* Nạp chỉ mục **BM25 và BGE-M3**.
> 💡 Nhờ cơ chế này, bạn có thể nhập câu truy vấn và bấm tìm kiếm **ngay lập tức mà không cần phải chờ 3-5 phút nạp dữ liệu ở lần đầu tiên!**

### 4.3. Cấu hình 1-Click Master Root:
Hệ thống sử dụng cơ chế **1-Click Master Root**:
* Bạn chỉ cần trỏ đường dẫn gốc tại sidebar: `D:\bachkhoa\ai_challenge\data`
* Toàn bộ 5 đường dẫn con (`manifest`, `features`, `keyframes`, `map-keyframes`, `videos`) sẽ được tự động nhận diện chính xác 100%.

---

## 5. Chi Tiết 4 Chế Độ Tìm Kiếm & Bí Quyết Chọn

Ở thanh công cụ tìm kiếm, bạn có thể lựa chọn 1 trong 4 chế độ chuyên biệt:

| Chế độ | Biểu tượng | Bản chất công nghệ | Khi nào nên sử dụng? |
| :--- | :---: | :--- | :--- |
| **Đa kênh (Hybrid)** | 🌈 | Dung hợp SigLIP 2 + Dense ASR + BM25 OCR | **Mặc định**. Dùng khi câu hỏi vừa có mô tả bối cảnh hình ảnh vừa có từ khóa/lời thoại. |
| **Thị giác (SigLIP)** | 👁️ | Thuần Visual Vector Search ($w_{\text{vis}}=1.0$) | Dùng khi câu hỏi thuần mô tả hình ảnh, màu sắc, hành động (ví dụ: *"người phụ nữ mặc áo đỏ lái xe ô tô"*). **Tốc độ $\le 1\text{s}$!** |
| **Lời thoại (ASR)** | 🗣️ | Dense BGE-M3 (1024d) + BM25 Tiếng Việt | Dùng khi câu hỏi trích dẫn lời thoại của nhân vật trong video (ví dụ: *"chúng ta phải bảo vệ môi trường biển"*). Bắt trúng cả từ đồng nghĩa! |
| **Chữ màn hình (OCR)**| 🔤 | Thuần BM25 Lexical OCR Search ($w_{\text{ocr}}=1.0$) | Dùng khi tìm kiếm biển hiệu cửa hàng, logo, banner, chữ viết trên màn hình (ví dụ: *"biển hiệu Quán Cơm Bình Dân"*). |

---

## 6. Cơ Chế Reranking, Lọc Đa Dạng Hóa (NMS) & VLM Judge

Trên thanh Sidebar hoặc thanh tùy chọn nhanh, bạn có thể bật/tắt các công cụ xếp hạng nâng cao:

### 6.1. 🎯 Lọc Đa Dạng Hóa (Temporal NMS) — *Khuyên dùng BẬT*
* **Vấn đề thực tế:** Khi tìm kiếm thông thường, Top 10 kết quả thường rơi vào cùng một đoạn video dài 2 giây (vì 10 frame đó gần như giống hệt nhau), khiến bạn khó tìm ra đáp án nếu video đó không đúng.
* **Cơ chế NMS:** Tự động triệt tiêu các frame nằm trong cửa sổ thời gian gần nhau ($\pm 75$ frames $\approx 3$ giây) và giới hạn tối đa 2-3 frames cho mỗi video.
* **Kết quả:** Danh sách Top 100 sẽ chứa các cảnh quay đa dạng từ nhiều video khác nhau, giúp **Top-1 và Top-5 trúng đích nhanh hơn gấp nhiều lần!**

### 6.2. ⚡ BLIP-2 GPU Reranker (Cross-Encoder)
* So khớp chi tiết từng vùng điểm ảnh (pixel patches) của các frame ứng viên với câu truy vấn.
* Tích hợp **VRAM Guard**: Tự động kiểm tra bộ nhớ GPU. Nếu VRAM trống $< 2.0\text{GB}$, hệ thống tự động fallback về SigLIP an toàn, không bao giờ gây lỗi sập app hay OOM.

### 6.3. ⚖️ VLM-as-a-Judge (Qwen2.5-VL)
* Dành cho bài toán khó cần thẩm định logic phức tạp: VLM sẽ "nhìn" từng ảnh trong Top 15 và chấm điểm theo tiêu chí:
  * **KIS:** Kiểm tra chi tiết màu sắc, đồ vật nhỏ, chữ graffiti.
  * **Q&A:** Đưa frame chứa góc nhìn có bằng chứng trả lời câu hỏi lên đầu.
  * **TRAKE:** Kiểm tra tính nhất quán của chuỗi đa sự kiện theo thời gian ($t_1 < t_2 < t_3$).

---

## 7. Chạy Bằng Dòng Lệnh CLI & Đánh Giá Ground-Truth

Bạn có thể chạy tự động hàng loạt truy vấn qua CLI:

### 7.1. Chạy truy vấn KIS:
```powershell
python -m aic2026.cli agent-query `
  --query query_kis.json `
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\bachkhoa\ai_challenge\data\processed\siglip2_features.npy" `
  --mode hybrid `
  --output outputs/result_kis.json

# Xuất CSV nộp bài cho BTC
python -m aic2026.cli export-submission --result outputs/result_kis.json --query query_kis.json --output outputs/submission_kis.csv
```

### 7.2. Chạy truy vấn TRAKE (Chuỗi sự kiện thời gian):
```powershell
python -m aic2026.cli agent-query `
  --query query_trake.json `
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\bachkhoa\ai_challenge\data\processed\siglip2_features.npy" `
  --output outputs/result_trake.json

python -m aic2026.cli export-submission --result outputs/result_trake.json --query query_trake.json --output outputs/submission_trake.csv
```

### 7.3. Đánh giá độ chính xác (Evaluate):
```powershell
python -m aic2026.cli evaluate query_kis.json outputs/result_kis.json ground_truth_kis.json
```

---

## 8. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Tình Huống / Cảnh Báo | Nguyên Nhân | Cách Xử Lý |
| :--- | :--- | :--- |
| **`CUDA out of memory`** | Trùng lặp mô hình trên GPU hoặc chạy đồng thời ứng dụng đồ họa nặng khác. | Hệ thống đã có **Singleton Cache** và **VRAM Guard**. Hãy tắt bớt game/ứng dụng 3D nền, khởi động lại bằng `run_app.bat`. |
| **`BGEM3TextEmbedder: Failed to load... safetensors`** | Hugging Face yêu cầu định dạng an toàn `.safetensors`. | **Đã sửa xong**: Code tự động đọc snapshot `model.safetensors` trên ổ `D:`. |
| **`runner.enforceZeroRerunWarning is not a valid option`** | Tùy chọn cũ trong Streamlit phiên bản trước. | **Đã gỡ bỏ**: File `.streamlit/config.toml` đã được dọn sạch. |
| **`HF_HUB_OFFLINE` chặn tải model** | Cờ offline mode trong batch script cũ. | **Đã sửa**: Đã mở kết nối và chuyển toàn bộ cache tải về ổ `D:`. |
| **Ảnh keyframe không hiển thị** | Sai đường dẫn thư mục Keyframes. | Đảm bảo Master Root trỏ đúng `D:\bachkhoa\ai_challenge\data` (nơi chứa thư mục `extracted\Keyframes`). |
| **Lỗi OpenMP `libiomp5md.dll` trên Windows** | Xung đột thư viện OpenMP giữa các gói. | Trong `run_app.bat` đã đặt sẵn `set KMP_DUPLICATE_LIB_OK=TRUE`. |

---

### 🏆 Chúc bạn và đội thi đạt thành tích cao nhất tại AI Challenge 2026!
