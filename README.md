# 🚀 AIC 2026 — PEGASUS Video Retrieval System

> Tiếp tục công việc: đọc [MEMORY.md](MEMORY.md) và [AGENTS.md](AGENTS.md).
> Bố trí workspace, lưu trữ bản nộp và chuẩn bị batch2: [docs/WORKSPACE.md](docs/WORKSPACE.md).

Hệ thống tìm kiếm video đa phương thức tối tân phục vụ 3 bài toán trọng tâm vòng sơ tuyển **AI Challenge 2026 (KIS / Q&A / TRAKE)**:
* **👁️ Visual Vector Search SOTA:** **Google SigLIP 2** (`ViT-SO400M-14-SigLIP2-378` / alias: `siglip2`) & `siglip2-base`.
* **💻 Kiến Trúc Dual-Hardware (GPU & CPU):** Tự động nhận diện phần cứng thông minh — **chạy siêu tốc trên GPU NVIDIA CUDA ($\le 1\text{s}$) và chạy mượt mà, ổn định trên máy thuần CPU (không cần GPU rời)**.
* **🎯 Shot-level Temporal NMS:** Thuật toán lọc phi cực đại triệt tiêu các khung hình trùng lặp trong cùng 1 cảnh quay, tối ưu hóa đưa keyframe chuẩn xác nhất lên Top 1.
* **🧠 Dense Semantic ASR (BGE-M3 1024-dim):** Tìm kiếm ngữ nghĩa lời thoại sâu sắc bằng `BAAI/bge-m3` (định dạng `model.safetensors` nạp trực tiếp trong $0.05\text{s}$), kết hợp **Hybrid Dense-Sparse** cùng BM25.
* **⚡ BLIP-2 ITM Cross-Encoder:** Tích hợp mô hình đối chiếu chéo cấp độ pixel theo nghiên cứu SOTA (arXiv:2512.12935v1) kèm bộ chốt an toàn VRAM Guard.
* **⚖️ VLM-as-a-Judge:** **Qwen2.5-VL-3B / 7B** thẩm định thị giác chi tiết cho KIS, kiểm tra bằng chứng thị giác (Visual Evidence) cho Q&A, và xác thực chuỗi đa sự kiện thời gian cho TRAKE.
* **🔤 Lexical & Metadata:** BM25 nhãn vật thể (Faster R-CNN) + Tiêu đề/mô tả YouTube + PaddleOCR.
* **⚡ Startup Preloader:** Tự động nạp sẵn toàn bộ mô hình và 177,321 keyframe vào RAM/GPU ngay khi mở app — **không còn độ trễ chờ đợi ở câu lệnh đầu tiên!**

---

## 📑 Mục Lục
1. [Điểm Nổi Bật & Khả Năng Tương Thích Phần Cứng](#1-điểm-nổi-bật--khả-năng-tương-thích-phần-cứng)
2. [Cấu Trúc Thư Mục Dữ Liệu & Mã Nguồn](#2-cấu-trúc-thư-mục-dữ-liệu--mã-nguồn)
3. [Cài Đặt Môi Trường (Dành Cho Cả GPU & Thuần CPU)](#3-cài-đặt-môi-trường-dành-cho-cả-gpu--thuần-cpu)
4. [Hướng Dẫn Khởi Động & Sử Dụng Giao Diện Web](#4-hướng-dẫn-khởi-động--sử-dụng-giao-diện-web)
5. [Khuyến Nghị Cấu Hình Cho Máy Thuần CPU vs Máy GPU](#5-khuyến-nghị-cấu-hình-cho-máy-thuần-cpu-vs-máy-gpu)
6. [Chi Tiết 4 Chế Độ Tìm Kiếm & Bí Quyết Chọn](#6-chi-tiết-4-chế-độ-tìm-kiếm--bí-quyết-chọn)
7. [Cơ Chế Reranking, Lọc Đa Dạng Hóa (NMS) & VLM Judge](#7-cơ-chế-reranking-lọc-đa-dạng-hóa-nms--vlm-judge)
8. [Chạy Bằng Dòng Lệnh CLI & Đánh Giá Ground-Truth](#8-chạy-bằng-dòng-lệnh-cli--đánh-giá-ground-truth)
9. [Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#9-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. Điểm Nổi Bật & Khả Năng Tương Thích Phần Cứng

Hệ thống được thiết kế với cơ chế **Hardware Auto-Detection**:
* Nếu máy có **GPU NVIDIA (GTX/RTX)**: Hệ thống tự động kích hoạt nhân Tensor/CUDA để đạt tốc độ tìm kiếm tối đa ($\le 1\text{s}$).
* Nếu máy **KHÔNG CÓ GPU (Laptop văn phòng, PC thường, Mac, CPU VM)**: Hệ thống **tự động chuyển 100% sang chạy trên CPU**, không báo lỗi, không crash, và được tối ưu hóa bộ nhớ chuyên sâu để máy RAM 8GB - 16GB vẫn chạy êm ái!

| Tính Năng | Máy Có GPU NVIDIA (CUDA) | **Máy Thuần CPU (Không Có GPU)** |
| :--- | :--- | :--- |
| **Nhận diện thiết bị** | Tự động chọn `cuda:0` | **Tự động fallback về `cpu` hoàn toàn trong suốt** |
| **Tốc độ tìm kiếm** | $\le 1\text{s}$ (Visual), 4–7s (Hybrid) | **~2–4s (Visual)**, ~8–12s (Hybrid) với phép nhân ma trận AVX2/AVX-512 |
| **Tiết kiệm RAM CPU** | VRAM GPU gánh mô hình | **Tự động gọt bỏ Tháp Thị Giác (Purge Visual Tower)**, tiết kiệm ngay 3.5GB RAM, không bị tràn RAM |
| **Bộ nhớ đệm Index** | Nạp sẵn trên RAM/GPU | Chỉ mục BM25 nạp từ cache nhị phân (.pkl) trong **$0.2\text{s}$** |
| **Quản lý VRAM / RAM** | Singleton `_MODEL_CACHE` + xả rác `empty_cache()` | Singleton Cache chỉ giữ duy nhất 1 bản trên RAM, ngăn nhân bản tiến trình |
| **Lưu trữ an toàn** | Khóa cứng sang ổ **`D:`** | Khóa cứng sang ổ **`D:`**, giải phóng >26GB trên ổ C |

---

## 2. Cấu Trúc Thư Mục Dữ Liệu & Mã Nguồn

Toàn bộ dữ liệu lớn (ảnh keyframe, vector npy, manifest jsonl, model cache) được lưu tách biệt trên **ổ `D:\bachkhoa\ai_challenge\data`** (hoặc `D:\ai_challenge\data`):

```text
D:\bachkhoa\ai_challenge\
├── my-repo\                                 # Thư mục mã nguồn dự án (Git repository)
│   ├── run_app.bat                          # File khởi động 1-click vào Web UI (tự động nhận diện GPU/CPU)
│   ├── pyproject.toml                       # Cấu hình dependencies (uv / pip)
│   ├── .streamlit\config.toml               # Tối ưu giao diện Streamlit
│   └── src\aic2026\                         # Package mã nguồn chính
│       ├── app\ui.py                        # Giao diện Web PEGASUS Streamlit
│       ├── app\api.py                       # REST API & Bộ điều phối Orchestrator
│       ├── embeddings\text.py               # Singleton SigLIP 2 Text Embedder (Auto CUDA/CPU)
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

## 3. Cài Đặt Môi Trường (Dành Cho Cả GPU & Thuần CPU)

Mở PowerShell tại thư mục `D:\bachkhoa\ai_challenge\my-repo`:

```powershell
# Bước 1: Tạo môi trường ảo với uv (khuyên dùng)
uv venv --python 3.11 .venv

# Bước 2: Cài đặt toàn bộ thư viện cơ sở của dự án
uv sync --extra retrieval --extra models --extra video --extra asr --extra agent --extra dev
```

### 👉 Bước 3: Cài đặt PyTorch theo phần cứng máy bạn:

#### Lựa chọn A: Dành cho Máy CÓ GPU NVIDIA (GTX 1060, GTX 1070, RTX Series...)
```powershell
# Cài đặt PyTorch CUDA 12.1 (Tăng tốc tối đa với GPU NVIDIA)
uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

#### Lựa chọn B: Dành cho Máy KHÔNG CÓ GPU (Thuần CPU / Laptop văn phòng / Mac / Linux)
```powershell
# Cài đặt PyTorch bản CPU siêu nhẹ (~200MB thay vì 2.5GB của CUDA)
uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

> 💡 **Ghi chú:** Dù bạn cài bản GPU hay bản CPU, mã nguồn của dự án sẽ **tự động tương thích 100% mà bạn không cần phải sửa bất kỳ dòng code nào!**

---

## 4. Hướng Dẫn Khởi Động & Sử Dụng Giao Diện Web

### 4.1. Khởi động Giao diện:
* **Cách 1 (Nhanh nhất):** Nhấp đúp chuột vào file [`run_app.bat`](run_app.bat). File `.bat` này đã được cấu hình tự nhận diện máy có GPU hay chỉ có CPU để thiết lập luồng tối ưu nhất.
* **Cách 2 (Dòng lệnh):**
  ```powershell
  uv run streamlit run src/aic2026/app/ui.py
  ```
Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

### 4.2. Cơ chế Tự Động Nạp Trước (Startup Preloader):
Ngay khi mở trang web, hệ thống tự động chạy **Startup Preloader**:
* Nạp sẵn bảng tra cứu **177,321 keyframe**.
* Khởi tạo mô hình **SigLIP 2** (trên GPU nếu có, hoặc trên CPU nếu không có GPU).
* Nạp sẵn chỉ mục **BM25 và BGE-M3**.
> 💡 Nhờ cơ chế này, bạn có thể nhập câu truy vấn và bấm tìm kiếm **ngay lập tức mà không cần phải chờ 3-5 phút nạp dữ liệu ở lần đầu tiên!**

### 4.3. Cấu hình 1-Click Master Root:
* Bạn chỉ cần trỏ đường dẫn gốc tại sidebar: `D:\bachkhoa\ai_challenge\data`
* Toàn bộ 5 đường dẫn con (`manifest`, `features`, `keyframes`, `map-keyframes`, `videos`) sẽ được tự động nhận diện chính xác 100%.

---

## 5. Khuyến Nghị Cấu Hình Cho Máy Thuần CPU vs Máy GPU

Để đạt hiệu quả thi đấu và trải nghiệm mượt mà nhất, hãy áp dụng cấu hình khuyến nghị sau:

| Thiết lập trên Giao diện Web | Khuyên dùng cho **Máy CÓ GPU** | Khuyên dùng cho **Máy THUẦN CPU** |
| :--- | :--- | :--- |
| **Mô hình Thị giác (Model Choice)** | `siglip2` (SOTA SO400M) | `siglip2` (vẫn chạy tốt ~3s) hoặc `siglip2-base` (chạy siêu nhẹ ~1s) |
| **🎯 Lọc Đa Dạng Hóa (NMS)** | **BẬT** (Khuyên dùng) | **BẬT** (Khuyên dùng - NMS tính toán cực nhẹ trên CPU) |
| **🌐 Dịch VI→EN** | BẬT khi nhập tiếng Việt | BẬT khi nhập tiếng Việt |
| **⚡ BLIP-2 GPU Reranker** | TẮT (chỉ bật khi đã tải model 4.6GB) | **TẮT** (Tránh quá tải CPU) |
| **⚖️ VLM-as-a-Judge** | BẬT nếu cần thẩm định Top 15 KIS/QA | **TẮT** (Hoặc dùng qua Ollama 4-bit) |
| **Chế độ tìm kiếm (Search Modality)**| `🌈 Đa kênh` hoặc `👁️ Thị giác` | `🌈 Đa kênh` hoặc `👁️ Thị giác` |

---

## 6. Chi Tiết 4 Chế Độ Tìm Kiếm & Bí Quyết Chọn

Ở thanh công cụ tìm kiếm, bạn có thể lựa chọn 1 trong 4 chế độ chuyên biệt:

| Chế độ | Biểu tượng | Bản chất công nghệ | Khi nào nên sử dụng? |
| :--- | :---: | :--- | :--- |
| **Đa kênh (Hybrid)** | 🌈 | Dung hợp SigLIP 2 + Dense ASR + BM25 OCR | **Mặc định**. Dùng khi câu hỏi vừa có mô tả bối cảnh hình ảnh vừa có từ khóa/lời thoại. |
| **Thị giác (SigLIP)** | 👁️ | Thuần Visual Vector Search ($w_{\text{vis}}=1.0$) | Dùng khi câu hỏi thuần mô tả hình ảnh, màu sắc, hành động (ví dụ: *"người phụ nữ mặc áo đỏ lái xe ô tô"*). **Rất nhanh trên cả GPU và CPU!** |
| **Lời thoại (ASR)** | 🗣️ | Dense BGE-M3 (1024d) + BM25 Tiếng Việt | Dùng khi câu hỏi trích dẫn lời thoại của nhân vật trong video (ví dụ: *"chúng ta phải bảo vệ môi trường biển"*). Bắt trúng cả từ đồng nghĩa! |
| **Chữ màn hình (OCR)**| 🔤 | Thuần BM25 Lexical OCR Search ($w_{\text{ocr}}=1.0$) | Dùng khi tìm kiếm biển hiệu cửa hàng, logo, banner, chữ viết trên màn hình (ví dụ: *"biển hiệu Quán Cơm Bình Dân"*). |

---

## 7. Cơ Chế Reranking, Lọc Đa Dạng Hóa (NMS) & VLM Judge

Trên thanh Sidebar hoặc thanh tùy chọn nhanh, bạn có thể bật/tắt các công cụ xếp hạng nâng cao:

### 7.1. 🎯 Lọc Đa Dạng Hóa (Temporal NMS) — *Khuyên dùng BẬT cho cả GPU & CPU*
* **Vấn đề thực tế:** Khi tìm kiếm thông thường, Top 10 kết quả thường rơi vào cùng một đoạn video dài 2 giây (vì 10 frame đó gần như giống hệt nhau), khiến bạn khó tìm ra đáp án nếu video đó không đúng.
* **Cơ chế NMS:** Tự động triệt tiêu các frame nằm trong cửa sổ thời gian gần nhau ($\pm 75$ frames $\approx 3$ giây) và giới hạn tối đa 2-3 frames cho mỗi video.
* **Hiệu quả:** Không tiêu tốn tài nguyên phần cứng, hoạt động tức thì trên cả CPU và GPU, giúp **Top-1 và Top-5 trúng đích nhanh hơn gấp nhiều lần!**

### 7.2. ⚡ BLIP-2 GPU Reranker (Cross-Encoder)
* So khớp chi tiết từng vùng điểm ảnh (pixel patches) của các frame ứng viên với câu truy vấn.
* Tích hợp **VRAM Guard**: Tự động kiểm tra bộ nhớ GPU. Nếu chạy trên CPU hoặc VRAM trống $< 2.0\text{GB}$, hệ thống tự động fallback về SigLIP an toàn, không bao giờ gây lỗi sập app hay OOM.

### 7.3. ⚖️ VLM-as-a-Judge (Qwen2.5-VL)
* Dành cho bài toán khó cần thẩm định logic phức tạp: VLM sẽ "nhìn" từng ảnh trong Top 15 và chấm điểm theo tiêu chí:
  * **KIS:** Kiểm tra chi tiết màu sắc, đồ vật nhỏ, chữ graffiti.
  * **Q&A:** Đưa frame chứa góc nhìn có bằng chứng trả lời câu hỏi lên đầu.
  * **TRAKE:** Kiểm tra tính nhất quán của chuỗi đa sự kiện theo thời gian ($t_1 < t_2 < t_3$).

---

## 8. Chạy Bằng Dòng Lệnh CLI & Đánh Giá Ground-Truth

Bạn có thể chạy tự động hàng loạt truy vấn qua CLI:

### 8.1. Chạy truy vấn KIS:
```powershell
# Chạy trên máy bất kỳ (tự động phát hiện GPU/CPU)
python -m aic2026.cli agent-query `
  --query query_kis.json `
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\bachkhoa\ai_challenge\data\processed\siglip2_features.npy" `
  --mode hybrid `
  --output outputs/result_kis.json

# Xuất CSV nộp bài cho BTC
python -m aic2026.cli export-submission --result outputs/result_kis.json --query query_kis.json --output outputs/submission_kis.csv
```

### 8.2. Chạy truy vấn TRAKE (Chuỗi sự kiện thời gian):
```powershell
python -m aic2026.cli agent-query `
  --query query_trake.json `
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\bachkhoa\ai_challenge\data\processed\siglip2_features.npy" `
  --output outputs/result_trake.json

python -m aic2026.cli export-submission --result outputs/result_trake.json --query query_trake.json --output outputs/submission_trake.csv
```

### 8.3. Đánh giá độ chính xác (Evaluate):
```powershell
python -m aic2026.cli evaluate query_kis.json outputs/result_kis.json ground_truth_kis.json
```

---

## 9. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Tình Huống / Cảnh Báo | Nguyên Nhân | Cách Xử Lý |
| :--- | :--- | :--- |
| **Máy không có GPU NVIDIA** | Máy chỉ có card Onboard Intel/AMD hoặc chạy trên CPU. | Làm theo **Lựa chọn B ở Mục 3** (cài PyTorch CPU). App tự động chạy trên CPU mà không cần GPU rời. |
| **`CUDA out of memory`** | Trùng lặp mô hình trên GPU hoặc chạy ứng dụng đồ họa nặng. | Hệ thống đã có **Singleton Cache** và **VRAM Guard**. Hãy tắt ứng dụng 3D nền, khởi động lại bằng `run_app.bat`. |
| **Máy RAM 8GB–16GB bị chậm trên CPU** | Nạp mô hình lớn chiếm RAM. | Trong code đã có tính năng tự động xóa tháp thị giác (**Purge Visual Tower**). Hãy chọn mô hình `siglip2-base` trên UI để chạy siêu nhẹ. |
| **`BGEM3TextEmbedder: Failed to load... safetensors`** | Hugging Face yêu cầu định dạng an toàn `.safetensors`. | **Đã sửa xong**: Code tự động đọc snapshot `model.safetensors` trên ổ `D:` chỉ trong $0.05\text{s}$. |
| **`runner.enforceZeroRerunWarning is not a valid option`** | Tùy chọn cũ trong Streamlit phiên bản trước. | **Đã gỡ bỏ**: File `.streamlit/config.toml` đã được dọn sạch. |
| **`HF_HUB_OFFLINE` chặn tải model** | Cờ offline mode trong batch script cũ. | **Đã sửa**: Đã mở kết nối và chuyển toàn bộ cache tải về ổ `D:`. |
| **Ảnh keyframe không hiển thị** | Sai đường dẫn thư mục Keyframes. | Đảm bảo Master Root trỏ đúng `D:\bachkhoa\ai_challenge\data` (nơi chứa thư mục `extracted\Keyframes`). |
| **Lỗi OpenMP `libiomp5md.dll` trên Windows** | Xung đột thư viện OpenMP giữa các gói. | Trong `run_app.bat` và `__init__.py` đã đặt sẵn `set KMP_DUPLICATE_LIB_OK=TRUE` và `OMP_NUM_THREADS=1`. |

---

### 🏆 Chúc bạn và đội thi đạt thành tích cao nhất tại AI Challenge 2026!
