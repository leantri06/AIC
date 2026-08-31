# 🚀 AIC 2026 — PEGASUS Video Retrieval System

Hệ thống tìm kiếm video đa phương thức tối tân phục vụ 3 bài toán trọng tâm vòng sơ tuyển **AI Challenge 2026 (KIS / Q&A / TRAKE)**:
* **👁️ Visual Vector Search SOTA:** **Google SigLIP 2** (`ViT-SO400M-14-SigLIP2-378` / alias: `siglip2`) & SigLIP 1 (`ViT-SO400M-14-SigLIP-384`).
* **⚖️ VLM-as-a-Judge:** **Qwen2.5-VL-3B / 7B** thẩm định thị giác chính xác cho KIS, kiểm tra bằng chứng thị giác (Visual Evidence) cho Q&A, và xác thực chuỗi đa sự kiện thời gian cho TRAKE.
* **🧠 Dense Semantic ASR (BGE-M3 1024-dim):** Tìm kiếm ngữ nghĩa lời thoại sâu sắc bằng `BAAI/bge-m3`, kết hợp **Hybrid Dense-Sparse** cùng BM25 để bắt trúng $100\%$ từ đồng nghĩa và diễn giải gián tiếp.
* **🔤 Lexical & Metadata:** BM25 nhãn vật thể (Faster R-CNN) + Tiêu đề/mô tả YouTube + PaddleOCR.
* **🌈 Reranking & Filtering:** Hybrid SRRF (Score-Reflected Reciprocal Rank Fusion) + VLM Judge Rerank + Bộ lọc màu sắc.
* **⏱️ Chuỗi Hành Động TRAKE:** Căn chỉnh chuỗi sự kiện theo thứ tự thời gian bằng Dynamic Programming Monotonic Alignment + VLM Sequence Verifier.

---

## 📑 Mục lục
1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Cấu trúc thư mục Dữ liệu tách riêng](#2-cấu-trúc-thư-mục-dữ-liệu-tách-riêng)
3. [Chuẩn bị dữ liệu & Xây dựng Index](#3-chuẩn-bị-dữ-liệu--xây-dựng-index)
4. [Khởi động Giao diện PEGASUS Web & Cấu hình đường dẫn](#4-khởi-động-giao-diện-pegasus-web--cấu-hình-đường-dẫn)
5. [Chi tiết Tính năng VLM-as-a-Judge & Dense Semantic ASR](#5-chi-tiết-tính-năng-vlm-as-a-judge--dense-semantic-asr)
6. [Hướng dẫn Tìm kiếm & Xuất bài nộp BTC](#6-hướng-dẫn-tìm-kiếm--xuất-bài-nộp-btc)
7. [Chạy bằng CLI & Đánh giá (Evaluate)](#7-chạy-bằng-cli--đánh-giá-evaluate)
8. [Xử lý sự cố thường gặp (Troubleshooting)](#8-xử-lý-sự-cố-thường-gặp-troubleshooting)
9. [Chi tiết Công nghệ Hiện tại & Đề xuất Nâng cấp Tiên tiến](#9-chi-tiết-công-nghệ-hiện-tại--đề-xuất-nâng-cấp-tiên-tiến)

---

## 1. Cài đặt môi trường

### Yêu cầu tiên quyết:
* **Hệ điều hành:** Windows 10/11 hoặc Linux / macOS.
* **Python:** `3.10` – `3.12` (khuyên dùng Python 3.11).
* **GPU NVIDIA (Khuyên dùng):** RTX Series (ví dụ: RTX 4070 12GB VRAM) hoặc GTX 1060 trở lên để tăng tốc trích xuất và tìm kiếm.

### Các bước cài đặt:

Mở **PowerShell / Terminal** tại thư mục gốc dự án:

```powershell
# 1. Tạo môi trường ảo (chỉ chạy lần đầu)
uv venv --python 3.11 .venv
# Hoặc dùng python chuẩn: python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# 3. Cài đặt toàn bộ dependencies của dự án bằng uv (khuyên dùng)
uv sync --extra retrieval --extra models --extra video --extra asr --extra agent --extra dev
# Hoặc bằng pip:
# pip install -e ".[retrieval,models,video,asr,agent,dev]"

# 4. Cài đặt PyTorch CUDA cho GPU NVIDIA (khuyên dùng CUDA 12.4)
uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
```

> **(Tùy chọn) Cài đặt Ollama nếu muốn chạy Qwen2.5-VL 7B bản lượng tử hóa 4-bit:**
> ```powershell
> ollama pull qwen2.5vl:7b
> ollama serve
> ```

---

## 2. Cấu trúc thư mục Dữ liệu tách riêng

> [!NOTE]
> **Toàn bộ thư mục dữ liệu (`extracted` và `processed`) được lưu riêng biệt ngoài mã nguồn `AIC`** tại `D:\ai_challenge\data\` để tránh nặng repo Git và tiện quản lý.

Sơ đồ cấu trúc thư mục data bên ngoài:

```text
D:\ai_challenge\data\ (Thư mục dữ liệu bên ngoài)
├── extracted/                            # Dữ liệu giải nén do BTC cung cấp
│   ├── Keyframes/                        # Thư mục ảnh keyframe (L21_V001/001.jpg, ...)
│   ├── map-keyframes/                    # File CSV ánh xạ frame_idx sang timestamp pts_time
│   ├── asr/                              # File JSON lời thoại ASR từng video (L21_V001.json, ...)
│   ├── Objects/                          # File JSON nhãn vật thể Faster R-CNN
│   ├── Metadata/                         # File JSON thông tin video YouTube
│   ├── clip_features/                    # (Tùy chọn) File .npy CLIP gốc của BTC
│   └── Videos/                           # (Tùy chọn) File video .mp4 gốc
└── processed/                            # Dữ liệu index đã xử lý (đầu ra của hệ thống)
    ├── siglip2_manifest.jsonl            # Manifest SigLIP 2 (vector_id, frame_id, labels, asr_text)
    ├── siglip2_features.npy              # Ma trận vector SigLIP 2 (177.321 x 1152)
    ├── asr_dense_features.npy            # Ma trận vector ngữ nghĩa lời thoại BGE-M3 (177.321 x 1024)
    ├── siglip_manifest.jsonl             # Manifest SigLIP 1
    ├── siglip_features.npy               # Ma trận vector SigLIP 1
    ├── derived_manifest.jsonl            # Manifest dự phòng
    ├── derived_features.npy              # Ma trận vector dự phòng
    └── colour_features.jsonl             # Đặc trưng màu sắc (tùy chọn)
```

---

## 3. Chuẩn bị dữ liệu & Xây dựng Index

### 3.1. Tự Encode Keyframes bằng GPU (Mô hình Google SigLIP 2 mới nhất)
```powershell
python -m aic2026.cli embed-keyframes `
  --keyframes-dir "D:\ai_challenge\data\extracted\Keyframes" `
  --map-keyframes-dir "D:\ai_challenge\data\extracted\map-keyframes" `
  --features-dir "D:\ai_challenge\data\processed\siglip2_features_dir" `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\ai_challenge\data\processed\siglip2_features.npy" `
  --clip-model siglip2 `
  --device cuda `
  --batch-size 64
```
*(Hỗ trợ resume tự động nếu bị ngắt quãng).*

### 3.2. Ánh xạ lời thoại ASR vào Manifest
```powershell
python -m aic2026.cli asr-manifest `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --asr-dir "D:\ai_challenge\data\extracted\asr" `
  --map-keyframes-dir "D:\ai_challenge\data\extracted\map-keyframes"
```

### 3.3. Mã hóa Ngữ nghĩa Lời thoại Dense Semantic ASR (BGE-M3 1024-dim)
```powershell
python -m aic2026.cli embed-asr `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --output "D:\ai_challenge\data\processed\asr_dense_features.npy" `
  --model-name "BAAI/bge-m3" `
  --device cuda `
  --batch-size 128
```
*(Chỉ mất ~2-3 phút trên GPU RTX 4070 để mã hóa toàn bộ 177.321 keyframe, xuất ra file `asr_dense_features.npy` ~360 MB).*

### 3.4. Tiền trích xuất đặc trưng màu sắc (Tùy chọn)
```powershell
python -m aic2026.cli compute-colour-features `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --raw-dir "D:\ai_challenge\data\extracted" `
  --output "D:\ai_challenge\data\processed\colour_features.jsonl"
```

### 3.5. Kiểm tra tính đồng bộ của Index
```powershell
python -m aic2026.cli check-features "D:\ai_challenge\data\processed\siglip2_features.npy" --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl"
```

---

## 4. Khởi động Giao diện PEGASUS Web & Cấu hình đường dẫn

### Khởi động Web UI:
* **Cách 1 (Nhanh nhất):** Nhấp đúp chuột vào file [`run_app.bat`](file:///d:/ai_challenge/AIC/run_app.bat).
* **Cách 2 (Terminal):** `streamlit run src/aic2026/app/ui.py`
* **Cách 3 (Backend API + UI):** Terminal 1: `python -m aic2026.cli serve` | Terminal 2: `streamlit run src/aic2026/app/ui.py`

### ⚙️ Đường dẫn Data mặc định tự động nhận diện:
Giao diện đã được cấu hình tự động nhận diện chính xác các đường dẫn trên máy bạn:
* **Manifest Path:** `D:\ai_challenge\data\processed\siglip2_manifest.jsonl`
* **Features Path:** `D:\ai_challenge\data\processed\siglip2_features.npy`
* **Dense ASR Features Path:** `D:\ai_challenge\data\processed\asr_dense_features.npy` (mã hóa ngữ nghĩa BGE-M3)
* **Keyframes Dir:** `D:\ai_challenge\data\extracted\Keyframes`
* **Map Keyframes Dir:** `D:\ai_challenge\data\extracted\map-keyframes`
* **Videos Dir:** `D:\ai_challenge\data\extracted\Videos` (nếu có)

---

## 5. Chi Tiết Tính Năng VLM-as-a-Judge & Dense Semantic ASR

### 5.1. VLM-as-a-Judge (Qwen2.5-VL)
Hệ thống tích hợp **Qwen2.5-VL-3B-Instruct** (hoặc 7B) làm "Trọng tài thị giác" (Visual Verifier & Reranker) cho cả 3 bài toán:

* **🔍 KIS (Known Item Search)**: VLM "nhìn" từng frame trong Top 15, phân tích chi tiết điều kiện (màu sắc, vật thể, hành động, graffiti, tranh vẽ...) và chấm điểm 0–100 $\rightarrow$ Đưa frame đúng nhất lên Top 1.
* **🎯 Q&A (Visual Evidence Verifier)**: Thẩm định xem ảnh ứng viên có chứa **bằng chứng trực tiếp** để trả lời câu hỏi hay không $\rightarrow$ Đưa frame có đáp án lên Top 1 để bạn tự quan sát và gõ câu trả lời vào bài nộp (tránh AI hallucinate).
* **⏱️ TRAKE (Temporal Sequence Verifier)**: Thẩm định chuỗi đa ảnh $(f_1 \to f_2 \to \dots \to f_N)$, kiểm tra từng frame có khớp với từng event trong chuỗi thời gian hay không.

#### ⚙️ Cách Kích Hoạt VLM Judge:
* **Trên Web UI**: Mở **Cấu hình nâng cao** (Sidebar) $\rightarrow$ Tích chọn `[x] ⚖️ Kích hoạt VLM-as-a-Judge (Qwen2.5-VL)`. Mặc định sử dụng model `Qwen/Qwen2.5-VL-3B-Instruct` (cực nhanh ~0.3s/ảnh, tối ưu hoàn hảo cho 12GB VRAM RTX 4070).
* **Trên CLI**: Thêm cờ `--vlm-judge --vlm-judge-model "Qwen/Qwen2.5-VL-3B-Instruct" --vlm-judge-top-k 15`.

### 5.2. Dense Semantic Speech Retrieval (BGE-M3 1024-dim)
* Sử dụng mô hình **`BAAI/bge-m3`** mã hóa toàn bộ câu thoại ASR thành vector ngữ nghĩa 1024 chiều.
* Tự động kết hợp **Hybrid Dense-Sparse**:
  $$\text{Score}_{\text{ASR}} = 0.65 \cdot \text{Dense}_{\text{BGE-M3}} + 0.35 \cdot \text{Sparse}_{\text{BM25}}$$
* Bắt trúng câu thoại kể cả khi query dùng từ đồng nghĩa hoặc diễn giải gián tiếp (ví dụ: *"hoàn cảnh nghèo khó"* $\leftrightarrow$ *"mẹ mất sớm, không có tiền đóng học"*).

---

## 6. Hướng dẫn Tìm kiếm & Xuất bài nộp BTC

### 6.1. Bốn Chế độ Tìm kiếm Chuyên biệt (Search Modality):
Hệ thống cho phép bạn linh hoạt chọn kênh tìm kiếm phù hợp với từng dạng câu hỏi:

| Chế độ | Ý nghĩa | Khi nào nên chọn? |
| :--- | :--- | :--- |
| 🌈 **Đa kênh (Hybrid)** | Dung hợp SigLIP 2 + Dense ASR + BM25 OCR | Mặc định, dùng khi câu hỏi vừa có mô tả hình ảnh vừa có từ khóa/lời thoại. |
| 👁️ **Thị giác (SigLIP)** | Thuần Visual Vector Search ($w_{\text{vis}}=1.0$) | Dùng khi câu hỏi thuần mô tả thị giác/hình ảnh (ví dụ: *"người phụ nữ mặc áo đỏ lái xe"*). |
| 🗣️ **Lời thoại (ASR)** | Thuần Dense & Sparse Speech Search ($w_{\text{asr}}=1.0$) | Dùng khi câu hỏi trích dẫn câu thoại của nhân vật (ví dụ: *"bản tin thông báo thời tiết"*). Giữ nguyên tiếng Việt 100%, tìm kiếm tức thì. |
| 🔤 **Chữ màn hình (OCR)** | Thuần OCR Text Search ($w_{\text{ocr}}=1.0$) | Dùng khi câu hỏi tìm biển báo, banner, logo, chữ trên màn hình (ví dụ: *"biển hiệu Phở 24"*). |

### 6.2. Các luồng bài toán:
1. **Tìm kiếm KIS (Known-Item Search):** Tìm 1 khung hình chính xác theo mô tả.
2. **Tìm kiếm Q&A:** Nhập ngữ cảnh và câu hỏi, VLM Judge đưa frame chứa đáp án lên đầu.
3. **Tìm kiếm TRAKE:** Tìm chuỗi sự kiện theo thứ tự thời gian.

### 6.3. Chọn kết quả & Xuất file nộp bài:
1. Click trực tiếp lên các card ảnh kết quả để chọn đáp án đúng.
2. Các ảnh được chọn sẽ có khung viền màu hồng đánh dấu thứ tự click.
3. Bấm nút **"Xuất CSV nộp bài"** để tải file CSV đúng định dạng nộp cho BTC.

---

## 7. Chạy bằng CLI & Đánh giá (Evaluate)

### 7.1. Chạy truy vấn KIS bằng CLI:
Tạo file `query_kis.json`:
```json
{
  "query_id": "kis_001",
  "type": "kis",
  "text": "bản tin thời sự sạt lở bờ sông"
}
```

Thực thi tìm kiếm (`--mode` có thể chọn `hybrid`, `siglip`, `asr`, `ocr`):
```powershell
# 1. Tìm kiếm -> JSON
python -m aic2026.cli agent-query `
  --query query_kis.json `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\ai_challenge\data\processed\siglip2_features.npy" `
  --mode asr `
  --output outputs/result_kis.json

# 2. Xuất CSV nộp bài
python -m aic2026.cli export-submission --result outputs/result_kis.json --query query_kis.json --output outputs/submission_kis.csv
```

### 7.2. Chạy truy vấn TRAKE bằng CLI:
Tạo file `query_trake.json`:
```json
{
  "query_id": "trake_001",
  "type": "trake",
  "text": "Một người mở tủ lạnh, lấy chai nước và uống",
  "events": [
    "A person opens the refrigerator door",
    "Taking out a bottle of water",
    "Drinking water from the bottle"
  ]
}
```

Xuất kết quả TRAKE (kết quả CSV chứa `video_id,f1,f2,f3` theo thứ tự các event):
```powershell
python -m aic2026.cli agent-query `
  --query query_trake.json `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\ai_challenge\data\processed\siglip2_features.npy" `
  --output outputs/result_trake.json

python -m aic2026.cli export-submission --result outputs/result_trake.json --query query_trake.json --output outputs/submission_trake.csv
```

### 7.3. Đánh giá Ground-Truth (Evaluate):
```powershell
# Đánh giá R@k và Final Score
python -m aic2026.cli evaluate query_kis.json outputs/result_kis.json ground_truth_kis.json
```

---

## 8. Xử lý sự cố thường gặp (Troubleshooting)

| Vấn đề / Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **`activate.ps1 cannot be loaded...`** | PowerShell chặn script thực thi. | Chạy lệnh: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` |
| **`CUDA Available: False`** | Chưa cài bản PyTorch có CUDA. | Chạy: `uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124` |
| **Lỗi OpenMP `libiomp5md.dll`** | Trùng thư viện runtime giữa PyTorch và các gói khác trên Windows. | Trong PowerShell chạy: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` trước khi chạy lệnh. |
| **Ảnh không hiển thị trên Web UI** | Đường dẫn `Keyframes Dir` trỏ chưa đúng thư mục data bên ngoài. | Vào mục **Cấu hình nâng cao** trên Sidebar của Web UI, kiểm tra lại đường dẫn `Keyframes Dir` (`D:\ai_challenge\data\extracted\Keyframes`). |
| **Lỗi Access Violation `-1073741819`** | RAM hệ thống bị tràn khi nạp model 7B unquantized. | Dùng model `Qwen/Qwen2.5-VL-3B-Instruct` (mặc định) hoặc chạy 7B qua Ollama 4-bit (`ollama run qwen2.5vl:7b`). |
| **Chạy kiểm thử toàn bộ hệ thống** | Kiểm tra tính ổn định của mã nguồn. | Chạy lệnh: `pytest` (Toàn bộ 152 tests sẽ pass). |

---

## 9. Chi tiết Công nghệ Hiện tại & Đề xuất Nâng cấp Tiên tiến

### 9.1. Bảng Tổng quan Công nghệ theo từng Chế độ

| Chế độ / Nhiệm vụ | 🛠️ Công nghệ Hiện tại (Current PEGASUS) | 🚀 Đề xuất Nâng cấp Tiên tiến (SOTA Upgrades) |
| :--- | :--- | :--- |
| **👁️ Thị giác (SigLIP 2 & Vector Search)** | • **Google SigLIP 2** `ViT-SO400M-14-SigLIP2-378` (SOTA) & SigLIP 1 `ViT-SO400M-14-SigLIP-384`<br>• Multi-template Prompt Ensembling (5 prompts)<br>• Vectorized Cosine Similarity / FAISS Index<br>• GQE 4-Facet SRRF Decomposition (`q_core`, `q_entity`, `q_action`, `q_scene`)<br>• Temporal Gaussian Smoothing ($\sigma=1.5, w=3$) | • **Video-Language Models (VLM video-native):** `InternVideo2`, `VideoLLaMA3`, `Video-LLaVA` hiểu trực tiếp chuỗi chuyển động spatio-temporal thay vì frame tĩnh.<br>• **CLIP chuyên biệt Tiếng Việt & Đa ngữ:** `EVA-02-CLIP-E/14+`, `DFN-5B`, hoặc Fine-tune `ViCLIP` / `PhoCLIP`.<br>• **Fine-grained Region Grounding:** `Grounding DINO` + `SAM2` trích xuất embedding từng vùng đối tượng nhỏ/chìm. |
| **⚖️ VLM-as-a-Judge (Phán Quyết Thị Giác)** | • **Qwen2.5-VL-3B / 7B Instruct** làm Trọng tài Thị giác Giai đoạn 2 (2nd-stage High-Precision Reranker):<br>  - **KIS:** Thẩm định chi tiết màu sắc, vật thể, hành động, graffiti, tranh vẽ.<br>  - **QA:** Thẩm định bằng chứng thị giác trực tiếp (Visual Evidence) đưa frame chứa đáp án lên Top 1.<br>  - **TRAKE:** Thẩm định tính nhất quán chuỗi đa sự kiện $(f_1 \to f_2 \to \dots \to f_N)$.<br>• Score Blending: $S_{\text{blended}} = S_{\text{retrieval}} \cdot (1 - w) + \frac{S_{\text{vlm}}}{100} \cdot w$ | • **Domain-specific Fine-tuning:** Fine-tune Qwen2.5-VL trên tập dữ liệu AIC các năm trước để nhận diện chính xác các câu hỏi đặc thù BTC.<br>• **Direct Video Chunk Verification:** Truyền 4–8 keyframe liên tiếp quanh mốc thời gian để VLM kiểm tra chuyển động ngữ cảnh. |
| **🗣️ Lời thoại (Dense & Sparse ASR)** | • **Dense Semantic ASR (BGE-M3 1024-dim):** Mã hóa ngữ nghĩa toàn bộ câu thoại, tìm kiếm bằng tích vô hướng ma trận GPU ($< 2\text{ms}$).<br>• **Sparse Lexical BM25Okapi:** Tokenizer Tiếng Việt nâng cao (NFC, Bigram ghép từ phức, Accent-folding).<br>• **Hybrid Fusion:** $S_{\text{ASR}} = 0.65 \cdot \text{Dense} + 0.35 \cdot \text{Sparse}$.<br>• Khớp timestamp PTS thời gian thực từ `map-keyframes`. | • **ASR Re-transcription & Diarization:** `Whisper-large-v3-turbo` + `PyAnnote` phân tách từng người nói và tự thêm dấu câu chuẩn xác.<br>• **Sliding Dialogue Context:** Gắn ngữ cảnh đoạn hội thoại mở rộng ($\pm 5\text{s} - 10\text{s}$) vào metadata keyframe. |
| **🔤 Chữ màn hình & Vật thể (OCR/Objects)** | • PaddleOCR (Tiếng Việt/Anh)<br>• Nhãn Faster R-CNN Google OpenImages (873 videos)<br>• Dual-threshold filtering (score $\ge 0.3$, presence $\ge 0.4$)<br>• BM25 Lexical Index với IDF Floor ($0.01$) | • **SOTA Vision OCR:** `GOT-OCR2.0` hoặc `Qwen2-VL` (đọc được chữ xoay nghiêng, logo mờ, biển số xe xa, chữ cách điệu).<br>• **Fuzzy Token Matching / Levenshtein N-gram:** Tự động sửa lỗi sai 1-2 ký tự do OCR nhận diện nhầm (ví dụ: `VTVl` $\rightarrow$ `VTV1`, `Hà Nọi` $\rightarrow$ `Hà Nội`).<br>• **Scene Text VQA:** Trả lời trực tiếp nội dung văn bản trên ảnh mà không cần bóc tách OCR riêng rẽ. |
| **🌈 Đa kênh (Hybrid Fusion)** | • Dynamic Modality Routing<br>• Score-Reflected Reciprocal Rank Fusion (SRRF)<br>• Trọng số động ($w_{\text{vis}}, w_{\text{ocr}}, w_{\text{asr}}$)<br>• Metadata / Text-in-Scene Reranking + VLM Judge Reranking | • **Learned Late-Interaction (ColPali):** Tính tương đồng multi-vector giữa từng từ của câu hỏi và từng token visual của frame.<br>• **Cross-Encoder Reranker:** Mô hình rerank đa phương thức (ví dụ: `BGE-Reranker-Large`, `Cohere Rerank v3`) xếp hạng lại Top 100 candidate.<br>• **Adaptive Meta-Classifier:** Mô hình phân loại tự động gán trọng số $w_{\text{vis}}, w_{\text{ocr}}, w_{\text{asr}}$ dựa trên đặc trưng ngôn ngữ của câu query. |
| **🎬 Chuỗi hành động (TRAKE)** | • Dynamic Programming Monotonic Alignment ($O(T \cdot N)$)<br>• Sequence Time-Gap Regularization ($t_1 < t_2 < \dots < t_n$)<br>• VLM Judge thẩm định chuỗi đa ảnh theo thời gian<br>• Trực quan hóa timeline liên hoàn trên Web UI | • **Temporal Action Localization:** `ActionFormer` hoặc `TriDet` phát hiện chính xác ranh giới bắt đầu/kết thúc của từng hành động đơn lẻ.<br>• **Video Event Graph Reasoning:** Xây dựng đồ thị liên kết đối tượng qua các frame để kiểm tra tính nhất quán (người A cầm chai $\rightarrow$ mở nắp $\rightarrow$ uống). |
| **❓ Truy vấn Hỏi đáp (QA Retrieval)** | • Visual Evidence Verification bằng Qwen2.5-VL (đưa frame có góc nhìn chứa đáp án lên Top 1)<br>• Query Synthesis (Context text + Question keywords)<br>• Trực quan hóa Badge OCR `🔤`, ASR `🗣️` & Evidence Score `🎯` để người dùng xác thực và trả lời tức thì | • **Bounding-box Zoom-in VQA:** Tự động crop vùng vật thể nghi vấn trong keyframe và phóng to cho VLM đọc các chi tiết siêu nhỏ (chữ trên đồng hồ, logo túi xách). |

---

### 9.2. Chi tiết & Hướng dẫn Triển khai Nâng cấp

#### 1. Nâng cấp Kênh Thị giác (Visual Search)
* **Hiện trạng:** Đã nâng cấp lên **Google SigLIP 2** (`ViT-SO400M-14-SigLIP2-378`) mang lại độ chính xác zero-shot vượt trội so với SigLIP 1 và CLIP.
* **Hướng nâng cấp tiếp theo:**
  * Thử nghiệm **`InternVideo2`** hoặc **`VideoLLaMA3`**: Trích xuất spatio-temporal feature cho từng đoạn video ngắn (shot 2-4 giây) thay vì ảnh tĩnh.
  * Vector đặc trưng video được lưu dưới dạng `.npy` và tìm kiếm bằng FAISS IndexIVFFlat hoặc HNSW để đạt tốc độ $< 50\text{ms}$.

#### 2. Nâng cấp Thẩm định Thị giác (VLM-as-a-Judge)
* **Hiện trạng:** Đã triển khai **Qwen2.5-VL-3B / 7B** thẩm định Top 15 ứng viên cho KIS, QA Evidence và TRAKE Sequence.
* **Hướng nâng cấp tiếp theo:**
  * Bổ sung cơ chế **Region of Interest (RoI) Zooming**: Khi câu hỏi nhắm vào chi tiết nhỏ (ví dụ: *"chữ trên cổ áo"*, *"màu sắc chiếc nhẫn"*), VLM Judge tự động crop các vùng candidate có độ phân giải cao để thẩm định chính xác hơn.

#### 3. Nâng cấp Kênh Lời thoại ASR (Speech Transcript)
* **Hiện trạng:** Đã triển khai **Dense Semantic ASR** bằng `BAAI/bge-m3` kết hợp Hybrid với BM25 ($S = 0.65 \cdot \text{Dense} + 0.35 \cdot \text{Sparse}$).
* **Hướng nâng cấp tiếp theo:**
  * Tái trích xuất lời thoại bằng `faster-whisper-large-v3-turbo` với Initial Prompt tiếng Việt chuẩn để nâng cao độ chính xác địa danh và tên riêng.
  * Phân tách người nói (Speaker Diarization) với `PyAnnote`.

#### 4. Nâng cấp Kênh Chữ màn hình & Biển hiệu (OCR)
* **Hiện trạng:** PaddleOCR + BM25 Lexical Index.
* **Hướng nâng cấp tiếp theo:**
  * Tích hợp **`GOT-OCR2.0`** (General OCR Theory) có khả năng đọc văn bản siêu nhỏ, định dạng phức tạp và chữ viết tay với độ chính xác cao hơn 30% so với PaddleOCR.
  * Bổ sung cơ chế **Fuzzy String Matching** (Levenshtein Distance / Token Set Ratio) trong `bm25_index.py` để vẫn bắt trúng kết quả khi OCR bị sai dấu hoặc thiếu chữ cái.


