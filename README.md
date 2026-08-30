# 🚀 AIC 2026 — PEGASUS Video Retrieval System

Hệ thống tìm kiếm video đa phương thức tối tân cho 3 bài toán vòng sơ tuyển **AI Challenge 2026 (KIS / Q&A / TRAKE)**:
* **Vector Search SOTA:** **Google SigLIP 2** (`ViT-SO400M-14-SigLIP2-378` / alias: `siglip2`) & SigLIP 1 (`ViT-SO400M-14-SigLIP-384`).
* **VLM-as-a-Judge:** **Qwen2.5-VL-7B** thẩm định thị giác chính xác cho KIS, kiểm tra bằng chứng (Visual Evidence) cho Q&A, và xác thực chuỗi đa sự kiện cho TRAKE.
* **Lexical & Metadata:** BM25 nhãn vật thể (Faster R-CNN) + Tiêu đề/mô tả YouTube.
* **Lời thoại ASR:** Whisper trích xuất lời thoại tiếng Việt có mốc thời gian, gắn trực tiếp vào từng keyframe.
* **Reranking & Filtering:** Hybrid SRRF (Score-Reflected Reciprocal Rank Fusion) + VLM Judge Rerank + Bộ lọc màu sắc.
* **TRAKE:** Căn chỉnh chuỗi sự kiện theo thứ tự thời gian bằng Dynamic Programming + VLM Sequence Verifier.

---

## 📑 Mục lục
1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Cấu trúc thư mục Dữ liệu tách riêng](#2-cấu-trúc-thư-mục-dữ-liệu-tách-riêng)
3. [Chuẩn bị dữ liệu & Encode SigLIP 2](#3-chuẩn-bị-dữ-liệu--encode-siglip-2)
4. [Khởi động Giao diện PEGASUS Web & Cấu hình đường dẫn](#4-khởi-động-giao-diện-pegasus-web--cấu-hình-đường-dẫn)
5. [Tính Năng VLM-as-a-Judge (Qwen2.5-VL)](#5-tính-năng-vlm-as-a-judge-qwen25-vl)
6. [Hướng dẫn Tìm kiếm & Xuất bài nộp BTC](#6-hướng-dẫn-tìm-kiếm--xuất-bài-nộp-btc)
7. [Chạy bằng CLI & Đánh giá (Evaluate)](#7-chạy-bằng-cli--đánh-giá-evaluate)
8. [Xử lý sự cố thường gặp (Troubleshooting)](#8-xử-lý-sự-cố-thường-gặp-troubleshooting)
9. [Chi tiết Công nghệ Hiện tại & Đề xuất Nâng cấp Tiên tiến](#9-chi-tiết-công-nghệ-hiện-tại--đề-xuất-nâng-cấp-tiên-tiến)

---

## 1. Cài đặt môi trường

### Yêu cầu tiên quyết:
* **Hệ điều hành:** Windows 10/11 hoặc Linux / macOS.
* **Python:** `3.10` – `3.12` (khuyên dùng Python 3.11).
* **GPU NVIDIA (Khuyên dùng):** GTX 1060 trở lên hoặc RTX Series để tăng tốc trích xuất và tìm kiếm.

### Các bước cài đặt:

Mở **PowerShell / Terminal** tại thư mục gốc `my-repo`:

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

# 4. (Tùy chọn) Cài đặt PyTorch CUDA cho GPU NVIDIA
uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
```

> **Cài đặt Ollama (cho Agent / VLM):**
> ```bash
> ollama pull qwen3.5:4b
> ollama serve
> ```

---

## 2. Cấu trúc thư mục Dữ liệu tách riêng

> [!NOTE]
> **Toàn bộ thư mục dữ liệu (`extracted` và `processed`) được lưu riêng biệt ngoài `AIC`** tại `D:\ai_challenge\data\` để tránh nặng repo và tiện quản lý.

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
    ├── siglip2_features.npy              # Ma trận vector SigLIP 2 tương ứng
    ├── siglip_manifest.jsonl             # Manifest SigLIP 1
    ├── siglip_features.npy               # Ma trận vector SigLIP 1
    ├── derived_manifest.jsonl            # Manifest hiện có
    ├── derived_features.npy              # Ma trận vector hiện có
    └── colour_features.jsonl             # Đặc trưng màu sắc (tùy chọn)
```

---

## 3. Chuẩn bị dữ liệu & Index

### 3.1. Tự Encode Keyframes bằng GPU (Mô hình SigLIP 2 mới nhất)
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

### 3.3. Tiền trích xuất đặc trưng màu sắc (Tùy chọn)
```powershell
python -m aic2026.cli compute-colour-features `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --raw-dir "D:\ai_challenge\data\extracted" `
  --output "D:\ai_challenge\data\processed\colour_features.jsonl"
```

### 3.4. Kiểm tra tính đồng bộ của Index
```powershell
python -m aic2026.cli check-features "D:\ai_challenge\data\processed\siglip2_features.npy" "D:\ai_challenge\data\processed\siglip2_manifest.jsonl"
```

---

## 4. Khởi động Giao diện PEGASUS Web & Cấu hình đường dẫn

### Khởi động Web UI:
* **Cách 1 (Nhanh):** Nhấp đúp chuột vào [`run_app.bat`](file:///d:/ai_challenge/AIC/run_app.bat).
* **Cách 2 (Terminal):** `streamlit run src/aic2026/app/ui.py`
* **Cách 3 (Backend API + UI):** Terminal 1: `python -m aic2026.cli serve` | Terminal 2: `streamlit run src/aic2026/app/ui.py`

### ⚙️ Đường dẫn Data mặc định tự động nhận diện:
Giao diện đã được cấu hình tự động nhận diện chính xác các đường dẫn trên máy bạn:
* **Manifest Path:** `D:\ai_challenge\data\processed\siglip2_manifest.jsonl` (hoặc `siglip_manifest.jsonl` / `derived_manifest.jsonl`)
* **Features Path:** `D:\ai_challenge\data\processed\siglip2_features.npy` (hoặc `siglip_features.npy` / `derived_features.npy`)
* **Keyframes Dir:** `D:\ai_challenge\data\extracted\Keyframes`
* **Map Keyframes Dir:** `D:\ai_challenge\data\extracted\map-keyframes`
* **Videos Dir:** `D:\ai_challenge\data\extracted\Videos` (nếu có)

---

## 5. Tính Năng VLM-as-a-Judge (Qwen2.5-VL)

Hệ thống tích hợp **Qwen2.5-VL-7B** làm "Trọng tài thị giác" (Visual Verifier & Reranker) cho cả 3 bài toán:

* **🔍 KIS (Known Item Search)**: VLM "nhìn" từng frame trong Top 15, phân tích chi tiết điều kiện (màu sắc, vật thể, hành động, graffiti, tranh vẽ...) và chấm điểm 0–100 $\rightarrow$ Đưa frame đúng nhất lên Top 1.
* **🎯 Q&A (Visual Evidence Verifier)**: Thẩm định xem ảnh ứng viên có chứa **bằng chứng trực tiếp** để trả lời câu hỏi hay không $\rightarrow$ Đưa frame có đáp án lên Top 1 để bạn tự quan sát và gõ câu trả lời vào bài nộp (tránh AI hallucinate).
* **⏱️ TRAKE (Temporal Sequence Verifier)**: Thẩm định chuỗi đa ảnh $(f_1 \to f_2 \to \dots \to f_N)$, kiểm tra từng frame có khớp với từng event trong chuỗi thời gian hay không.

### ⚙️ Cách Kích Hoạt VLM Judge:
* **Trên Web UI**: Mở **Cấu hình nâng cao** (Sidebar) $\rightarrow$ Tích chọn `[x] ⚖️ Kích hoạt VLM-as-a-Judge (Qwen2.5-VL)`.
* **Trên CLI**: Thêm cờ `--vlm-judge --vlm-judge-model "Qwen/Qwen2.5-VL-7B-Instruct" --vlm-judge-top-k 15`.

---

## 6. Hướng dẫn Tìm kiếm & Xuất bài nộp BTC

### 6.1. Bốn Chế độ Tìm kiếm Chuyên biệt (Search Modality):
Hệ thống cho phép bạn linh hoạt chọn kênh tìm kiếm phù hợp với từng dạng câu hỏi:

| Chế độ | Ý nghĩa | Khi nào nên chọn? |
| :--- | :--- | :--- |
| 🌈 **Đa kênh (Hybrid)** | Dung hợp SigLIP 2 + ASR + OCR | Mặc định, dùng khi câu hỏi vừa có mô tả hình ảnh vừa có từ khóa/lời thoại. |
| 👁️ **Thị giác (SigLIP)** | Thuần Visual Vector Search ($w_{\text{vis}}=1.0$) | Dùng khi câu hỏi thuần mô tả thị giác/hình ảnh (ví dụ: *"người phụ nữ mặc áo đỏ lái xe"*). |
| 🗣️ **Lời thoại (ASR)** | Thuần Speech Search ($w_{\text{asr}}=1.0$) | Dùng khi câu hỏi trích dẫn câu thoại của nhân vật (ví dụ: *"bản tin thông báo thời tiết"*). Giữ nguyên tiếng Việt 100%, tìm kiếm tức thì. |
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

### 7.1. Chạy truy vấn KIS bằng CLI (có chọn chế độ `--mode` và VLM Judge):
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
# 1. Tìm kiếm -> JSON (ví dụ tìm thuần theo lời thoại ASR)
python -m aic2026.cli agent-query `
  --query query_kis.json `
  --manifest "D:\ai_challenge\data\processed\siglip2_manifest.jsonl" `
  --features "D:\ai_challenge\data\processed\siglip2_features.npy" `
  --mode asr `
  --output outputs/result_kis.json

# 2. Xuất CSV nộp bài
python -m aic2026.cli export-submission --result outputs/result_kis.json --query query_kis.json --output outputs/submission_kis.csv
```

### 6.2. Chạy truy vấn TRAKE bằng CLI:
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
# Trích mảng candidates từ kết quả JSON
python -c "import json; d=json.load(open('outputs/result_kis.json', encoding='utf-8')); json.dump(d['candidates'], open('outputs/candidates_kis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)"

# Đánh giá R@k và Final Score
python -m aic2026.cli evaluate query_kis.json outputs/candidates_kis.json ground_truth_kis.json
```

---

## 8. Xử lý sự cố thường gặp (Troubleshooting)

| Vấn đề / Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **`activate.ps1 cannot be loaded...`** | PowerShell chặn script thực thi. | Chạy lệnh: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` |
| **`CUDA Available: False`** | Chưa cài bản PyTorch có CUDA. | Chạy: `uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124` |
| **Lỗi OpenMP `libiomp5md.dll`** | Trùng thư viện runtime giữa PyTorch và các gói khác trên Windows. | Trong PowerShell chạy: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` trước khi chạy lệnh. |
| **Ảnh không hiển thị trên Web UI** | Đường dẫn `Keyframes Dir` trỏ chưa đúng thư mục data bên ngoài. | Vào mục **Cấu hình nâng cao** trên Sidebar của Web UI, kiểm tra lại đường dẫn `Keyframes Dir` (`D:\ai_challenge\data\extracted\Keyframes`). |
| **Chạy kiểm thử toàn bộ hệ thống** | Kiểm tra tính ổn định của mã nguồn. | Chạy lệnh: `pytest` |

---

## 9. Chi tiết Công nghệ Hiện tại & Đề xuất Nâng cấp Tiên tiến

### 8.1. Bảng Tổng quan Công nghệ theo từng Chế độ

| Chế độ / Nhiệm vụ | 🛠️ Công nghệ Hiện tại (Current PEGASUS) | 🚀 Đề xuất Nâng cấp Tiên tiến (SOTA Upgrades) |
| :--- | :--- | :--- |
| **👁️ Thị giác (SigLIP)** | • OpenCLIP `ViT-SO400M-14-SigLIP-384` (`webli`)<br>• Multi-template Prompt Ensembling (5 prompts)<br>• Vectorized Cosine Similarity / FAISS Index<br>• Temporal Gaussian Smoothing ($\sigma=1.5, w=3$)<br>• Video-level LogSumExp aggregation | • **Video-Language Models (VLM video-native):** `InternVideo2`, `VideoLLaMA3`, `Video-LLaVA` hiểu trực tiếp chuỗi chuyển động spatio-temporal thay vì frame tĩnh.<br>• **CLIP chuyên biệt Tiếng Việt & Đa ngữ:** `EVA-02-CLIP-E/14+`, `DFN-5B`, hoặc Fine-tune `ViCLIP` / `PhoCLIP`.<br>• **Fine-grained Region Grounding:** `Grounding DINO` + `SAM2` trích xuất embedding từng vùng đối tượng nhỏ/chìm. |
| **🗣️ Lời thoại (ASR)** | • BM25Okapi Lexical Indexing<br>• Tokenizer Tiếng Việt nâng cao (NFC, Bigram ghép từ phức, Accent-folded variants)<br>• Khớp timestamp PTS thời gian thực từ `map-keyframes` | • **Dense Semantic Dialogue Retrieval:** `BGE-M3` (1024-dim đa ngữ), `Vietnamese-Bi-Encoder` hoặc `PhoBERT-v2` để hiểu ngữ cảnh tương đồng (ví dụ: *"hoàn cảnh éo le"* $\leftrightarrow$ *"mẹ mất sớm, nhà nghèo"*).<br>• **ASR Re-transcription & Diarization:** `Whisper-large-v3-turbo` + `PyAnnote` phân tách từng người nói và tự thêm dấu câu chuẩn xác.<br>• **Hybrid Dense-Sparse (SPLADE):** Kết hợp BM25 + Learned Lexical Expansion. |
| **🔤 Chữ màn hình & Vật thể (OCR/Objects)** | • PaddleOCR (Tiếng Việt/Anh)<br>• Nhãn Faster R-CNN Google OpenImages (873 videos)<br>• Dual-threshold filtering (score $\ge 0.3$, presence $\ge 0.4$)<br>• BM25 Lexical Index với IDF Floor ($0.01$) | • **SOTA Vision OCR:** `GOT-OCR2.0` hoặc `Qwen2-VL` (đọc được chữ xoay nghiêng, logo mờ, biển số xe xa, chữ cách điệu).<br>• **Fuzzy Token Matching / Levenshtein N-gram:** Tự động sửa lỗi sai 1-2 ký tự do OCR nhận diện nhầm (ví dụ: `VTVl` $\rightarrow$ `VTV1`, `Hà Nọi` $\rightarrow$ `Hà Nội`).<br>• **Scene Text VQA:** Trả lời trực tiếp nội dung văn bản trên ảnh mà không cần bóc tách OCR riêng rẽ. |
| **🌈 Đa kênh (Hybrid Fusion)** | • Dynamic Modality Routing<br>• Score-Reflected Reciprocal Rank Fusion (SRRF)<br>• Trọng số động ($w_{\text{vis}}, w_{\text{ocr}}, w_{\text{asr}}$)<br>• Metadata / Text-in-Scene Reranking | • **Learned Late-Interaction (ColBERT / ColPali):** Tính tương đồng multi-vector giữa từng từ của câu hỏi và từng token visual của frame.<br>• **Cross-Encoder Reranker:** Mô hình rerank đa phương thức (ví dụ: `BGE-Reranker-Large`, `Cohere Rerank v3`) xếp hạng lại Top 100 candidate.<br>• **Adaptive Meta-Classifier:** Mô hình phân loại tự động gán trọng số $w_{\text{vis}}, w_{\text{ocr}}, w_{\text{asr}}$ dựa trên đặc trưng ngôn ngữ của câu query. |
| **🎬 Chuỗi hành động (TRAKE)** | • Dynamic Programming Monotonic Alignment ($O(T \cdot N)$)<br>• Sequence Time-Gap Regularization ($t_1 < t_2 < \dots < t_n$)<br>• Trực quan hóa cụm timeline liên hoàn trên Web UI | • **Temporal Action Localization:** `ActionFormer` hoặc `TriDet` phát hiện chính xác ranh giới bắt đầu/kết thúc của từng hành động đơn lẻ.<br>• **Video Event Graph Reasoning:** Xây dựng đồ thị liên kết đối tượng qua các frame để kiểm tra tính nhất quán (người A cầm chai $\rightarrow$ mở nắp $\rightarrow$ uống). |
| **❓ Truy vấn Hỏi đáp (QA Retrieval)** | • Zero-VLM Fast Retrieval ($<1$s)<br>• Query Synthesis (Context text + Question keywords)<br>• Trực quan hóa Badge OCR `🔤` & ASR `🗣️` trên từng frame card để người dùng xác thực trực quan tức thì | • **Ultra-lightweight On-Demand VLM:** Tích hợp `PaliGemma-2` (3B) hoặc `SmolVLM` (2B) — chỉ chạy khi người dùng chủ động nhấn *"Phân tích đối tượng"* trên frame cụ thể để crop và trả lời câu hỏi chi tiết. |

---

### 8.2. Chi tiết & Hướng dẫn Triển khai Nâng cấp

#### 1. Nâng cấp Kênh Thị giác (Visual Search)
* **Vấn đề hiện tại:** CLIP/SigLIP mã hóa từng keyframe độc lập, đôi khi bỏ lỡ mối quan hệ hành động có tính chuyển động cao (ví dụ: *"đang rót nước"* vs *"đã rót xong"*).
* **Giải pháp nâng cấp:**
  * Sử dụng **`InternVideo2`** hoặc **`VideoLLaMA3`**: Trích xuất spatio-temporal feature cho từng đoạn video ngắn (shot 2-4 giây) thay vì ảnh tĩnh.
  * Vector đặc trưng video được lưu dưới dạng `.npy` và tìm kiếm bằng FAISS IndexIVFFlat hoặc HNSW để đạt tốc độ $< 50\text{ms}$.

#### 2. Nâng cấp Kênh Lời thoại ASR (Speech Transcript)
* **Vấn đề hiện tại:** BM25 chỉ tìm được khi từ ngữ trong query xuất hiện chính xác trong lời thoại. Nếu người hỏi dùng từ đồng nghĩa hoặc mô tả gián tiếp, BM25 có thể bỏ sót.
* **Giải pháp nâng cấp:**
  * Xây dựng **Dense Speech Vector Index** bằng `BGE-M3` (đa ngữ, tối ưu cho tiếng Việt).
  * Khi tìm kiếm ASR: Tính điểm kết hợp $\text{Score}_{\text{ASR}} = 0.5 \cdot \text{Dense}(\text{BGE-M3}) + 0.5 \cdot \text{Sparse}(\text{BM25})$.

#### 3. Nâng cấp Kênh Chữ màn hình & Biển hiệu (OCR)
* **Vấn đề hiện tại:** Chữ trên áo, logo nhỏ hoặc biển hiệu bị che khuất một phần có thể bị bỏ qua bởi OCR truyền thống.
* **Giải pháp nâng cấp:**
  * Tích hợp **`GOT-OCR2.0`** (General OCR Theory) có khả năng đọc văn bản siêu nhỏ, định dạng phức tạp và chữ viết tay với độ chính xác cao hơn 30% so với PaddleOCR.
  * Bổ sung cơ chế **Fuzzy String Matching** (Levenshtein Distance / Token Set Ratio) trong `bm25_index.py` để vẫn bắt trúng kết quả khi OCR bị sai dấu hoặc thiếu chữ cái.

#### 4. Nâng cấp Khâu Xếp hạng Đa kênh (Cross-Modal Reranking)
* **Vấn đề hiện tại:** RRF (Reciprocal Rank Fusion) kết hợp thứ hạng theo công thức tĩnh $\frac{1}{k + r}$.
* **Giải pháp nâng cấp:**
  * Sau khi lấy ra Top 100 candidate bằng Vector + BM25, đưa Top 100 vào một mô hình **Cross-Encoder Reranker** đa phương thức để tính toán điểm tương thích tương tác sâu (Deep Late-Interaction) trước khi hiển thị lên Web UI.

