# 🚀 AIC 2026 — PEGASUS Video Retrieval System

Hệ thống tìm kiếm video đa phương thức cho 3 bài toán vòng sơ tuyển **AI Challenge 2026 (KIS / Q&A / TRAKE)**:
* **Vector Search:** SigLIP (`ViT-SO400M-14-SigLIP-384` / `ViT-L-14`) trích xuất embedding đa phương thức.
* **Lexical & Metadata:** BM25 nhãn vật thể (Faster R-CNN) + Tiêu đề/mô tả YouTube.
* **Lời thoại ASR:** Whisper trích xuất lời thoại tiếng Việt có mốc thời gian, gắn trực tiếp vào từng keyframe.
* **Reranking & Filtering:** Hybrid RRF (Reciprocal Rank Fusion) + Bộ lọc màu sắc.
* **VLM Q&A:** Trả lời câu hỏi hình ảnh bằng mô hình cục bộ (Qwen2.5-VL / Ollama / Florence-2).
* **TRAKE:** Căn chỉnh chuỗi sự kiện theo thứ tự thời gian bằng Dynamic Programming.

---

## 📑 Mục lục
1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Cấu trúc thư mục Dữ liệu tách riêng](#2-cấu-trúc-thư-mục-dữ-liệu-tách-riêng)
3. [Chuẩn bị dữ liệu & Index (Chỉ định đường dẫn ngoài)](#3-chuẩn-bị-dữ-liệu--index-chỉ-định-đường-dẫn-ngoài)
4. [Khởi động Giao diện PEGASUS Web & Cấu hình đường dẫn](#4-khởi-động-giao-diện-pegasus-web--cấu-hình-đường-dẫn)
5. [Hướng dẫn Tìm kiếm & Xuất bài nộp BTC](#5-hướng-dẫn-tìm-kiếm--xuất-bài-nộp-btc)
6. [Chạy bằng CLI & Đánh giá (Evaluate)](#6-chạy-bằng-cli--đánh-giá-evaluate)
7. [Xử lý sự cố thường gặp (Troubleshooting)](#7-xử-lý-sự-cố-thường-gặp-troubleshooting)

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
> **Toàn bộ thư mục dữ liệu (`extracted` và `processed`) được lưu riêng biệt ngoài `my-repo`** tại `D:\bachkhoa\ai_challenge\data\` để tránh nặng repo và tiện quản lý.

Sơ đồ cấu trúc thư mục data bên ngoài:

```text
D:\bachkhoa\ai_challenge\data\ (Thư mục dữ liệu bên ngoài)
├── extracted/                            # Dữ liệu giải nén do BTC cung cấp
│   ├── Keyframes/                        # Thư mục ảnh keyframe (L21_V001/001.jpg, ...)
│   ├── map-keyframes/                    # File CSV ánh xạ frame_idx sang timestamp pts_time
│   ├── asr/                              # File JSON lời thoại ASR từng video (L21_V001.json, ...)
│   ├── Objects/                          # File JSON nhãn vật thể Faster R-CNN
│   ├── Metadata/                         # File JSON thông tin video YouTube
│   ├── clip_features/                    # (Tùy chọn) File .npy CLIP gốc của BTC
│   └── Videos/                           # (Tùy chọn) File video .mp4 gốc
└── processed/                            # Dữ liệu index đã xử lý (đầu ra của hệ thống)
    ├── siglip_manifest.jsonl             # Manifest chính (vector_id, frame_id, labels, asr_text)
    ├── siglip_features.npy               # Ma trận vector đặc trưng tương ứng
    └── colour_features.jsonl             # Đặc trưng màu sắc (tùy chọn)
```

---

## 3. Chuẩn bị dữ liệu & Index

### 3.1. Tự Encode Keyframes bằng GPU (Mô hình SigLIP mặc định)
```powershell
python -m aic2026.cli embed-keyframes `
  --keyframes-dir "D:\bachkhoa\ai_challenge\data\extracted\Keyframes" `
  --map-keyframes-dir "D:\bachkhoa\ai_challenge\data\extracted\map-keyframes" `
  --features-dir "D:\bachkhoa\ai_challenge\data\processed\siglip_features_dir" `
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip_manifest.jsonl" `
  --features "D:\bachkhoa\ai_challenge\data\processed\siglip_features.npy" `
  --clip-model "ViT-SO400M-14-SigLIP-384" `
  --clip-pretrained "webli" `
  --device cuda `
  --batch-size 64
```
*(Hỗ trợ resume tự động nếu bị ngắt quãng).*

### 3.2. Ánh xạ lời thoại ASR vào Manifest
```powershell
python -m aic2026.cli asr-manifest `
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip_manifest.jsonl" `
  --asr-dir "D:\bachkhoa\ai_challenge\data\extracted\asr" `
  --map-keyframes-dir "D:\bachkhoa\ai_challenge\data\extracted\map-keyframes"
```

### 3.3. Tiền trích xuất đặc trưng màu sắc (Tùy chọn)
```powershell
python -m aic2026.cli compute-colour-features `
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip_manifest.jsonl" `
  --raw-dir "D:\bachkhoa\ai_challenge\data\extracted" `
  --output "D:\bachkhoa\ai_challenge\data\processed\colour_features.jsonl"
```

### 3.4. Kiểm tra tính đồng bộ của Index
```powershell
python -m aic2026.cli check-features "D:\bachkhoa\ai_challenge\data\processed\siglip_features.npy" "D:\bachkhoa\ai_challenge\data\processed\siglip_manifest.jsonl"
```

---

## 4. Khởi động Giao diện PEGASUS Web & Cấu hình đường dẫn

### Khởi động Web UI:
* **Cách 1 (Nhanh):** Nhấp đúp chuột vào [`run_app.bat`](file:///d:/bachkhoa/ai_challenge/my-repo/run_app.bat).
* **Cách 2 (Terminal):** `streamlit run src/aic2026/app/ui.py`
* **Cách 3 (Backend API + UI):** Terminal 1: `python -m aic2026.cli serve` | Terminal 2: `streamlit run src/aic2026/app/ui.py`

### ⚙️ Đường dẫn Data mặc định tự động nhận diện:
Giao diện đã được cấu hình tự động nhận diện chính xác các đường dẫn trên máy bạn:
* **Manifest Path:** `D:\bachkhoa\ai_challenge\data\processed\siglip_manifest.jsonl`
* **Features Path:** `D:\bachkhoa\ai_challenge\data\processed\siglip_features.npy`
* **Keyframes Dir:** `D:\bachkhoa\ai_challenge\data\extracted\Keyframes`
* **Map Keyframes Dir:** `D:\bachkhoa\ai_challenge\data\extracted\map-keyframes`
* **Videos Dir:** `D:\bachkhoa\ai_challenge\data\extracted\Videos` (nếu có)

---

## 5. Hướng dẫn Tìm kiếm & Xuất bài nộp BTC

### 5.1. Bốn Chế độ Tìm kiếm Chuyên biệt (Search Modality):
Hệ thống cho phép bạn linh hoạt chọn kênh tìm kiếm phù hợp với từng dạng câu hỏi:

| Chế độ | Ý nghĩa | Khi nào nên chọn? |
| :--- | :--- | :--- |
| 🌈 **Đa kênh (Hybrid)** | Dung hợp SigLIP + ASR + OCR | Mặc định, dùng khi câu hỏi vừa có mô tả hình ảnh vừa có từ khóa/lời thoại. |
| 👁️ **Thị giác (SigLIP)** | Thuần Visual Vector Search ($w_{\text{vis}}=1.0$) | Dùng khi câu hỏi thuần mô tả thị giác/hình ảnh (ví dụ: *"người phụ nữ mặc áo đỏ lái xe"*). |
| 🗣️ **Lời thoại (ASR)** | Thuần Speech Search ($w_{\text{asr}}=1.0$) | Dùng khi câu hỏi trích dẫn câu thoại của nhân vật (ví dụ: *"bản tin thông báo thời tiết"*). Giữ nguyên tiếng Việt 100%, tìm kiếm tức thì. |
| 🔤 **Chữ màn hình (OCR)** | Thuần OCR Text Search ($w_{\text{ocr}}=1.0$) | Dùng khi câu hỏi tìm biển báo, banner, logo, chữ trên màn hình (ví dụ: *"biển hiệu Phở 24"*). |

### 5.2. Các luồng bài toán:
1. **Tìm kiếm KIS (Known-Item Search):** Tìm 1 khung hình chính xác theo mô tả.
2. **Tìm kiếm Q&A:** Nhập ngữ cảnh và câu hỏi, VLM cục bộ sẽ đọc ảnh và trả lời.
3. **Tìm kiếm TRAKE:** Tìm chuỗi sự kiện theo thứ tự thời gian.

### 5.3. Chọn kết quả & Xuất file nộp bài:
1. Click trực tiếp lên các card ảnh kết quả để chọn đáp án đúng.
2. Các ảnh được chọn sẽ có khung viền màu hồng đánh dấu thứ tự click.
3. Bấm nút **"Xuất CSV nộp bài"** để tải file CSV đúng định dạng nộp cho BTC.

---

## 6. Chạy bằng CLI & Đánh giá (Evaluate)

### 6.1. Chạy truy vấn KIS bằng CLI (có chọn chế độ `--mode`):
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
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip_manifest.jsonl" `
  --features "D:\bachkhoa\ai_challenge\data\processed\siglip_features.npy" `
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
  --manifest "D:\bachkhoa\ai_challenge\data\processed\siglip_manifest.jsonl" `
  --features "D:\bachkhoa\ai_challenge\data\processed\siglip_features.npy" `
  --output outputs/result_trake.json

python -m aic2026.cli export-submission --result outputs/result_trake.json --query query_trake.json --output outputs/submission_trake.csv
```

### 6.3. Đánh giá Ground-Truth (Evaluate):
```powershell
# Trích mảng candidates từ kết quả JSON
python -c "import json; d=json.load(open('outputs/result_kis.json', encoding='utf-8')); json.dump(d['candidates'], open('outputs/candidates_kis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)"

# Đánh giá R@k và Final Score
python -m aic2026.cli evaluate query_kis.json outputs/candidates_kis.json ground_truth_kis.json
```

---

## 7. Xử lý sự cố thường gặp (Troubleshooting)

| Vấn đề / Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **`activate.ps1 cannot be loaded...`** | PowerShell chặn script thực thi. | Chạy lệnh: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` |
| **`CUDA Available: False`** | Chưa cài bản PyTorch có CUDA. | Chạy: `uv pip install --reinstall-package torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124` |
| **Lỗi OpenMP `libiomp5md.dll`** | Trùng thư viện runtime giữa PyTorch và các gói khác trên Windows. | Trong PowerShell chạy: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` trước khi chạy lệnh. |
| **Ảnh không hiển thị trên Web UI** | Đường dẫn `Keyframes Dir` trỏ chưa đúng thư mục data bên ngoài. | Vào mục **Cấu hình nâng cao** trên Sidebar của Web UI, kiểm tra lại đường dẫn `Keyframes Dir` (`D:\bachkhoa\ai_challenge\data\extracted\Keyframes`). |
| **Chạy kiểm thử toàn bộ hệ thống** | Kiểm tra tính ổn định của mã nguồn. | Chạy lệnh: `pytest` |
