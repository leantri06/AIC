# PEGASUS — SigLIP2 Video Retrieval

Tìm kiếm trên **keyframe chính thức BTC**, dùng embedding **SigLIP2 SO400M-14-378 / webli** đã tạo sẵn. Không cần chạy lại extraction hoặc embedding.
Đọc [MEMORY.md](MEMORY.md) khi tiếp tục công việc.

## Chạy ứng dụng

Nhấp đúp `run_app.bat`, hoặc tại thư mục repository:

```powershell
.venv/Scripts/python.exe -m streamlit run src/aic2026/app/ui.py --server.port 8501
```

Mở http://localhost:8501. UI chạy retrieval trực tiếp khi không có API riêng.
Mô hình truy vấn được nạp ở lần tìm đầu tiên; các lần sau dùng cache.
Nếu chưa có checkpoint, cần mạng để tải SigLIP2 về `data/.cache/huggingface`.
App tự chọn CUDA nếu PyTorch hỗ trợ GPU, còn lại dùng CPU.

Môi trường mới (Python >=3.11):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[models,dev]"
```

Cài bản PyTorch CUDA tương ứng nếu muốn dùng GPU. Launcher ưu tiên `.venv` hiện có, không tự đồng bộ lại và thay bản PyTorch đã cài.

## Dữ liệu

```text
ai_challenge/
├── AIC/                         # repository
└── data/
    ├── processed/
    │   ├── siglip2_manifest.jsonl
    │   └── siglip2_features.npy
    └── extracted/
        ├── keyframes/VIDEO/001.jpg
        ├── map_keyframes/VIDEO.csv
        └── Videos/              # tùy chọn, .mp4 / .mov
```

App dò `data` cạnh repository. Có thể chọn thư mục khác ở **Cấu hình Thư mục Dữ liệu Gốc** hoặc đặt `AIC_DATA_ROOT`.
Hỗ trợ tên thư mục `Keyframes` / `keyframes`, `map-keyframes` / `map_keyframes`, và đường dẫn ổ D cũ trong manifest.
Không tự fallback sang CLIP, SigLIP1 hoặc SigLIP2 Base vì chúng không cùng không gian embedding.
Thứ tự `vector_id` phải đúng hàng tương ứng trong ma trận.

Kiểm tra index/ảnh/mapping mà không sửa dữ liệu:

```powershell
.venv/Scripts/python.exe scripts/audit_siglip2.py
```

Báo cáo lưu vào `data/processed/siglip2_audit.json`. Audit kiểm tra cấu trúc, tên ảnh, CSV mapping và giá trị vector; không thay thế đánh giá chất lượng retrieval hay giải mã toàn bộ JPEG.

## Sử dụng

- **KIS:** nhập mô tả hình ảnh để tìm keyframe.
- **Q&A:** tìm ảnh theo mô tả/câu hỏi; nhập đáp án thủ công trước khi xuất CSV.
- **TRAKE:** mỗi dòng là một sự kiện theo thứ tự thời gian.
- Tùy chọn dịch VI→EN và lọc khung hình gần nhau (NMS).
- Xem keyframe lân cận, chọn kết quả, tải CSV. Không có video gốc vẫn tìm/xem keyframe được.

ASR, OCR, BGE-M3, BM25 và các lệnh CLI xử lý tương ứng đã được gỡ.
Các trường cũ trong manifest được bỏ qua; file dữ liệu gốc được giữ nguyên.

API riêng (không bắt buộc):

```powershell
.venv/Scripts/python.exe -m uvicorn aic2026.app.api:app --host 127.0.0.1 --port 8000
```

CLI:

```powershell
.venv/Scripts/python.exe -m aic2026.cli agent-query --query examples/query_kis.json --mode siglip --output outputs/result.json
```

Tên file query phải trỏ tới JSON của bạn, gồm `query_id`, `type`, `text`; Q&A thêm `question`, TRAKE thêm `events`.
