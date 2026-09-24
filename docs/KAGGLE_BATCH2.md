# SigLIP2 batch 2 trên Kaggle T4 ×2

Người dùng đã chọn bộ keyframe CPU. Notebook sẵn tại
`kaggle/batch2_siglip2_t4x2.ipynb`; runner độc lập tại `kaggle/siglip2_runner.py`.
Mỗi GPU chạy một image encoder; chia shard theo số ảnh. Inference FP16,
normalize và lưu float32. Giữ model, resize PIL bicubic 378x378, mean/std 0.5.

## Bắt đầu với gói benchmark

Gói local: `data/batches/batch2/kaggle-sample/` tính từ workspace.
2.354 ảnh (M05_V001: 1.005; N001-V001: 325; S01-V001: 1.024), tổng shard
373.005.447 byte (~373 MB); ảnh phân bố theo thứ tự keyframe của từng video.

1. Tạo **private dataset** thứ nhất từ các file trong `control/`:
   `catalog.json`, `parity.aicshard`, `siglip2_runner.py`.
2. Tạo **private dataset** thứ hai từ ba file `.aicshard` trong `images-000/`.
   Upload file nguyên trạng. Đuôi `.aicshard` là ZIP_STORED có chủ ý để
   tránh dịch vụ tự giải nén thành hàng nghìn ảnh. Không đổi tên/giải nén.
3. Tạo private Notebook, **Import Notebook** file ipynb ở trên; chọn
   **GPU T4 ×2**, bật Internet để cài OpenCLIP 3.3.0 và tải checkpoint pin
   revision. Add Input hai dataset vừa tạo.
4. Giữ `RUN_FULL=False`, dùng **Save Version → Save & Run All**.
   Notebook xác nhận hai GPU và chạy kiểm chứng batch1 trên từng GPU,
   rồi đo tốc độ đọc/tiền xử lý/inference/ghi shard của cả hai GPU.
5. Xem `benchmark/gpu-*-parity.json`, `gpu-*-summary.json`, `run-summary.json`.
   Cosine phải đạt >=0.999; vector phải hữu hạn, norm gần 1. Chưa có số đo
   T4 local nên không coi ước tính 3–9 giờ là kết quả đã kiểm chứng.

Nếu thiếu quyền chọn GPU, quota hoặc Internet, xử lý trong tài khoản Kaggle.
Không đưa token/API key vào notebook, repo hay hội thoại. Dataset/Notebook
chứa dữ liệu camera không được công khai; việc dùng cloud cần phù hợp quy
định BTC. Bộ công cụ này chỉ chuẩn bị local, không tự upload hoặc publish.

## Chuẩn bị bộ đầy đủ

Từ repo:

```powershell
& .\.venv\Scripts\python.exe -u scripts/prepare_kaggle_batch2.py `
  --output ..\data\batches\batch2\kaggle-full `
  --shard-frames 2048 --group-gib 8
```

Có thể chạy lại cùng lệnh để resume. Packer kiểm tra source/report/map;
JPEG được chép nguyên byte. Mỗi shard tối đa 2.048 ảnh, thường nhỏ hơn
vì không trộn hai video. Mỗi nhóm `images-000`, `images-001`, ... khoảng
8 GiB. File shard có SHA256, mapping có SHA256, gói có catalog. State chỉ
ghi sau khi shard được ghi nguyên tử và kiểm tra. Giữ ít nhất 30 GiB trống.
Không chạy hai packer cùng output. Không upload thư mục `state/`.

Nguồn đầy đủ: 495 video, 670.151 ảnh, 112.653.200.707 byte ảnh; bao gồm
overhead ZIP/mapping thì gói lớn hơn một ít. Việc đóng gói cũng cần đọc toàn
bộ ảnh và thêm khoảng 113 GB dung lượng local. Không thay đổi/xóa ảnh gốc.
`control/catalog.json` chỉ được tạo khi đóng gói toàn bộ thành công.

Xem tiến độ mà không khởi chạy thêm packer:

```powershell
& .\.venv\Scripts\python.exe scripts/prepare_kaggle_batch2.py `
  --output ..\data\batches\batch2\kaggle-full --status
```

Chỉ dùng control full khi `catalog_ready` là true. Không chạy lại lệnh
đóng gói khi process cũ vẫn còn hoạt động. File `scripts/prepare_kaggle_full.bat`
giúp tiếp tục nếu process bị ngắt; giữ `.aicshard`/state hoàn tất, file tạm
được tạo lại khi resume.

Sau benchmark, tháo dataset sample và **output sample** khỏi notebook.
Thêm control của bộ full cùng các nhóm ảnh full. Chỉ gắn một catalog.
Các shard sample và full có thể cùng tên nhưng khác nội dung; resume sẽ
từ chối nếu vô tình gắn output của sample vào phiên full.

Nếu tài khoản không đủ dung lượng input cho tất cả nhóm cùng lúc, xử lý
từng nhóm/đợt. Notebook bật `--allow-partial-input`; phần chưa gắn vẫn được
ghi trong `remaining_shards`, không được coi là hoàn thành.

## Chạy full và tiếp tục qua phiên

- Đặt `RUN_FULL=True`, batch size 32; nếu OOM dùng 16. Giữ precision FP16.
- Giới hạn xử lý 9,5 giờ sau bước benchmark, rồi thoát bình thường để Kaggle
  lưu output khi **Save & Run All** thành công. Giới hạn thời gian được kiểm
  tra giữa các shard, vì vậy một shard đang chạy sẽ được làm xong.
- Không dựa vào việc ghi file trong phiên interactive để cho rằng đã lưu
  bền vững. Khi một version kết thúc, giữ output đó và gắn bằng **Add Input**
  vào phiên tiếp theo. Gắn **tất cả output cũ của cùng catalog**, vì phiên
  mới không sao chép lại các shard đã làm trước đó.
- Runner xác minh checksum nguồn, mapping, cấu hình checkpoint/preprocess,
  phiên bản thư viện, shape/norm trước khi bỏ qua output cũ. Đổi cấu hình
  giữa chừng bị từ chối; nếu Kaggle đổi môi trường cần khôi phục phiên bản
  thư viện cũ hoặc dùng bộ output riêng và kiểm chứng lại.
- Kết quả ghi `.npz.tmp` rồi đổi tên nguyên tử. File tạm không được skip.
  `.run.lock` chặn ghi đồng thời; chỉ xóa lock cũ khi chắc chắn process đã dừng.
- Model tải vào `/tmp/aic-model-cache`, không chiếm output được lưu.
  Toàn bộ vector float32 khoảng 3,09 GB, cộng mapping/metadata và báo cáo.
- Khi `remaining_shards` rỗng mới đủ bộ. Notebook benchmark cũng tạo shard
  hợp lệ; giữ cả thư mục `benchmark/` trong output để merge.

## Tải và ghép về máy local

Tải output mọi phiên vào thư mục riêng ngoài Git rồi chạy (thay đường dẫn
`kaggle-results` bằng thư mục chứa các phiên đã tải):

```powershell
& .\.venv\Scripts\python.exe scripts/merge_kaggle_batch2.py `
  --catalog ..\data\batches\batch2\kaggle-full\control\catalog.json `
  --results ..\data\batches\batch2\kaggle-results `
  --extracted ..\data\batches\batch2\extracted `
  --output ..\data\batches\batch2\processed\siglip2-kaggle
```

Output phải là thư mục mới. Merger từ chối thiếu shard, config khác nhau,
mapping sai, vector không hợp lệ, khóa trùng, đường dẫn không tồn tại và
catalog sample. Nó tạo manifest với đường dẫn local, `vector_id` liên tục,
ma trận theo đúng thứ tự và `merge-report.json`. Không tự ghép batch1 hoặc
chuyển app; các bước kiểm thử retrieval và timestamp nộp VFR vẫn còn riêng.

## Kiểm chứng tại máy local

- Test unit kiểm tra JPEG không đổi, mapping VFR, checksum, chia tải,
  resume và kết quả sai/hỏng.
- Smoke test thực chạy FP16 trên GTX 1070, 4 ảnh và 6 ảnh parity batch1:
  cosine tối thiểu 0,9999899. Khởi chạy child process thành công, output
  4x1152 hợp lệ; lần resume từ folder output cũ bỏ qua đúng shard và làm 0 ảnh.
- Chưa chạy trên T4 ×2: tài khoản Kaggle do người dùng thao tác/import.

Nguồn giới hạn nền tảng: https://www.kaggle.com/docs/notebooks (12 giờ/phiên
GPU, tối đa 20 GB output được lưu theo tài liệu đã tra ngày 2026-09-24).
