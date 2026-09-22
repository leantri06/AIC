"""
Script tải dữ liệu Batch 2 cho AIC 2026.
Hỗ trợ:
- Resumable download qua HTTP Range header.
- Cập nhật tiến độ trực tiếp vào progress.txt và download_status.json.
- Kiểm tra tính toàn vẹn zip và kích thước sau khi tải xong.
- Theo dõi dung lượng ổ đĩa an toàn.
- Lệnh --status để xem tiến độ hiện tại.
"""

import sys
import os
import time
import json
import zipfile
import shutil
import urllib.request
import urllib.error
from datetime import datetime, timezone

# Fix Windows console cp1252 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Cấu hình danh sách file
DOWNLOAD_ITEMS = [
    {
        "filename": "media-info-aic25-b2.zip",
        "url": "https://aic-data.ledo.io.vn/media-info-aic25-b2.zip",
        "expected_bytes": 771155,
        "description": "Metadata thông tin media AIC 2026 (Batch 2)",
    },
    {
        "filename": "Video_N001-N010.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N001-N010.zip",
        "expected_bytes": 10549210472,
        "description": "Video camera giao thông N001-N010",
    },
    {
        "filename": "Video_N011-N020.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N011-N020.zip",
        "expected_bytes": 11425892934,
        "description": "Video camera giao thông N011-N020",
    },
    {
        "filename": "Video_N021-N030.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N021-N030.zip",
        "expected_bytes": 10306243576,
        "description": "Video camera giao thông N021-N030",
    },
    {
        "filename": "Video_N031-N040.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N031-N040.zip",
        "expected_bytes": 10924602184,
        "description": "Video camera giao thông N031-N040 (phục hồi từ server)",
    },
    {
        "filename": "Video_N041-N050.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N041-N050.zip",
        "expected_bytes": 9432748608,
        "description": "Video camera giao thông N041-N050",
    },
    {
        "filename": "Video_N051-N060.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N051-N060.zip",
        "expected_bytes": 9408137747,
        "description": "Video camera giao thông N051-N060",
    },
    {
        "filename": "Video_N061-N070.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N061-N070.zip",
        "expected_bytes": 9234704702,
        "description": "Video camera giao thông N061-N070",
    },
    {
        "filename": "Video_N071-N080.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N071-N080.zip",
        "expected_bytes": 10623121918,
        "description": "Video camera giao thông N071-N080",
    },
    {
        "filename": "Video_N081-N090.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N081-N090.zip",
        "expected_bytes": 10984589208,
        "description": "Video camera giao thông N081-N090",
    },
    {
        "filename": "Video_N091-N100.zip",
        "url": "https://aic-data.ledo.io.vn/Video_N091-N100.zip",
        "expected_bytes": 9952427936,
        "description": "Video camera giao thông N091-N100",
    },
    {
        "filename": "Video_S01.zip",
        "url": "https://aic-data.ledo.io.vn/Video_S01.zip",
        "expected_bytes": 53896100459,
        "description": "Video đua xe đạp S01",
    },
    {
        "filename": "Videos_M05.zip",
        "url": "https://aic-data.ledo.io.vn/Videos_M05.zip",
        "expected_bytes": 10161147478,
        "description": "Video tin tức M05",
    },
    {
        "filename": "Videos_M06.zip",
        "url": "https://aic-data.ledo.io.vn/Videos_M06.zip",
        "expected_bytes": 10889052849,
        "description": "Video tin tức M06",
    },
    {
        "filename": "Videos_M07.zip",
        "url": "https://aic-data.ledo.io.vn/Videos_M07.zip",
        "expected_bytes": 11840337561,
        "description": "Video tin tức M07",
    },
    {
        "filename": "Videos_M08.zip",
        "url": "https://aic-data.ledo.io.vn/Videos_M08.zip",
        "expected_bytes": 12632959226,
        "description": "Video tin tức M08",
    },
    {
        "filename": "Videos_M09.zip",
        "url": "https://aic-data.ledo.io.vn/Videos_M09.zip",
        "expected_bytes": 13294614361,
        "description": "Video tin tức M09",
    },
    {
        "filename": "Videos_M10.zip",
        "url": "https://aic-data.ledo.io.vn/Videos_M10.zip",
        "expected_bytes": 12918192698,
        "description": "Video tin tức M10",
    },
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIC2026-Batch2-Downloader"
BUFFER_SIZE = 2 * 1024 * 1024  # 2MB chunk buffer
MIN_FREE_DISK_GB = 30  # Giữ tối thiểu 30GB trống


def get_default_target_dir():
    # Thư mục đích: data/batches/batch2/raw
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(cur_dir)  # my-repo
    parent_dir = os.path.dirname(workspace_dir)  # ai_challenge root
    target_dir = os.path.join(parent_dir, "data", "batches", "batch2", "raw")
    return target_dir


def format_bytes(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_time(seconds):
    if seconds is None or seconds < 0 or seconds > 86400 * 30:
        return "--:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def make_progress_bar(percent, width=30):
    filled = int(width * percent / 100)
    bar = "=" * filled + (">" if filled < width else "")
    return f"[{bar:<{width}}]"


def check_disk_free_gb(path):
    try:
        total, used, free = shutil.disk_usage(path)
        return free / (1024**3)
    except Exception:
        return 999.0


def log_message(log_path, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def write_status_files(target_dir, status_data):
    status_json_path = os.path.join(target_dir, "download_status.json")
    progress_txt_path = os.path.join(target_dir, "progress.txt")

    try:
        with open(status_json_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    try:
        lines = [
            "================================================================================",
            "                   TIẾN ĐỘ TẢI DỮ LIỆU BATCH 2 - AIC 2026",
            "================================================================================",
            f"Trạng thái tổng thể: {status_data.get('overall_status', 'RUNNING')}",
            f"Thời gian cập nhật: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Đã hoàn thành: {status_data.get('completed_files_count', 0)}/{status_data.get('total_files_count', 0)} file",
            f"Tổng dung lượng: {format_bytes(status_data.get('overall_downloaded_bytes', 0))} / {format_bytes(status_data.get('overall_total_bytes', 0))} ({status_data.get('overall_percent', 0.0):.1f}%)",
            f"{make_progress_bar(status_data.get('overall_percent', 0.0), width=50)}",
            "",
            "--- FILE ĐANG XỬ LÝ ---",
            f"File: {status_data.get('current_file', 'None')}",
            f"Mô tả: {status_data.get('current_description', '')}",
            f"Tiến độ file: {format_bytes(status_data.get('current_downloaded_bytes', 0))} / {format_bytes(status_data.get('current_total_bytes', 0))} ({status_data.get('current_percent', 0.0):.1f}%)",
            f"Tốc độ tải: {status_data.get('current_speed_mb_s', 0.0):.2f} MB/s",
            f"Thời gian còn lại (ETA): {format_time(status_data.get('current_eta_seconds', 0))}",
            f"Dung lượng ổ D còn trống: {status_data.get('disk_free_gb', 0.0):.1f} GB",
            "",
            "--- DANH SÁCH FILE ---",
        ]

        for item in status_data.get("files", []):
            st = item.get("status", "PENDING")
            fn = item.get("filename", "")
            sz = format_bytes(item.get("expected_bytes", 0))
            pct = item.get("percent", 0.0)
            mark = "[✓]" if st == "COMPLETED" else ("[->]" if st == "DOWNLOADING" else "[ ]")
            lines.append(f"  {mark} {fn:<25} | {sz:>10} | {pct:>5.1f}% | {st}")

        lines.append("================================================================================")
        with open(progress_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def print_status(target_dir):
    progress_txt_path = os.path.join(target_dir, "progress.txt")
    if os.path.exists(progress_txt_path):
        with open(progress_txt_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print(f"Chưa tìm thấy {progress_txt_path}. Quá trình tải có thể chưa khởi chạy.")


def download_file(item, target_dir, log_path, status_data, overall_stats):
    filename = item["filename"]
    url = item["url"]
    expected_bytes = item["expected_bytes"]
    final_path = os.path.join(target_dir, filename)
    part_path = final_path + ".part"

    # Kiểm tra nếu file đã tải trọn vẹn
    if os.path.exists(final_path):
        actual_size = os.path.getsize(final_path)
        if actual_size == expected_bytes:
            log_message(log_path, f"File {filename} đã tồn tại với đầy đủ dung lượng ({format_bytes(actual_size)}). Bỏ qua.")
            item["status"] = "COMPLETED"
            item["percent"] = 100.0
            overall_stats["downloaded_base"] += expected_bytes
            status_data["completed_files_count"] += 1
            return True
        else:
            log_message(log_path, f"File {filename} có dung lượng lệch ({actual_size} != {expected_bytes}). Sẽ tải lại.")

    item["status"] = "DOWNLOADING"
    status_data["current_file"] = filename
    status_data["current_description"] = item["description"]
    status_data["current_total_bytes"] = expected_bytes

    # Kiểm tra dung lượng ổ đĩa
    free_gb = check_disk_free_gb(target_dir)
    status_data["disk_free_gb"] = free_gb
    if free_gb < MIN_FREE_DISK_GB:
        err_msg = f"CẢNH BÁO NGUY HIỂM: Dung lượng trống ổ D ({free_gb:.1f} GB) thấp hơn ngưỡng an toàn ({MIN_FREE_DISK_GB} GB)! Dừng tải."
        log_message(log_path, err_msg)
        status_data["overall_status"] = "HALTED_LOW_DISK"
        write_status_files(target_dir, status_data)
        return False

    max_retries = 15
    retry_count = 0

    while retry_count < max_retries:
        existing_bytes = 0
        if os.path.exists(part_path):
            existing_bytes = os.path.getsize(part_path)
            if existing_bytes > expected_bytes:
                log_message(log_path, f"File .part lớn hơn file gốc ({existing_bytes} > {expected_bytes}), xóa tải lại từ đầu.")
                os.remove(part_path)
                existing_bytes = 0

        headers = {
            "User-Agent": USER_AGENT,
        }
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            log_message(log_path, f"Tiếp tục tải (Resume) {filename} từ byte {existing_bytes} ({format_bytes(existing_bytes)} / {format_bytes(expected_bytes)})")
        else:
            log_message(log_path, f"Bắt đầu tải {filename} ({format_bytes(expected_bytes)})")

        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing_bytes > 0 else "wb"

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                status_code = resp.status
                if existing_bytes > 0 and status_code != 206:
                    log_message(log_path, f"Server không trả mã 206 Partial Content (mã: {status_code}), tải lại từ đầu.")
                    mode = "wb"
                    existing_bytes = 0

                downloaded_file = existing_bytes
                last_update_time = time.time()
                last_bytes = downloaded_file
                t_start = time.time()

                with open(part_path, mode) as out_file:
                    while True:
                        chunk = resp.read(BUFFER_SIZE)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded_file += len(chunk)

                        now = time.time()
                        if now - last_update_time >= 1.5:
                            dt = now - last_update_time
                            db = downloaded_file - last_bytes
                            speed_mb = (db / dt) / (1024 * 1024) if dt > 0 else 0.0
                            remaining_bytes = expected_bytes - downloaded_file
                            eta = remaining_bytes / (db / dt) if (dt > 0 and db > 0) else 0

                            file_pct = (downloaded_file / expected_bytes) * 100.0 if expected_bytes > 0 else 0
                            curr_overall_downloaded = overall_stats["downloaded_base"] + downloaded_file
                            curr_overall_pct = (curr_overall_downloaded / overall_stats["total_bytes"]) * 100.0

                            status_data["current_downloaded_bytes"] = downloaded_file
                            status_data["current_percent"] = file_pct
                            status_data["current_speed_mb_s"] = speed_mb
                            status_data["current_eta_seconds"] = eta
                            status_data["overall_downloaded_bytes"] = curr_overall_downloaded
                            status_data["overall_percent"] = curr_overall_pct
                            status_data["disk_free_gb"] = check_disk_free_gb(target_dir)
                            item["percent"] = file_pct

                            write_status_files(target_dir, status_data)
                            last_update_time = now
                            last_bytes = downloaded_file

            # Hoàn thành download luồng này
            actual_size = os.path.getsize(part_path)
            if actual_size == expected_bytes:
                log_message(log_path, f"Tải xong {filename} ({format_bytes(actual_size)}). Đang kiểm tra tính toàn vẹn ZIP...")
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(part_path, final_path)

                # Kiểm tra zip integrity
                try:
                    if zipfile.is_zipfile(final_path):
                        log_message(log_path, f"[THÀNH CÔNG] File {filename} là ZIP hợp lệ!")
                    else:
                        log_message(log_path, f"[CẢNH BÁO] File {filename} tải xong nhưng không phải định dạng ZIP chuẩn.")
                except Exception as ze:
                    log_message(log_path, f"[CẢNH BÁO] Kiểm tra zip {filename} gặp lỗi: {ze}")

                item["status"] = "COMPLETED"
                item["percent"] = 100.0
                overall_stats["downloaded_base"] += expected_bytes
                status_data["completed_files_count"] += 1
                write_status_files(target_dir, status_data)
                return True
            else:
                log_message(log_path, f"Kích thước sau khi tải chưa đủ ({actual_size}/{expected_bytes}). Thử lại...")
                retry_count += 1
                time.sleep(3)

        except (urllib.error.URLError, TimeoutError, ConnectionResetError, Exception) as e:
            retry_count += 1
            wait_time = min(5 * retry_count, 30)
            log_message(log_path, f"Lỗi tải {filename}: {e}. Sẽ tự động thử lại lần {retry_count}/{max_retries} sau {wait_time}s...")
            time.sleep(wait_time)

    item["status"] = "FAILED"
    log_message(log_path, f"[THẤT BẠI] Không thể tải xong {filename} sau {max_retries} lần thử.")
    write_status_files(target_dir, status_data)
    return False


def main():
    target_dir = get_default_target_dir()
    os.makedirs(target_dir, exist_ok=True)

    if "--status" in sys.argv:
        print_status(target_dir)
        return

    log_path = os.path.join(target_dir, "download.log")
    log_message(log_path, f"=== Bắt đầu phiên tải dữ liệu Batch 2 ===")
    log_message(log_path, f"Thư mục lưu trữ: {target_dir}")

    total_bytes = sum(x["expected_bytes"] for x in DOWNLOAD_ITEMS)
    status_data = {
        "overall_status": "RUNNING",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "total_files_count": len(DOWNLOAD_ITEMS),
        "completed_files_count": 0,
        "overall_total_bytes": total_bytes,
        "overall_downloaded_bytes": 0,
        "overall_percent": 0.0,
        "current_file": "",
        "current_description": "",
        "current_downloaded_bytes": 0,
        "current_total_bytes": 0,
        "current_percent": 0.0,
        "current_speed_mb_s": 0.0,
        "current_eta_seconds": 0,
        "disk_free_gb": check_disk_free_gb(target_dir),
        "files": DOWNLOAD_ITEMS,
    }

    overall_stats = {
        "total_bytes": total_bytes,
        "downloaded_base": 0,
    }

    # Bắt đầu tải từng file
    for item in DOWNLOAD_ITEMS:
        success = download_file(item, target_dir, log_path, status_data, overall_stats)
        if not success and status_data.get("overall_status") == "HALTED_LOW_DISK":
            break

    if status_data["completed_files_count"] == len(DOWNLOAD_ITEMS):
        status_data["overall_status"] = "ALL_COMPLETED"
        log_message(
            log_path,
            f"=== TOÀN BỘ {len(DOWNLOAD_ITEMS)} FILE BATCH 2 ĐÃ ĐƯỢC TẢI VỀ "
            "THÀNH CÔNG VÀ KIỂM TRA TOÀN VẸN ===",
        )
    else:
        status_data["overall_status"] = "FINISHED_WITH_SOME_FAILURES"
        log_message(log_path, f"Hoàn tất phiên. Đã xong {status_data['completed_files_count']}/{len(DOWNLOAD_ITEMS)} file.")

    status_data["finish_time"] = datetime.now(timezone.utc).isoformat()
    write_status_files(target_dir, status_data)


if __name__ == "__main__":
    main()
