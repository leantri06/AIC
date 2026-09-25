"""Resumable sequential downloader for SigLIP2 Giant text model checkpoint."""
from __future__ import annotations

import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

from aic2026.paths import default_data_root

REPO_ID = "google/siglip2-giant-opt-patch16-384"
FILENAME = "model-00001-of-00002.safetensors"
EXPECTED_BYTES = 4994453512
SHA256 = "9f6da5ad2e3178b0c1197b740c9ffefee0bc733ee8759095a5ea8c4c845a3f4e"
URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"
MIRROR_URL = f"https://hf-mirror.com/{REPO_ID}/resolve/main/{FILENAME}"


def get_target_paths() -> tuple[Path, Path]:
    hf_hub = default_data_root() / ".cache" / "huggingface" / "hub"
    model_dir = hf_hub / "models--google--siglip2-giant-opt-patch16-384"
    blobs_dir = model_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    final_blob = blobs_dir / SHA256
    part_blob = blobs_dir / f"{SHA256}.part"
    return final_blob, part_blob


def setup_snapshot_links(final_blob: Path) -> None:
    snap_dir = final_blob.parent.parent / "snapshots" / "a713301b217d38485fb2204c808367d10bc3cc40"
    if not snap_dir.exists():
        snap_dir.mkdir(parents=True, exist_ok=True)
    
    snap_target = snap_dir / FILENAME
    if not snap_target.exists():
        try:
            os.link(final_blob, snap_target)
            print(f"[LINK] Da tao hardlink sang {snap_target}")
        except Exception:
            try:
                import shutil
                shutil.copyfile(final_blob, snap_target)
                print(f"[COPY] Da sao chep sang {snap_target}")
            except Exception as e:
                print(f"[CANH BAO] Khong the link: {e}")

    # Prune model.safetensors.index.json to only reference text shard
    idx_file = snap_dir / "model.safetensors.index.json"
    if idx_file.exists():
        try:
            idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
            idx_data["weight_map"] = {
                k: v for k, v in idx_data.get("weight_map", {}).items() if v == FILENAME
            }
            idx_file.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")
            print("[INDEX] Da cap nhat model.safetensors.index.json chi giu shard text")
        except Exception as e:
            print(f"[CANH BAO] Loi cap nhat index.json: {e}")


def download_file(url: str) -> bool:
    final_blob, part_file = get_target_paths()
    if final_blob.exists() and final_blob.stat().st_size == EXPECTED_BYTES:
        print(f"[OK] File {FILENAME} da ton tai day du tai {final_blob}")
        setup_snapshot_links(final_blob)
        return True

    existing = 0
    if part_file.exists():
        existing = part_file.stat().st_size
        if existing > EXPECTED_BYTES:
            print(f"[RESET] Part file lon hon expected ({existing} > {EXPECTED_BYTES}), xoa tai lai.")
            part_file.unlink()
            existing = 0

    print(f"[START] Dang tai {FILENAME} ({EXPECTED_BYTES / 1024 / 1024 / 1024:.2f} GB) tu {url}...")
    if existing > 0:
        print(f"[RESUME] Tiep tuc tu byte {existing} ({existing / 1024 / 1024:.1f} MB / {EXPECTED_BYTES / 1024 / 1024:.1f} MB)...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIC/PEGASUS",
    }
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"

    max_retries = 20
    retry = 0
    while retry < max_retries:
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                downloaded = existing
                with open(part_file, mode) as fh:
                    last_log = time.time()
                    last_bytes = downloaded
                    while True:
                        buf = resp.read(256 * 1024)
                        if not buf:
                            break
                        fh.write(buf)
                        downloaded += len(buf)

                        now = time.time()
                        if now - last_log >= 2.5:
                            dt = now - last_log
                            db = downloaded - last_bytes
                            speed = (db / dt) / (1024 * 1024)
                            pct = (downloaded / EXPECTED_BYTES) * 100.0
                            eta = (EXPECTED_BYTES - downloaded) / (db / dt) if db > 0 else 0
                            print(
                                f"\r[Tien do] {downloaded/1024/1024:.1f}/{EXPECTED_BYTES/1024/1024:.1f} MB "
                                f"({pct:.1f}%) - Toc do: {speed:.2f} MB/s - ETA: {int(eta//60)}m{int(eta%60)}s",
                                end="",
                                flush=True,
                            )
                            last_log = now
                            last_bytes = downloaded

            if part_file.stat().st_size == EXPECTED_BYTES:
                print(f"\n[HOAN TAT] Tai xong {FILENAME}!")
                if final_blob.exists():
                    final_blob.unlink()
                part_file.rename(final_blob)
                setup_snapshot_links(final_blob)
                return True
            else:
                existing = part_file.stat().st_size
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                req = urllib.request.Request(url, headers=headers)
                retry += 1
                print(f"\n[RETRY] Ket noi bi ngat, dang thu lai lan {retry}/{max_retries} tu {existing/1024/1024:.1f} MB...")
                time.sleep(2)
        except Exception as exc:
            retry += 1
            print(f"\n[LOI] {exc}. Thu lai lan {retry}/{max_retries} sau 3s...")
            time.sleep(3)
            if part_file.exists():
                existing = part_file.stat().st_size
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                req = urllib.request.Request(url, headers=headers)
    return False


if __name__ == "__main__":
    url_to_use = MIRROR_URL if "--mirror" in sys.argv else URL
    download_file(url_to_use)
