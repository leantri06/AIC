"""Run the verified one-video TransNet V2 pipeline over selected Batch 2 videos."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS = ROOT / "data/batches/batch2/extracted/Videos"
DEFAULT_OUTPUT = ROOT / "data/batches/batch2/transnetv2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-id", action="append", help="Exact filename stem; repeat as needed.")
    parser.add_argument("--reserve-gib", type=float, default=30.0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.reserve_gib) or args.reserve_gib <= 0:
        parser.error("--reserve-gib must be positive")

    videos = sorted(
        path for path in args.videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}
    )
    if len({path.stem.casefold() for path in videos}) != len(videos):
        parser.error("Duplicate video IDs")
    if args.video_id:
        requested = set(args.video_id)
        missing = requested - {path.stem for path in videos}
        if missing:
            parser.error(f"Unknown video IDs: {sorted(missing)}")
        videos = [path for path in videos if path.stem in requested]
    if not videos:
        parser.error("No videos selected")
    if args.list:
        print("\n".join(path.name for path in videos))
        print(f"{len(videos)} videos")
        return 0

    output = args.output_root.resolve()
    completed = output / "completed"
    partial = output / "partial"
    failed = output / "failed"
    for path in (completed, partial, failed):
        path.mkdir(parents=True, exist_ok=True)

    lock_path = output / ".batch.lock"
    failures = []
    with lock_path.open("a+b") as lock:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        for number, video in enumerate(videos, 1):
            destination = completed / video.stem
            report = destination / "benchmark.json"
            if report.is_file():
                data = json.loads(report.read_text(encoding="utf-8"))
                if Path(data["video"]).resolve() != video.resolve():
                    raise RuntimeError(f"Completed report source mismatch: {video.stem}")
                print(f"[{number}/{len(videos)}] SKIP {video.stem}", flush=True)
                continue
            if destination.exists():
                raise RuntimeError(f"Incomplete final directory must be inspected: {destination}")
            free = shutil.disk_usage(output).free
            if free < args.reserve_gib * 1024**3:
                raise RuntimeError(f"Less than {args.reserve_gib:g} GiB free; stopped safely")

            working = partial / f"{video.stem}-{uuid.uuid4().hex[:8]}"
            command = [
                sys.executable, "-u", str(Path(__file__).with_name("benchmark_transnetv2.py")),
                "--video", str(video), "--output", str(working),
            ]
            print(f"[{number}/{len(videos)}] RUN {video.name}", flush=True)
            try:
                subprocess.run(command, check=True)
                if not (working / "benchmark.json").is_file():
                    raise RuntimeError("Child completed without benchmark.json")
                working.replace(destination)
                print(f"DONE {video.stem} -> {destination}", flush=True)
            except KeyboardInterrupt:
                print(f"Stopped; partial output kept at {working}", flush=True)
                return 130
            except Exception as exc:
                preserved = failed / f"{video.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
                if working.exists():
                    working.replace(preserved)
                failures.append({"video": video.name, "error": str(exc), "output": str(preserved)})
                print(f"ERROR {video.name}: {exc}; kept at {preserved}", flush=True)

    summary = {"selected": len(videos), "failures": failures}
    (output / "last-run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
