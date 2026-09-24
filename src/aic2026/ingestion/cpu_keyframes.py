"""Strict manifest reader for extract_keyframes_cpu.py outputs (not BTC ordinals)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from aic2026.models import FrameRecord


def cpu_video_records(root: Path, video_id: str) -> list[FrameRecord]:
    root = Path(root).resolve()
    if Path(video_id).name != video_id or video_id in (".", ".."):
        raise ValueError("Invalid video ID")
    image_dir = root / "Keyframes" / video_id
    owner = json.loads((image_dir / ".owner.json").read_text(encoding="utf-8"))
    report = json.loads((root / "keyframe-reports" / f"{video_id}.json").read_text(encoding="utf-8"))
    if owner != {"source": report["source"], "settings": report["settings"]}:
        raise ValueError(f"{video_id}: owner/report mismatch")
    videos = [p for p in (root / "Videos").iterdir()
              if p.stem == video_id and p.suffix.lower() in (".mp4", ".mov")]
    if len(videos) != 1:
        raise ValueError(f"{video_id}: expected exactly one source video")
    video = videos[0]
    stat = video.stat()
    if stat.st_size != report["source"]["bytes"] or stat.st_mtime_ns != report["source"]["mtime_ns"]:
        raise ValueError(f"{video_id}: source changed since extraction")
    with (root / "map-keyframes" / f"{video_id}.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != report["kept_frames"]:
        raise ValueError(f"{video_id}: incomplete map")
    records = []
    previous_frame, previous_time = -1, -1.0
    for ordinal, row in enumerate(rows, 1):
        frame = int(row["frame_idx"])
        pts, source_pts = float(row["pts_time"]), float(row["source_pts_time"])
        if (int(row["n"]) != ordinal or frame <= previous_frame
                or frame >= report["decoded_frames"] or not math.isfinite(pts)
                or not math.isfinite(source_pts) or pts < 0 or pts < previous_time):
            raise ValueError(f"{video_id}: invalid mapping at row {ordinal}")
        image = image_dir / f"{frame:09d}.jpg"
        if not image.is_file() or image.stat().st_size == 0:
            raise ValueError(f"{video_id}: missing/empty image {image.name}")
        records.append(FrameRecord(vector_id=len(records), video_id=video_id,
            frame_id=frame, pts_time=pts, source_pts_time=source_pts,
            keyframe_path=str(image), video_path=str(video), clip_feature_index=len(records)))
        previous_frame, previous_time = frame, pts
    if {p.name for p in image_dir.glob("*.jpg")} != {Path(r.keyframe_path).name for r in records}:
        raise ValueError(f"{video_id}: unexpected images")
    return records
