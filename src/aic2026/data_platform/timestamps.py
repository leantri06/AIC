"""BTC timestamps keyed by image ordinal, preserving duplicate frame IDs."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PureWindowsPath


@dataclass
class TimestampMap:
    by_frame: dict[int, float]
    by_ordinal: dict[int, tuple[int, float]]

    def timestamp(self, frame_id: int, keyframe_path: str | Path | None = None) -> float | None:
        if keyframe_path:
            stem = PureWindowsPath(str(keyframe_path)).stem
            if stem.isdecimal():
                row = self.by_ordinal.get(int(stem))
                if row is not None and row[0] == frame_id:
                    return row[1]
        return self.by_frame.get(frame_id)


def load_timestamp_map(path: Path) -> TimestampMap:
    path = path.resolve()
    stat = path.stat()
    return _load(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2048)
def _load(path: str, mtime_ns: int, size: int) -> TimestampMap:
    by_frame: dict[int, float] = {}
    by_ordinal: dict[int, tuple[int, float]] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame, pts = int(row["frame_idx"]), float(row["pts_time"])
                if not math.isfinite(pts) or pts < 0:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            by_frame[frame] = pts  # Legacy callers without image identity.
            try:
                by_ordinal[int(row["n"])] = (frame, pts)
            except (KeyError, ValueError, TypeError):
                pass
    return TimestampMap(by_frame, by_ordinal)
