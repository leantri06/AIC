"""Read-only audit of the precomputed index and relocated BTC keyframes."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np

from aic2026.paths import default_data_root, resolve_data_root_and_paths


def main():
    paths = resolve_data_root_and_paths(str(default_data_root()))
    vectors = np.load(paths["features"], mmap_mode="r")
    counts = Counter()
    missing = []
    mismatches = []
    seen = set()
    duplicates = 0
    maps = {}
    available = {}
    rows = 0
    with open(paths["manifest"], encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["vector_id"] != rows:
                raise ValueError(f"Misaligned vector_id at row {rows}")
            vid, frame = rec["video_id"], rec["frame_id"]
            counts[vid] += 1
            pair = (vid, frame)
            duplicates += pair in seen
            seen.add(pair)
            name = PureWindowsPath(rec["keyframe_path"]).name
            if vid not in available:
                folder = Path(paths["keyframes"]) / vid
                available[vid] = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
                csv_file = Path(paths["map_keyframes"]) / f"{vid}.csv"
                if csv_file.is_file():
                    with csv_file.open(encoding="utf-8-sig") as csv_handle:
                        maps[vid] = {int(r["n"]): int(r["frame_idx"]) for r in csv.DictReader(csv_handle)}
            if name not in available[vid]:
                missing.append((vid, name))
            if vid in maps and name.rsplit('.', 1)[0].isdigit():
                expected = maps[vid].get(int(name.rsplit('.', 1)[0]))
                if expected is not None and expected != frame:
                    mismatches.append((rows, vid, name, frame, expected))
            rows += 1
    norm_min, norm_max = float('inf'), 0.0
    for start in range(0, len(vectors), 8192):
        block = np.asarray(vectors[start:start+8192], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite vectors near row {start}")
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    report = dict(rows=rows, shape=list(vectors.shape), dtype=str(vectors.dtype),
                  videos=len(counts), video_prefixes=dict(Counter(v[:1] for v in counts)),
                  norm_min=norm_min, norm_max=norm_max, duplicate_frame_keys=duplicates,
                  missing_images=len(missing), missing_image_examples=missing[:10],
                  mapping_mismatches=len(mismatches), mapping_examples=mismatches[:10],
                  videos_with_maps=len(maps), paths=paths)
    output = Path(paths['data_dir']) / 'processed' / 'siglip2_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if rows != len(vectors) or missing or mismatches or norm_min <= 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
