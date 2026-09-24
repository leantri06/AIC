"""Verify and merge downloaded shard outputs; does not switch the application index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kaggle"))
from siglip2_runner import atomic_json, digest, validate_result


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--results", type=Path, action="append", required=True)
    p.add_argument("--extracted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog["sample"]:
        raise ValueError("Cannot promote a sample catalog to a full index")
    chosen, config_id = [], None
    for job in catalog["shards"]:
        matches = [path for root in args.results for path in root.rglob(f"{job['id']}.npz")]
        if not matches:
            raise ValueError(f"Missing result {job['id']}")
        for path in matches:
            meta = validate_result(path, job, config_id)
            config_id = meta["config_id"]
        if len({digest(path) for path in matches}) != 1:
            raise ValueError(f"Conflicting duplicate output {job['id']}")
        chosen.append((job, matches[0]))
    # New destination only; an interrupted merge is never an active index.
    args.output.mkdir(parents=True, exist_ok=False)
    features_tmp = args.output / "siglip2_features.partial.npy"
    manifest_tmp = args.output / "siglip2_manifest.partial.jsonl"
    matrix = np.lib.format.open_memmap(features_tmp, mode="w+", dtype=np.float32,
        shape=(catalog["frames"], 1152))
    cursor, seen = 0, set()
    with manifest_tmp.open("w", encoding="utf-8") as manifest:
        for job, path in chosen:
            with np.load(path, allow_pickle=False) as z:
                rows = [json.loads(line) for line in str(z["records"].item()).splitlines()]
                if len(rows) != job["count"]:
                    raise ValueError("Wrong row count")
                matrix[cursor:cursor+len(rows)] = z["features"]
                for row in rows:
                    key = row["video_id"], row["frame_id"]
                    if key in seen:
                        raise ValueError(f"Duplicate frame {key}")
                    seen.add(key)
                    row.pop("member")
                    for name in ("keyframe_path", "video_path"):
                        target = (args.extracted / row[name]).resolve()
                        if not target.is_relative_to(args.extracted.resolve()) or not target.is_file():
                            raise ValueError(f"Missing or unsafe local path {target}")
                        row[name] = str(target)
                    row.update(vector_id=cursor, clip_feature_index=cursor)
                    manifest.write(json.dumps(row, ensure_ascii=False)+"\n")
                    cursor += 1
    if cursor != catalog["frames"]:
        raise ValueError("Incomplete merge")
    matrix.flush()
    del matrix
    os.replace(features_tmp, args.output / "siglip2_features.npy")
    os.replace(manifest_tmp, args.output / "siglip2_manifest.jsonl")
    atomic_json(args.output / "merge-report.json", dict(frames=cursor, config_id=config_id,
        catalog_sha256=digest(args.catalog), complete=True, app_switched=False))
    print(f"Verified {cursor} frames; app index unchanged")


if __name__ == "__main__":
    main()
