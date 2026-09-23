"""Audit the canonical local Batch 2 layout without changing its data files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_VIDEO_COUNT = 495
EXPECTED_VIDEO_BYTES = 228_473_998_977
EXPECTED_METADATA_COUNT = 605
VIDEO_NAME = re.compile(
    r"^(?P<collection>[A-Z]\d+)[_-](?P<ordinal>V\d+)\.(?P<suffix>mp4|mov)$",
    re.IGNORECASE,
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_inventory(extracted: Path) -> dict:
    videos = sorted(
        path
        for pattern in ("*.mp4", "*.mov")
        for path in (extracted / "Videos").rglob(pattern)
    )
    metadata = sorted((extracted / "Metadata").rglob("*.json"))
    invalid_names = [path.name for path in videos if not VIDEO_NAME.fullmatch(path.name)]
    identifiers = [path.stem for path in videos]
    duplicate_ids = sorted(name for name, count in Counter(identifiers).items() if count > 1)
    collections = Counter()
    collection_bytes = Counter()
    for path in videos:
        match = VIDEO_NAME.fullmatch(path.name)
        if match:
            collection = match["collection"]
            collections[collection] += 1
            collection_bytes[collection] += path.stat().st_size

    video_bytes = sum(path.stat().st_size for path in videos)
    checks = {
        "expected_video_count": len(videos) == EXPECTED_VIDEO_COUNT,
        "expected_video_bytes": video_bytes == EXPECTED_VIDEO_BYTES,
        "expected_metadata_count": len(metadata) == EXPECTED_METADATA_COUNT,
        "valid_video_names": not invalid_names,
        "unique_video_ids": not duplicate_ids,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(extracted),
        "layout": {"videos": "Videos/*.{mp4,mov}", "metadata": "Metadata/**/*.json"},
        "videos": {
            "count": len(videos),
            "bytes": video_bytes,
            "collections": [
                {"id": key, "video_count": collections[key], "bytes": collection_bytes[key]}
                for key in sorted(collections)
            ],
            "invalid_names": invalid_names,
            "duplicate_ids": duplicate_ids,
        },
        "metadata": {"json_count": len(metadata)},
        "checks": checks,
        "valid": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=workspace_root() / "data" / "batches" / "batch2" / "extracted",
    )
    parser.add_argument("--write", action="store_true", help="Write inventory.json beside data.")
    args = parser.parse_args()

    inventory = build_inventory(args.extracted_dir.resolve())
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    if args.write:
        output = args.extracted_dir / "inventory.json"
        output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output}")
    return 0 if inventory["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
