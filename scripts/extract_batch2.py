"""Safely extract completed Batch 2 archives into a canonical layout.

The downloader may run at the same time. Only final ``*.zip`` files are read;
temporary ``*.part`` files are ignored. Archive members are normalized into
folders understood by the project, for example ``videos`` -> ``Videos`` and
``media-info`` -> ``Metadata``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


MIN_FREE_BYTES = 30 * 1024**3
STATE_FILE = ".extraction_state.json"
TOP_LEVEL_NAMES = {
    "videos": "Videos",
    "video": "Videos",
    "media-info": "Metadata",
    "media_info": "Metadata",
    "metadata": "Metadata",
    "keyframes": "Keyframes",
    "objects": "Objects",
    "map-keyframes": "map-keyframes",
    "clip-features-32": "clip_features",
    "clip-features": "clip_features",
    "clip_features": "clip_features",
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_raw_dir() -> Path:
    return workspace_root() / "data" / "batches" / "batch2" / "raw"


def default_destination() -> Path:
    return workspace_root() / "data" / "batches" / "batch2" / "extracted"


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def load_state(destination: Path) -> dict:
    path = destination / STATE_FILE
    if not path.exists():
        return {"schema_version": 1, "archives": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "archives": {}}
    data.setdefault("schema_version", 1)
    data.setdefault("archives", {})
    return data


def save_state(destination: Path, state: dict) -> None:
    path = destination / STATE_FILE
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def safe_relative_path(member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe archive member: {member_name}")
    parts = list(pure.parts)
    parts[0] = TOP_LEVEL_NAMES.get(parts[0].casefold(), parts[0])
    return Path(*parts)


def archive_summary(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as zf:
        members = [item for item in zf.infolist() if not item.is_dir()]
        for item in members:
            safe_relative_path(item.filename)
        return {
            "archive": archive.name,
            "entries": len(members),
            "uncompressed_bytes": sum(item.file_size for item in members),
            "categories": sorted(
                {safe_relative_path(item.filename).parts[0] for item in members}
            ),
        }


def verify_archive_output(archive: Path, destination: Path) -> tuple[int, int]:
    checked = 0
    total_bytes = 0
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            if item.is_dir():
                continue
            target = destination / safe_relative_path(item.filename)
            if not target.is_file():
                raise FileNotFoundError(f"Missing extracted file: {target}")
            actual_size = target.stat().st_size
            if actual_size != item.file_size:
                raise ValueError(
                    f"Size mismatch for {target}: {actual_size} != {item.file_size}"
                )
            checked += 1
            total_bytes += actual_size
    return checked, total_bytes


def extract_archive(archive: Path, destination: Path) -> tuple[int, int]:
    extracted = 0
    extracted_bytes = 0
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            relative = safe_relative_path(item.filename)
            target = destination / relative
            resolved = target.resolve()
            if not resolved.is_relative_to(destination_resolved):
                raise ValueError(f"Unsafe extraction target: {target}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size == item.file_size:
                    continue
                # A previous extraction may have been interrupted while writing
                # this member. Remove only that incomplete destination file and
                # recreate it from the still-verified source archive.
                target.unlink()
            with zf.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            extracted += 1
            extracted_bytes += item.file_size
    return extracted, extracted_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    parser.add_argument("--destination", type=Path, default=default_destination())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--delete-archives-after-verify",
        action="store_true",
        help="Delete each source ZIP only after all extracted files pass verification.",
    )
    parser.add_argument(
        "--only-support",
        action="store_true",
        help="Extract only small support archives such as media-info, map and objects.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state = load_state(destination)

    archives = sorted(raw_dir.glob("*.zip"))
    if args.only_support:
        prefixes = ("media-info", "map-keyframes", "objects", "clip-features")
        archives = [p for p in archives if p.name.casefold().startswith(prefixes)]
    summaries = [archive_summary(path) for path in archives]
    total_uncompressed = sum(item["uncompressed_bytes"] for item in summaries)
    free_bytes = shutil.disk_usage(destination).free

    print(f"Completed ZIP files: {len(archives)}")
    print(f"Uncompressed size: {format_bytes(total_uncompressed)}")
    print(f"Free disk space: {format_bytes(free_bytes)}")
    for item in summaries:
        print(
            f"- {item['archive']}: {item['entries']} files, "
            f"{format_bytes(item['uncompressed_bytes'])}, "
            f"{', '.join(item['categories'])}"
        )

    if args.status or args.dry_run:
        return 0
    if not archives:
        print("No completed archives to extract.")
        return 0
    if not args.delete_archives_after_verify and free_bytes < total_uncompressed + MIN_FREE_BYTES:
        print(
            "Insufficient disk space while preserving ZIP files: need "
            f"{format_bytes(total_uncompressed + MIN_FREE_BYTES)}, have "
            f"{format_bytes(free_bytes)}.",
            file=sys.stderr,
        )
        return 2

    for archive, summary in zip(archives, summaries, strict=True):
        completed = state["archives"].get(archive.name)
        if completed and completed.get("verified"):
            if archive.exists() and args.delete_archives_after_verify:
                archive.unlink()
                completed["archive_deleted_at"] = datetime.now(timezone.utc).isoformat()
                save_state(destination, state)
            print(f"Already verified: {archive.name}")
            continue

        if args.delete_archives_after_verify:
            current_free = shutil.disk_usage(destination).free
            required = summary["uncompressed_bytes"] + MIN_FREE_BYTES
            if current_free < required:
                print(
                    f"Insufficient space for {archive.name}: need "
                    f"{format_bytes(required)}, have {format_bytes(current_free)}.",
                    file=sys.stderr,
                )
                return 2

        print(f"Extracting {archive.name}...")
        new_files, new_bytes = extract_archive(archive, destination)
        checked_files, checked_bytes = verify_archive_output(archive, destination)
        state["archives"][archive.name] = {
            "verified": True,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "archive_bytes": archive.stat().st_size,
            "files": checked_files,
            "uncompressed_bytes": checked_bytes,
            "new_files": new_files,
            "new_bytes": new_bytes,
            "categories": summary["categories"],
        }
        save_state(destination, state)
        print(
            f"Verified {archive.name}: {checked_files} files, "
            f"{format_bytes(checked_bytes)}"
        )
        if args.delete_archives_after_verify:
            archive.unlink()
            state["archives"][archive.name]["archive_deleted_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            save_state(destination, state)
            print(f"Deleted verified source ZIP: {archive.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
