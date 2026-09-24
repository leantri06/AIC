"""Package unchanged CPU JPEGs into portable shards; never uploads anything."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "kaggle")]
from aic2026.ingestion.cpu_keyframes import cpu_video_records
from siglip2_runner import SCHEMA, atomic_json, digest, fingerprint, shard_records


class SequentialZipOutput:
    """Non-seekable ZIP output avoids flushing a buffered HDD write per JPEG."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)

    def tell(self):
        return self.stream.tell()

    def flush(self):
        return self.stream.flush()


def portable_record(record, member):
    return dict(video_id=record.video_id, frame_id=record.frame_id,
        pts_time=record.pts_time, source_pts_time=record.source_pts_time,
        keyframe_path=f"Keyframes/{record.video_id}/{Path(record.keyframe_path).name}",
        video_path=f"Videos/{Path(record.video_path).name}", member=member)


def write_shard(target, records):
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(portable_record(r, f"images/{i:06d}.jpg"),
        sort_keys=True) + "\n" for i, r in enumerate(records)).encode()
    tmp = target.with_suffix(".tmp")
    with tmp.open("wb", buffering=64 * 1024**2) as stream:
        with zipfile.ZipFile(SequentialZipOutput(stream), "w",
                compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            z.writestr("records.jsonl", payload)
            for i, record in enumerate(records):
                z.write(record.keyframe_path, f"images/{i:06d}.jpg")
    os.replace(tmp, target)
    import hashlib
    job = dict(id=target.stem, file=target.name, count=len(records),
        sha256=digest(target), records_sha256=hashlib.sha256(payload).hexdigest(),
        bytes=target.stat().st_size)
    shard_records(target, job)
    return job


def parity_fixture(data, target):
    import numpy as np
    features = np.load(data / "processed/siglip2_features.npy", mmap_mode="r")
    selected = set(np.linspace(0, len(features)-1, 6, dtype=int).tolist())
    rows = []
    with (data / "processed/siglip2_manifest.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in selected:
                row = json.loads(line)
                suffix = row["keyframe_path"].replace("\\", "/").split("/Keyframes/")[-1]
                rows.append((i, data / "extracted/Keyframes" / suffix))
    tmp = target.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        memory = io.BytesIO()
        np.save(memory, np.asarray(features[[i for i, _ in rows]], dtype=np.float32))
        z.writestr("expected.npy", memory.getvalue())
        for n, (_, path) in enumerate(rows):
            z.write(path, f"{n:03d}.jpg")
    os.replace(tmp, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=REPO.parent)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard-frames", type=int, default=2048)
    p.add_argument("--group-gib", type=float, default=8)
    p.add_argument("--video", action="append")
    p.add_argument("--sample-per-video", type=int, default=0)
    p.add_argument("--status", action="store_true", help="Read package progress without starting work")
    args = p.parse_args()
    if args.shard_frames < 1 or args.group_gib <= 0 or args.sample_per_video < 0:
        p.error("Invalid package size")
    data = args.workspace.resolve() / "data"
    root = data / "batches/batch2/extracted"
    output = args.output.resolve()
    if not output.is_relative_to(data) or output == root or output.is_relative_to(root):
        p.error("Output must be a separate directory under workspace/data")
    if args.status:
        jobs = [json.loads(path.read_text()) for path in (output / "state").glob("*.json")]
        print(json.dumps(dict(output=str(output), completed_shards=len(jobs),
            completed_frames=sum(j["count"] for j in jobs),
            completed_gib=sum(j["bytes"] for j in jobs)/2**30,
            catalog_ready=(output / "control/catalog.json").exists()), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    config = dict(schema=SCHEMA, source=str(root), shard_frames=args.shard_frames,
        group_gib=args.group_gib, videos=sorted(args.video or []), sample_per_video=args.sample_per_video)
    config_file = output / "package-config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Package configuration changed; use a new output directory")
    atomic_json(config_file, config)
    control = output / "control"
    control.mkdir(exist_ok=True)
    ids = args.video or sorted(p.stem for p in (root / "keyframe-reports").glob("*.json"))
    jobs, inventory, group, group_bytes = [], [], 0, 0
    for video_id in ids:
        records = cpu_video_records(root, video_id)
        report_file = root / "keyframe-reports" / f"{video_id}.json"
        source_id = fingerprint(dict(report=digest(report_file),
            csv=digest(root / "map-keyframes" / f"{video_id}.csv")))
        if args.sample_per_video:
            import numpy as np
            records = [records[i] for i in np.linspace(0, len(records)-1,
                min(args.sample_per_video, len(records)), dtype=int)]
        inventory.append(dict(video_id=video_id, frames=len(records), source_id=source_id))
        for start in range(0, len(records), args.shard_frames):
            chunk = records[start:start + args.shard_frames]
            estimated = sum(Path(r.keyframe_path).stat().st_size for r in chunk) + len(chunk)*1024
            if group_bytes and group_bytes + estimated > args.group_gib*2**30:
                group += 1
                group_bytes = 0
            name = f"{video_id}-{start//args.shard_frames:04d}"
            target = output / f"images-{group:03d}" / f"{name}.aicshard"
            state_file = output / "state" / f"{name}.json"
            if state_file.exists() and target.exists():
                job = json.loads(state_file.read_text())
                if job["source_id"] != source_id:
                    raise ValueError(f"Source mapping changed: {name}")
                shard_records(target, job)
            else:
                if shutil.disk_usage(output).free < estimated + 30*2**30:
                    raise RuntimeError("Keep 30 GiB free; package is resumable")
                job = write_shard(target, chunk)
                job.update(source_id=source_id, group=target.parent.name)
                state_file.parent.mkdir(exist_ok=True)
                atomic_json(state_file, job)
            jobs.append(job)
            group_bytes += job["bytes"]
        print(f"{video_id}: {len(records)} images, {len(jobs)} shards total", flush=True)
    parity_fixture(data, control / "parity.aicshard")
    catalog = dict(schema=SCHEMA, keyframe_source="CPU", sample=bool(args.sample_per_video),
        frames=sum(j["count"] for j in jobs), videos=inventory, shards=jobs,
        parity_sha256=digest(control / "parity.aicshard"))
    atomic_json(control / "catalog.json", catalog)
    shutil.copy2(REPO / "kaggle/siglip2_runner.py", control / "siglip2_runner.py")
    if (REPO / "kaggle/batch2_siglip2_t4x2.ipynb").exists():
        shutil.copy2(REPO / "kaggle/batch2_siglip2_t4x2.ipynb", output / "batch2_siglip2_t4x2.ipynb")
    print(json.dumps(dict(output=str(output), frames=catalog["frames"], shards=len(jobs),
        groups=group+1, bytes=sum(j["bytes"] for j in jobs)), indent=2), flush=True)


if __name__ == "__main__":
    main()
