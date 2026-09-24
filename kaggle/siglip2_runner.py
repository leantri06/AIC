"""Portable, resumable image-only SigLIP2 inference. No project installation needed."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import zipfile

MODEL = "ViT-SO400M-14-SigLIP2-378"
REVISION = "93e0e6ad1063507a7777d0b570292a29a3d94c5f"
WEIGHTS_SHA = "7ed1f8fb32da2d18fd71028ca2c6d61ea5fec9a6a8f99d1d9159b62c0c1b7a9f"
SCHEMA = 1


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def balanced(jobs, workers):
    groups, loads = [[] for _ in range(workers)], [0] * workers
    for job in sorted(jobs, key=lambda x: (-x["count"], x["id"])):
        idx = min(range(workers), key=loads.__getitem__)
        groups[idx].append(job)
        loads[idx] += job["count"]
    return groups


def shard_records(path, job):
    if digest(path) != job["sha256"]:
        raise ValueError(f"Corrupt input shard {job['id']}")
    with zipfile.ZipFile(path) as z:
        payload = z.read("records.jsonl")
        if hashlib.sha256(payload).hexdigest() != job["records_sha256"]:
            raise ValueError("Mapping digest mismatch")
        rows = [json.loads(line) for line in payload.splitlines()]
        expected = [r["member"] for r in rows]
        names = z.namelist()
        if (len(rows) != job["count"] or len(set(expected)) != len(rows)
                or len(set(names)) != len(names) or set(names) != set(expected) | {"records.jsonl"}):
            raise ValueError("Shard entries do not match mapping")
        return rows


def validate_result(path, job, config_id=None):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        if (meta["schema"] != SCHEMA or meta["input_sha256"] != job["sha256"]
                or meta["records_sha256"] != job["records_sha256"]
                or meta["job_id"] != job["id"] or meta["count"] != job["count"]):
            raise ValueError(f"Wrong provenance: {path}")
        if fingerprint(meta["config"]) != meta["config_id"]:
            raise ValueError("Corrupt config fingerprint")
        if config_id is not None and meta["config_id"] != config_id:
            raise ValueError("Resume config mismatch; use a new output directory")
        if meta["config"]["checkpoint_sha256"] != WEIGHTS_SHA:
            raise ValueError("Wrong checkpoint")
        values = z["features"]
        if (values.shape != (job["count"], 1152) or values.dtype != np.float32
                or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)):
            raise ValueError(f"Invalid features: {path}")
        # Mapping is byte-for-byte the mapping packaged with the input shard.
        payload = str(z["records"].item()).encode()
        if hashlib.sha256(payload).hexdigest() != job["records_sha256"]:
            raise ValueError("Output mapping mismatch")
    return meta


def save_result(path, values, records_payload, meta):
    import numpy as np
    path = Path(path)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez(f, features=values, records=records_payload, metadata=json.dumps(meta))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_model(checkpoint, gpu, lock):
    import gc
    import open_clip
    import torch
    # Serialize CPU model construction so two full text towers do not coexist.
    with lock:
        cfg = open_clip.get_pretrained_cfg(MODEL, "webli")
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=str(checkpoint), device="cpu",
            image_mean=cfg["mean"], image_std=cfg["std"],
            image_interpolation=cfg["interpolation"], image_resize_mode=cfg["resize_mode"])
        del model.text
        gc.collect()
        model.eval().to(f"cuda:{gpu}")
    return model, preprocess


def encode_payloads(payloads, model, preprocess, gpu, batch_size, precision, pool):
    import numpy as np
    import torch
    from PIL import Image

    def prepare(payload):
        with Image.open(io.BytesIO(payload)) as image:
            return preprocess(image.convert("RGB"))

    outputs = []
    for start in range(0, len(payloads), batch_size):
        batch = torch.stack(list(pool.map(prepare, payloads[start:start + batch_size])))
        context = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()
        with torch.inference_mode(), context:
            features = model.encode_image(batch.to(f"cuda:{gpu}"))
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        outputs.append(features.cpu().numpy())
    return np.concatenate(outputs)


def worker(gpu, jobs, args, checkpoint, lock, deadline):
    import numpy as np
    import torch
    import open_clip
    import PIL
    import timm
    import torchvision
    torch.set_num_threads(2)
    torch.cuda.set_device(gpu)
    model, preprocess = load_model(checkpoint, gpu, lock)
    config = dict(schema=SCHEMA, model=MODEL, checkpoint_sha256=WEIGHTS_SHA,
        precision=args.precision, normalization="L2-float32", preprocess=repr(preprocess),
        torch=torch.__version__, open_clip=open_clip.__version__, pillow=PIL.__version__,
        torchvision=torchvision.__version__, timm=timm.__version__)
    config_id = fingerprint(config)
    output = Path(args.output)
    with ThreadPoolExecutor(max_workers=args.decode_threads) as pool:
        # Gate every GPU against old batch1 vectors before accepting any shard.
        with zipfile.ZipFile(Path(args.catalog).parent / "parity.aicshard") as z:
            reference = np.load(io.BytesIO(z.read("expected.npy")), allow_pickle=False)
            payloads = [z.read(f"{i:03d}.jpg") for i in range(len(reference))]
        computed = encode_payloads(payloads, model, preprocess, gpu, args.batch_size, args.precision, pool)
        cosine = np.sum(computed * (reference / np.linalg.norm(reference, axis=1, keepdims=True)), axis=1)
        if not np.isfinite(cosine).all() or np.min(cosine) < 0.999:
            raise ValueError(f"GPU {gpu}: batch1 parity failed: {cosine.tolist()}")
        atomic_json(output / f"gpu-{gpu}-parity.json", dict(config=config,
            config_id=config_id, cosine=cosine.tolist(), device=torch.cuda.get_device_name(gpu)))
        done, skipped, seconds, frames = [], [], 0.0, 0
        start_worker = time.monotonic()
        for job in jobs:
            if time.monotonic() >= deadline:
                break
            target = output / f"{job['id']}.npz"
            candidates = ([target] if target.exists() else [])
            for root in args.resume_root:
                candidates.extend(Path(root).rglob(f"{job['id']}.npz"))
            if candidates:
                for candidate in candidates:
                    validate_result(candidate, job, config_id)
                skipped.append(job["id"])
                continue
            started = time.monotonic()
            rows = shard_records(job["path"], job)
            vectors = []
            # Read bounded batches; never unpack all images or copy the dataset.
            with zipfile.ZipFile(job["path"]) as z:
                records_payload = z.read("records.jsonl").decode()
                for start in range(0, len(rows), args.batch_size):
                    payloads = [z.read(r["member"]) for r in rows[start:start + args.batch_size]]
                    vectors.append(encode_payloads(payloads, model, preprocess, gpu,
                        args.batch_size, args.precision, pool))
            values = np.concatenate(vectors)
            elapsed = time.monotonic() - started
            meta = dict(schema=SCHEMA, job_id=job["id"], count=job["count"],
                input_sha256=job["sha256"], records_sha256=job["records_sha256"],
                config=config, config_id=config_id, seconds=elapsed)
            save_result(target, values, records_payload, meta)
            validate_result(target, job, config_id)
            done.append(job["id"])
            frames += len(rows)
            seconds += elapsed
            print(f"GPU {gpu}: {job['id']} {len(rows)} images, {elapsed:.1f}s, "
                  f"{len(rows)/elapsed:.2f} img/s", flush=True)
            atomic_json(output / f"gpu-{gpu}-progress.json", dict(done=done,
                skipped=skipped, frames=frames, seconds=seconds))
        atomic_json(output / f"gpu-{gpu}-summary.json", dict(done=done, skipped=skipped,
            frames=frames, inference_seconds=seconds, worker_seconds=time.monotonic()-start_worker,
            peak_allocated_gib=torch.cuda.max_memory_allocated(gpu)/2**30))


def run(args):
    import torch
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    if catalog["schema"] != SCHEMA:
        raise ValueError("Unsupported catalog")
    if digest(Path(args.catalog).parent / "parity.aicshard") != catalog["parity_sha256"]:
        raise ValueError("Corrupt parity fixture")
    available, missing = [], []
    paths = {}
    for path in Path(args.input_root).rglob("*.aicshard"):
        paths.setdefault(path.name, []).append(path)
    for job in catalog["shards"]:
        found = paths.get(job["file"], [])
        if len(found) > 1:
            raise ValueError(f"Ambiguous input shard: {job['file']}")
        if found:
            available.append(dict(job, path=str(found[0])))
        else:
            missing.append(job["id"])
    if missing and not args.allow_partial_input:
        raise ValueError(f"Missing {len(missing)} shards; attach datasets or explicitly allow partial input")
    if args.max_shards:
        import numpy as np
        available = [available[i] for i in np.linspace(0, len(available)-1,
            min(args.max_shards, len(available)), dtype=int)]
    if not available:
        raise ValueError("No input shards found")
    if torch.cuda.device_count() < args.gpus:
        raise ValueError(f"Need {args.gpus} CUDA GPUs, got {torch.cuda.device_count()}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Refuse concurrent writers to one output. A hard-killed run may leave this
    # marker: remove it only after confirming the previous process has stopped.
    lock_path = out / ".run.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        checkpoint = args.checkpoint
        if not checkpoint:
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download("timm/" + MODEL, "open_clip_model.safetensors",
                revision=REVISION, cache_dir=args.model_cache)
        if digest(checkpoint) != WEIGHTS_SHA:
            raise ValueError("Checkpoint SHA256 mismatch")
        ctx = mp.get_context("spawn")
        lock = ctx.Lock()
        started = time.monotonic()
        deadline = started + args.hours * 3600
        processes = [ctx.Process(target=worker, args=(gpu, jobs, args, checkpoint, lock, deadline))
                     for gpu, jobs in enumerate(balanced(available, args.gpus)) if jobs]
        for process in processes:
            process.start()
        try:
            while any(p.is_alive() for p in processes):
                if any(p.exitcode not in (None, 0) for p in processes):
                    for other in processes:
                        if other.is_alive():
                            other.terminate()
                    break
                time.sleep(0.5)
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            for process in processes:
                process.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Workers failed {failed}; completed shard outputs remain reusable")
        roots = [out, *(Path(p) for p in args.resume_root)]
        complete = []
        expected_config = None
        for job in catalog["shards"]:
            matches = [p for root in roots for p in root.rglob(f"{job['id']}.npz")]
            if matches:
                for match in matches:
                    meta = validate_result(match, job, expected_config)
                    expected_config = meta["config_id"]
                complete.append(job["id"])
        result = dict(catalog_sha256=digest(args.catalog), total_shards=len(catalog["shards"]),
            complete_shards=complete, remaining_shards=[j["id"] for j in catalog["shards"] if j["id"] not in complete],
            wall_seconds=time.monotonic()-started, missing_input_shards=missing,
            warning="Persist this output version and attach ALL earlier outputs on resume.")
        atomic_json(out / "run-summary.json", result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        lock_path.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume-root", action="append", default=[])
    p.add_argument("--checkpoint")
    p.add_argument("--model-cache", default="/tmp/aic-model-cache")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--decode-threads", type=int, default=2)
    p.add_argument("--hours", type=float, default=9.5)
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--allow-partial-input", action="store_true")
    args = p.parse_args()
    if min(args.gpus, args.batch_size, args.decode_threads, args.hours) <= 0 or args.max_shards < 0:
        p.error("Positive GPU/batch/thread/time values required")
    run(args)


if __name__ == "__main__":
    main()
