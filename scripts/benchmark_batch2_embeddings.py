"""Bounded offline SigLIP2 trial; never modifies production indexes."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-video", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.per_video <= 128 or not 1 <= args.batch_size <= 8:
        parser.error("Bounded trial requires 1..128 images/video and batch size 1..8")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import open_clip
    import transformers
    from PIL import Image
    from aic2026.ingestion.cpu_keyframes import cpu_video_records

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing unplanned CPU benchmark")
    data = args.workspace / "data"
    root = data / "batches/batch2/extracted"
    records = []
    counts = {}
    for video in ("M05_V001", "N001-V001", "S01-V001"):
        items = cpu_video_records(root, video)
        counts[video] = len(items)
        for i in np.linspace(0, len(items) - 1, min(args.per_video, len(items)), dtype=int):
            r = items[i].model_copy(update={"vector_id": len(records), "clip_feature_index": len(records)})
            records.append(r)
    (args.output / "manifest.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    print(f"Validated maps: {counts}; sampled {len(records)} images", flush=True)
    cache = data / ".cache/huggingface/hub/models--timm--ViT-SO400M-14-SigLIP2-378/snapshots"
    checkpoints = list(cache.glob("*/open_clip_model.safetensors"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected exactly one cached checkpoint; select provenance explicitly")
    checkpoint = checkpoints[0]
    with checkpoint.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    model_name = "ViT-SO400M-14-SigLIP2-378"
    cfg = open_clip.get_pretrained_cfg(model_name, "webli")
    start = time.perf_counter()
    print("Loading cached checkpoint on CPU (FP32)...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=str(checkpoint), device="cpu",
        image_mean=cfg["mean"], image_std=cfg["std"],
        image_interpolation=cfg["interpolation"], image_resize_mode=cfg["resize_mode"])
    # Image-only benchmark: remove text tower before moving to the 8 GiB GPU.
    del model.text
    gc.collect()
    model = model.eval().to("cuda")
    torch.cuda.synchronize()
    setup = time.perf_counter() - start

    def encode(paths):
        outputs = []
        for i in range(0, len(paths), args.batch_size):
            images = []
            for path in paths[i:i + args.batch_size]:
                with Image.open(path) as image:
                    images.append(preprocess(image.convert("RGB")))
            with torch.inference_mode():
                value = model.encode_image(torch.stack(images).to("cuda"))
                value = value / value.norm(dim=-1, keepdim=True)
            outputs.append(value.cpu().float().numpy())
        return np.concatenate(outputs)

    # Compare recomputed vectors with six spaced rows of the existing index.
    old_vectors = np.load(data / "processed/siglip2_features.npy", mmap_mode="r")
    chosen = set(np.linspace(0, len(old_vectors) - 1, 6, dtype=int).tolist())
    old_rows = []
    with (data / "processed/siglip2_manifest.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in chosen:
                row = json.loads(line)
                parts = row["keyframe_path"].replace("\\", "/").split("/Keyframes/")
                path = data / "extracted/Keyframes" / parts[-1]
                old_rows.append((i, path))
    print("Checking six batch1 vectors...", flush=True)
    recomputed = encode([p for _, p in old_rows])
    old = np.asarray(old_vectors[[i for i, _ in old_rows]], dtype=np.float32)
    old_norms = np.linalg.norm(old, axis=1)
    cosine = np.sum(recomputed * (old / old_norms[:, None]), axis=1)
    print(f"Batch1 cosine: {cosine.tolist()}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    features = encode([r.keyframe_path for r in records])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or not np.allclose(norms, 1, atol=1e-5):
        raise RuntimeError("Invalid vectors")
    np.save(args.output / "features.npy", features)
    result = dict(model=model_name, checkpoint=str(checkpoint), checkpoint_sha256=sha,
        preprocess=repr(preprocess), precision="fp32", normalization="L2",
        torch=torch.__version__, transformers=transformers.__version__, open_clip=open_clip.__version__,
        device=torch.cuda.get_device_name(), batch_size=args.batch_size,
        setup_seconds=setup, images=len(records), sampled_videos=counts,
        embedding_seconds=elapsed, images_per_second=len(records)/elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        shape=list(features.shape), norm_range=[float(norms.min()), float(norms.max())],
        batch1_rows=[i for i, _ in old_rows], batch1_cosine=cosine.tolist(),
        batch1_norms=old_norms.tolist(),
        compatibility_sample_pass=bool(np.all(cosine >= 0.999)),
        note="Sample only; not a full index or retrieval-quality evaluation. PTS convention for submissions remains to verify.")
    (args.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
