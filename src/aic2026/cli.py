from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Fix OpenMP duplicate lib runtime conflict on Windows (Error #15 / 0xC0000005)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows console defaults to a legacy code page (cp1252) that cannot encode
# Vietnamese (or any non-Latin) output, which makes every `typer.echo` of a
# progress message crash with UnicodeEncodeError. Force UTF-8 on the std
# streams so the CLI is runnable on Windows terminals as well as Colab/Linux.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (OSError, ValueError):
    pass

import typer

from aic2026.evaluation import evaluate_query
from aic2026.ingestion import build_manifest, resolve_feature_sources
from aic2026.models import Candidate, GroundTruth, Query

logger = logging.getLogger(__name__)

app = typer.Typer(help="AIC 2026 retrieval toolkit")


def _setup_run_logging(log_file: Path) -> None:
    """Cấu hình logging ghi LIÊN TỤC ra file (và console) để theo dõi lúc chạy.

    Không đợi chạy xong mới ghi — mỗi ``logger.info`` được flush ngay vào file,
    nên user có thể ``tail``/mở file log bất cứ lúc nào để xem tiến độ.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=False)
    file_handler.setLevel(logging.INFO)
    # dòng format có timestamp để dễ biết đang ở thời điểm nào.
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Tránh nhân đôi handler mỗi lần gọi lệnh.
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file.resolve())
               for h in root.handlers):
        root.addHandler(file_handler)
    # Console handler để vẫn thấy trên terminal.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        root.addHandler(stream)

@app.command()
def prepare(raw_dir: Path = typer.Option(Path("data/raw")), output: Path = typer.Option(Path("data/processed/manifest.jsonl"))) -> None:
    """Discover raw assets and create the frame manifest."""
    typer.echo(f"Wrote {build_manifest(raw_dir, output)} frame records to {output}")

@app.command("inspect-data")
def inspect_data(raw_dir: Path = typer.Option(Path("data/raw"))) -> None:
    """Kiểm tra 5 nhóm dữ liệu BTC: Video, Keyframes, Objects, CLIP features, Metadata."""
    from aic2026.data_platform import inspect_official_assets
    assets = inspect_official_assets(raw_dir)
    typer.echo(json.dumps({
        "Videos": str(assets.videos) if assets.videos else None,
        "Keyframes": str(assets.keyframes) if assets.keyframes else None,
        "Objects": str(assets.objects) if assets.objects else None,
        "CLIP_features": [str(item) for item in assets.clip_features],
        "Metadata": str(assets.metadata) if assets.metadata else None,
        "support_archives": [str(item) for item in assets.support_archives],
        "missing_required": assets.missing_required(),
    }, ensure_ascii=False, indent=2))

@app.command("unpack-support-assets")
def unpack_support_assets(
    archives_dir: Path = typer.Option(Path("data/downloads"), help="Nơi chứa ZIP BTC tải về"),
    destination: Path = typer.Option(Path("data/raw/support")),
) -> None:
    """Giải nén clip-features/map-keyframes/media-info/objects ZIP do BTC cấp."""
    from aic2026.data_platform import extract_support_archives
    extracted = extract_support_archives(archives_dir, destination)
    typer.echo(json.dumps({"extracted": [str(path) for path in extracted]}, ensure_ascii=False, indent=2))

@app.command("extract-keyframes")
def extract_keyframes(
    video: Path = typer.Option(..., exists=True, help="Video chính thức do BTC cung cấp"),
    keyframes_dir: Path = typer.Option(Path("data/processed/keyframes"), help="Nơi ghi keyframe đã lọc vào processed"),
    features_dir: Path = typer.Option(Path("data/processed/clip_features"), help="Thư mục output để ghi .npz/.npy embedding tự tạo (không phải thư mục CLIP features BTC đầu vào)"),
    cosine_threshold: float = typer.Option(0.985, min=0.0, max=1.0),
    batch_size: int = typer.Option(32, min=1),
    clip_pretrained: str = typer.Option("webli", help="Checkpoint OpenCLIP: webli, openai, hoặc laion2b_s32b_b79k"),
) -> None:
    """Trích toàn bộ frame, CLIP-embed, rồi loại near-duplicate bằng cosine similarity."""
    from aic2026.data_platform import OpenCLIPFrameEncoder, extract_deduplicated_keyframes
    image_encoder = OpenCLIPFrameEncoder(model_name="ViT-SO400M-14-SigLIP-384", pretrained=clip_pretrained)
    report = extract_deduplicated_keyframes(video, keyframes_dir, features_dir, image_encoder, cosine_threshold, batch_size)
    typer.echo(json.dumps({"video_id": report.video_id, "decoded_frames": report.decoded_frames, "kept_frames": report.kept_frames, "keyframes": str(report.output_dir), "features": str(report.feature_path)}, ensure_ascii=False, indent=2))

@app.command("build-derived-index-input")
def build_derived_index_input(
    keyframes_dir: Path = typer.Option(Path("data/processed/keyframes"), help="Thư mục keyframe đã sinh ra ở processed"),
    features_dir: Path = typer.Option(Path("data/processed/clip_features"), help="Thư mục embedding đã sinh ra ở processed"),
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest output ở processed"),
    features: Path = typer.Option(Path("data/processed/derived_features.npy"), help="Feature index output ở processed"),
    map_keyframes_dir: Path | None = typer.Option(None, help="Thư mục map-keyframes CSV để gắn đúng frame_idx video gốc"),
) -> None:
    """Gộp embedding frame đã giữ thành manifest và `.npy` dùng trực tiếp cho FAISS/retrieval."""
    from aic2026.data_platform import build_derived_artifacts
    count = build_derived_artifacts(keyframes_dir, features_dir, manifest, features, map_keyframes_dir=map_keyframes_dir)
    typer.echo(f"Wrote {count} retained frame records to {manifest} and {features}")

@app.command("embed-keyframes")
def embed_keyframes(
    keyframes_dir: Path = typer.Option(Path("data/raw/Keyframes"), help="Thư mục keyframe BTC đầu vào, nên đặt ở data/raw/Keyframes"),
    features_dir: Path = typer.Option(Path("data/processed/clip_features"), help="Thư mục output để ghi .npz/.npy embedding tự tạo (không phải thư mục CLIP features BTC đầu vào)"),
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest output ở processed"),
    features: Path = typer.Option(Path("data/processed/derived_features.npy"), help="Feature index output ở processed"),
    clip_model: str = typer.Option("ViT-SO400M-14-SigLIP2-378", help="Mô hình CLIP/SigLIP: ViT-SO400M-14-SigLIP2-378 (alias: siglip2), ViT-B-16-SigLIP2-384 (alias: siglip2-base), ViT-SO400M-14-SigLIP-384 (alias: siglip), ViT-L-14"),
    clip_pretrained: str = typer.Option("webli", help="Checkpoint: webli, openai, metaclip_fullcc, metaclip_400m, laion2b_s32b_b82k"),
    device: str = typer.Option("cuda", help="Thiết bị: cuda, cpu, hoặc auto"),
    batch_size: int = typer.Option(32, min=1, help="Số ảnh mỗi batch để encode trên GPU"),
    video_id: str | None = typer.Option(None, help="Chỉ encode 1 thư mục keyframe, ví dụ L21_V001"),
    map_keyframes_dir: Path | None = typer.Option(None, help="Thư mục map-keyframes CSV để gắn đúng frame_idx video gốc"),
    resume: bool = typer.Option(True, help="Bỏ qua các video đã encode xong"),
    with_chroma: bool = typer.Option(False, help="Sau khi build .npy/.manifest, build luôn Chroma collection tại --chroma-dir"),
    chroma_dir: Path = typer.Option(Path("data/indexes/chroma"), help="Thư mục lưu Chroma index (dùng khi --with-chroma)"),
) -> None:
    """Encode existing JPG keyframes on disk into .npz archives and build the retrieval index inputs."""
    from aic2026.data_platform import (
        OpenCLIPFrameEncoder,
        build_derived_artifacts,
        embed_existing_keyframes,
    )
    dev = None if device == "auto" else device
    image_encoder = OpenCLIPFrameEncoder(model_name=clip_model, pretrained=clip_pretrained, device=dev)

    archives = embed_existing_keyframes(keyframes_dir, features_dir, image_encoder, batch_size=batch_size, video_id=video_id, resume=resume)
    count = build_derived_artifacts(keyframes_dir, features_dir, manifest, features, video_ids=[video_id] if video_id else None, map_keyframes_dir=map_keyframes_dir)
    typer.echo(f"Created {len(archives)} feature archives under {features_dir}; wrote {count} records to {manifest} and {features}")

    if with_chroma:
        from aic2026.ingestion import load_manifest
        from aic2026.retrieval.factory import build_vector_index
        build_vector_index(features, load_manifest(manifest), backend="chroma", chroma_dir=chroma_dir)
        typer.echo(f"Chroma collection ready at {chroma_dir}")

@app.command("probe-official-features")
def probe_official_features(
    features: Path = typer.Option(..., help="File CLIP features chính thức BTC (.npy)"),
    manifest: Path | None = typer.Option(None, help="Manifest hiện có để so count/thứ tự (tùy chọn)"),
    metadata_dir: Path | None = typer.Option(None, help="Thư mục Metadata BTC"),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu để phát hiện 5 nhóm asset"),
) -> None:
    """Probe file CLIP .npy chính thức: dim/count, frame_indices từng video, thứ tự video."""
    from aic2026.ingestion.official_index import probe_official_features as _probe
    report = _probe(features, manifest, metadata_dir, raw_dir)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

@app.command("prepare-official")
def prepare_official(
    features: Path = typer.Option(..., help="File CLIP features chính thức BTC (.npy)"),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu chứa Keyframes/Objects/Metadata"),
    output_manifest: Path = typer.Option(Path("data/processed/official_manifest.jsonl")),
    output_features: Path = typer.Option(Path("data/processed/official_features.npy")),
    log_file: Path = typer.Option(
        Path("data/processed/prepare_official.log"),
        help="File log tiến độ LIÊN TỤC (ghi từng video, không đợi xong). Mở/tail bất cứ lúc nào để theo dõi.",
    ),
) -> None:
    """Build manifest + aligned .npy từ CLIP features BTC chính thức (thứ tự khớp keyframe)."""
    _setup_run_logging(log_file)
    from aic2026.data_platform import inspect_official_assets
    from aic2026.ingestion.official_index import build_official_index
    # Fail fast if the map-keyframes CSV (nguồn frame_idx gốc của video) is
    # missing. Without it, `build_official_index` silently falls back to keyframe
    # ordinal as `frame_id`, and TRAKE would then emit out-of-range frames versus
    # the BTC ground-truth `ranges`. BTC Metadata for this corpus does not carry
    # `frame_indices` (verified on L21_V001), so the CSV is the only source of
    # correct coordinates. This guard prevents building a subtly-wrong manifest.
    assets = inspect_official_assets(raw_dir)
    if assets.map_keyframes is None:
        raise typer.BadParameter(
            "Không tìm thấy thư mục map-keyframes (CSV frame_idx gốc). "
            "TRAKE cần tọa độ frame gốc; thiếu CSV sẽ sinh manifest sai tọa độ. "
            "Hãy giải nén ZIP hỗ trợ BTC vào data/raw (vd map-keyframes-aic25-b1)."
        )
    logger.info("BẮT ĐẦU prepare-official: manifest=%s features=%s", output_manifest, features)
    count = build_official_index(raw_dir, features, output_manifest, output_features)
    logger.info("HOÀN TẤT prepare-official: %d records", count)
    typer.echo(f"Wrote {count} records to {output_manifest} and {output_features}")


@app.command("compute-colour-features")
def compute_colour_features(
    manifest: Path = typer.Option(
        Path("data/processed/official_manifest.jsonl"),
        help="Manifest JSONL (có keyframe_path + object_path mỗi record).",
    ),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu chứa Keyframes/Objects."),
    output: Path = typer.Option(
        Path("data/processed/colour_features.jsonl"),
        help="Output sidecar JSONL (key vector_id).",
    ),
    limit: int = typer.Option(0, help="Chỉ xử lý N record đầu (dev; 0 = toàn bộ)."),
) -> None:
    """Precompute offline colour evidence per keyframe.

    CLIP toàn cảnh gần như 'mù màu', nên truy vấn như 'xe đỏ' và 'xe xanh'
    trả kết quả gần như giống nhau. Bước rerank màu gốc giải quyết bằng cách
    decode ảnh keyframe mỗi lần query — chậm và chỉ xét top-N. Lệnh này
    chuyển toàn bộ công việc pixel sang một lần chạy offline, ghi một sidecar
    nhẹ (vector_id → {dom, fracs, obj}); query chỉ là dict lookup tức thì,
    áp dụng cho toàn bộ candidate pool, và bind màu với bất kỳ object nào.

    Chạy một lần trước khi serve; rerank sẽ tự động dùng sidecar nếu tồn tại.
    """
    from aic2026.ingestion.colour_features import build_colour_features

    count = build_colour_features(
        manifest_path=manifest,
        keyframes_root=raw_dir / "Keyframes",
        objects_root=raw_dir / "Objects",
        output_path=output,
        limit=limit,
    )
    typer.echo(f"Wrote {count} colour-feature records to {output}")

@app.command("build-from-clip")
def build_from_clip_cmd(
    features: Path = typer.Option(..., help="File hoặc thư mục CLIP features BTC (.npy)"),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu chứa Objects/Metadata (Keyframes tùy chọn)"),
    output_manifest: Path = typer.Option(Path("data/processed/official_manifest.jsonl")),
    output_features: Path = typer.Option(Path("data/processed/official_features.npy")),
    keyframes_dir: Path | None = typer.Option(None, help="Thư mục Keyframes (tùy chọn, chỉ để hiển thị/VLM). Thiếu vẫn chạy được."),
    objects_dir: Path | None = typer.Option(None, help="Thư mục Objects (tùy chọn)"),
    metadata_dir: Path | None = typer.Option(None, help="Thư mục Metadata (tùy chọn)"),
) -> None:
    """Build manifest + .npy CHỈ từ CLIP features BTC (không bắt buộc có Keyframes ảnh).

    Dùng khi đã có CLIP features của toàn bộ video nhưng Keyframes mới tải 1 phần.
    Keyframes nếu có sẽ dùng làm keyframe_path (gallery/VLM), thiếu thì để trống.
    """
    from aic2026.ingestion.official_index import build_from_clip
    count = build_from_clip(
        raw_dir, features, output_manifest, output_features,
        keyframes_dir=keyframes_dir, objects_dir=objects_dir, metadata_dir=metadata_dir,
    )
    typer.echo(f"Wrote {count} records to {output_manifest} and {output_features}")

@app.command("check-coverage")
def check_coverage(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Gốc dữ liệu"),
) -> None:
    """So sánh CLIP features vs Keyframes: video nào có cả 2, số lượng có khớp không.

    In ra: tổng video CLIP, tổng video Keyframes, danh sách video chỉ có CLIP (thiếu ảnh),
    và với mỗi video có cả 2, số keyframe ảnh vs số row CLIP.
    """
    import sys

    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from aic2026.data_platform import inspect_official_assets
    assets = inspect_official_assets(raw_dir)

    clip_videos: dict[str, int] = {}
    for f in assets.clip_features:
        n = int(np.load(f, mmap_mode="r").shape[0])
        clip_videos[f.stem] = clip_videos.get(f.stem, 0) + n
    kf_videos: dict[str, int] = {}
    if assets.keyframes is not None and assets.keyframes.exists():
        for image in sorted([*assets.keyframes.rglob("*.jpg"), *assets.keyframes.rglob("*.png")]):
            vid = image.parent.name
            kf_videos[vid] = kf_videos.get(vid, 0) + 1

    clip_set = set(clip_videos)
    kf_set = set(kf_videos)
    both = sorted(clip_set & kf_set)
    clip_only = sorted(clip_set - kf_set)
    kf_only = sorted(kf_set - clip_set)

    typer.echo(f"CLIP features videos : {len(clip_set)} (total rows {sum(clip_videos.values())})")
    typer.echo(f"Keyframes videos      : {len(kf_set)} (total images {sum(kf_videos.values())})")
    typer.echo(f"Video có CẢ 2         : {len(both)}")
    typer.echo(f"Video CHỈ có CLIP (thiếu ảnh hiển thị/VLM): {len(clip_only)}")
    if clip_only:
        typer.echo("  -> " + ", ".join(clip_only[:20]) + (" ..." if len(clip_only) > 20 else ""))
    if kf_only:
        typer.echo(f"Video CHỈ có Keyframes (không có CLIP, sẽ bị bỏ qua): {len(kf_only)}")
        typer.echo("  -> " + ", ".join(kf_only[:20]) + (" ..." if len(kf_only) > 20 else ""))

    mismatched = [(v, kf_videos[v], clip_videos[v]) for v in both if kf_videos[v] != clip_videos[v]]
    if mismatched:
        typer.echo(f"\nCẢNH BÁO: {len(mismatched)} video có số keyframe ảnh != số row CLIP:")
        for v, k, c in mismatched[:30]:
            typer.echo(f"  {v}: {k} ảnh vs {c} rows CLIP")
    else:
        typer.echo("\nTất cả video có cả 2 đều KHỚP số lượng (ảnh == rows CLIP).")

@app.command("check-features")
def check_features(features: Path, manifest: Path = Path("data/processed/derived_manifest.jsonl")) -> None:
    """Fail early unless provided CLIP features and the manifest have the same order/count."""
    import numpy as np

    from aic2026.ingestion import load_manifest
    # O(1) shape check via mmap — no need to load/normalize the full matrix.
    rows = int(np.load(features, mmap_mode="r").shape[0])
    count = len(load_manifest(manifest))
    if rows != count:
        raise typer.BadParameter(f"features has {rows} vectors; manifest has {count} records")
    typer.echo(f"OK: {count} feature vectors match the manifest")

@app.command("build-chroma-index")
def build_chroma_index(
    features: Path = typer.Option(Path("data/processed/derived_features.npy"), help="Feature .npy input"),
    manifest: Path = typer.Option(Path("data/processed/derived_manifest.jsonl"), help="Manifest input"),
    chroma_dir: Path = typer.Option(Path("data/indexes/chroma"), help="Thư mục lưu Chroma collection"),
    collection_name: str = typer.Option("aic2026_frames"),
    space: str = typer.Option("cosine", help="Khoảng cách Chroma: cosine | l2 | ip"),
) -> None:
    """Build/populate the on-disk ChromaDB collection from .npy + manifest."""
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval.factory import build_vector_index

    manifest_records = load_manifest(manifest)
    store = build_vector_index(
        features,
        manifest_records,
        backend="chroma",
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        space=space,
    )
    typer.echo(
        f"Chroma collection '{collection_name}' ready: {len(store)} vectors @ {chroma_dir}"
    )


@app.command("agent-query")
def agent_query(
    query: Path = typer.Option(..., help="JSON file matching the Query schema"),
    features: Path | None = typer.Option(None, help="BTC CLIP .npy feature file. Mặc định auto: ưu tiên official_features.npy (CLIP sẵn BTC) nếu có, fallback derived_features.npy (encode keyframe)."),
    manifest: Path | None = typer.Option(None, help="Manifest JSONL. Mặc định auto: ưu tiên official_manifest.jsonl nếu có, fallback derived_manifest.jsonl."),
    llm_model: str = typer.Option("qwen3.5:4b"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434"),
    llm_timeout: int = typer.Option(600, help="Timeout (giây) cho mỗi lần gọi LLM local (CPU chậm cần lớn)"),
    backend: str = typer.Option("faiss", help="Index backend: faiss (mặc định) | numpy | chroma"),
    chroma_dir: Path = typer.Option(Path("data/indexes/chroma"), help="Thư mục Chroma (chỉ dùng khi backend=chroma)"),
    metadata_filter: str = typer.Option("", help="Từ khóa metadata cách nhau dấu phẩy để lọc video TRƯỚC retrieval (vd: cửa hàng,siêu thị). Để trống = không lọc."),
    vlm_backend: str = typer.Option("none", help="Backend VLM cho Q&A: florence | none (Florence-2 thay thế QwenVLM/Ollama)"),
    vlm_model: str = typer.Option("microsoft/Florence-2-base-ft", help="Model VLM cho Q&A (HuggingFace id Florence-2; để trống để tắt)"),
    vlm_device: str | None = typer.Option(None, help="Device VLM (mặc định auto / cpu)"),
    vlm_dtype: str = typer.Option("float32", help="Precision VLM: float32 (CPU an toàn) | float16 | bfloat16 (GPU)"),
    vlm_timeout: int = typer.Option(120, help="Timeout (giây) mỗi lần gọi VLM"),
    coarse_top_k: int = typer.Option(200, help="TRAKE: giới hạn số video đưa vào DP alignment (coarse filter). 0 = xét hết."),
    late_interaction_weight: float = typer.Option(0.0, help="KIS: bật late-interaction (ColBERT-style MaxSim theo từng facet query). 0 = tắt (mặc định). Thử 0.5 để CLIP không chọn nhầm frame đúng cảnh sai vật thể. Nặng hơn RRF nhưng chỉ vài chục encode text, vẫn ms."),
    object_evidence_weight: float = typer.Option(0.0, help="KIS: độ lớn reward/penalty object detector khi query hỏi vật thể (RRF scale). Lớn hơn = ép object mạnh hơn CLIP."),
    object_penalty_scale: float = typer.Option(2.0, help="KIS: hệ số phạt frame THIẾU vật thể so với thưởng frame có vật thể (mặc định 2.0)."),
    drop_empty_object_frames: bool = typer.Option(False, help="KIS: loại hẳn frame KHÔNG có vật thể nào (ảnh mờ, không entity đạt ngưỡng 0.4) khi query có hỏi vật thể. Giảm truy xuất đến frame nhiễu. Tắt nếu sợ mất recall."),
    trake_preferred_prefixes: str = typer.Option("", help="Ưu tiên (soft bias, KHÔNG loại trừ) TRAKE vào tiền tố video_id, cách nhau dấu phẩy (vd: L26). KIS/Q&A luôn xét TOÀN BỘ video. Để trống = mặc định ưu tiên L26. Truyền '.' để tắt ưu tiên."),
    search_mode: str = typer.Option("siglip", "--mode", "-m", help="SigLIP2 visual search (siglip)"),
    clip_model: str = typer.Option("siglip2", help="Mô hình CLIP/SigLIP (ViT-SO400M-14-SigLIP2-378 / siglip2, ViT-SO400M-14-SigLIP-384 / siglip, ViT-B-16-SigLIP2-384 / siglip2-base, ViT-L-14)"),
    clip_pretrained: str = typer.Option("webli", help="Checkpoint (webli, openai, metaclip_fullcc, metaclip_400m, laion2b_s32b_b82k)"),
    vlm_judge: bool = typer.Option(False, help="Kích hoạt VLM-as-a-Judge (Qwen2.5-VL) thẩm định thị giác chính xác cho KIS, QA và TRAKE"),
    vlm_judge_model: str = typer.Option("Qwen/Qwen2.5-VL-7B-Instruct", help="Model VLM Judge: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-3B-Instruct, hoặc Ollama model"),
    vlm_judge_backend: str = typer.Option("qwen", help="Backend VLM Judge: qwen (transformers CUDA/CPU), ollama, mock"),
    vlm_judge_top_k: int = typer.Option(15, help="Số lượng ứng viên hàng đầu được VLM Judge thẩm định chi tiết"),
    output: Path | None = typer.Option(None),
) -> None:
    """Run the bounded local-LLM agent and save auditable candidates/trace."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools
    from aic2026.embeddings import OpenCLIPTextEmbedder
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval import RetrievalPipeline
    from aic2026.retrieval.factory import load_index_for_query

    if search_mode != "siglip":
        raise typer.BadParameter("Only mode siglip is supported.")
    parsed_query = Query.model_validate_json(query.read_text(encoding="utf-8-sig"))
    resolved_manifest, resolved_features = resolve_feature_sources(manifest, features)
    if manifest is None or features is None:
        typer.echo(
            f"Auto-selected feature sources: manifest={resolved_manifest}, "
            f"features={resolved_features}"
        )
    manifest_records = load_manifest(resolved_manifest)
    # Per-video metadata (title/description/keywords) sống cùng thư mục với
    # manifest. Load vào store để pipeline chạy được metadata pre-filter
    # (filter_videos_by_metadata) — nếu không truyền, store rỗng và pre-filter
    # bị skip dù tools.video_filter_terms có set.
    from aic2026.retrieval import VideoMetadataStore

    video_metadata = VideoMetadataStore.load(
        resolved_manifest.parent / "video_metadata.jsonl"
    )

    index = load_index_for_query(resolved_features, manifest_records, backend=backend, chroma_dir=chroma_dir)

    pipeline = RetrievalPipeline(
        index,
        manifest_records,
        frames_per_video=20,
        video_metadata=video_metadata,
    )
    text_encoder = OpenCLIPTextEmbedder(model_name=clip_model, pretrained=clip_pretrained)
    encode_text = text_encoder.encode
    tools = RetrievalTools(pipeline, encode_text)
    # coarse_top_k áp dụng cho cả KIS (video-level rerank) và TRAKE (DP coarse
    # filter). TRAKE dùng agent.coarse_top_k; KIS dùng tools.coarse_top_k.
    tools.coarse_top_k = coarse_top_k
    # KIS evidence tuning (object detector + late-interaction facet rerank).
    tools.late_interaction_weight = late_interaction_weight
    tools.object_evidence_weight = object_evidence_weight
    tools.object_penalty_scale = object_penalty_scale
    tools.drop_empty_object_frames = drop_empty_object_frames
    # Metadata pre-filter: chỉ retrieve trong các video có chứa từ khóa.
    tools.video_filter_terms = [t.strip() for t in metadata_filter.split(",") if t.strip()] or None
    # TRAKE soft preference: "." disables the default L26 bias; empty uses the
    # agent's built-in default (L26); otherwise split on commas into the
    # explicit preferred-prefix list. KIS/Q&A are intentionally never scoped —
    # they search the full corpus.
    def _parse_prefixes(raw: str) -> list[str] | None:
        if raw == ".":
            return None
        return [p.strip() for p in raw.split(",") if p.strip()] or None
    # Pass the VLM model id to tools; it is built lazily ONLY on the QA answer
    # step (after the CLIP encoder is freed) so CLIP + Florence are never both
    # resident (avoids OOM). For KIS/TRAKE we pass None so the VLM is never
    # built/loaded (saves the ~52s + ~600MB Florence load).
    if parsed_query.type == "qa" and vlm_backend != "none" and vlm_model:
        tools.vlm_model = vlm_model
        tools.vlm_device = vlm_device
        tools.vlm_dtype = vlm_dtype
    if vlm_judge:
        tools.vlm_judge_enabled = True
        tools.vlm_judge_model = vlm_judge_model
        tools.vlm_judge_backend = vlm_judge_backend
        tools.vlm_judge_top_k = vlm_judge_top_k
    agent = RetrievalAgent(
        tools,
        llm=OllamaLLM(model=llm_model, base_url=ollama_url, timeout_seconds=llm_timeout),
        trake_preferred_prefixes=_parse_prefixes(trake_preferred_prefixes),
    )
    agent.coarse_top_k = coarse_top_k
    agent.search_mode = search_mode
    result = agent.run(parsed_query)
    if parsed_query.type == "qa" and not any(c.answer for c in result.candidates):
        typer.echo(
            "⚠️ Cảnh báo: task Q&A nhưng không candidate nào có answer "
            "(VLM không gọi được hoặc không sinh nội dung). Submission sẽ không hợp lệ "
            "— kiểm tra: ollama đang serve model vision chưa (`ollama pull qwen2.5vl:3b`), "
            "--vlm-model, hoặc kết nối."
        )
    serialized = result.model_dump_json(indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        typer.echo(f"Wrote agent result to {output}")
    else:
        typer.echo(serialized)

@app.command("export-submission")
def export_submission(
    result: Path = typer.Option(..., help="Agent JSON output của `agent-query` (--output ...)"),
    query: Path = typer.Option(..., help="File Query JSON gốc (cùng query_id/type/events)"),
    output: Path | None = typer.Option(None, help="CSV đầu ra. Mặc định <result_stem>.csv cạnh file result."),
) -> None:
    """Chuyển kết quả JSON của `agent-query` thành 1 dòng CSV nộp bài BTC.

    Đây là bước bạn đang thiếu: `agent-query` chỉ xuất JSON (có event_frames ở
    trong), còn BTC yêu cầu file CSV theo đúng định dạng mỗi task (trường cách
    nhau bởi ", " — dấu phẩy + 1 dấu cách):
      - KIS : ``<video_id>, <frame_id>``
      - Q&A : ``<video_id>, <frame_id>, <answer>``
      - TRAKE: ``<video_id>, <frame_1>, <frame_2>, ..., <frame_N>``  (N = số event)

    Với TRAKE, dòng này CHỨA ĐỦ các frame event (khác với KIS chỉ 1 frame),
    nên kết quả sẽ KHÔNG còn 'giống KIS'. Dùng sau mỗi `agent-query`:

        aic2026 export-submission --result outputs/result_trake.json \\
            --query query_trake.json --output outputs/result_trake.csv
    """
    from aic2026.agent.types import AgentResult
    from aic2026.submission import csv_row

    parsed_query = Query.model_validate_json(query.read_text(encoding="utf-8"))
    agent_result = AgentResult.model_validate_json(
        result.read_text(encoding="utf-8")
    )

    if not agent_result.candidates:
        raise typer.BadParameter(
            f"Result rỗng — query '{parsed_query.query_id}' không có candidate nào."
        )

    ranked = sorted(
        agent_result.candidates,
        key=lambda c: c.score,
        reverse=True,
    )

    out_path = output or result.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        csv_row(parsed_query, ranked[0]) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        f"Wrote BTC submission CSV ({parsed_query.type}) -> {out_path}\n"
        f"  {out_path.read_text(encoding='utf-8').strip()}"
    )


@app.command()
def evaluate(query: Path, candidates: Path, ground_truth: Path) -> None:
    """Evaluate one query using AIC R@k scoring."""
    q = Query.model_validate_json(query.read_text(encoding="utf-8"))
    gt = GroundTruth.model_validate_json(ground_truth.read_text(encoding="utf-8"))
    items = [Candidate.model_validate(row) for row in json.loads(candidates.read_text(encoding="utf-8"))]
    typer.echo(json.dumps(evaluate_query(q, items, gt), ensure_ascii=False, indent=2))

@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the internal FastAPI service."""
    import uvicorn
    uvicorn.run("aic2026.app.api:app", host=host, port=port, reload=False)

if __name__ == "__main__": app()
