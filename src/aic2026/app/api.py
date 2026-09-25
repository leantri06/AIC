from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from aic2026.agent.types import AgentResult, AgentTrace
from aic2026.ingestion import resolve_feature_sources
from aic2026.models import Query

app = FastAPI(title="AIC 2026 Agent API", version="0.2.0")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_retrieval_pipeline(manifest_path: str, features_path: str, backend: str, chroma_dir: str):
    """Share the large read-only corpus across task and search-option changes."""
    from aic2026.ingestion import load_manifest
    from aic2026.retrieval import RetrievalPipeline, VideoMetadataStore
    from aic2026.retrieval.factory import load_index_for_query

    manifest = load_manifest(Path(manifest_path))
    index = load_index_for_query(Path(features_path), manifest, backend=backend, chroma_dir=chroma_dir)
    metadata = VideoMetadataStore.load(Path(manifest_path).parent / "video_metadata.jsonl")
    return RetrievalPipeline(index, manifest, frames_per_video=20, video_metadata=metadata)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """Trang chào để mở localhost:8000 không trả 404."""
    return """<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>AIC 2026 Agent API</title>
    <style>body{font-family:system-ui;max-width:760px;margin:4rem auto;line-height:1.6}a{color:#2563eb}</style></head>
    <body><h1>AIC 2026 Agent API đang chạy</h1><p>Backend ở cổng 8000. Giao diện thi Streamlit chạy riêng ở cổng 8501.</p>
    <ul><li><a href='/docs'>API docs (/docs)</a></li><li><a href='/health'>Health check (/health)</a></li><li><a href='http://127.0.0.1:8501'>Giao diện Streamlit (cần chạy lệnh riêng)</a></li></ul>
    <pre>aic2026 serve
streamlit run src/aic2026/app/ui.py</pre></body></html>"""


# RAM Cache lưu trữ ảnh để giảm thiểu I/O đĩa
_IMAGE_CACHE: dict[str, bytes] = {}


@app.get("/images", include_in_schema=True)
def get_image(path: str) -> Response:
    """Trả về file ảnh trực tiếp. Cache lại dạng byte trong RAM để tăng tốc tối đa."""
    safe_path = Path(path).resolve()
    if not safe_path.exists():
        raise HTTPException(404, "Không tìm thấy file ảnh.")

    path_str = str(safe_path)
    if path_str in _IMAGE_CACHE:
        return Response(content=_IMAGE_CACHE[path_str], media_type="image/jpeg")

    try:
        with open(safe_path, "rb") as handle:
            img_bytes = handle.read()
    except Exception as exc:
        raise HTTPException(500, f"Không thể đọc file ảnh: {exc}")

    # Giới hạn kích thước cache khoảng 300 ảnh để tránh OOM
    if len(_IMAGE_CACHE) > 300:
        _IMAGE_CACHE.clear()

    _IMAGE_CACHE[path_str] = img_bytes
    return Response(content=img_bytes, media_type="image/jpeg")



class RuntimeConfig(BaseModel):
    """Cấu hình nội bộ; UI gửi để backend nạp đúng index/encoder."""
    manifest_path: str | None = None
    features_path: str | None = None
    clip_model: Literal["siglip2", "ViT-SO400M-14-SigLIP2-378", "siglip2-giant", "google/siglip2-giant-opt-patch16-384"] = "siglip2"
    clip_pretrained: str = "webli"
    llm_model: str = "qwen3.5:4b"
    ollama_url: str = "http://127.0.0.1:11434"
    # The official CLIP matrix is static; FAISS is faster than starting a
    # Chroma collection and is the default execution path.
    backend: str = "faiss"
    chroma_dir: str = "data/indexes/chroma"
    late_interaction_weight: float = 0.0
    metadata_filter: str = ""
    search_mode: Literal["siglip"] = "siglip"
    vlm_backend: str = "none"
    vlm_model: str = "microsoft/Florence-2-base-ft"
    vlm_device: str | None = None
    vlm_dtype: str = "float32"
    vlm_timeout: int = 120
    coarse_top_k: int = 200
    vlm_judge_enabled: bool = False
    vlm_judge_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_judge_backend: str = "qwen"
    vlm_judge_top_k: int = 15
    # Số video gửi VLM tối đa (top-N video score cao nhất). 0 = tất cả.
    # Giới hạn này giúp giảm thời gian VLM khi có nhiều candidate video.
    vlm_top_videos: int = 20
    # Số thread song song gọi VLM (mỗi video 1 thread). Giảm xuống 3 để tránh
    # nghẽn Ollama local — Ollama xử lý tuần tự nên thread quá nhiều chỉ tăng queue.
    vlm_max_workers: int = 3
    # BLIP-2 Cross-Encoder ITM Reranker (arXiv:2512.12935v1)
    blip2_rerank_enabled: bool = False
    blip2_top_k: int = 30
    # Shot-level diversity / Temporal NMS
    temporal_nms_enabled: bool = False
    temporal_nms_window: int = 60


class TaskRequest(BaseModel):
    query_id: str = "live-query"
    text: str = ""
    question: str | None = None
    events: list[str] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class AnswerRequest(BaseModel):
    """Phase-2 QA: gửi lại candidates đã retrieve, chỉ chạy VLM để sinh đáp án."""
    question: str = Field(min_length=1)
    candidates: list[dict] = Field(default_factory=list)  # Candidate.model_dump() list
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


@lru_cache(maxsize=4)
def load_orchestrator(
    manifest_path: str,
    features_path: str,
    clip_pretrained: str,
    llm_model: str,
    ollama_url: str,
    backend: str = "faiss",
    chroma_dir: str = "data/indexes/chroma",
    metadata_filter: str = "",
    vlm_backend: str = "none",
    late_interaction_weight: float = 0.0,
    vlm_model: str = "microsoft/Florence-2-base-ft",
    vlm_device: str | None = None,
    vlm_dtype: str = "float32",
    vlm_timeout: int = 120,
    coarse_top_k: int = 200,
    query_type: str | None = None,
    clip_model: Literal["siglip2", "ViT-SO400M-14-SigLIP2-378", "siglip2-giant", "google/siglip2-giant-opt-patch16-384"] = "siglip2",
    vlm_judge_enabled: bool = False,
    vlm_judge_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_judge_backend: str = "qwen",
    vlm_judge_top_k: int = 15,
    blip2_rerank_enabled: bool = False,
    blip2_top_k: int = 30,
    temporal_nms_enabled: bool = False,
):
    """Shared layer: index + encoders load once per runtime configuration."""
    from aic2026.agent import OllamaLLM, RetrievalAgent
    from aic2026.agent.tools import RetrievalTools

    giant = clip_model in ("siglip2-giant", "google/siglip2-giant-opt-patch16-384")
    if giant:
        from aic2026.paths import default_data_root
        folder = Path(manifest_path or features_path).parent if (manifest_path or features_path) else default_data_root() / "processed"
        manifest_path = manifest_path or str(folder / "siglip2_giant_manifest.jsonl")
        features_path = features_path or str(folder / "siglip2_giant_features.npy")
    manifest_file, features_file = resolve_feature_sources(manifest_path, features_path)
    known_pairs = {
        "siglip2_features.npy": "siglip2_manifest.jsonl",
        "siglip2_giant_features.npy": "siglip2_giant_manifest.jsonl",
    }
    expected_manifest = known_pairs.get(features_file.name)
    if expected_manifest and manifest_file.name != expected_manifest:
        raise ValueError(f"{features_file.name} must be paired with {expected_manifest}; row orders differ.")
    missing = [str(path) for path in (manifest_file, features_file) if not path.exists()]
    if missing:
        hint = "Cần cặp siglip2_manifest.jsonl và siglip2_features.npy đã embed sẵn."
        raise FileNotFoundError(f"Thiếu index retrieval: {', '.join(missing)}. {hint}")
    pipeline = load_retrieval_pipeline(
        str(manifest_file.resolve()), str(features_file.resolve()), backend, chroma_dir
    )
    expected_dimension = 1536 if giant else 1152
    if pipeline.index.vectors.shape[1] != expected_dimension:
        raise ValueError(f"Encoder {clip_model} requires {expected_dimension}-D features; select its matching manifest and features.")
    from aic2026.embeddings.factory import load_text_encoder
    text_encoder = load_text_encoder(clip_model, clip_pretrained)
    tools = RetrievalTools(
        pipeline, text_encoder.encode, rerank_weight=0.0, object_evidence_weight=0.0,
        colour_rerank_weight=0.0, contrastive_colour_rerank_weight=0.0
    )
    tools.coarse_top_k = coarse_top_k
    tools.video_filter_terms = [t.strip() for t in metadata_filter.split(",") if t.strip()] or None
    tools.late_interaction_weight = late_interaction_weight
    # Pass the VLM model id to tools; it is built lazily ONLY on the QA answer
    # step (after the CLIP encoder is freed) so CLIP + Florence are never both
    # resident (avoids OOM). For KIS/TRAKE we force "none" so the VLM is never
    # built/loaded (saves the ~52s + ~600MB Florence load).
    effective_vlm_backend = vlm_backend
    if query_type and query_type != "qa":
        effective_vlm_backend = "none"
    if effective_vlm_backend != "none" and vlm_model:
        tools.vlm_model = vlm_model
        tools.vlm_device = vlm_device
        tools.vlm_dtype = vlm_dtype
    if vlm_judge_enabled:
        tools.vlm_judge_enabled = True
        tools.vlm_judge_model = vlm_judge_model
        tools.vlm_judge_backend = vlm_judge_backend
        tools.vlm_judge_top_k = vlm_judge_top_k
    if blip2_rerank_enabled:
        tools.blip2_rerank_enabled = True
        tools.blip2_top_k = blip2_top_k
    if temporal_nms_enabled:
        tools.temporal_nms_enabled = True
    agent = RetrievalAgent(
        tools,
        llm=OllamaLLM(
            model=llm_model,
            base_url=ollama_url,
            timeout_seconds=600,
            num_predict=1500,
            keep_alive="0",
        ),
    )
    agent.coarse_top_k = coarse_top_k
    return agent


def trim_process_memory() -> None:
    """Run garbage collection without dumping resident memory to disk."""
    import gc
    gc.collect()


def run_task(task_type: str, request: TaskRequest) -> AgentResult:
    if task_type == "qa" and not request.question:
        request.question = request.text
    if task_type == "trake" and not request.events:
        import re
        parts = [p.strip() for p in re.split(r'[\n;.]+', request.text) if p.strip()]
        request.events = parts or [request.text]
    started = perf_counter()
    try:
        config = request.runtime
        orchestrator = load_orchestrator(
            config.manifest_path,
            config.features_path,
            config.clip_pretrained,
            config.llm_model,
            config.ollama_url,
            backend=config.backend,
            chroma_dir=config.chroma_dir,
            metadata_filter=config.metadata_filter,
            vlm_backend=config.vlm_backend,
            vlm_model=config.vlm_model,
            vlm_device=getattr(config, "vlm_device", None),
            vlm_dtype=getattr(config, "vlm_dtype", "float32"),
            vlm_timeout=config.vlm_timeout,
            late_interaction_weight=config.late_interaction_weight,
            coarse_top_k=config.coarse_top_k,
            query_type=task_type,
            clip_model=config.clip_model,
            vlm_judge_enabled=config.vlm_judge_enabled,
            vlm_judge_model=config.vlm_judge_model,
            vlm_judge_backend=config.vlm_judge_backend,
            vlm_judge_top_k=config.vlm_judge_top_k,
            blip2_rerank_enabled=getattr(config, "blip2_rerank_enabled", False),
            blip2_top_k=getattr(config, "blip2_top_k", 30),
            temporal_nms_enabled=getattr(config, "temporal_nms_enabled", False),
        )
        loaded = perf_counter()
        res = orchestrator.run(Query(query_id=request.query_id, type=task_type, text=request.text, question=request.question, events=request.events))
        finished = perf_counter()
        res.trace.append(AgentTrace(step="timing", detail=f"load={loaded-started:.3f}s; search={finished-loaded:.3f}s; total={finished-started:.3f}s"))
        return res
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        # Log the full traceback so the exact failing line is visible in the
        # server log instead of just the client-facing message.
        logger.exception("run_task failed for task=%s: %s", task_type, error)
        raise HTTPException(500, f"{type(error).__name__}: {error}") from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "flows": "kis,qa,trake"}


@app.get("/health/ready")
def ready(
    manifest_path: str | None = None,
    features_path: str | None = None,
    chroma_dir: str = "data/indexes/chroma",
) -> dict[str, object]:
    """Kiểm tra nhanh trước khi UI gọi agent (tránh chờ load model rồi mới báo thiếu file)."""
    from aic2026.ingestion import resolve_feature_sources
    from aic2026.retrieval import ChromaVectorStore

    resolved_manifest, resolved_features = resolve_feature_sources(manifest_path, features_path)
    manifest_ok = resolved_manifest.exists()
    features_ok = resolved_features.exists()
    chroma_ok = ChromaVectorStore.available() and (Path(chroma_dir) / "chroma.sqlite3").exists()
    return {
        "ready": manifest_ok and features_ok,
        "manifest_path": str(resolved_manifest),
        "manifest_exists": manifest_ok,
        "features_path": str(resolved_features),
        "features_exists": features_ok,
        "chroma_dir": chroma_dir,
        "chroma_ok": chroma_ok,
    }


@app.post("/tasks/kis/run", response_model=AgentResult)
def run_kis(request: TaskRequest) -> AgentResult:
    """KIS flow: plan -> retrieval -> judge -> KIS format candidates."""
    return run_task("kis", request)


@app.post("/tasks/qa/run", response_model=AgentResult)
def run_qa(request: TaskRequest) -> AgentResult:
    """Q&A flow: plan -> retrieval -> judge -> local VLM/manual answer stage."""
    return run_task("qa", request)


@app.post("/tasks/qa/candidates", response_model=AgentResult)
def run_qa_candidates(request: TaskRequest) -> AgentResult:
    """Phase-1 QA: chỉ retrieval (SigLIP2), không VLM. Trả candidates ngay.

    UI gọi endpoint này trước: hiển thị gallery ảnh ngay (giống KIS), sau đó mới
    gọi /tasks/qa/answers để điền đáp án từ VLM mà không block UI.
    """
    if not request.question:
        request.question = request.text
    try:
        config = request.runtime
        orchestrator = load_orchestrator(
            config.manifest_path,
            config.features_path,
            config.clip_pretrained,
            config.llm_model,
            config.ollama_url,
            backend=config.backend,
            chroma_dir=config.chroma_dir,
            metadata_filter=config.metadata_filter,
            # VLM không cần cho phase-1: truyền vlm_backend="none" để bỏ qua
            vlm_backend="none",
            vlm_model=getattr(config, "vlm_model", "microsoft/Florence-2-base-ft"),
            vlm_device=getattr(config, "vlm_device", None),
            vlm_dtype=getattr(config, "vlm_dtype", "float32"),
            vlm_timeout=getattr(config, "vlm_timeout", 120),
            late_interaction_weight=config.late_interaction_weight,
            query_type="qa",
            clip_model=config.clip_model,
        )
        # Phase-1 QA: retrieval only, no VLM. We pass vlm_backend="none" above so
        # tools.vlm_model is unset and ensure_vlm() is a no-op in _finalize — no
        # need to null the callables (that previously corrupted the cached agent).
        orchestrator.search_mode = config.search_mode
        query = Query(
            query_id=request.query_id,
            type="qa",
            text=request.text,
            question=request.question,
            events=request.events,
        )
        return orchestrator.run(query)
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(500, str(error)) from error


@app.post("/tasks/qa/answers")
def run_qa_answers(request: AnswerRequest) -> dict[str, str]:
    """Phase-2 QA: nhận candidates đã có, chạy VLM song song. Trả {vector_id: answer}.

    Chỉ xử lý top-N video (config.vlm_top_videos) theo score giảm dần để kiểm soát
    thời gian VLM. Số thread song song: config.vlm_max_workers.
    """
    from aic2026.models import Candidate

    config = request.runtime

    if not config.vlm_backend or config.vlm_backend == "none":
        return {}

    # Parse candidates từ dict
    try:
        candidates = [Candidate.model_validate(c) for c in request.candidates]
    except Exception as exc:
        raise HTTPException(422, f"candidates không hợp lệ: {exc}") from exc

    if not candidates:
        return {}

    # Giới hạn top-N video đưa vào VLM
    if config.vlm_top_videos and config.vlm_top_videos > 0:
        seen_videos: set[str] = set()
        top_candidates: list[Candidate] = []
        for c in sorted(candidates, key=lambda x: x.score, reverse=True):
            if c.video_id not in seen_videos:
                seen_videos.add(c.video_id)
                if len(seen_videos) > config.vlm_top_videos:
                    break
            top_candidates.append(c)
        candidates = top_candidates

    try:
        from aic2026.qa.florence import FlorenceVLM

        florence = FlorenceVLM(
            model_name=getattr(config, "vlm_model", "microsoft/Florence-2-base-ft"),
            device=getattr(config, "vlm_device", None),
            torch_dtype=getattr(config, "vlm_dtype", "float32"),
        )
        answers = florence.answer_question(request.question, candidates)
    except Exception as error:
        raise HTTPException(500, str(error)) from error

    # Trả về string key (JSON không hỗ trợ int key)
    return {str(k): v for k, v in answers.items()}


@app.post("/tasks/trake/run", response_model=AgentResult)
def run_trake(request: TaskRequest) -> AgentResult:
    """TRAKE flow: plan -> per-event retrieval -> judge -> deterministic temporal alignment."""
    return run_task("trake", request)
