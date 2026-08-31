from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from aic2026.embeddings.asr import BGEM3TextEmbedder
from aic2026.models import FrameRecord
from aic2026.retrieval.bm25_index import BM25Index
from aic2026.retrieval.index import VectorIndex
from aic2026.retrieval.pipeline import RetrievalPipeline


def test_bgem3_embedder_mock():
    embedder = BGEM3TextEmbedder(backend="mock")
    assert embedder.available is True

    texts = ["bản tin thời sự hôm nay", "thời tiết mưa lớn"]
    embeddings = embedder.encode(texts)
    assert embeddings.shape == (2, 1024)
    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)


def test_pipeline_hybrid_dense_asr():
    manifest = [
        FrameRecord(
            vector_id=0,
            video_id="L21_V001",
            frame_id=1,
            keyframe_path="L21_V001/001.jpg",
            asr_text=["bản tin thời sự thông báo mưa bão"],
        ),
        FrameRecord(
            vector_id=1,
            video_id="L21_V001",
            frame_id=2,
            keyframe_path="L21_V001/002.jpg",
            asr_text=["chương trình ca nhạc giải trí"],
        ),
        FrameRecord(
            vector_id=2,
            video_id="L21_V002",
            frame_id=10,
            keyframe_path="L21_V002/010.jpg",
            asr_text=[],
        ),
    ]

    embedder = BGEM3TextEmbedder(backend="mock")
    # Encode mock asr features
    f0 = embedder.encode("bản tin thời sự thông báo mưa bão")[0]
    f1 = embedder.encode("chương trình ca nhạc giải trí")[0]
    f2 = np.zeros(1024, dtype=np.float32)
    asr_features = np.vstack([f0, f1, f2])

    clip_vectors = np.random.randn(3, 512).astype(np.float32)
    clip_vectors /= np.linalg.norm(clip_vectors, axis=1, keepdims=True)
    index = VectorIndex(clip_vectors)

    pipeline = RetrievalPipeline(
        index=index,
        manifest=manifest,
        frames_per_video=3,
        asr_features=asr_features,
        encode_asr=embedder.encode,
    )

    bm25 = BM25Index(manifest)

    # Search for weather news (matching frame 0)
    candidates = pipeline.adaptive_multimodal_retrieve_raw(
        visual_query="bão lụt",
        text_embedding=np.random.randn(512).astype(np.float32),
        bm25_index=bm25,
        asr_query="thời tiết mưa bão",
        w_vis=0.1,
        w_ocr=0.0,
        w_asr=0.9,
        top_frames=3,
    )

    assert len(candidates) > 0
    # Frame 0 should rank high due to high ASR matching
    assert any(c.video_id == "L21_V001" and c.frame_id == 1 for c in candidates)
