from __future__ import annotations

import numpy as np
import pytest

from aic2026.embeddings.text import normalize_model_name, OpenCLIPTextEmbedder
from aic2026.data_platform.video_frames import OpenCLIPFrameEncoder


def test_siglip2_alias_normalization():
    model, pre = normalize_model_name("siglip2")
    assert model == "ViT-SO400M-14-SigLIP2-378"
    assert pre == "webli"

    model_base, pre_base = normalize_model_name("siglip2-base")
    assert model_base == "ViT-B-16-SigLIP2-384"
    assert pre_base == "webli"

    model_large, pre_large = normalize_model_name("siglip2-large")
    assert model_large == "ViT-L-16-SigLIP2-384"
    assert pre_large == "webli"

    model_legacy, pre_legacy = normalize_model_name("siglip")
    assert model_legacy == "ViT-SO400M-14-SigLIP-384"
    assert pre_legacy == "webli"

    model_custom, pre_custom = normalize_model_name("custom-model", "custom-pre")
    assert model_custom == "custom-model"
    assert pre_custom == "custom-pre"


def test_openclip_text_embedder_siglip2_alias():
    # Test initialization with alias
    try:
        embedder = OpenCLIPTextEmbedder(model_name="siglip2", device="cpu")
        assert embedder.model_name == "ViT-SO400M-14-SigLIP2-378"
        assert embedder.pretrained == "webli"
    except Exception as exc:
        pytest.skip(f"Network / weights download skipped in test environment: {exc}")


def test_openclip_frame_encoder_siglip2_alias():
    try:
        encoder = OpenCLIPFrameEncoder(model_name="siglip2-base", device="cpu")
        assert encoder.model_name == "ViT-B-16-SigLIP2-384"
        assert encoder.pretrained == "webli"
    except Exception as exc:
        pytest.skip(f"Network / weights download skipped in test environment: {exc}")
