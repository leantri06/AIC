from pathlib import Path

import numpy as np
import pytest

from aic2026.app.api import load_orchestrator
from aic2026.paths import resolve_data_root_and_paths


def test_model_paths_keep_manifests_paired(tmp_path: Path):
    old = resolve_data_root_and_paths(tmp_path, "siglip2")
    giant = resolve_data_root_and_paths(tmp_path, "siglip2-giant")
    assert Path(old["features"]).name == "siglip2_features.npy"
    assert Path(giant["features"]).name == "siglip2_giant_features.npy"
    assert Path(giant["manifest"]).name == "siglip2_giant_manifest.jsonl"
    assert giant["clip_model"] == "siglip2-giant"


def test_reject_mixed_known_pair_before_loading():
    with pytest.raises(ValueError, match="row orders differ"):
        load_orchestrator("siglip2_manifest.jsonl", "siglip2_giant_features.npy",
                          "webli", "unused", "unused", clip_model="siglip2-giant")


def test_text_only_checkpoint_matches_full_model(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from transformers import SiglipConfig, SiglipModel, SiglipTextModel

    config = SiglipConfig(
        text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2,
                     "max_position_embeddings": 8, "projection_size": 24,
                     "bos_token_id": 1, "eos_token_id": 2},
        vision_config={"hidden_size": 24, "intermediate_size": 32,
                       "num_hidden_layers": 1, "num_attention_heads": 2,
                       "image_size": 16, "patch_size": 8},
    )
    model = SiglipModel(config).eval()
    model.save_pretrained(tmp_path)
    text_model = SiglipTextModel.from_pretrained(tmp_path).eval()
    inputs = torch.tensor([[3, 4, 2, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        expected = model.text_model(input_ids=inputs).pooler_output
        actual = text_model(input_ids=inputs).pooler_output
    assert actual.shape == (1, 24)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6)
