"""Text-only Transformers encoder for the supplied SigLIP2 Giant corpus."""
from __future__ import annotations

import numpy as np

GIANT_MODEL = "google/siglip2-giant-opt-patch16-384"


class SigLIP2GiantTextEmbedder:
    def __init__(self, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load()

    def load(self) -> None:
        if self.model is not None:
            self.model.to(self.device)
            return
        from transformers import AutoTokenizer, SiglipTextModel

        # Load only the text tower; image vectors are already computed.
        # Prefer local cache so startup never hangs on slow/unstable competition internet.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(GIANT_MODEL, local_files_only=True)
            self.model = SiglipTextModel.from_pretrained(
                GIANT_MODEL,
                torch_dtype=self.torch.float16 if self.device.startswith("cuda") else self.torch.float32,
                use_safetensors=True,
                local_files_only=True,
            ).eval().to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(GIANT_MODEL)
            self.model = SiglipTextModel.from_pretrained(
                GIANT_MODEL,
                torch_dtype=self.torch.float16 if self.device.startswith("cuda") else self.torch.float32,
                use_safetensors=True,
            ).eval().to(self.device)

    def encode(self, text: str) -> np.ndarray:
        self.load()
        tokens = self.tokenizer(
            [(text or "").strip() or "a photo"],
            padding="max_length", truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        # SigLIP uses the final padded position, with bidirectional attention.
        # Match get_text_features: take the projected pooler output, not tokens.
        with self.torch.inference_mode():
            vector = self.model(input_ids=tokens["input_ids"].to(self.device)).pooler_output.float()
            vector = self.torch.nn.functional.normalize(vector, dim=-1)
        return vector[0].cpu().numpy()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
