from __future__ import annotations

import logging
import os
from typing import Any

# Fix OpenMP duplicate lib runtime conflict on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

logger = logging.getLogger(__name__)


class BGEM3TextEmbedder:
    """Multilingual & Vietnamese Dense Semantic Embedder powered by BAAI/bge-m3.

    Produces normalized 1024-dimensional dense semantic vectors for video speech transcripts (ASR)
    and user queries.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        max_length: int = 512,
        backend: str = "transformers",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.backend = backend
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded or self.backend == "mock":
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            logger.warning("BGEM3TextEmbedder: transformers/torch not installed: %s", exc)
            return

        resolved_device = self.device or "cpu"

        target_path = self.model_name
        from pathlib import Path
        for candidate_dir in [
            Path(r"D:\bachkhoa\ai_challenge\data\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"),
            Path("data/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"),
        ]:
            if (candidate_dir / "model.safetensors").exists():
                target_path = str(candidate_dir)
                break

        try:
            logger.info("BGEM3TextEmbedder: Loading %s on %s...", self.model_name, resolved_device)
            self._tokenizer = AutoTokenizer.from_pretrained(target_path)
            self._model = AutoModel.from_pretrained(
                target_path,
                low_cpu_mem_usage=True,
                dtype=torch.float32,
            )
            self._model.to(resolved_device)
            self._model.eval()
            self._torch = torch
            self._loaded = True
            logger.info("BGEM3TextEmbedder: Successfully loaded %s", self.model_name)
        except Exception as exc:
            logger.warning("BGEM3TextEmbedder: Failed to load %s: %s", self.model_name, exc)
            self._loaded = False

    @property
    def available(self) -> bool:
        if self.backend == "mock":
            return True
        self._ensure_loaded()
        return self._loaded and self._model is not None

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a single text or list of texts into L2-normalized 1024-dim dense vectors."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        if self.backend == "mock":
            # Deterministic pseudo-embedding for testing
            vectors = []
            for t in texts:
                np.random.seed(abs(hash(t)) % (2**31))
                vec = np.random.randn(1024).astype(np.float32)
                if normalize:
                    vec /= np.linalg.norm(vec) + 1e-9
                vectors.append(vec)
            return np.vstack(vectors)

        self._ensure_loaded()
        if not self._loaded or self._model is None or self._tokenizer is None:
            logger.warning("BGEM3TextEmbedder is not loaded, returning zeros.")
            return np.zeros((len(texts), 1024), dtype=np.float32)

        torch = self._torch
        all_embeddings: list[np.ndarray] = []
        device = next(self._model.parameters()).device

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for b_idx in range(total_batches):
            batch_texts = texts[b_idx * batch_size : (b_idx + 1) * batch_size]
            # Replace empty strings with a single space to avoid tokenizer failure
            clean_texts = [t.strip() if t and t.strip() else " " for t in batch_texts]

            encoded_input = self._tokenizer(
                clean_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                model_output = self._model(**encoded_input)
                # Mean pooling over attention mask
                token_embeddings = model_output[0]  # First element contains hidden states
                input_mask_expanded = (
                    encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)

    def unload(self) -> None:
        """Unload model and free GPU VRAM."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:
                pass
