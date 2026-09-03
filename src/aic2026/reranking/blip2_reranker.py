"""BLIP-2 Image-Text Matching (ITM) Cross-Encoder Reranker.

Implements fine-grained cross-modal attention reranking based on
arXiv:2512.12935v1 ("Unified Interactive Multimodal Moment Retrieval via
Cascaded Embedding-Reranking and Temporal-Aware Score Fusion").

Uses the Q-Former cross-attention head (Salesforce/blip2-itm-vit-g) to compute
fine-grained matching probabilities between the textual query and visual keyframes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from aic2026.models import Candidate

logger = logging.getLogger(__name__)


class BLIP2ITMReranker:
    """Fine-grained Cross-Encoder Reranker using BLIP-2 ITM (Image-Text Matching)."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-itm-vit-g",
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self._requested_device = device
        self.batch_size = batch_size
        self.model: Any = None
        self.processor: Any = None
        self.device: str = "cpu"
        self.dtype: Any = None

    def load(self) -> None:
        """Load the BLIP-2 ITM model and processor lazily."""
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import AutoProcessor, Blip2ForImageTextRetrieval

        if self._requested_device:
            self.device = self._requested_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # On CUDA GTX 1070 (Pascal), use float16 to conserve VRAM (takes ~2.5GB)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(
            "Loading BLIP-2 ITM model '%s' on %s (%s)...",
            self.model_name,
            self.device,
            self.dtype,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        logger.info("BLIP-2 ITM loaded successfully on %s.", self.device)

    def unload(self) -> None:
        """Free model from VRAM/RAM when not in active use."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def compute_itm_scores(
        self,
        query: str,
        image_paths: list[str | Path],
    ) -> list[float]:
        """Compute P(match) for a list of keyframe images against a query string."""
        if not image_paths or not query.strip():
            return [0.0] * len(image_paths)

        self.load()
        import torch
        from PIL import Image

        scores: list[float] = []
        n_images = len(image_paths)

        for start_idx in range(0, n_images, self.batch_size):
            batch_paths = image_paths[start_idx : start_idx + self.batch_size]
            valid_images = []
            valid_indices = []

            for i, p in enumerate(batch_paths):
                path = Path(p)
                if path.exists():
                    try:
                        img = Image.open(path).convert("RGB")
                        valid_images.append(img)
                        valid_indices.append(i)
                    except Exception as exc:
                        logger.warning("Failed to open image %s: %s", path, exc)

            batch_scores = [0.0] * len(batch_paths)

            if valid_images:
                try:
                    inputs = self.processor(
                        images=valid_images,
                        text=[query] * len(valid_images),
                        return_tensors="pt",
                        padding=True,
                    )
                    device_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            if v.is_floating_point():
                                device_inputs[k] = v.to(device=self.device, dtype=self.dtype)
                            else:
                                device_inputs[k] = v.to(device=self.device)
                        else:
                            device_inputs[k] = v

                    with torch.no_grad():
                        outputs = self.model(**device_inputs, use_itm_head=True)
                        probs = torch.softmax(outputs.itm_score, dim=-1)[:, 1]
                        probs_list = probs.float().cpu().tolist()

                    for local_idx, prob in zip(valid_indices, probs_list):
                        batch_scores[local_idx] = float(prob)
                except Exception as exc:
                    logger.error("Error during BLIP-2 ITM inference: %s", exc)

            scores.extend(batch_scores)

        return scores

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int = 30,
        gamma: float = 0.8,
        min_coarse_weight: float = 0.3,
    ) -> list[Candidate]:
        """Rerank top-k candidates using BLIP-2 ITM cross-attention probabilities."""
        if not candidates or top_k <= 0 or not query.strip():
            return candidates

        rerank_pool = candidates[:top_k]
        remainder = candidates[top_k:]

        image_paths: list[Path] = []
        for c in rerank_pool:
            p = getattr(c, "keyframe_path", None) or getattr(c, "image_path", None)
            if p:
                image_paths.append(Path(p))
            else:
                image_paths.append(Path())

        itm_scores = self.compute_itm_scores(query, image_paths)

        coarse_vals = [float(c.score) for c in rerank_pool]
        c_min, c_max = min(coarse_vals), max(coarse_vals)
        c_range = c_max - c_min if c_max > c_min else 1.0

        reranked: list[Candidate] = []
        for cand, itm_p in zip(rerank_pool, itm_scores):
            norm_c = (float(cand.score) - c_min) / c_range
            fused_score = (min_coarse_weight + (1.0 - min_coarse_weight) * norm_c) * (itm_p ** gamma)

            cand.score = float(fused_score)
            cand.metadata = cand.metadata or {}
            cand.metadata["blip2_itm_score"] = round(float(itm_p), 4)
            cand.metadata["pre_rerank_score"] = round(float(cand.score), 4)
            reranked.append(cand)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked + remainder


def apply_temporal_nms(
    candidates: list[Candidate],
    frame_window: int = 60,
    max_per_video: int = 3,
) -> list[Candidate]:
    """Shot-level diversity non-maximum suppression (NMS)."""
    if not candidates:
        return []

    accepted: list[Candidate] = []
    video_frames: dict[str, list[int]] = {}
    video_counts: dict[str, int] = {}

    for cand in candidates:
        vid = str(cand.video_id)
        fid = int(cand.frame_id)

        curr_count = video_counts.get(vid, 0)
        if curr_count >= max_per_video:
            continue

        past_fids = video_frames.get(vid, [])
        is_too_close = any(abs(fid - pf) <= frame_window for pf in past_fids)
        if is_too_close:
            continue

        accepted.append(cand)
        video_frames.setdefault(vid, []).append(fid)
        video_counts[vid] = curr_count + 1

    return accepted
