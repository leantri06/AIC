from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aic2026.models import Candidate, Query
from aic2026.qa.answers import resolve_keyframe_path

logger = logging.getLogger(__name__)

_SYSTEM_JUDGE_PROMPT = (
    "You are an expert AI Visual Judge for a video retrieval competition. "
    "Your job is to objectively analyze video keyframe images against user queries and return structured evaluations. "
    "Always return your verdict in valid JSON."
)


@dataclass
class JudgeResult:
    score: float = 0.0
    matched: bool = False
    reason: str = ""
    event_scores: list[float] = field(default_factory=list)


def _extract_json_block(text: str) -> dict[str, Any]:
    """Robustly extract a JSON object from model generation text."""
    if not text:
        return {}
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    # Fallback score extraction via regex
    score_match = re.search(r'(?:score|điểm)\s*(?:is|là|:|=|")*\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    score = float(score_match.group(1)) if score_match else 50.0
    matched = "true" in text.lower() or score >= 70.0
    return {"score": score, "matched": matched, "reason": text[:150]}


class VLMJudge:
    """Multi-modal Visual Judge for KIS, QA (Evidence Verification), and TRAKE sequence verification.

    Powered by Qwen2.5-VL / Ollama. Operates as a Second-Stage High-Precision Reranker
    over the top candidate keyframes or video trajectories retrieved by SigLIP 2.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        backend: str = "qwen",
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model_name = model_name
        self.backend = backend.lower()
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.ollama_url = ollama_url

        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._loaded: bool = False

    def load(self) -> None:
        """Load the VLM weights if using transformers backend."""
        if self._loaded or self.backend == "mock" or self.backend == "ollama":
            self._loaded = True
            return

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            try:
                import torch
                from transformers import AutoModelForVision2Seq, AutoProcessor
                AutoModelForImageTextToText = AutoModelForVision2Seq  # type: ignore
            except ImportError as exc:
                logger.warning("VLMJudge: transformers/torch not available: %s", exc)
                self._loaded = False
                return

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16) if isinstance(self.torch_dtype, str) else self.torch_dtype
        try:
            logger.info("VLMJudge: loading processor and model %s", self.model_name)
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            device_map = self.device if self.device else ("cuda" if torch.cuda.is_available() else "auto")
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            self._torch = torch
            self._loaded = True
            logger.info("VLMJudge: successfully loaded %s on %s", self.model_name, device_map)
        except Exception as exc:
            logger.warning("VLMJudge: failed to load %s: %s", self.model_name, exc)
            self._loaded = False

    def close(self) -> None:
        """Unload model and free GPU VRAM."""
        self._model = None
        self._processor = None
        self._torch = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core Evaluation Methods
    # ------------------------------------------------------------------

    def judge_kis_candidate(self, image_input: Any, query_text: str) -> JudgeResult:
        """Evaluate how well an image matches all detailed conditions of a KIS query."""
        prompt = (
            "Bạn là một giám khảo thị giác AI chính xác cho cuộc thi truy vấn video.\n"
            f"Hãy quan sát bức ảnh và đánh giá xem bức ảnh này có khớp với mô tả sau không:\n"
            f"Mô tả KIS: \"{query_text}\"\n\n"
            "Hãy kiểm tra các điều kiện sau:\n"
            "- Các đối tượng/vật thể/con người chính có xuất hiện không?\n"
            "- Màu sắc và đặc điểm ngoại hình cụ thể có khớp không?\n"
            "- Hành động và không gian bối cảnh có đúng không?\n\n"
            "Trả về kết quả dưới định dạng JSON duy nhất:\n"
            "{\"score\": <điểm số từ 0 đến 100>, \"matched\": <true hoặc false>, \"reason\": \"<giải thích ngắn gọn 1-2 câu>\"}"
        )
        return self._run_vision_query([image_input], prompt)

    def judge_qa_evidence(self, image_input: Any, context_text: str, question: str) -> JudgeResult:
        """Evaluate if an image contains visual evidence to answer the QA question."""
        prompt = (
            "Bạn là một giám khảo thị giác AI thẩm định bằng chứng.\n"
            f"Ngữ cảnh video: \"{context_text}\"\n"
            f"Câu hỏi cần trả lời: \"{question}\"\n\n"
            "Nhiệm vụ của bạn: Hãy quan sát bức ảnh và đánh giá xem bức ảnh này CÓ CHỨA BẰNG CHỨNG TRỰC TIẾP "
            "(thông tin, chi tiết, vật thể, bảng biểu, con số, chữ viết...) để trả lời được câu hỏi trên hay không.\n"
            "Lưu ý: Bạn KHÔNG cần phải trả lời câu hỏi, chỉ cần đánh giá bức ảnh có đủ bằng chứng thị giác hay không.\n\n"
            "Trả về kết quả dưới định dạng JSON duy nhất:\n"
            "{\"score\": <điểm từ 0 đến 100 về mức độ xuất hiện bằng chứng rõ ràng>, \"matched\": <true hoặc false>, \"reason\": \"<chỉ ra chi tiết trong ảnh làm bằng chứng trả lời>\"}"
        )
        return self._run_vision_query([image_input], prompt)

    def judge_trake_sequence(self, image_inputs: list[Any], events: list[str]) -> JudgeResult:
        """Evaluate an ordered multi-frame sequence against an ordered list of events."""
        if not image_inputs or not events:
            return JudgeResult(score=0.0, matched=False, reason="Missing images or events")

        events_str = "\n".join(f"Event {i + 1}: {e}" for i, e in enumerate(events))
        prompt = (
            "Bạn là một giám khảo thị giác AI thẩm định chuỗi sự kiện thời gian (TRAKE).\n"
            f"Chuỗi sự kiện theo thứ tự thời gian cần kiểm tra ({len(events)} sự kiện):\n{events_str}\n\n"
            f"Được cung cấp {len(image_inputs)} bức ảnh tương ứng với {len(events)} mốc thời gian (Ảnh 1 -> Ảnh {len(image_inputs)}).\n"
            "Hãy thẩm định từng ảnh xem có đúng là sự kiện tương ứng đang diễn ra hay không.\n\n"
            "Trả về kết quả dưới định dạng JSON duy nhất:\n"
            "{\"score\": <điểm trung bình toàn chuỗi từ 0 đến 100>, \"matched\": <true hoặc false>, \"event_scores\": [<điểm 0-100 cho từng event>], \"reason\": \"<nhận xét ngắn gọn từng event trong chuỗi>\"}"
        )
        res = self._run_vision_query(image_inputs, prompt)
        if not res.event_scores and res.score > 0:
            res.event_scores = [res.score] * len(events)
        return res

    # ------------------------------------------------------------------
    # Unified Reranking Flow
    # ------------------------------------------------------------------

    def rerank_candidates(
        self,
        candidates: list[Candidate],
        query: Query,
        top_k: int = 15,
        weight: float = 0.5,
        keyframe_root: Path | None = None,
    ) -> list[Candidate]:
        """Rerank the top-K candidates using VLM Judge scoring according to query type."""
        if not candidates or top_k <= 0:
            return candidates

        self.load()
        if not self._loaded and self.backend != "mock":
            logger.warning("VLMJudge: not loaded, returning original candidates")
            return candidates

        eval_pool = candidates[:top_k]
        rest = candidates[top_k:]

        evaluated: list[Candidate] = []
        for cand in eval_pool:
            cand_copy = cand.model_copy(deep=True)
            res = self._evaluate_single_candidate(cand_copy, query, keyframe_root)
            # Store VLM judge signals in candidate metadata
            cand_copy.metadata["vlm_judge_score"] = float(res.score)
            cand_copy.metadata["vlm_judge_matched"] = bool(res.matched)
            cand_copy.metadata["vlm_judge_reason"] = str(res.reason)
            if res.event_scores:
                cand_copy.metadata["vlm_event_scores"] = res.event_scores

            # Score blend: original score + normalized VLM score bonus
            vlm_normalized = res.score / 100.0
            cand_copy.score = cand.score * (1.0 - weight) + vlm_normalized * weight
            evaluated.append(cand_copy)

        # Sort evaluated candidates by blended score
        evaluated.sort(key=lambda c: c.score, reverse=True)
        return evaluated + rest

    def _evaluate_single_candidate(
        self,
        candidate: Candidate,
        query: Query,
        keyframe_root: Path | None = None,
    ) -> JudgeResult:
        from PIL import Image

        # 1. TRAKE sequence judging
        if query.type == "trake" and query.events:
            if self.backend == "mock":
                return self.judge_trake_sequence([None] * len(query.events), list(query.events))
            if candidate.event_frames:
                images: list[Image.Image] = []
                for ef in candidate.event_frames:
                    if isinstance(ef, int):
                        frame_path = f"{candidate.video_id}/{ef:03d}.jpg"
                    else:
                        frame_path = getattr(ef, "keyframe_path", None) or candidate.keyframe_path
                    resolved = resolve_keyframe_path(frame_path)
                    if not resolved or not resolved.exists():
                        resolved = resolve_keyframe_path(f"{candidate.video_id}/{ef}.jpg")
                    if resolved and resolved.exists():
                        try:
                            images.append(Image.open(resolved).convert("RGB"))
                        except Exception:
                            pass
                if images and len(images) == len(query.events):
                    return self.judge_trake_sequence(images, list(query.events))

        # 2. Single-image judging (KIS or QA)
        if self.backend == "mock":
            if query.type == "qa":
                return self.judge_qa_evidence(None, query.text, query.question or "")
            return self.judge_kis_candidate(None, query.text)

        resolved_path = resolve_keyframe_path(candidate.keyframe_path)
        if not resolved_path or not resolved_path.exists():
            return JudgeResult(score=50.0, matched=False, reason="Keyframe image file not found")

        try:
            img = Image.open(resolved_path).convert("RGB")
        except Exception as exc:
            return JudgeResult(score=50.0, matched=False, reason=f"Cannot open image: {exc}")

        if query.type == "qa":
            return self.judge_qa_evidence(img, query.text, query.question or "")
        return self.judge_kis_candidate(img, query.text)

    # ------------------------------------------------------------------
    # Backend Inference Execution
    # ------------------------------------------------------------------

    def _run_vision_query(self, images: list[Any], prompt: str) -> JudgeResult:
        if self.backend == "mock":
            return JudgeResult(score=92.0, matched=True, reason="[Mock] Perfect visual match confirmed.")

        if self.backend == "ollama":
            return self._run_ollama(images, prompt)

        return self._run_transformers(images, prompt)

    def _run_transformers(self, images: list[Any], prompt: str) -> JudgeResult:
        if not self._loaded or self._processor is None or self._model is None:
            return JudgeResult(score=50.0, matched=False, reason="VLM model not loaded")

        from PIL import Image
        pil_images = [img if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in images]

        content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": _SYSTEM_JUDGE_PROMPT},
            {"role": "user", "content": content},
        ]

        try:
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=pil_images,
                return_tensors="pt",
            ).to(self._model.device)

            with self._torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature,
                )
            generated = self._processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            parsed = _extract_json_block(generated)
            return JudgeResult(
                score=float(parsed.get("score", 50.0)),
                matched=bool(parsed.get("matched", False)),
                reason=str(parsed.get("reason", generated[:150])),
                event_scores=[float(s) for s in parsed.get("event_scores", [])],
            )
        except Exception as exc:
            logger.warning("VLMJudge transformers inference failed: %s", exc)
            return JudgeResult(score=50.0, matched=False, reason=f"Inference error: {exc}")

    def _run_ollama(self, images: list[Any], prompt: str) -> JudgeResult:
        import base64
        import io
        import httpx
        from PIL import Image

        b64_images: list[str] = []
        for img in images:
            pil_img = img if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            b64_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        payload = {
            "model": self.model_name if ":" in self.model_name else "qwen2.5vl:7b",
            "messages": [
                {"role": "system", "content": _SYSTEM_JUDGE_PROMPT},
                {"role": "user", "content": prompt, "images": b64_images},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            response = httpx.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60.0)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            parsed = _extract_json_block(content)
            return JudgeResult(
                score=float(parsed.get("score", 50.0)),
                matched=bool(parsed.get("matched", False)),
                reason=str(parsed.get("reason", content[:150])),
                event_scores=[float(s) for s in parsed.get("event_scores", [])],
            )
        except Exception as exc:
            logger.warning("VLMJudge ollama call failed: %s", exc)
            return JudgeResult(score=50.0, matched=False, reason=f"Ollama error: {exc}")
