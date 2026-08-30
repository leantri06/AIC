from __future__ import annotations

import pytest
from pathlib import Path

from aic2026.models import Candidate, Query
from aic2026.reranking.vlm_judge import JudgeResult, VLMJudge, _extract_json_block


def test_extract_json_block_markdown():
    raw = "Here is the result:\n```json\n{\"score\": 95, \"matched\": true, \"reason\": \"Perfect match found\"}\n```"
    parsed = _extract_json_block(raw)
    assert parsed["score"] == 95
    assert parsed["matched"] is True
    assert parsed["reason"] == "Perfect match found"


def test_extract_json_block_raw_and_regex_fallback():
    raw_plain = "Verdict: {\"score\": 82.5, \"matched\": true, \"reason\": \"Good match\"}"
    parsed_plain = _extract_json_block(raw_plain)
    assert parsed_plain["score"] == 82.5
    assert parsed_plain["matched"] is True

    raw_fallback = "The candidate score is 78.0 and looks very plausible."
    parsed_fallback = _extract_json_block(raw_fallback)
    assert parsed_fallback["score"] == 78.0
    assert parsed_fallback["matched"] is True


def test_vlm_judge_kis_mock():
    judge = VLMJudge(backend="mock")
    res = judge.judge_kis_candidate(None, "5 people playing with yellow dog")
    assert isinstance(res, JudgeResult)
    assert res.score == 92.0
    assert res.matched is True


def test_vlm_judge_qa_evidence_mock():
    judge = VLMJudge(backend="mock")
    res = judge.judge_qa_evidence(None, "Bác sĩ khám bệnh", "Bác sĩ mặc áo màu gì?")
    assert isinstance(res, JudgeResult)
    assert res.score == 92.0
    assert res.matched is True


def test_vlm_judge_trake_sequence_mock():
    judge = VLMJudge(backend="mock")
    res = judge.judge_trake_sequence([None, None], ["Event 1", "Event 2"])
    assert isinstance(res, JudgeResult)
    assert res.score == 92.0
    assert res.matched is True
    assert len(res.event_scores) == 2


def test_rerank_candidates_flow():
    judge = VLMJudge(backend="mock")
    candidates = [
        Candidate(vector_id=0, video_id="L21_V001", frame_id=10, score=0.8),
        Candidate(vector_id=1, video_id="L21_V001", frame_id=20, score=0.6),
        Candidate(vector_id=2, video_id="L21_V002", frame_id=30, score=0.4),
    ]
    query = Query(query_id="q1", type="kis", text="Test KIS query")
    reranked = judge.rerank_candidates(candidates, query, top_k=2, weight=0.5)

    assert len(reranked) == 3
    # Check that metadata is populated on top_k
    assert "vlm_judge_score" in reranked[0].metadata
    assert reranked[0].metadata["vlm_judge_score"] == 92.0
    assert "vlm_judge_reason" in reranked[0].metadata
    assert reranked[0].metadata["vlm_judge_matched"] is True
