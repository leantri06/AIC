from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from aic2026.models import Candidate

Action = Literal["retrieve", "align_events", "answer_question", "finish"]


class ModalityDecomposition(BaseModel):
    """The visual query used by the SigLIP2 encoder."""
    visual_query: str
    w_vis: Literal[1.0] = 1.0


class VisualQueryFacets(BaseModel):
    """Generalized Query Expansion (GQE) & Multi-Facet Decomposition (ECCV 2022 / GQE 2024).

    Deconstructs a user query into 4 complementary visual facets:
    - q_core: High-level overview visual caption summarizing the whole scene.
    - q_entity: Fine-grained subjects, objects, clothing, colors, and attributes.
    - q_action: Dynamic actions, gestures, movements, and interactions.
    - q_scene: Background environment, setting, lighting, and camera perspective.
    """

    q_core: str = Field(description="Visual overview caption summarizing the whole scene")
    q_entity: str = Field(default="", description="Subjects, objects, clothing, colors, and attributes")
    q_action: str = Field(default="", description="Actions, movements, gestures, and dynamic interactions")
    q_scene: str = Field(default="", description="Environment, background, setting, lighting, and camera angle")

    def to_list(self) -> list[str]:
        """Return non-empty facets in fixed canonical order: [core, entity, action, scene]."""
        items: list[str] = []
        for q in (self.q_core, self.q_entity, self.q_action, self.q_scene):
            clean = (q or "").strip()
            if clean and clean not in items:
                items.append(clean)
        return items or ([self.q_core.strip()] if self.q_core.strip() else ["a video frame"])


class AgentPlan(BaseModel):
    """Kế hoạch nhỏ, có schema; tránh agent tự sinh truy vấn/tool tùy ý."""
    query_variants: list[str] = Field(min_length=1, max_length=6)
    events: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(max_length=500)
    modality: ModalityDecomposition | None = None
    facets: VisualQueryFacets | None = None


class AgentDecision(BaseModel):
    action: Action
    selected_vector_ids: list[int] = Field(default_factory=list, max_length=100)
    answer: str | None = Field(default=None, max_length=160)
    rationale: str = Field(default="", max_length=500)


class AgentTrace(BaseModel):
    step: str
    detail: str


class AgentResult(BaseModel):
    candidates: list[Candidate]
    plan: AgentPlan
    trace: list[AgentTrace]
