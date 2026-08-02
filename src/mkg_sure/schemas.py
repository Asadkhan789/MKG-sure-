from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QAExample:
    qid: str
    split: str
    video_id: str
    question: str
    options: list[str]
    answer_idx: int | None = None
    gold_temporal_span: tuple[float, float] | None = None
    gold_boxes: list[dict[str, Any]] = field(default_factory=list)
    question_type: str | None = None
    knowledge_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceUnit:
    source_id: str
    video_id: str
    source_type: str
    text: str
    start_time: float | None = None
    end_time: float | None = None
    frame_id: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    media_path: str | None = None
    feature_path: str | None = None
    confidence: float = 1.0
    oracle_annotation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Fact:
    fact_id: str
    video_id: str
    head: str
    relation: str
    tail: str
    source_type: str
    source_ids: list[str]
    extraction_log_likelihood: float = 0.0
    reliability: float = 0.5
    start_time: float | None = None
    end_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphNode:
    node_id: str
    video_id: str
    node_type: str
    label: str
    source_ids: list[str] = field(default_factory=list)
    temporal_position: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphEdge:
    edge_id: str
    video_id: str
    head_id: str
    relation: str
    tail_id: str
    source_type: str
    confidence: float
    source_ids: list[str]
    start_time: float | None = None
    end_time: float | None = None

    def verbalize(self, nodes: dict[str, GraphNode]) -> str:
        return f"{nodes[self.head_id].label} {self.relation} {nodes[self.tail_id].label}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphPath:
    path_id: str
    video_id: str
    node_ids: list[str]
    edge_ids: list[str]
    source_ids: list[str]
    serialized_text: str
    hop_count: int
    seed_score: float = 0.0
    utility: float = 0.0
    grounding: float = 0.0
    matched_anchors: list[str] = field(default_factory=list)
    text_token_ids: list[str] = field(default_factory=list)
    visual_token_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UtilityLabel:
    qid: str
    path_id: str
    baseline_margin: float
    path_margin: float
    utility: float
    helpful: bool
    question_vector: list[float]
    path_vector: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectionRecord:
    qid: str
    path_id: str
    utility: float
    grounding: float
    coverage_gain: float
    redundancy: float
    marginal_cost: int
    marginal_gain: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SufficiencyExample:
    qid: str
    features: list[float]
    label: int
    corruption: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Prediction:
    qid: str
    predicted_idx: int | None
    gold_idx: int | None
    option_scores: list[float]
    selected_path_ids: list[str]
    selected_source_ids: list[str]
    sufficiency_probability: float
    calibrated_probability: float
    decision: str
    recovery_invoked: bool
    before_recovery_idx: int | None = None
    latency_ms: float = 0.0
    candidate_path_count: int = 0
    selected_path_count: int = 0
    text_tokens: int = 0
    visual_tokens: int = 0
    decoder_calls: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["correct"] = None if self.gold_idx is None or self.predicted_idx is None else self.predicted_idx == self.gold_idx
        return payload
