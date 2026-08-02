from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SourceType = Literal["subtitle", "clip", "frame", "region", "external", "feature"]


@dataclass(slots=True)
class ProvenanceRecord:
    source_id: str
    source_type: SourceType
    video_id: str
    subtitle_index: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    frame_id: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    external_document_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceUnit:
    source_id: str
    video_id: str
    source_type: SourceType
    text: str = ""
    start_time: float | None = None
    end_time: float | None = None
    frame_ids: list[int] = field(default_factory=list)
    media_path: str | None = None
    feature_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphNode:
    node_id: str
    video_id: str
    node_type: str
    label: str
    provenance_ids: list[str] = field(default_factory=list)
    temporal_position: float | None = None
    embedding: list[float] | None = None

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
    extraction_confidence: float
    provenance_ids: list[str] = field(default_factory=list)
    temporal_span: tuple[float, float] | None = None
    embedding: list[float] | None = None

    def verbalize(self, nodes: dict[str, GraphNode]) -> str:
        head = nodes[self.head_id].label
        tail = nodes[self.tail_id].label
        return f"{head} {self.relation} {tail}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphPath:
    path_id: str
    video_id: str
    node_ids: list[str]
    edge_ids: list[str]
    source_ids: list[str]
    hop_count: int
    serialized_text: str
    embedding: list[float] | None = None
    utility: float = 0.0
    grounding: float = 0.0
    matched_anchors: list[str] = field(default_factory=list)
    text_token_ids: list[str] = field(default_factory=list)
    visual_token_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LocalizationResult:
    video_id: str
    selected_source_ids: list[str]
    windows: list[tuple[float, float]]
    scores: dict[str, float]


@dataclass(slots=True)
class SelectionTrace:
    path_id: str
    utility: float
    grounding: float
    coverage_gain: float
    redundancy: float
    marginal_cost: int
    marginal_gain: float
    normalized_gain: float


@dataclass(slots=True)
class SufficiencyFeatures:
    coverage: float
    connectivity: float
    grounding: float
    utility: float
