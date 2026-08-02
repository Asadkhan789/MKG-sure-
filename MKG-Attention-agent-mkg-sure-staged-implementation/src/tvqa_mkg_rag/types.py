from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BBoxRegion:
    frame_id: int
    label: str
    top: int | None = None
    left: int | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QAExample:
    qid: int
    split: str
    vid_name: str
    question: str
    options: list[str]
    answer_idx: int | None = None
    ts: tuple[float, float] | None = None
    bbox: dict[int, list[BBoxRegion]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = {
            str(frame_id): [region.to_dict() for region in regions]
            for frame_id, regions in self.bbox.items()
        }
        return payload


@dataclass(slots=True)
class SubtitleTurn:
    index: int
    text: str
    start_time: float
    end_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceChunk:
    qid: int
    doc_id: str
    modality: str
    text: str
    vid_name: str
    start_time: float | None = None
    end_time: float | None = None
    frame_ids: list[int] = field(default_factory=list)
    source_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Triplet:
    subject: str
    relation: str
    object: str

    def to_tuple(self) -> tuple[str, str, str]:
        return self.subject, self.relation, self.object

    def to_string(self) -> str:
        return f"({self.subject}, {self.relation}, {self.object})"

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
        }


@dataclass(slots=True)
class TripletRecord:
    qid: int
    doc_id: str
    triplet: Triplet | None
    relation_norm: str | None
    raw_llm_output: str
    source_hash: str
    parse_status: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["triplet"] = self.triplet.to_dict() if self.triplet else None
        return payload


@dataclass(slots=True)
class GraphEdge:
    qid: int
    head: str
    relation: str
    tail: str
    confidence: float
    modalities: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    timestamps: list[tuple[float | None, float | None]] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    source_hashes: list[str] = field(default_factory=list)
    support_count: int = 1

    def key(self) -> tuple[str, str, str]:
        return (self.head.lower(), self.relation.lower(), self.tail.lower())

    def verbalize(self) -> str:
        return f"{self.head} {self.relation} {self.tail}"

    def to_triplet(self) -> str:
        return f"({self.head}, {self.relation}, {self.tail})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditDiagnostics:
    low_edges: list[str] = field(default_factory=list)
    conflicting_edges: list[str] = field(default_factory=list)
    bridge_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditResult:
    qid: int
    quality_score: float
    quality_components: dict[str, float]
    repaired: bool
    selected_edges: list[GraphEdge]
    diagnostics: AuditDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "quality_score": self.quality_score,
            "quality_components": self.quality_components,
            "repaired": self.repaired,
            "selected_edges": [edge.to_dict() for edge in self.selected_edges],
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(slots=True)
class PredictionRecord:
    qid: int
    predicted_idx: int
    gold_idx: int | None
    selected_doc_ids: list[str] = field(default_factory=list)
    selected_subtitle_doc_ids: list[str] = field(default_factory=list)
    selected_triplets: list[str] = field(default_factory=list)
    audit_score: float = 0.0
    llm_raw: str = ""
    verifier_used: bool = False
    risk_flags: list[str] = field(default_factory=list)
    confidence: float | None = None
    correct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
