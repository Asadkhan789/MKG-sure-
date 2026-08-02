from __future__ import annotations

from collections import defaultdict

from ..types import EvidenceChunk, GraphEdge, TripletRecord
from ..utils import bounded, dedupe_preserve_order, tokenize


def _edge_confidence(records: list[TripletRecord]) -> float:
    support_count = len(records)
    confidence = min(0.95, 0.55 + (support_count - 1) * 0.1)
    exemplar = records[0]
    triplet = exemplar.triplet
    assert triplet is not None

    relation = (exemplar.relation_norm or triplet.relation).lower()
    if "unkname" in triplet.subject.lower() or "unkname" in triplet.object.lower():
        confidence -= 0.12
    if relation in {"says", "states", "asks", "tells"}:
        confidence -= 0.08
    if relation in {"feels", "emotion"}:
        confidence -= 0.05
    if len(tokenize(triplet.object)) >= 6:
        confidence -= 0.05
    return bounded(confidence, minimum=0.2, maximum=0.95)


def build_question_graph(
    qid: int,
    triplet_records: list[TripletRecord],
    evidence_by_doc_id: dict[str, EvidenceChunk],
) -> list[GraphEdge]:
    grouped: dict[tuple[str, str, str], list[TripletRecord]] = defaultdict(list)
    for record in triplet_records:
        if record.triplet is None or record.parse_status != "ok":
            continue
        grouped[
            (
                record.triplet.subject.lower(),
                (record.relation_norm or record.triplet.relation).lower(),
                record.triplet.object.lower(),
            )
        ].append(record)

    edges: list[GraphEdge] = []
    for _, records in grouped.items():
        exemplar = records[0]
        triplet = exemplar.triplet
        assert triplet is not None
        modalities: list[str] = []
        doc_ids: list[str] = []
        timestamps: list[tuple[float | None, float | None]] = []
        frame_ids: list[int] = []
        source_hashes: list[str] = []
        for record in records:
            chunk = evidence_by_doc_id.get(record.doc_id)
            modalities.append(record.provenance.get("modality", "unknown"))
            doc_ids.append(record.doc_id)
            timestamps.append((record.provenance.get("start_time"), record.provenance.get("end_time")))
            frame_ids.extend(record.provenance.get("frame_ids", []))
            source_hashes.append(record.source_hash)
            if chunk and chunk.frame_ids:
                frame_ids.extend(chunk.frame_ids)
        support_count = len(records)
        confidence = _edge_confidence(records)
        edges.append(
            GraphEdge(
                qid=qid,
                head=triplet.subject,
                relation=exemplar.relation_norm or triplet.relation,
                tail=triplet.object,
                confidence=confidence,
                modalities=dedupe_preserve_order(modalities),
                doc_ids=dedupe_preserve_order(doc_ids),
                timestamps=timestamps,
                frame_ids=sorted(set(frame_ids)),
                source_hashes=dedupe_preserve_order(source_hashes),
                support_count=support_count,
            )
        )
    edges.sort(key=lambda edge: (-edge.confidence, edge.head.lower(), edge.relation.lower(), edge.tail.lower()))
    return edges


def build_graph_rows(
    triplet_records: list[TripletRecord],
    evidence_by_doc_id: dict[str, EvidenceChunk],
) -> list[dict]:
    records_by_qid: dict[int, list[TripletRecord]] = defaultdict(list)
    for record in triplet_records:
        records_by_qid[record.qid].append(record)

    rows: list[dict] = []
    for qid in sorted(records_by_qid):
        edges = build_question_graph(qid, records_by_qid[qid], evidence_by_doc_id)
        rows.append({"qid": qid, "edges": [edge.to_dict() for edge in edges]})
    return rows
