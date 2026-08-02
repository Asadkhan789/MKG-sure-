from __future__ import annotations

from ..config import AuditConfig
from ..types import AuditDiagnostics, AuditResult, GraphEdge, QAExample
from ..utils import bounded, is_causal_or_affective_question, tokenize


def _find_conflicts(edges: list[GraphEdge]) -> list[str]:
    conflicts: list[str] = []
    seen_by_head_relation: dict[tuple[str, str], str] = {}
    for edge in edges:
        key = (edge.head.lower(), edge.relation.lower())
        tail = edge.tail.lower()
        existing_tail = seen_by_head_relation.get(key)
        if existing_tail is not None and existing_tail != tail:
            conflicts.append(f"{edge.head}|{edge.relation}")
        else:
            seen_by_head_relation[key] = tail
    return sorted(set(conflicts))


def _question_anchors(example: QAExample) -> set[str]:
    return set(tokenize(example.question + " " + " ".join(example.options)))


def _coverage_score(example: QAExample, edges: list[GraphEdge]) -> float:
    anchors = _question_anchors(example)
    if not anchors:
        return 0.0
    edge_tokens: set[str] = set()
    for edge in edges:
        edge_tokens.update(tokenize(edge.verbalize()))
    return len(anchors & edge_tokens) / len(anchors)


def audit_and_repair(
    example: QAExample,
    selected_edges: list[GraphEdge],
    candidate_edges: list[GraphEdge],
    edge_scores: dict[tuple[str, str, str], float],
    config: AuditConfig,
) -> AuditResult:
    if not selected_edges:
        return AuditResult(
            qid=example.qid,
            quality_score=0.0,
            quality_components={"relevance": 0.0, "provenance": 0.0, "consistency": 1.0, "coverage": 0.0},
            repaired=False,
            selected_edges=[],
            diagnostics=AuditDiagnostics(low_edges=[], conflicting_edges=[], bridge_nodes=[]),
        )

    conflicts = _find_conflicts(selected_edges)
    relevance = sum(edge_scores.get(edge.key(), 0.0) for edge in selected_edges) / len(selected_edges)
    provenance = sum(1 for edge in selected_edges if edge.timestamps or edge.frame_ids) / len(selected_edges)
    consistency = 1.0 - (len(conflicts) / len(selected_edges))
    coverage = _coverage_score(example, selected_edges)
    quality_components = {
        "relevance": bounded(relevance),
        "provenance": bounded(provenance),
        "consistency": bounded(consistency),
        "coverage": bounded(coverage),
    }
    quality_score = sum(quality_components.values()) / len(quality_components)

    bridge_nodes = sorted(
        {
            token
            for token in _question_anchors(example)
            if not any(token in tokenize(edge.verbalize()) for edge in selected_edges)
        }
    )
    low_edges = [
        edge.to_triplet()
        for edge in selected_edges
        if edge_scores.get(edge.key(), 0.0) < config.low_score_threshold
        or edge.confidence < config.low_confidence_threshold
    ]
    bbox_only_penalty = 0.0
    if selected_edges and is_causal_or_affective_question(example.question):
        modalities = [{modality.lower() for modality in edge.modalities} for edge in selected_edges]
        if modalities and all(modality_set == {"bbox"} for modality_set in modalities if modality_set):
            bbox_only_penalty = 0.2

    diagnostics = AuditDiagnostics(
        low_edges=low_edges,
        conflicting_edges=conflicts,
        bridge_nodes=bridge_nodes,
    )

    quality_score = bounded(quality_score - bbox_only_penalty)

    if not config.enabled or quality_score >= config.quality_threshold:
        return AuditResult(
            qid=example.qid,
            quality_score=quality_score,
            quality_components=quality_components,
            repaired=False,
            selected_edges=selected_edges,
            diagnostics=diagnostics,
        )

    repaired: list[GraphEdge] = []
    low_edge_keys = {edge.to_triplet() for edge in selected_edges if edge.to_triplet() in low_edges}
    bridge_tokens = set(bridge_nodes)
    for edge in selected_edges:
        edge_tokens = set(tokenize(edge.verbalize()))
        covers_bridge = bool(edge_tokens & bridge_tokens)
        if edge.to_triplet() in low_edge_keys and len(selected_edges) > 2 and not covers_bridge:
            continue
        repaired.append(edge)

    existing_keys = {edge.key() for edge in repaired}
    patch_candidates: list[tuple[float, GraphEdge]] = []
    for edge in candidate_edges:
        if edge.key() in existing_keys:
            continue
        edge_tokens = set(tokenize(edge.verbalize()))
        bridge_bonus = 0.25 if edge_tokens & bridge_tokens else 0.0
        conflict_penalty = 0.2 if f"{edge.head}|{edge.relation}" in conflicts else 0.0
        bbox_penalty = 0.12 if (
            is_causal_or_affective_question(example.question)
            and {modality.lower() for modality in edge.modalities} == {"bbox"}
        ) else 0.0
        patch_candidates.append((edge_scores.get(edge.key(), 0.0) + bridge_bonus - conflict_penalty - bbox_penalty, edge))

    patch_candidates.sort(key=lambda item: (-item[0], item[1].head.lower(), item[1].relation.lower(), item[1].tail.lower()))
    for _, edge in patch_candidates[: config.patch_budget]:
        if edge.key() in existing_keys:
            continue
        repaired.append(edge)
        existing_keys.add(edge.key())

    repaired = repaired[: len(selected_edges)]
    return AuditResult(
        qid=example.qid,
        quality_score=quality_score,
        quality_components=quality_components,
        repaired=True,
        selected_edges=repaired,
        diagnostics=diagnostics,
    )
