from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

from ..utils import stable_hash
from .config import PathConfig, SelectionConfig
from .types import GraphEdge, GraphNode, GraphPath, SelectionTrace

Similarity = Callable[[list[float] | None, list[float] | None], float]


def cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def temporal_iou(span: tuple[float, float] | None, windows: list[tuple[float, float]]) -> float:
    if span is None or not windows:
        return 0.0
    start, end = span
    best = 0.0
    for window_start, window_end in windows:
        intersection = max(0.0, min(end, window_end) - max(start, window_start))
        union = max(end, window_end) - min(start, window_start)
        best = max(best, intersection / union if union > 0 else 0.0)
    return best


def score_seed_edges(
    question_embedding: list[float],
    edges: list[GraphEdge],
    windows: list[tuple[float, float]],
    similarity_weight: float = 0.5,
    confidence_weight: float = 0.25,
    provenance_weight: float = 0.25,
) -> list[tuple[GraphEdge, float]]:
    scored = []
    for edge in edges:
        semantic = (1.0 + cosine(question_embedding, edge.embedding)) / 2.0
        score = (
            similarity_weight * semantic
            + confidence_weight * edge.extraction_confidence
            + provenance_weight * temporal_iou(edge.temporal_span, windows)
        )
        scored.append((edge, score))
    return sorted(scored, key=lambda item: (-item[1], item[0].edge_id))


def generate_candidate_paths(
    video_id: str,
    nodes: dict[str, GraphNode],
    edges: list[GraphEdge],
    ranked_edges: list[tuple[GraphEdge, float]],
    config: PathConfig,
) -> list[GraphPath]:
    adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
    edge_by_id = {edge.edge_id: edge for edge in edges}
    for edge in edges:
        adjacency[edge.head_id].append(edge)

    candidates: dict[tuple[str, ...], GraphPath] = {}
    for seed, _ in ranked_edges[: config.seed_edges]:
        first_key = (seed.edge_id,)
        candidates[first_key] = _make_path(video_id, nodes, edge_by_id, first_key)
        if config.max_path_length < 2:
            continue
        expansions = sorted(
            adjacency.get(seed.tail_id, []),
            key=lambda edge: (-edge.extraction_confidence, edge.edge_id),
        )[: config.beam_width]
        for second in expansions:
            if second.edge_id == seed.edge_id:
                continue
            if not seed.provenance_ids or not second.provenance_ids:
                continue
            key = (seed.edge_id, second.edge_id)
            candidates[key] = _make_path(video_id, nodes, edge_by_id, key)
    ordered = sorted(candidates.values(), key=lambda path: (path.hop_count, path.path_id))
    return ordered[: config.max_candidates]


def _make_path(
    video_id: str,
    nodes: dict[str, GraphNode],
    edge_by_id: dict[str, GraphEdge],
    edge_ids: tuple[str, ...],
) -> GraphPath:
    selected_edges = [edge_by_id[edge_id] for edge_id in edge_ids]
    node_ids = [selected_edges[0].head_id] + [edge.tail_id for edge in selected_edges]
    source_ids = sorted({source_id for edge in selected_edges for source_id in edge.provenance_ids})
    parts = []
    for edge in selected_edges:
        parts.append(edge.verbalize(nodes))
    return GraphPath(
        path_id=f"{video_id}:path:{stable_hash(*edge_ids, limit=16)}",
        video_id=video_id,
        node_ids=node_ids,
        edge_ids=list(edge_ids),
        source_ids=source_ids,
        hop_count=len(edge_ids),
        serialized_text=" -> ".join(parts),
    )


def _redundancy(path: GraphPath, selected: list[GraphPath]) -> float:
    if not selected:
        return 0.0
    path_edges = set(path.edge_ids)
    values = []
    for other in selected:
        other_edges = set(other.edge_ids)
        union = path_edges | other_edges
        structural = len(path_edges & other_edges) / len(union) if union else 0.0
        semantic = (1.0 + cosine(path.embedding, other.embedding)) / 2.0
        values.append(0.5 * semantic + 0.5 * structural)
    return max(values)


def _marginal_cost(path: GraphPath, selected: list[GraphPath]) -> int:
    used_text = {token for item in selected for token in item.text_token_ids}
    used_visual = {token for item in selected for token in item.visual_token_ids}
    return len(set(path.text_token_ids) - used_text) + len(set(path.visual_token_ids) - used_visual)


def select_paths(
    candidates: list[GraphPath],
    anchors: list[str],
    config: SelectionConfig,
) -> tuple[list[GraphPath], list[SelectionTrace]]:
    selected: list[GraphPath] = []
    trace: list[SelectionTrace] = []
    covered: set[str] = set()
    remaining = list(candidates)
    spent = 0

    while remaining:
        best: tuple[float, GraphPath, SelectionTrace] | None = None
        for path in remaining:
            new_anchors = set(path.matched_anchors) - covered
            coverage_gain = len(new_anchors) / max(1, len(anchors))
            redundancy = _redundancy(path, selected)
            cost = max(1, _marginal_cost(path, selected))
            gain = (
                path.utility
                + config.grounding_weight * path.grounding
                + config.coverage_weight * coverage_gain
                - config.redundancy_weight * redundancy
            )
            normalized = gain / cost
            item = SelectionTrace(
                path_id=path.path_id,
                utility=path.utility,
                grounding=path.grounding,
                coverage_gain=coverage_gain,
                redundancy=redundancy,
                marginal_cost=cost,
                marginal_gain=gain,
                normalized_gain=normalized,
            )
            if best is None or normalized > best[0]:
                best = (normalized, path, item)
        if best is None:
            break
        _, path, item = best
        if item.marginal_gain < config.minimum_gain or spent + item.marginal_cost > config.evidence_budget:
            remaining.remove(path)
            if all(
                max(1, _marginal_cost(candidate, selected)) + spent > config.evidence_budget
                for candidate in remaining
            ):
                break
            continue
        selected.append(path)
        trace.append(item)
        spent += item.marginal_cost
        covered.update(path.matched_anchors)
        remaining.remove(path)
    return selected, trace
