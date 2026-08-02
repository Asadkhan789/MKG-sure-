from __future__ import annotations

import re
from collections import defaultdict

from .config import GraphConfig
from .io_utils import stable_id, tokenize
from .schemas import Fact, GraphEdge, GraphNode


def canonical_label(text: str) -> str:
    value = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(value.split())


def _entity_similarity(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a or not b:
        return 0.0
    lexical = len(a & b) / len(a | b)
    exact = 1.0 if canonical_label(left) == canonical_label(right) else 0.0
    return 0.7 * lexical + 0.3 * exact


def _noisy_or(values: list[float]) -> float:
    product = 1.0
    for value in values:
        product *= 1.0 - min(1.0, max(0.0, value))
    return 1.0 - product


def build_graph(facts: list[Fact], config: GraphConfig) -> tuple[list[GraphNode], list[GraphEdge]]:
    labels_by_video: dict[str, list[str]] = defaultdict(list)
    canonical_map: dict[tuple[str, str], str] = {}
    for fact in facts:
        for label in (fact.head, fact.tail):
            existing = next((item for item in labels_by_video[fact.video_id] if _entity_similarity(label, item) >= config.entity_threshold), None)
            representative = existing or canonical_label(label)
            if existing is None:
                labels_by_video[fact.video_id].append(representative)
            canonical_map[(fact.video_id, label)] = representative

    node_map: dict[tuple[str, str], GraphNode] = {}
    for (video_id, _), label in canonical_map.items():
        key = (video_id, label)
        node_map.setdefault(key, GraphNode(node_id=f"n_{stable_id(video_id, label)}", video_id=video_id, node_type="entity", label=label))

    grouped: dict[tuple[str, str, str, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        head = canonical_map[(fact.video_id, fact.head)]
        tail = canonical_map[(fact.video_id, fact.tail)]
        relation = canonical_label(fact.relation).replace(" ", "_")
        grouped[(fact.video_id, head, relation, tail)].append(fact)

    edges: list[GraphEdge] = []
    for (video_id, head, relation, tail), rows in grouped.items():
        by_source: dict[str, float] = {}
        for row in rows:
            for source_id in row.source_ids:
                by_source[source_id] = max(by_source.get(source_id, 0.0), row.reliability)
        confidence = _noisy_or(list(by_source.values()))
        if confidence < config.minimum_edge_confidence:
            continue
        source_ids = sorted(by_source)
        start_values = [row.start_time for row in rows if row.start_time is not None]
        end_values = [row.end_time for row in rows if row.end_time is not None]
        edge = GraphEdge(
            edge_id=f"e_{stable_id(video_id, head, relation, tail)}", video_id=video_id,
            head_id=node_map[(video_id, head)].node_id, relation=relation, tail_id=node_map[(video_id, tail)].node_id,
            source_type=rows[0].source_type, confidence=confidence, source_ids=source_ids,
            start_time=min(start_values) if start_values else None,
            end_time=max(end_values) if end_values else None,
        )
        edges.append(edge)
        node_map[(video_id, head)].source_ids = sorted(set(node_map[(video_id, head)].source_ids + source_ids))
        node_map[(video_id, tail)].source_ids = sorted(set(node_map[(video_id, tail)].source_ids + source_ids))
    return sorted(node_map.values(), key=lambda node: node.node_id), sorted(edges, key=lambda edge: edge.edge_id)
