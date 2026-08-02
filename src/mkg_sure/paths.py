from __future__ import annotations

from collections import defaultdict

import torch

from .config import GraphConfig, SelectionConfig
from .embeddings import Embedder, cosine
from .io_utils import stable_id, tokenize
from .localization import LocalizedEvidence
from .schemas import GraphEdge, GraphNode, GraphPath, SourceUnit

_STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "did", "does", "do", "to", "of", "in", "on", "at", "when", "who", "why", "where", "what", "which", "and", "or", "he", "she", "they", "it"}


def question_anchors(question: str) -> list[str]:
    anchors = []
    for token in tokenize(question):
        if token not in _STOPWORDS and len(token) > 2 and token not in anchors:
            anchors.append(token)
    return anchors or tokenize(question)[:3]


def temporal_iou(edge: GraphEdge, windows: list[tuple[float, float]]) -> float:
    if edge.start_time is None or not windows:
        return 0.0
    edge_end = edge.end_time if edge.end_time is not None else edge.start_time
    best = 0.0
    for start, end in windows:
        intersection = max(0.0, min(edge_end, end) - max(edge.start_time, start))
        union = max(edge_end, end) - min(edge.start_time, start)
        best = max(best, intersection / union if union > 0 else 0.0)
    return best


def seed_edges(question: str, nodes: list[GraphNode], edges: list[GraphEdge], localized: LocalizedEvidence, embedder: Embedder, config: GraphConfig) -> list[tuple[float, GraphEdge]]:
    node_map = {node.node_id: node for node in nodes}
    query = embedder.encode_texts([question])[0]
    texts = [edge.verbalize(node_map) for edge in edges]
    vectors = embedder.encode_texts(texts)
    rows = []
    active_sources = set(localized.source_ids)
    for edge, vector in zip(edges, vectors):
        semantic = (cosine(query, vector) + 1.0) / 2.0
        provenance = max(temporal_iou(edge, localized.windows), 1.0 if active_sources & set(edge.source_ids) else 0.0)
        score = 0.55 * semantic + 0.25 * edge.confidence + 0.20 * provenance
        rows.append((score, edge))
    rows.sort(key=lambda item: (-item[0], item[1].edge_id))
    return rows[:config.seed_edges]


def generate_paths(question: str, nodes: list[GraphNode], edges: list[GraphEdge], localized: LocalizedEvidence, sources: dict[str, SourceUnit], embedder: Embedder, graph_config: GraphConfig, selection_config: SelectionConfig) -> list[GraphPath]:
    node_map = {node.node_id: node for node in nodes}
    edge_map = {edge.edge_id: edge for edge in edges}
    adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.head_id].append(edge)
        adjacency[edge.tail_id].append(edge)
    seeds = seed_edges(question, nodes, edges, localized, embedder, graph_config)
    candidates: list[tuple[float, list[str], list[str]]] = []
    for seed_score, edge in seeds:
        candidates.append((seed_score, [edge.head_id, edge.tail_id], [edge.edge_id]))
        if graph_config.max_path_length >= 2:
            for endpoint in (edge.head_id, edge.tail_id):
                extensions = sorted(adjacency[endpoint], key=lambda item: (-item.confidence, item.edge_id))[:graph_config.beam_width]
                for second in extensions:
                    if second.edge_id == edge.edge_id:
                        continue
                    other = second.tail_id if second.head_id == endpoint else second.head_id
                    first = edge.tail_id if edge.head_id == endpoint else edge.head_id
                    candidates.append((seed_score + 0.5 * second.confidence, [first, endpoint, other], [edge.edge_id, second.edge_id]))
    unique: dict[tuple[str, ...], tuple[float, list[str], list[str]]] = {}
    for row in candidates:
        key = tuple(row[2])
        if key not in unique or row[0] > unique[key][0]:
            unique[key] = row
    anchors = question_anchors(question)
    anchor_vectors = embedder.encode_texts(anchors)
    output: list[GraphPath] = []
    for score, node_ids, edge_ids in sorted(unique.values(), key=lambda item: -item[0])[:graph_config.max_candidate_paths]:
        path_edges = [edge_map[edge_id] for edge_id in edge_ids]
        source_ids = sorted({source_id for edge in path_edges for source_id in edge.source_ids})
        if not source_ids:
            continue
        labels = [node_map[node_id].label for node_id in node_ids]
        pieces = [f"{labels[index]} --{edge.relation}--> {labels[index + 1]}" for index, edge in enumerate(path_edges)]
        serialized = " ; ".join(pieces)
        path_vector = embedder.encode_texts([serialized])[0]
        matched = []
        node_vectors = embedder.encode_texts(labels)
        for anchor, anchor_vector in zip(anchors, anchor_vectors):
            if max((cosine(anchor_vector, vector) + 1.0) / 2.0 for vector in node_vectors) >= selection_config.anchor_threshold:
                matched.append(anchor)
        source_texts = [sources[source_id].text for source_id in source_ids if source_id in sources]
        source_vectors = embedder.encode_texts(source_texts) if source_texts else torch.empty((0, embedder.dim))
        grounding_values = [(cosine(path_vector, vector) + 1.0) / 2.0 for vector in source_vectors]
        edge_reliability = sum(edge.confidence for edge in path_edges) / len(path_edges)
        grounding = edge_reliability * (sum(grounding_values) / len(grounding_values) if grounding_values else 0.0)
        text_tokens = sorted({token for token in tokenize(serialized + " " + " ".join(source_texts))})
        visual_tokens = sorted({source_id for source_id in source_ids if source_id in sources and sources[source_id].source_type in {"clip", "frame", "region", "visual_concept"}})
        output.append(GraphPath(
            path_id=f"p_{stable_id(question, *edge_ids)}", video_id=path_edges[0].video_id,
            node_ids=node_ids, edge_ids=edge_ids, source_ids=source_ids, serialized_text=serialized,
            hop_count=len(edge_ids), seed_score=score, grounding=grounding, matched_anchors=matched,
            text_token_ids=text_tokens, visual_token_ids=visual_tokens,
        ))
    return output
