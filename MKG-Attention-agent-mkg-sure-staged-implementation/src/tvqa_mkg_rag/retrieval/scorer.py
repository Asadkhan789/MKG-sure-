from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from math import log, sqrt

from ..config import RetrievalConfig
from ..types import EvidenceChunk, GraphEdge, QAExample
from ..utils import bounded, is_causal_or_affective_question, lexical_overlap_score, tokenize


def _question_text(example: QAExample) -> str:
    return example.question + " " + " ".join(example.options)


def _timestamp_overlap(
    example_ts: tuple[float, float] | None,
    start: float | None,
    end: float | None,
    *,
    frame_backoff: float = 0.0,
) -> float:
    if example_ts is None:
        return frame_backoff
    if start is None:
        return frame_backoff
    example_start, example_end = example_ts
    final_end = end if end is not None else start
    if start <= example_end + 2.0 and final_end >= example_start - 2.0:
        return 1.0
    return frame_backoff


def _lexical_score(example: QAExample, evidence_text: str) -> float:
    question_overlap = lexical_overlap_score(example.question, evidence_text)
    options_overlap = lexical_overlap_score(" ".join(example.options), evidence_text)
    return (0.6 * question_overlap) + (0.4 * options_overlap)


def _tfidf_cosine_similarity(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0

    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    docs = [set(left_counts), set(right_counts)]
    df = Counter(token for doc in docs for token in doc)
    weights = {token: log((1 + len(docs)) / (1 + df[token])) + 1.0 for token in set(df)}

    dot = sum((left_counts[token] * weights[token]) * (right_counts[token] * weights[token]) for token in weights)
    left_norm = sqrt(sum((count * weights[token]) ** 2 for token, count in left_counts.items()))
    right_norm = sqrt(sum((count * weights[token]) ** 2 for token, count in right_counts.items()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return bounded(dot / (left_norm * right_norm))


@lru_cache(maxsize=4)
def _load_dense_model(model_name: str) -> object | None:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _dense_similarity(query: str, text: str, config: RetrievalConfig) -> float:
    model = _load_dense_model(config.dense_model_name)
    if model is None:
        return _tfidf_cosine_similarity(query, text)

    try:
        query_vec, text_vec = model.encode([query, text], normalize_embeddings=True)
    except Exception:
        return _tfidf_cosine_similarity(query, text)

    try:
        similarity = float(query_vec @ text_vec)
    except Exception:
        return _tfidf_cosine_similarity(query, text)
    return bounded((similarity + 1.0) / 2.0)


def _weighted_score(
    *,
    dense: float,
    lexical: float,
    provenance: float,
    confidence: float,
    subtitle_prior: float,
    config: RetrievalConfig,
) -> float:
    total_weight = (
        config.weight_dense
        + config.weight_lexical
        + config.weight_provenance
        + config.weight_confidence
        + config.weight_subtitle_prior
    )
    if total_weight <= 0.0:
        return 0.0
    score = (
        config.weight_dense * dense
        + config.weight_lexical * lexical
        + config.weight_provenance * provenance
        + config.weight_confidence * confidence
        + config.weight_subtitle_prior * subtitle_prior
    )
    return bounded(score / total_weight)


def _edge_subtitle_prior(example: QAExample, edge: GraphEdge) -> float:
    modalities = {modality.lower() for modality in edge.modalities}
    if "subtitle" in modalities:
        return 1.0
    if is_causal_or_affective_question(example.question) and modalities == {"bbox"}:
        return 0.0
    return 0.2 if modalities else 0.0


def _edge_reasoning_penalty(example: QAExample, edge: GraphEdge) -> float:
    modalities = {modality.lower() for modality in edge.modalities}
    if is_causal_or_affective_question(example.question) and modalities == {"bbox"}:
        return 0.12
    return 0.0


def _chunk_confidence(chunk: EvidenceChunk) -> float:
    raw_score = chunk.source_meta.get("score")
    if isinstance(raw_score, (int, float)):
        return bounded(float(raw_score) * 4.0)
    return 0.6 if chunk.modality == "subtitle" else 0.3


def _provenance_score(example: QAExample, edge: GraphEdge) -> float:
    if edge.timestamps:
        best = max(_timestamp_overlap(example.ts, start, end, frame_backoff=0.0) for start, end in edge.timestamps)
    else:
        best = 0.0
    if best > 0.0:
        return best
    return 0.3 if edge.frame_ids else 0.0


def score_edge(example: QAExample, edge: GraphEdge, config: RetrievalConfig) -> float:
    evidence_text = edge.verbalize()
    query_text = _question_text(example)
    score = _weighted_score(
        dense=_dense_similarity(query_text, evidence_text, config),
        lexical=_lexical_score(example, evidence_text),
        provenance=_provenance_score(example, edge),
        confidence=bounded(edge.confidence),
        subtitle_prior=_edge_subtitle_prior(example, edge),
        config=config,
    )
    return bounded(score - _edge_reasoning_penalty(example, edge))


def score_subtitle_chunk(example: QAExample, chunk: EvidenceChunk, config: RetrievalConfig) -> float:
    query_text = _question_text(example)
    provenance = _timestamp_overlap(example.ts, chunk.start_time, chunk.end_time, frame_backoff=0.0)
    score = _weighted_score(
        dense=_dense_similarity(query_text, chunk.text, config),
        lexical=_lexical_score(example, chunk.text),
        provenance=provenance,
        confidence=_chunk_confidence(chunk),
        subtitle_prior=1.0,
        config=config,
    )
    if is_causal_or_affective_question(example.question):
        score += 0.05
    return bounded(score)


def _anchor_tokens(example: QAExample) -> set[str]:
    return set(tokenize(_question_text(example)))


def retrieve_subtitle_context(
    example: QAExample,
    evidence_chunks: list[EvidenceChunk],
    config: RetrievalConfig,
    *,
    limit: int,
) -> list[EvidenceChunk]:
    subtitle_chunks = [chunk for chunk in evidence_chunks if chunk.modality == "subtitle"]
    if not subtitle_chunks or limit <= 0:
        return []

    scored = [(chunk, score_subtitle_chunk(example, chunk, config)) for chunk in subtitle_chunks]
    scored.sort(key=lambda item: (-item[1], item[0].doc_id))
    selected = [chunk for chunk, _ in scored[:limit]]
    selected.sort(key=lambda chunk: (chunk.start_time is None, chunk.start_time or 0.0, chunk.doc_id))
    return selected


def retrieve_question_edges(
    example: QAExample,
    edges: list[GraphEdge],
    config: RetrievalConfig,
) -> tuple[list[GraphEdge], dict[tuple[str, str, str], float]]:
    scored = [(edge, score_edge(example, edge, config)) for edge in edges]
    scored.sort(key=lambda item: (-item[1], item[0].head.lower(), item[0].relation.lower(), item[0].tail.lower()))

    seed_edges = [edge for edge, _ in scored[: config.max_seed_edges]]
    edge_scores = {edge.key(): score for edge, score in scored}

    adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.head.lower()].append(edge)
        adjacency[edge.tail.lower()].append(edge)

    selected: list[GraphEdge] = []
    seen_keys: set[tuple[str, str, str]] = set()
    anchor_tokens = _anchor_tokens(example)
    frontier_nodes = {edge.head.lower() for edge in seed_edges} | {edge.tail.lower() for edge in seed_edges}
    for edge in seed_edges:
        if edge.key() in seen_keys:
            continue
        selected.append(edge)
        seen_keys.add(edge.key())

    for _ in range(config.expansion_hops):
        if len(selected) >= config.max_graph_edges:
            break
        candidates: list[tuple[float, GraphEdge]] = []
        for node in sorted(frontier_nodes):
            for edge in adjacency.get(node, [])[: config.node_degree_cap]:
                if edge.key() in seen_keys:
                    continue
                connectivity_bonus = 0.15 if (
                    edge.head.lower() in anchor_tokens or edge.tail.lower() in anchor_tokens
                ) else 0.0
                subtitle_bonus = 0.05 if "subtitle" in {modality.lower() for modality in edge.modalities} else 0.0
                candidates.append((edge_scores.get(edge.key(), 0.0) + connectivity_bonus + subtitle_bonus, edge))
        candidates.sort(key=lambda item: (-item[0], item[1].head.lower(), item[1].relation.lower(), item[1].tail.lower()))
        for _, edge in candidates:
            if len(selected) >= config.max_graph_edges:
                break
            if edge.key() in seen_keys:
                continue
            selected.append(edge)
            seen_keys.add(edge.key())
            frontier_nodes.add(edge.head.lower())
            frontier_nodes.add(edge.tail.lower())

    selected.sort(key=lambda edge: (-edge_scores.get(edge.key(), 0.0), edge.head.lower(), edge.relation.lower(), edge.tail.lower()))
    return selected[: config.max_selected_evidence_units], edge_scores
