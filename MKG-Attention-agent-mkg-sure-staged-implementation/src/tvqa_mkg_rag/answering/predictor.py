from __future__ import annotations

from ..config import AnswerConfig
from ..llm.client import LLMClient
from ..llm.prompts import PromptRepository
from ..types import AuditResult, EvidenceChunk, GraphEdge, PredictionRecord, QAExample
from ..utils import (
    bounded,
    first_json_object,
    is_causal_or_affective_question,
    lexical_overlap_score,
    question_reasoning_type,
)


def parse_answer_index(raw_text: str) -> int | None:
    payload = first_json_object(raw_text)
    if payload is None:
        return None
    answer_idx = payload.get("answer_idx")
    if isinstance(answer_idx, int) and 0 <= answer_idx <= 4:
        return answer_idx
    if isinstance(answer_idx, str) and answer_idx.isdigit():
        value = int(answer_idx)
        return value if 0 <= value <= 4 else None
    return None


def _heuristic_option_scores(example: QAExample, evidence_text: str) -> dict[int, float]:
    return {
        index: lexical_overlap_score(option, evidence_text) + (0.2 * lexical_overlap_score(example.question, option))
        for index, option in enumerate(example.options)
    }


def fallback_answer_index(example: QAExample, evidence_text: str) -> int:
    scores = _heuristic_option_scores(example, evidence_text)
    best_index = max(range(len(example.options)), key=lambda index: (scores[index], -index))
    return int(best_index)


def _heuristic_confidence(option_scores: dict[int, float]) -> float:
    ranked = sorted(option_scores.values(), reverse=True)
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return bounded(0.5 + (ranked[0] / 2.0))
    return bounded(0.5 + (ranked[0] - ranked[1]))


def _normalize_confidence(raw_value: object) -> float | None:
    if isinstance(raw_value, (int, float)):
        return bounded(float(raw_value))
    if isinstance(raw_value, str):
        try:
            return bounded(float(raw_value))
        except ValueError:
            return None
    return None


def _normalize_option_scores(raw_scores: object) -> dict[int, float]:
    if not isinstance(raw_scores, dict):
        return {}
    normalized: dict[int, float] = {}
    for key, value in raw_scores.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if not 0 <= index <= 4:
            continue
        if isinstance(value, (int, float)):
            normalized[index] = bounded(float(value))
        elif isinstance(value, str):
            try:
                normalized[index] = bounded(float(value))
            except ValueError:
                continue
    return normalized


def _normalize_string_list(raw_values: object) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    normalized: list[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def parse_answer_payload(raw_text: str) -> dict[str, object]:
    payload = first_json_object(raw_text) or {}
    answer_idx = parse_answer_index(raw_text)
    option_scores = _normalize_option_scores(payload.get("option_scores"))
    if answer_idx is None and option_scores:
        answer_idx = max(option_scores, key=lambda index: (option_scores[index], -index))
    return {
        "answer_idx": answer_idx,
        "supporting_subtitle_doc_ids": _normalize_string_list(
            payload.get("supporting_subtitle_doc_ids", payload.get("subtitle_doc_ids"))
        ),
        "supporting_triplets": _normalize_string_list(
            payload.get("supporting_triplets", payload.get("selected_triplets"))
        ),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "option_scores": option_scores,
    }


def _format_subtitle_context(selected_chunks: list[EvidenceChunk]) -> tuple[str, list[str]]:
    if not selected_chunks:
        return "None.", []
    lines = [f"- [{chunk.doc_id}] {chunk.text}" for chunk in selected_chunks]
    return "\n".join(lines), [chunk.doc_id for chunk in selected_chunks]


def _format_graph_evidence(
    selected_edges: list[GraphEdge],
    evidence_by_doc_id: dict[str, EvidenceChunk],
) -> tuple[str, list[str], list[str]]:
    if not selected_edges:
        return "None.", [], []

    graph_lines: list[str] = []
    doc_ids: list[str] = []
    triplets: list[str] = []
    for edge in selected_edges:
        triplet = edge.to_triplet()
        modalities = ",".join(edge.modalities) if edge.modalities else "unknown"
        doc_refs = ",".join(edge.doc_ids) if edge.doc_ids else "none"
        graph_lines.append(
            f"- {triplet} [confidence={edge.confidence:.2f}; modalities={modalities}; doc_ids={doc_refs}]"
        )
        triplets.append(triplet)
        for doc_id in edge.doc_ids:
            if doc_id not in evidence_by_doc_id or doc_id in doc_ids:
                continue
            doc_ids.append(doc_id)
    return "\n".join(graph_lines), doc_ids, triplets


def _select_supported_values(candidates: list[str], allowed: list[str], fallback: list[str]) -> list[str]:
    allowed_set = {value.strip() for value in allowed}
    selected = [value for value in candidates if value.strip() in allowed_set]
    return selected or fallback


def _best_option_from_evidence(example: QAExample, evidence_text: str) -> int:
    return fallback_answer_index(example, evidence_text)


def _is_bbox_only_graph(selected_edges: list[GraphEdge]) -> bool:
    if not selected_edges:
        return False
    for edge in selected_edges:
        modalities = {modality.lower() for modality in edge.modalities}
        if modalities != {"bbox"}:
            return False
    return True


def _risk_flags(
    example: QAExample,
    audit_result: AuditResult,
    selected_edges: list[GraphEdge],
    selected_subtitle_doc_ids: list[str],
    subtitle_text: str,
    graph_text: str,
    confidence: float,
    config: AnswerConfig,
) -> list[str]:
    flags: list[str] = []
    question_type = question_reasoning_type(example.question)

    if audit_result.quality_score < config.verify_if_audit_below and selected_edges:
        flags.append("low_audit")
    if confidence < config.verify_if_confidence_below:
        flags.append("low_confidence")
    if question_type in {"why", "feeling"} and confidence < 0.75:
        flags.append("reasoning_exactness")
    if question_type in {"where", "when"} and confidence < 0.9:
        flags.append("temporal_spatial_exactness")
    if question_type in {"why", "feeling", "when"} and len(selected_subtitle_doc_ids) <= 1 and confidence < 0.85:
        flags.append("sparse_subtitle_context")
    if (
        config.verify_on_stream_disagreement
        and subtitle_text != "None."
        and graph_text != "None."
        and _best_option_from_evidence(example, subtitle_text) != _best_option_from_evidence(example, graph_text)
    ):
        flags.append("stream_disagreement")
    if (
        config.verify_on_bbox_only_reasoning
        and config.prefer_subtitles_for_reasoning
        and is_causal_or_affective_question(example.question)
        and _is_bbox_only_graph(selected_edges)
        and not selected_subtitle_doc_ids
    ):
        flags.append("bbox_only_reasoning")

    return flags


def predict_answer(
    example: QAExample,
    audit_result: AuditResult,
    evidence_by_doc_id: dict[str, EvidenceChunk],
    selected_subtitle_chunks: list[EvidenceChunk],
    client: LLMClient,
    prompts: PromptRepository,
    config: AnswerConfig,
) -> PredictionRecord:
    selected_edges = audit_result.selected_edges[: config.max_graph_evidence_units]
    selected_subtitles = selected_subtitle_chunks[: config.max_subtitle_context_units]

    subtitle_text, subtitle_doc_ids = _format_subtitle_context(selected_subtitles)
    graph_text, graph_doc_ids, triplets = _format_graph_evidence(selected_edges, evidence_by_doc_id)
    combined_text = "\n".join(part for part in [subtitle_text, graph_text] if part and part != "None.")
    if not combined_text:
        combined_text = example.question + " " + " ".join(example.options)

    prompt = prompts.render(
        "answer_mc",
        question=example.question,
        options=PromptRepository.format_options(example.options),
        subtitle_context=subtitle_text,
        graph_evidence=graph_text,
    )
    raw = client.query(prompt)
    parsed = parse_answer_payload(raw)

    heuristic_scores = _heuristic_option_scores(example, combined_text)
    option_scores = parsed["option_scores"] if parsed["option_scores"] else heuristic_scores
    predicted_idx = parsed["answer_idx"]
    if predicted_idx is None:
        predicted_idx = max(option_scores, key=lambda index: (option_scores[index], -index))
    confidence = parsed["confidence"]
    if confidence is None:
        confidence = _heuristic_confidence(option_scores)

    selected_subtitle_doc_ids = _select_supported_values(
        parsed["supporting_subtitle_doc_ids"],
        subtitle_doc_ids,
        subtitle_doc_ids,
    )
    selected_triplets = _select_supported_values(
        parsed["supporting_triplets"],
        triplets,
        triplets,
    )

    risk_flags = _risk_flags(
        example,
        audit_result,
        selected_edges,
        selected_subtitle_doc_ids,
        subtitle_text,
        graph_text,
        confidence,
        config,
    )

    verifier_used = False
    if risk_flags:
        verifier_prompt = prompts.render(
            "verify_answer",
            question=example.question,
            options=PromptRepository.format_options(example.options),
            subtitle_context=subtitle_text,
            graph_evidence=graph_text,
            draft_answer=str(predicted_idx),
            risk_flags=", ".join(risk_flags),
        )
        verification_raw = client.query(verifier_prompt)
        raw = raw + "\n---VERIFY---\n" + verification_raw
        verified = parse_answer_payload(verification_raw)
        verified_scores = verified["option_scores"] if verified["option_scores"] else option_scores
        verified_idx = verified["answer_idx"]
        if verified_idx is None:
            verified_idx = max(verified_scores, key=lambda index: (verified_scores[index], -index))
        predicted_idx = verified_idx
        if verified["confidence"] is not None:
            confidence = verified["confidence"]
        else:
            confidence = _heuristic_confidence(verified_scores)
        selected_subtitle_doc_ids = _select_supported_values(
            verified["supporting_subtitle_doc_ids"],
            subtitle_doc_ids,
            selected_subtitle_doc_ids,
        )
        selected_triplets = _select_supported_values(
            verified["supporting_triplets"],
            triplets,
            selected_triplets,
        )
        verifier_used = True

    correct = None if example.answer_idx is None else predicted_idx == example.answer_idx
    return PredictionRecord(
        qid=example.qid,
        predicted_idx=predicted_idx,
        gold_idx=example.answer_idx,
        selected_doc_ids=graph_doc_ids,
        selected_subtitle_doc_ids=selected_subtitle_doc_ids,
        selected_triplets=selected_triplets,
        audit_score=audit_result.quality_score,
        llm_raw=raw,
        verifier_used=verifier_used,
        risk_flags=risk_flags,
        confidence=confidence,
        correct=correct,
    )
