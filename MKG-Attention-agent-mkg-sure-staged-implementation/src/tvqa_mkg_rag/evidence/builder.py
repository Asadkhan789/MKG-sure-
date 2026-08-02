from __future__ import annotations

from ..config import EvidenceConfig
from ..types import EvidenceChunk, QAExample, SubtitleTurn
from ..utils import (
    chunked,
    dedupe_preserve_order,
    lexical_overlap_score,
    normalize_whitespace,
    question_reasoning_type,
    stable_hash,
)


def _window_subtitles(turns: list[SubtitleTurn], ts: tuple[float, float] | None, padding: float) -> list[SubtitleTurn]:
    if not turns:
        return []
    if ts is None:
        return turns
    start, end = ts
    lower = start - padding
    upper = end + padding
    windowed = [turn for turn in turns if lower <= turn.start_time <= upper]
    return windowed or turns


def _effective_subtitle_padding(example: QAExample, config: EvidenceConfig) -> float:
    base_padding = config.subtitle_window_padding_seconds
    question_type = question_reasoning_type(example.question)
    if question_type in {"why", "feeling"}:
        return max(base_padding, 8.0)
    if question_type in {"when", "where"}:
        return max(base_padding, 6.0)
    return base_padding


def _score_turns(example: QAExample, turns: list[SubtitleTurn]) -> float:
    joined = " ".join(turn.text for turn in turns)
    option_text = " ".join(example.options)
    return (0.7 * lexical_overlap_score(example.question, joined)) + (
        0.3 * lexical_overlap_score(option_text, joined)
    )


def _make_doc_id(example: QAExample, modality: str, ordinal: int, text: str) -> str:
    return f"{example.qid}:{modality}:{ordinal}:{stable_hash(example.qid, modality, text)}"


def build_subtitle_chunks(example: QAExample, turns: list[SubtitleTurn], config: EvidenceConfig) -> list[EvidenceChunk]:
    if not turns:
        return []
    scoped_turns = _window_subtitles(turns, example.ts, _effective_subtitle_padding(example, config))
    turn_groups = chunked(scoped_turns, config.max_subtitle_turns_per_chunk)
    scored_groups = sorted(
        ((index, group, _score_turns(example, group)) for index, group in enumerate(turn_groups)),
        key=lambda item: (-item[2], item[0]),
    )
    limited_groups = sorted(scored_groups[: config.max_subtitle_chunks_per_question], key=lambda item: item[0])
    chunks: list[EvidenceChunk] = []
    for ordinal, (_, group, score) in enumerate(limited_groups):
        start_time = group[0].start_time
        end_time = group[-1].end_time if group[-1].end_time is not None else group[-1].start_time
        text = " ".join(turn.text for turn in group)
        chunks.append(
            EvidenceChunk(
                qid=example.qid,
                doc_id=_make_doc_id(example, "subtitle", ordinal, text),
                modality="subtitle",
                text=text,
                vid_name=example.vid_name,
                start_time=start_time,
                end_time=end_time,
                source_meta={
                    "turn_indices": [turn.index for turn in group],
                    "score": score,
                    "selected_by": "ts_window" if example.ts else "lexical_rank",
                },
            )
        )
    return chunks


def _labels_to_text(frame_id: int, labels: list[str]) -> str:
    if not labels:
        return f"Frame {frame_id}: no salient labels"
    return f"Frame {frame_id}: " + ", ".join(labels)


def normalize_visual_tags(raw_frame_text: str) -> list[str]:
    parts = [normalize_whitespace(piece) for piece in raw_frame_text.split(",")]
    return dedupe_preserve_order(part for part in parts if part and part.lower() != "logo")


def build_bbox_chunks(example: QAExample, config: EvidenceConfig) -> list[EvidenceChunk]:
    if not example.bbox:
        return []
    grouped_frames = []
    for frame_id in sorted(example.bbox):
        labels = dedupe_preserve_order(region.label for region in example.bbox[frame_id] if region.label)
        grouped_frames.append((frame_id, labels))

    chunks: list[EvidenceChunk] = []
    for ordinal, frame_group in enumerate(chunked(grouped_frames, config.max_visual_frames_per_chunk)):
        frame_ids = [frame_id for frame_id, _ in frame_group]
        text = " ".join(_labels_to_text(frame_id, labels) for frame_id, labels in frame_group)
        chunks.append(
            EvidenceChunk(
                qid=example.qid,
                doc_id=_make_doc_id(example, "bbox", ordinal, text),
                modality="bbox",
                text=text,
                vid_name=example.vid_name,
                frame_ids=frame_ids,
                source_meta={"selected_by": "bbox_frames"},
            )
        )
    return chunks


def build_visual_concept_chunks(
    example: QAExample,
    visual_concepts: list[str],
    config: EvidenceConfig,
) -> list[EvidenceChunk]:
    if not visual_concepts:
        return []

    frame_ids = sorted(example.bbox) if example.bbox else list(range(1, len(visual_concepts) + 1))
    selected_frames: list[tuple[int, list[str]]] = []
    for frame_id in frame_ids:
        index = frame_id - 1
        if index < 0 or index >= len(visual_concepts):
            continue
        tags = normalize_visual_tags(visual_concepts[index])
        if tags:
            selected_frames.append((frame_id, tags))

    chunks: list[EvidenceChunk] = []
    for ordinal, frame_group in enumerate(chunked(selected_frames, config.max_visual_frames_per_chunk)):
        frame_numbers = [frame_id for frame_id, _ in frame_group]
        text = " ".join(
            f"Frame {frame_id}: " + "; ".join(tags[:8]) for frame_id, tags in frame_group
        )
        chunks.append(
            EvidenceChunk(
                qid=example.qid,
                doc_id=_make_doc_id(example, "visual", ordinal, text),
                modality="visual",
                text=text,
                vid_name=example.vid_name,
                frame_ids=frame_numbers,
                source_meta={"selected_by": "visual_concepts"},
            )
        )
    return chunks


def build_evidence_for_example(
    example: QAExample,
    subtitles: dict[str, list[SubtitleTurn]],
    visual_concepts: dict[str, list[str]],
    config: EvidenceConfig,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    if config.include_subtitles:
        chunks.extend(build_subtitle_chunks(example, subtitles.get(example.vid_name, []), config))
    if config.include_bbox:
        chunks.extend(build_bbox_chunks(example, config))
    if config.include_visual_concepts:
        chunks.extend(build_visual_concept_chunks(example, visual_concepts.get(example.vid_name, []), config))
    return chunks
