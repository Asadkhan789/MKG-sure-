from __future__ import annotations

from dataclasses import dataclass

from .config import LocalizationConfig
from .embeddings import Embedder, cosine
from .schemas import SourceUnit


@dataclass(slots=True)
class LocalizedEvidence:
    source_ids: list[str]
    windows: list[tuple[float, float]]
    scores: dict[str, float]


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        old_start, old_end = merged[-1]
        if start <= old_end:
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))
    return merged


def temporal_localize(question: str, sources: list[SourceUnit], embedder: Embedder, config: LocalizationConfig) -> LocalizedEvidence:
    candidates = [source for source in sources if source.source_type in {"subtitle", "clip", "visual_concept", "frame", "region"}]
    if not candidates:
        return LocalizedEvidence([], [], {})
    query = embedder.encode_texts([question])[0]
    source_vectors = embedder.encode_texts([source.text for source in candidates])
    scored: list[tuple[float, SourceUnit]] = []
    for source, vector in zip(candidates, source_vectors):
        base = (cosine(query, vector) + 1.0) / 2.0
        weight = config.subtitle_weight if source.source_type == "subtitle" else config.visual_weight
        scored.append((weight * base, source))
    scored.sort(key=lambda item: (-item[0], item[1].source_id))
    selected = scored[:config.top_b]
    windows = []
    for _, source in selected:
        if source.start_time is not None:
            end = source.end_time if source.end_time is not None else source.start_time
            windows.append((max(0.0, source.start_time - config.temporal_margin_seconds), end + config.temporal_margin_seconds))
    return LocalizedEvidence(
        source_ids=[source.source_id for _, source in selected],
        windows=_merge_windows(windows),
        scores={source.source_id: score for score, source in selected},
    )
