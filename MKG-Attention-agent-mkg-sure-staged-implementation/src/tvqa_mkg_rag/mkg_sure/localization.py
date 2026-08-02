from __future__ import annotations

from collections.abc import Callable

from .config import LocalizationConfig
from .types import LocalizationResult, SourceUnit

SimilarityFn = Callable[[str, SourceUnit], float]


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def localize_temporal_units(
    question: str,
    source_units: list[SourceUnit],
    subtitle_similarity: SimilarityFn,
    visual_similarity: SimilarityFn,
    config: LocalizationConfig,
) -> LocalizationResult:
    """Rank subtitle-aligned units without assuming raw video availability.

    Raw-video mode supplies clip similarity through ``visual_similarity``. In the
    no-video mode, the same function can score cached visual features or visual
    concept text. Subtitle-only items receive zero visual score.
    """

    scored: list[tuple[float, SourceUnit]] = []
    for unit in source_units:
        subtitle_score = subtitle_similarity(question, unit) if unit.source_type == "subtitle" else 0.0
        visual_score = visual_similarity(question, unit) if unit.source_type != "subtitle" else 0.0
        score = config.subtitle_weight * subtitle_score + config.visual_weight * visual_score
        scored.append((score, unit))

    scored.sort(key=lambda item: (-item[0], item[1].source_id))
    selected = scored[: config.top_b]
    windows: list[tuple[float, float]] = []
    for _, unit in selected:
        if unit.start_time is None:
            continue
        end = unit.end_time if unit.end_time is not None else unit.start_time
        windows.append(
            (
                max(0.0, unit.start_time - config.temporal_margin_seconds),
                end + config.temporal_margin_seconds,
            )
        )
    video_id = selected[0][1].video_id if selected else ""
    return LocalizationResult(
        video_id=video_id,
        selected_source_ids=[unit.source_id for _, unit in selected],
        windows=_merge_windows(windows),
        scores={unit.source_id: score for score, unit in scored},
    )
