from __future__ import annotations

import pickle
from pathlib import Path

from ..config import PipelineConfig
from ..types import BBoxRegion, QAExample, SubtitleTurn
from ..utils import load_json, normalize_whitespace


def _parse_bbox(raw_bbox: dict[str, list[dict]]) -> dict[int, list[BBoxRegion]]:
    bbox: dict[int, list[BBoxRegion]] = {}
    for frame_key, regions in (raw_bbox or {}).items():
        frame_id = int(frame_key)
        bbox[frame_id] = [
            BBoxRegion(
                frame_id=int(region.get("img_id", frame_id)),
                label=normalize_whitespace(str(region.get("label", ""))),
                top=region.get("top"),
                left=region.get("left"),
                width=region.get("width"),
                height=region.get("height"),
            )
            for region in regions
        ]
    return bbox


def _parse_answer_idx(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip()
    return int(value) if value.isdigit() else None


def _parse_ts(value: list[float] | tuple[float, float] | None) -> tuple[float, float] | None:
    if not value or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def load_annotations(path: str | Path, split: str, limit: int | None = None) -> list[QAExample]:
    rows = load_json(Path(path))
    examples: list[QAExample] = []
    for row in rows[:limit]:
        options = [normalize_whitespace(str(row.get(f"a{index}", ""))) for index in range(5)]
        example = QAExample(
            qid=int(row["qid"]),
            split=split,
            vid_name=normalize_whitespace(str(row["vid_name"])),
            question=normalize_whitespace(str(row["q"])),
            options=options,
            answer_idx=_parse_answer_idx(row.get("answer_idx")),
            ts=_parse_ts(row.get("ts")),
            bbox=_parse_bbox(row.get("bbox", {})),
        )
        examples.append(example)
    return examples


def load_subtitles(path: str | Path) -> dict[str, list[SubtitleTurn]]:
    raw = load_json(Path(path))
    subtitles: dict[str, list[SubtitleTurn]] = {}
    for vid_name, clip_data in raw.items():
        raw_text = clip_data.get("sub_text", "")
        parts = [normalize_whitespace(part) for part in raw_text.split("<eos>")]
        parts = [part for part in parts if part]
        starts = [float(value) for value in clip_data.get("sub_time", [])]
        turns: list[SubtitleTurn] = []
        for index, text in enumerate(parts):
            start = starts[index] if index < len(starts) else starts[-1] if starts else 0.0
            if index + 1 < len(starts):
                end = starts[index + 1]
            else:
                end = None
            turns.append(SubtitleTurn(index=index, text=text, start_time=start, end_time=end))
        subtitles[vid_name] = turns
    return subtitles


def load_visual_concepts(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("rb") as handle:
        try:
            raw = pickle.load(handle, encoding="latin1")
        except TypeError:
            raw = pickle.load(handle)
    visual_map: dict[str, list[str]] = {}
    for vid_name, frames in raw.items():
        visual_map[str(vid_name)] = [normalize_whitespace(str(frame)) for frame in frames]
    return visual_map


def load_tvqa_bundle(config: PipelineConfig, split: str | None = None) -> tuple[list[QAExample], dict[str, list[SubtitleTurn]], dict[str, list[str]]]:
    target_split = split or config.split
    annotations_path = config.data.annotations_val if target_split == "val" else config.data.annotations_test
    examples = load_annotations(annotations_path, target_split, limit=config.limit)
    subtitles = load_subtitles(config.data.subtitles)
    visual_concepts = load_visual_concepts(config.data.visual_concepts)
    return examples, subtitles, visual_concepts
