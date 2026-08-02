from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .config import MKGSureConfig
from .io_utils import stable_id, write_jsonl
from .schemas import QAExample, SourceUnit


def _parse_answer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_tvqa_examples(config: MKGSureConfig, split: str, limit: int | None = None) -> list[QAExample]:
    path_map = {"train": config.data.annotations_train, "val": config.data.annotations_val, "test": config.data.annotations_test}
    path = Path(path_map[split])
    if not path.exists():
        raise FileNotFoundError(f"TVQA+ annotation file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    examples: list[QAExample] = []
    for row in rows[:limit]:
        boxes: list[dict[str, Any]] = []
        for frame_key, frame_boxes in (row.get("bbox") or {}).items():
            for box in frame_boxes:
                item = dict(box); item["frame_id"] = int(frame_key); boxes.append(item)
        span = row.get("ts")
        examples.append(QAExample(qid=str(row["qid"]), split=split, video_id=str(row["vid_name"]), question=str(row["q"]), options=[str(row.get(f"a{i}", "")) for i in range(5)], answer_idx=_parse_answer(row.get("answer_idx")), gold_temporal_span=(float(span[0]), float(span[1])) if span and len(span) == 2 else None, gold_boxes=boxes))
    return examples


def _load_subtitles(path: Path) -> dict[str, list[tuple[str, float, float | None]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for video_id, record in raw.items():
        texts = [part.strip() for part in str(record.get("sub_text", "")).split("<eos>") if part.strip()]
        starts = [float(value) for value in record.get("sub_time", [])]
        rows = []
        for index, text in enumerate(texts):
            start = starts[index] if index < len(starts) else float(index)
            end = starts[index + 1] if index + 1 < len(starts) else None
            rows.append((text, start, end))
        output[str(video_id)] = rows
    return output


def _load_visual_concepts(path: Path) -> dict[str, list[str]]:
    if not path.exists(): return {}
    with path.open("rb") as handle:
        try: raw = pickle.load(handle, encoding="latin1")
        except TypeError: raw = pickle.load(handle)
    return {str(key): [str(item) for item in value] for key, value in raw.items()}


def _find_video(config: MKGSureConfig, video_id: str) -> str | None:
    if not config.data.raw_video_dir: return None
    root = Path(config.data.raw_video_dir)
    for suffix in config.data.video_extensions:
        candidate = root / f"{video_id}{suffix}"
        if candidate.exists(): return str(candidate)
    return None


def _find_feature(config: MKGSureConfig, video_id: str) -> str | None:
    if not config.data.feature_dir: return None
    root = Path(config.data.feature_dir)
    for suffix in (".pt", ".npy", ".npz", ".safetensors", ".h5"):
        candidate = root / f"{video_id}{suffix}"
        if candidate.exists(): return str(candidate)
    return None


def build_sources(config: MKGSureConfig, examples: list[QAExample]) -> list[SourceUnit]:
    subtitles = _load_subtitles(Path(config.data.subtitles)) if Path(config.data.subtitles).exists() else {}
    visuals = _load_visual_concepts(Path(config.data.visual_concepts))
    sources: list[SourceUnit] = []
    for video_id in sorted({example.video_id for example in examples}):
        media_path, feature_path = _find_video(config, video_id), _find_feature(config, video_id)
        if media_path:
            sources.append(SourceUnit(source_id=f"{video_id}:clip", video_id=video_id, source_type="clip", text=f"Raw video clip {video_id}", media_path=media_path, feature_path=feature_path))
        elif feature_path:
            sources.append(SourceUnit(source_id=f"{video_id}:clip_feature", video_id=video_id, source_type="clip", text=f"Precomputed clip feature {video_id}", feature_path=feature_path))
        for index, (text, start, end) in enumerate(subtitles.get(video_id, [])):
            sources.append(SourceUnit(source_id=f"{video_id}:sub:{index}", video_id=video_id, source_type="subtitle", text=text, start_time=start, end_time=end, metadata={"subtitle_index": index}))
        for frame_index, text in enumerate(visuals.get(video_id, []), start=1):
            sources.append(SourceUnit(source_id=f"{video_id}:visual:{frame_index}", video_id=video_id, source_type="visual_concept", text=text, frame_id=frame_index, start_time=(frame_index - 1) / 3.0, end_time=frame_index / 3.0, confidence=0.6))
        seen = set()
        for example in [item for item in examples if item.video_id == video_id]:
            for box in example.gold_boxes:
                key = (box.get("frame_id"), box.get("label"), box.get("left"), box.get("top"))
                if key in seen: continue
                seen.add(key)
                left, top = float(box.get("left", 0)), float(box.get("top", 0))
                width, height = float(box.get("width", 0)), float(box.get("height", 0))
                sources.append(SourceUnit(source_id=f"{video_id}:oracle_box:{stable_id(*key)}", video_id=video_id, source_type="region", text=str(box.get("label", "object")), frame_id=int(box.get("frame_id", box.get("img_id", 0))), bbox=(left, top, left + width, top + height), confidence=1.0, oracle_annotation=True))
    return sources


def write_source_stage(config: MKGSureConfig, examples: list[QAExample], sources: list[SourceUnit], split: str) -> tuple[Path, Path]:
    stage = config.stage_dir(1, "sources")
    return write_jsonl(stage / f"{split}_qa.jsonl", (item.to_dict() for item in examples)), write_jsonl(stage / f"{split}_sources.jsonl", (item.to_dict() for item in sources))


def make_dummy_dataset(count: int = 12) -> tuple[list[QAExample], list[SourceUnit]]:
    templates = [
        ("Who brought the notebook to the lab?", ["Penny", "Leonard", "Howard", "Raj", "Sheldon"], 2, ["Howard enters the lab carrying a blue notebook.", "Leonard thanks Howard for bringing the notebook."], ["Howard", "notebook", "lab"]),
        ("Why did Penny leave the room?", ["She was tired", "She received a call", "She was angry", "She needed food", "She was late"], 1, ["Penny's phone rings during the discussion.", "Penny says she must take the call and leaves."], ["Penny", "phone", "door"]),
        ("Where is Sheldon sitting?", ["Kitchen", "Office", "Couch", "Car", "Lab"], 2, ["Sheldon sits on his usual spot on the couch.", "Leonard stands beside the couch."], ["Sheldon", "couch", "Leonard"]),
        ("What does Raj give Howard?", ["A book", "A key", "A cup", "A ticket", "A phone"], 3, ["Raj hands Howard a concert ticket.", "Howard reads the ticket and smiles."], ["Raj", "Howard", "ticket"]),
    ]
    examples, sources = [], []
    for index in range(count):
        question, options, answer, subtitle_lines, visual_labels = templates[index % len(templates)]
        video_id = f"dummy_clip_{index // 2:03d}"; split = "train" if index < max(4, int(count * 0.7)) else "val"; qid = f"dummy_{index:04d}"
        examples.append(QAExample(qid=qid, split=split, video_id=video_id, question=question, options=options, answer_idx=answer, gold_temporal_span=(0.0, 6.0)))
        if not any(item.video_id == video_id for item in sources):
            for sub_index, text in enumerate(subtitle_lines):
                sources.append(SourceUnit(source_id=f"{video_id}:sub:{sub_index}", video_id=video_id, source_type="subtitle", text=text, start_time=float(sub_index * 3), end_time=float((sub_index + 1) * 3)))
            for frame_index, label in enumerate(visual_labels, start=1):
                sources.append(SourceUnit(source_id=f"{video_id}:visual:{frame_index}", video_id=video_id, source_type="visual_concept", text=label, frame_id=frame_index, start_time=(frame_index - 1) * 1.5, end_time=frame_index * 1.5, confidence=0.8))
    return examples, sources
