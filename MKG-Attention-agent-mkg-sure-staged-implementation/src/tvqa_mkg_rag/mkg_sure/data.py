from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from ..datasets.loaders import load_annotations, load_subtitles
from ..utils import normalize_whitespace, stable_hash
from .config import DataConfig
from .types import SourceUnit

_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".webm")
_FEATURE_EXTENSIONS = (".pt", ".npy", ".npz", ".safetensors")


def _find_existing(root: Path | None, stem: str, extensions: tuple[str, ...]) -> Path | None:
    if root is None or not root.exists():
        return None
    for extension in extensions:
        candidate = root / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    matches = sorted(path for path in root.rglob(f"{stem}.*") if path.suffix.lower() in extensions)
    return matches[0] if matches else None


def resolve_input_mode(config: DataConfig) -> str:
    if config.input_mode != "auto":
        return config.input_mode
    video_root = Path(config.raw_video_dir) if config.raw_video_dir else None
    if video_root and video_root.exists() and any(video_root.rglob("*.mp4")):
        return "raw_video"
    return "precomputed"


def _load_visual_concepts(path: str | None) -> dict[str, list[str]]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open("rb") as handle:
        try:
            raw = pickle.load(handle, encoding="latin1")
        except TypeError:
            raw = pickle.load(handle)
    return {
        str(video_id): [normalize_whitespace(str(value)) for value in values]
        for video_id, values in raw.items()
    }


def build_source_manifest(config: DataConfig, split: str = "val", limit: int | None = None) -> list[dict[str, Any]]:
    """Create query-independent video manifests.

    The manifest never requires raw clips. When a clip is missing, downstream stages
    can use subtitles, TVQA+ box labels, visual concepts, or cached feature tensors.
    """

    examples = load_annotations(config.annotations, split, limit=limit)
    subtitles = load_subtitles(config.subtitles)
    visual_concepts = _load_visual_concepts(config.visual_concepts)
    video_root = Path(config.raw_video_dir) if config.raw_video_dir else None
    feature_root = Path(config.precomputed_feature_dir) if config.precomputed_feature_dir else None
    requested_mode = resolve_input_mode(config)

    by_video: dict[str, dict[str, Any]] = {}
    for example in examples:
        row = by_video.setdefault(
            example.vid_name,
            {
                "video_id": example.vid_name,
                "requested_mode": requested_mode,
                "raw_video_path": None,
                "feature_path": None,
                "source_units": [],
            },
        )
        if row["raw_video_path"] is None:
            raw_path = _find_existing(video_root, example.vid_name, _VIDEO_EXTENSIONS)
            row["raw_video_path"] = str(raw_path) if raw_path else None
        if row["feature_path"] is None:
            feature_path = _find_existing(feature_root, example.vid_name, _FEATURE_EXTENSIONS)
            row["feature_path"] = str(feature_path) if feature_path else None

        existing_ids = {unit["source_id"] for unit in row["source_units"]}
        for turn in subtitles.get(example.vid_name, []):
            source_id = f"{example.vid_name}:subtitle:{turn.index}"
            if source_id in existing_ids:
                continue
            unit = SourceUnit(
                source_id=source_id,
                video_id=example.vid_name,
                source_type="subtitle",
                text=turn.text,
                start_time=turn.start_time,
                end_time=turn.end_time,
                media_path=row["raw_video_path"],
                feature_path=row["feature_path"],
                metadata={"subtitle_index": turn.index},
            )
            row["source_units"].append(unit.to_dict())
            existing_ids.add(source_id)

        for frame_id, regions in sorted(example.bbox.items()):
            labels = sorted({region.label for region in regions if region.label})
            text = f"Frame {frame_id}: " + (", ".join(labels) if labels else "unlabelled regions")
            source_id = f"{example.vid_name}:region:{frame_id}:{stable_hash(text, limit=10)}"
            if source_id not in existing_ids:
                row["source_units"].append(
                    SourceUnit(
                        source_id=source_id,
                        video_id=example.vid_name,
                        source_type="region",
                        text=text,
                        frame_ids=[frame_id],
                        media_path=row["raw_video_path"],
                        feature_path=row["feature_path"],
                        metadata={
                            "oracle_annotation": True,
                            "regions": [region.to_dict() for region in regions],
                        },
                    ).to_dict()
                )
                existing_ids.add(source_id)

        for frame_index, concept_text in enumerate(visual_concepts.get(example.vid_name, []), start=1):
            if not concept_text:
                continue
            source_id = f"{example.vid_name}:feature:{frame_index}"
            if source_id in existing_ids:
                continue
            row["source_units"].append(
                SourceUnit(
                    source_id=source_id,
                    video_id=example.vid_name,
                    source_type="feature",
                    text=f"Frame {frame_index}: {concept_text}",
                    frame_ids=[frame_index],
                    feature_path=row["feature_path"],
                    metadata={"source": "visual_concepts"},
                ).to_dict()
            )
            existing_ids.add(source_id)

    manifests = []
    for video_id in sorted(by_video):
        row = by_video[video_id]
        row["available_mode"] = "raw_video" if row["raw_video_path"] else "precomputed"
        row["source_units"].sort(key=lambda item: (item.get("start_time") is None, item.get("start_time") or 0.0, item["source_id"]))
        manifests.append(row)
    return manifests


def write_manifest(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
