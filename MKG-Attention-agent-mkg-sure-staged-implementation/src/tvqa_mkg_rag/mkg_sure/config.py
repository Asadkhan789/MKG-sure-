from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

InputMode = Literal["auto", "raw_video", "precomputed"]


@dataclass(slots=True)
class DataConfig:
    annotations: str
    subtitles: str
    visual_concepts: str | None = None
    raw_video_dir: str | None = None
    precomputed_feature_dir: str | None = None
    input_mode: InputMode = "auto"


@dataclass(slots=True)
class BackboneConfig:
    reader_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    extractor_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    multimodal_embedding_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    detector_name: str = "IDEA-Research/grounding-dino-base"
    dtype: str = "bfloat16"
    freeze_backbones: bool = True


@dataclass(slots=True)
class LocalizationConfig:
    top_b: int = 8
    temporal_margin_seconds: float = 2.0
    subtitle_weight: float = 0.5
    visual_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.top_b < 1:
            raise ValueError("top_b must be positive")
        total = self.subtitle_weight + self.visual_weight
        if total <= 0:
            raise ValueError("localization weights must have positive sum")
        self.subtitle_weight /= total
        self.visual_weight /= total


@dataclass(slots=True)
class PathConfig:
    seed_edges: int = 8
    beam_width: int = 5
    max_path_length: int = 2
    max_candidates: int = 16
    utility_label_candidates: int = 8


@dataclass(slots=True)
class SelectionConfig:
    evidence_budget: int = 1536
    grounding_weight: float = 0.35
    coverage_weight: float = 0.30
    redundancy_weight: float = 0.25
    minimum_gain: float = 0.0
    anchor_threshold: float = 0.55


@dataclass(slots=True)
class TrainingConfig:
    hidden_size: int = 512
    dropout: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate_stage1: float = 2e-4
    learning_rate_stage2: float = 1e-4
    gradient_checkpointing: bool = True
    use_cache: bool = False


@dataclass(slots=True)
class SufficiencyConfig:
    low_threshold: float = 0.35
    high_threshold: float = 0.65
    max_connecting_distance: int = 4
    recovery_paths: int = 4


@dataclass(slots=True)
class ArtifactConfig:
    root: str = "artifacts/mkg_sure"


@dataclass(slots=True)
class MKGSureConfig:
    run_id: str
    data: DataConfig
    backbones: BackboneConfig = field(default_factory=BackboneConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.artifacts.root) / self.run_id

    def stage_dir(self, stage: str) -> Path:
        path = self.run_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path


def _section(cls: type[Any], raw: dict[str, Any], key: str) -> Any:
    return cls(**raw.get(key, {}))


def load_mkg_sure_config(path: str | Path) -> MKGSureConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "run_id" not in raw or "data" not in raw:
        raise ValueError("MKG-Sure config requires run_id and data sections")
    config = MKGSureConfig(
        run_id=str(raw["run_id"]),
        data=DataConfig(**raw["data"]),
        backbones=_section(BackboneConfig, raw, "backbones"),
        localization=_section(LocalizationConfig, raw, "localization"),
        paths=_section(PathConfig, raw, "paths"),
        selection=_section(SelectionConfig, raw, "selection"),
        training=_section(TrainingConfig, raw, "training"),
        sufficiency=_section(SufficiencyConfig, raw, "sufficiency"),
        artifacts=_section(ArtifactConfig, raw, "artifacts"),
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    return config
