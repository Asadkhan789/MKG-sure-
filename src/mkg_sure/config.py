from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    annotations_train: str = "data/tvqa_plus_annotations/tvqa_plus_train.json"
    annotations_val: str = "data/tvqa_plus_annotations/tvqa_plus_val.json"
    annotations_test: str = "data/tvqa_plus_annotations_with_test/tvqa_plus_test_preprocessed_no_anno.json"
    subtitles: str = "data/tvqa_plus_subtitles.json"
    visual_concepts: str = "data/det_visual_concepts_hq.pickle"
    raw_video_dir: str | None = None
    feature_dir: str | None = None
    external_knowledge: str | None = None
    input_mode: str = "auto"
    allow_oracle_visual_input: bool = False
    video_extensions: list[str] = field(default_factory=lambda: [".mp4", ".mkv", ".avi", ".webm"])


@dataclass(slots=True)
class ModelConfig:
    reader_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    extractor_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    embedding_name: str = "hash"
    detector_name: str = "IDEA-Research/grounding-dino-base"
    backend: str = "dummy"
    dtype: str = "bfloat16"
    device: str = "auto"
    flash_attention_2: bool = True
    gradient_checkpointing: bool = True
    freeze_backbones: bool = True
    embedding_dim: int = 128


@dataclass(slots=True)
class LocalizationConfig:
    top_b: int = 8
    temporal_margin_seconds: float = 2.0
    subtitle_weight: float = 0.55
    visual_weight: float = 0.45


@dataclass(slots=True)
class GraphConfig:
    entity_threshold: float = 0.82
    seed_edges: int = 8
    beam_width: int = 5
    max_path_length: int = 2
    max_candidate_paths: int = 16
    utility_label_candidates: int = 8
    minimum_edge_confidence: float = 0.05


@dataclass(slots=True)
class SelectionConfig:
    evidence_budget: int = 900
    max_selected_paths: int = 4
    grounding_weight: float = 0.35
    coverage_weight: float = 0.35
    redundancy_weight: float = 0.25
    semantic_redundancy_weight: float = 0.5
    minimum_gain: float = 0.01
    anchor_threshold: float = 0.35


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    stage1_epochs: int = 12
    stage2_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    utility_margin: float = 0.0
    pairwise_tie_margin: float = 0.03
    pointwise_weight: float = 1.0
    pairwise_weight: float = 0.5
    generation_weight: float = 1.0
    sufficiency_weight: float = 0.5
    hidden_dim: int = 128


@dataclass(slots=True)
class SufficiencyConfig:
    low_threshold: float = 0.35
    high_threshold: float = 0.65
    max_connecting_distance: int = 4
    recovery_paths: int = 2
    temperature: float = 1.0
    support_threshold: float = 0.5


@dataclass(slots=True)
class RuntimeConfig:
    run_id: str = "mkg_sure_4b"
    artifacts_dir: str = "artifacts/mkg_sure"
    resume: bool = True
    max_examples: int | None = None
    frames_per_window: int = 4
    dummy_examples: int = 12


@dataclass(slots=True)
class MKGSureConfig:
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.runtime.artifacts_dir) / self.runtime.run_id

    def stage_dir(self, number: int, name: str) -> Path:
        path = self.run_dir / f"{number:02d}_{name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    return cls(**(raw or {}))


def load_config(path: str | Path) -> MKGSureConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = MKGSureConfig(
        data=_construct(DataConfig, raw.get("data")),
        models=_construct(ModelConfig, raw.get("models")),
        localization=_construct(LocalizationConfig, raw.get("localization")),
        graph=_construct(GraphConfig, raw.get("graph")),
        selection=_construct(SelectionConfig, raw.get("selection")),
        training=_construct(TrainingConfig, raw.get("training")),
        sufficiency=_construct(SufficiencyConfig, raw.get("sufficiency")),
        runtime=_construct(RuntimeConfig, raw.get("runtime")),
    )
    if config.data.input_mode not in {"auto", "raw_video", "precomputed", "dummy"}:
        raise ValueError(f"Unsupported input_mode: {config.data.input_mode}")
    if config.graph.max_path_length not in {1, 2}:
        raise ValueError("MKG-Sure supports one- and two-hop paths only")
    return config
