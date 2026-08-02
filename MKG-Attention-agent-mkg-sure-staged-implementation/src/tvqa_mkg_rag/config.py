from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import ensure_directory, project_root, resolve_path

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def normalize_run_id(value: str | None) -> str | None:
    """Return a safe artifact-namespace token, or None when unset."""
    if value is None:
        return None
    if not isinstance(value, str) or isinstance(value, bool):
        raise TypeError("run_id must be a string or null")
    stripped = value.strip()
    if not stripped:
        return None
    if not _RUN_ID_PATTERN.fullmatch(stripped):
        raise ValueError(
            "run_id must be 1-64 characters and only contain letters, digits, underscore, and hyphen"
        )
    return stripped


@dataclass(slots=True)
class DataPaths:
    annotations_val: str = "data/tvqa_plus_annotations_with_test/tvqa_plus_valid_preprocessed.json"
    annotations_test: str = "data/tvqa_plus_annotations_with_test/tvqa_plus_test_preprocessed_no_anno.json"
    subtitles: str = "data/tvqa_plus_subtitles.json"
    visual_concepts: str = "data/det_visual_concepts_hq.pickle"

    def resolve(self, root: Path) -> "DataPaths":
        return DataPaths(
            annotations_val=str(resolve_path(self.annotations_val, root)),
            annotations_test=str(resolve_path(self.annotations_test, root)),
            subtitles=str(resolve_path(self.subtitles, root)),
            visual_concepts=str(resolve_path(self.visual_concepts, root)),
        )


@dataclass(slots=True)
class ArtifactPaths:
    """Resolved layout is always ``<project>/artifacts/<run_id>/{cache,reports}`` (see PipelineConfig.resolve)."""

    cache_dir: str = "artifacts/cache"
    runs_dir: str = "artifacts/runs"
    reports_dir: str = "artifacts/reports"

    def resolve(self, root: Path) -> "ArtifactPaths":
        return ArtifactPaths(
            cache_dir=str(resolve_path(self.cache_dir, root)),
            runs_dir=str(resolve_path(self.runs_dir, root)),
            reports_dir=str(resolve_path(self.reports_dir, root)),
        )

    def ensure(self) -> None:
        ensure_directory(Path(self.cache_dir))
        ensure_directory(Path(self.runs_dir))
        ensure_directory(Path(self.reports_dir))


@dataclass(slots=True)
class LLMConfig:
    env_file: str = ".env"
    api_key_env: str = "AIGCBEST_API_KEY"
    host: str = "api2.aigcbest.top"
    path: str = "/v1/chat/completions"
    model: str = "gpt-5.1-thinking-all"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_seconds: int = 60
    sleep_seconds: float = 1.0
    max_retries: int = 3
    user_agents: list[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 6.1; WOW64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3)",
        ]
    )

    def resolve(self, root: Path) -> "LLMConfig":
        return LLMConfig(
            env_file=str(resolve_path(self.env_file, root)),
            api_key_env=self.api_key_env,
            host=self.host,
            path=self.path,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            sleep_seconds=self.sleep_seconds,
            max_retries=self.max_retries,
            user_agents=list(self.user_agents),
        )


@dataclass(slots=True)
class EvidenceConfig:
    include_subtitles: bool = True
    include_bbox: bool = True
    include_visual_concepts: bool = False
    subtitle_window_padding_seconds: float = 2.0
    max_subtitle_turns_per_chunk: int = 6
    max_visual_frames_per_chunk: int = 5
    max_subtitle_chunks_per_question: int = 8


@dataclass(slots=True)
class RetrievalConfig:
    max_seed_edges: int = 12
    max_graph_edges: int = 20
    max_selected_evidence_units: int = 5
    expansion_hops: int = 1
    node_degree_cap: int = 8
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    weight_dense: float = 0.35
    weight_lexical: float = 0.2
    weight_provenance: float = 0.15
    weight_confidence: float = 0.15
    weight_subtitle_prior: float = 0.15


@dataclass(slots=True)
class AuditConfig:
    enabled: bool = True
    quality_threshold: float = 0.55
    low_score_threshold: float = 0.2
    low_confidence_threshold: float = 0.55
    patch_budget: int = 3


@dataclass(slots=True)
class AnswerConfig:
    max_evidence_units: int | None = None
    max_subtitle_context_units: int = 3
    max_graph_evidence_units: int = 5
    prefer_subtitles_for_reasoning: bool = True
    verify_if_audit_below: float = 0.5
    verify_if_confidence_below: float = 0.55
    verify_on_stream_disagreement: bool = True
    verify_on_bbox_only_reasoning: bool = True
    require_json_answer: bool = True

    def __post_init__(self) -> None:
        if self.max_evidence_units is not None:
            self.max_graph_evidence_units = self.max_evidence_units


@dataclass(slots=True)
class PromptPaths:
    triplet_text: str = "prompts/triplet_text_prompt.txt"
    triplet_visual: str = "prompts/triplet_visual_prompt.txt"
    answer_mc: str = "prompts/answer_multiple_choice_prompt.txt"
    verify_answer: str = "prompts/verify_answer_prompt.txt"

    def resolve(self, root: Path) -> "PromptPaths":
        return PromptPaths(
            triplet_text=str(resolve_path(self.triplet_text, root)),
            triplet_visual=str(resolve_path(self.triplet_visual, root)),
            answer_mc=str(resolve_path(self.answer_mc, root)),
            verify_answer=str(resolve_path(self.verify_answer, root)),
        )


@dataclass(slots=True)
class PipelineConfig:
    project_root: str
    run_name: str = "val_default"
    # Required for resolve(): namespaces outputs under ``artifacts/<run_id>/``.
    run_id: str | None = None
    split: str = "val"
    limit: int | None = None
    data: DataPaths = field(default_factory=DataPaths)
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    answering: AnswerConfig = field(default_factory=AnswerConfig)
    prompts: PromptPaths = field(default_factory=PromptPaths)

    @property
    def root_path(self) -> Path:
        return Path(self.project_root)

    @property
    def evidence_mode_name(self) -> str:
        parts: list[str] = []
        if self.evidence.include_subtitles:
            parts.append("subtitles")
        if self.evidence.include_bbox:
            parts.append("bbox")
        if self.evidence.include_visual_concepts:
            parts.append("visual")
        if not parts:
            parts.append("none")
        return "_".join(parts)

    @property
    def stage_prefix(self) -> str:
        """Filename stem for JSONL and reports (run_id is only the parent directory under ``artifacts/``)."""
        parts = [self.run_name, self.split, self.evidence_mode_name]
        base = "_".join(parts)
        if self.limit is not None:
            return f"{base}_n{self.limit}"
        return base

    def cache_path(self, stage: str) -> Path:
        return Path(self.artifacts.cache_dir) / f"{self.stage_prefix}_{stage}.jsonl"

    def report_json_path(self) -> Path:
        return Path(self.artifacts.reports_dir) / f"{self.stage_prefix}_report.json"

    def report_markdown_path(self) -> Path:
        return Path(self.artifacts.reports_dir) / f"{self.stage_prefix}_report.md"

    def predicted_outputs_json_path(self) -> Path:
        return Path(self.artifacts.reports_dir) / f"{self.stage_prefix}_predicted_outputs.json"

    def resolve(self) -> "PipelineConfig":
        root = self.root_path
        if self.run_id is None:
            raise ValueError(
                "run_id is required: set \"run_id\" in the JSON config or pass --run-id "
                "(see README: write_run_config and pipeline stages)."
            )
        run_root = root / "artifacts" / self.run_id
        artifacts = ArtifactPaths(
            cache_dir=str(run_root / "cache"),
            reports_dir=str(run_root / "reports"),
            runs_dir=str(run_root),
        )
        artifacts.ensure()
        return PipelineConfig(
            project_root=str(root),
            run_name=self.run_name,
            run_id=self.run_id,
            split=self.split,
            limit=self.limit,
            data=self.data.resolve(root),
            artifacts=artifacts,
            llm=self.llm.resolve(root),
            evidence=self.evidence,
            retrieval=self.retrieval,
            audit=self.audit,
            answering=self.answering,
            prompts=self.prompts.resolve(root),
        )


def _merge_dataclass(dataclass_type: type[Any], values: dict[str, Any] | None) -> Any:
    if not values:
        return dataclass_type()
    return dataclass_type(**values)


def read_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load settings from JSON without ``resolve()`` (so callers can apply CLI overrides first)."""
    root = project_root()
    config_path = resolve_path(path, root)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    limit = raw.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("config 'limit' must be null or a positive integer")

    run_id_raw = raw.get("run_id")
    if run_id_raw is not None and (not isinstance(run_id_raw, str) or isinstance(run_id_raw, bool)):
        raise ValueError("config 'run_id' must be null or a string")
    run_id = normalize_run_id(run_id_raw)

    return PipelineConfig(
        project_root=str(root),
        run_name=raw.get("run_name", "val_default"),
        run_id=run_id,
        split=raw.get("split", "val"),
        limit=limit,
        data=_merge_dataclass(DataPaths, raw.get("data")),
        artifacts=_merge_dataclass(ArtifactPaths, raw.get("artifacts")),
        llm=_merge_dataclass(LLMConfig, raw.get("llm")),
        evidence=_merge_dataclass(EvidenceConfig, raw.get("evidence")),
        retrieval=_merge_dataclass(RetrievalConfig, raw.get("retrieval")),
        audit=_merge_dataclass(AuditConfig, raw.get("audit")),
        answering=_merge_dataclass(AnswerConfig, raw.get("answering")),
        prompts=_merge_dataclass(PromptPaths, raw.get("prompts")),
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Load JSON from disk and resolve paths (requires ``run_id`` in JSON or set before resolve)."""
    return read_pipeline_config(path).resolve()
