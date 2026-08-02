"""TVQA+ MKG-RAG validation-first pipeline."""

from __future__ import annotations

from .config import PipelineConfig, load_config, read_pipeline_config

__all__ = [
    "PipelineConfig",
    "build_evidence_cache",
    "extract_triplets",
    "load_config",
    "read_pipeline_config",
    "predict_val",
    "report_val",
]


def __getattr__(name: str):
    if name in ("build_evidence_cache", "extract_triplets", "predict_val", "report_val"):
        from .pipelines import main as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
