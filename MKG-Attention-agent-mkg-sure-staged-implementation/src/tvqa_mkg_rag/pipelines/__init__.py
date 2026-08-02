from __future__ import annotations

__all__ = ["build_evidence_cache", "extract_triplets", "predict_val", "report_val"]


def __getattr__(name: str):
    if name in __all__:
        from . import main as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
