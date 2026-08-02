from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import MKGSureConfig, load_mkg_sure_config
from .data import build_source_manifest, write_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prepare_sources(config: MKGSureConfig, split: str, limit: int | None = None) -> Path:
    rows = build_source_manifest(config.data, split=split, limit=limit)
    return write_manifest(rows, config.stage_dir("01_sources") / f"{split}_source_manifest.jsonl")


def inspect_sources(config: MKGSureConfig, split: str) -> dict[str, Any]:
    path = config.stage_dir("01_sources") / f"{split}_source_manifest.jsonl"
    rows = _read_jsonl(path)
    raw = sum(1 for row in rows if row.get("raw_video_path"))
    feature = sum(1 for row in rows if row.get("feature_path"))
    source_units = sum(len(row.get("source_units", [])) for row in rows)
    payload = {
        "videos": len(rows),
        "videos_with_raw_clips": raw,
        "videos_with_cached_features": feature,
        "source_units": source_units,
        "manifest": str(path),
    }
    output = config.stage_dir("01_sources") / f"{split}_source_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_stage_plan(config: MKGSureConfig) -> Path:
    stages = {
        "01_sources": "Build raw-video/precomputed source manifests.",
        "02_regions": "Run frozen Grounding DINO or import cached regions.",
        "03_facts": "Run frozen Qwen3-VL extraction per video.",
        "04_embeddings": "Cache frozen multimodal embeddings.",
        "05_graphs": "Build query-independent provenance-aware graphs.",
        "06_candidates": "Localize and generate one/two-hop candidate paths.",
        "07_utility_labels": "Generate frozen-reader margin labels on train only.",
        "08_stage1": "Train path encoder, anchor projection and utility head.",
        "09_selection": "Select complementary paths under the evidence budget.",
        "10_sufficiency_data": "Create positive and corrupted sufficiency examples.",
        "11_stage2": "Train source projection, injection and sufficiency modules.",
        "12_calibration": "Fit validation temperature and decision thresholds.",
        "13_evaluation": "Run answer, grounding, retrieval, calibration and cost metrics.",
    }
    output = config.run_dir / "stage_plan.json"
    output.write_text(json.dumps(stages, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged MKG-Sure runner")
    parser.add_argument("stage", choices=["prepare_sources", "inspect_sources", "write_stage_plan"])
    parser.add_argument("--config", default="configs/mkg_sure_tvqaplus_4b.json")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_mkg_sure_config(args.config)
    if args.stage == "prepare_sources":
        print(prepare_sources(config, args.split, args.limit))
    elif args.stage == "inspect_sources":
        print(json.dumps(inspect_sources(config, args.split), indent=2))
    else:
        print(write_stage_plan(config))


if __name__ == "__main__":
    main()
