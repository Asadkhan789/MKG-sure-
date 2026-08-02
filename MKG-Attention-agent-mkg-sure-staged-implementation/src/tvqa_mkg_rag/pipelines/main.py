from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _with_progress(iterable: Iterable[T], *, desc: str, unit: str) -> Iterable[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit)

from ..answering import predict_answer
from ..audit import audit_and_repair
from ..config import PipelineConfig, normalize_run_id, project_root, read_pipeline_config, resolve_path
from ..datasets import load_annotations, load_tvqa_bundle
from ..evaluation import build_prediction_outputs_payload, build_report_payload, render_markdown_report
from ..evidence import build_evidence_for_example
from ..graph import build_graph_rows
from ..llm import LLMClient, PromptRepository
from ..retrieval import retrieve_question_edges, retrieve_subtitle_context
from ..triplets import build_triplet_prompt_name
from ..triplets.extractor import build_triplet_records
from ..types import EvidenceChunk, GraphEdge, PredictionRecord, QAExample, TripletRecord
from ..utils import append_jsonl, dump_json, iter_jsonl, read_seen_values, stable_hash


def _config(config_or_path: PipelineConfig | str | Path | None) -> PipelineConfig:
    if isinstance(config_or_path, PipelineConfig):
        if config_or_path.run_id is None:
            raise ValueError(
                'PipelineConfig.run_id is required (set "run_id" in JSON or pass --run-id before resolve).'
            )
        return config_or_path.resolve()
    if config_or_path is None:
        raise ValueError("A PipelineConfig instance or config path is required.")
    return read_pipeline_config(config_or_path).resolve()


def _load_evidence_rows(path: Path) -> list[EvidenceChunk]:
    rows: list[EvidenceChunk] = []
    for row in iter_jsonl(path):
        rows.append(EvidenceChunk(**row))
    return rows


def _load_triplet_rows(path: Path) -> list[TripletRecord]:
    rows: list[TripletRecord] = []
    for row in iter_jsonl(path):
        triplet_payload = row.get("triplet")
        if triplet_payload:
            from ..types import Triplet

            row["triplet"] = Triplet(**triplet_payload)
        rows.append(TripletRecord(**row))
    return rows


def _graph_edges_from_rows(path: Path) -> dict[int, list[GraphEdge]]:
    edges_by_qid: dict[int, list[GraphEdge]] = {}
    for row in iter_jsonl(path):
        qid = int(row["qid"])
        edges_by_qid[qid] = [GraphEdge(**edge) for edge in row.get("edges", [])]
    return edges_by_qid


def _evidence_by_qid(evidence_rows: list[EvidenceChunk]) -> dict[int, list[EvidenceChunk]]:
    rows_by_qid: dict[int, list[EvidenceChunk]] = {}
    for row in evidence_rows:
        rows_by_qid.setdefault(row.qid, []).append(row)
    return rows_by_qid


def _load_examples_only(config: PipelineConfig) -> list[QAExample]:
    annotations_path = config.data.annotations_val if config.split == "val" else config.data.annotations_test
    return load_annotations(annotations_path, config.split, limit=config.limit)


def _write_prediction_outputs(
    config: PipelineConfig,
    examples: list[QAExample],
    predictions: list[PredictionRecord],
) -> Path:
    payload = build_prediction_outputs_payload(config.stage_prefix, examples, predictions)
    output_path = config.predicted_outputs_json_path()
    dump_json(output_path, payload)
    return output_path


def _load_prediction_records(path: Path) -> list[PredictionRecord]:
    return [PredictionRecord(**row) for row in iter_jsonl(path)]


def _refresh_prediction_outputs(config: PipelineConfig, examples: list[QAExample]) -> Path:
    return _write_prediction_outputs(
        config,
        examples,
        _load_prediction_records(config.cache_path("predictions")),
    )


def build_evidence_cache(config_or_path: PipelineConfig | str | Path | None = None) -> Path:
    config = _config(config_or_path)
    evidence_path = config.cache_path("evidence")
    seen_doc_ids = read_seen_values(evidence_path, "doc_id")
    examples, subtitles, visual_concepts = load_tvqa_bundle(config)

    rows: list[dict[str, Any]] = []
    for example in examples:
        chunks = build_evidence_for_example(example, subtitles, visual_concepts, config.evidence)
        for chunk in chunks:
            if chunk.doc_id in seen_doc_ids:
                continue
            rows.append(chunk.to_dict())
            seen_doc_ids.add(chunk.doc_id)
    append_jsonl(evidence_path, rows)
    return evidence_path


def _example_index(examples: list[QAExample]) -> dict[int, QAExample]:
    return {example.qid: example for example in examples}


def _filter_evidence_to_examples(
    evidence_rows: list[EvidenceChunk], examples: list[QAExample]
) -> list[EvidenceChunk]:
    allowed_qids = {example.qid for example in examples}
    return [chunk for chunk in evidence_rows if chunk.qid in allowed_qids]


def _filter_triplets_to_examples(
    triplet_rows: list[TripletRecord], examples: list[QAExample]
) -> list[TripletRecord]:
    allowed_qids = {example.qid for example in examples}
    return [row for row in triplet_rows if row.qid in allowed_qids]


def extract_triplets(
    config_or_path: PipelineConfig | str | Path | None = None,
    client: LLMClient | None = None,
) -> Path:
    config = _config(config_or_path)
    evidence_path = build_evidence_cache(config)
    triplet_path = config.cache_path("triplets")
    seen_hashes = read_seen_values(triplet_path, "source_hash")
    examples, subtitles, visual_concepts = load_tvqa_bundle(config)
    _ = subtitles, visual_concepts
    example_by_qid = _example_index(examples)
    evidence_rows = _filter_evidence_to_examples(_load_evidence_rows(evidence_path), examples)
    prompt_repository = PromptRepository(config.prompts)
    llm_client = client or LLMClient(config.llm)

    for chunk in _with_progress(evidence_rows, desc="extract_triplets", unit="chunk"):
        example = example_by_qid.get(chunk.qid)
        if example is None:
            continue
        source_hash = stable_hash(example.question, chunk.text, limit=32)
        if source_hash in seen_hashes:
            continue
        prompt_name = build_triplet_prompt_name(chunk)
        prompt = prompt_repository.render(
            prompt_name,
            question=example.question,
            options=PromptRepository.format_options(example.options),
            evidence=chunk.text,
        )
        raw_response = llm_client.query(prompt)
        records = build_triplet_records(example, chunk, raw_response)
        written = append_jsonl(triplet_path, (record.to_dict() for record in records))
        if written:
            seen_hashes.add(source_hash)
    return triplet_path


def _build_graph_cache(config: PipelineConfig) -> Path:
    graph_path = config.cache_path("graphs")
    evidence_path = build_evidence_cache(config)
    triplet_path = config.cache_path("triplets")
    examples, _, _ = load_tvqa_bundle(config)
    evidence_rows = _filter_evidence_to_examples(_load_evidence_rows(evidence_path), examples)
    triplet_rows = _filter_triplets_to_examples(_load_triplet_rows(triplet_path), examples)
    evidence_by_doc_id = {chunk.doc_id: chunk for chunk in evidence_rows}
    seen_qids = read_seen_values(graph_path, "qid")
    graph_rows = [row for row in build_graph_rows(triplet_rows, evidence_by_doc_id) if row["qid"] not in seen_qids]
    append_jsonl(graph_path, graph_rows)
    return graph_path


def predict_val(
    config_or_path: PipelineConfig | str | Path | None = None,
    client: LLMClient | None = None,
) -> Path:
    config = _config(config_or_path)
    if config.split != "val":
        raise ValueError("predict_val is validation-only. Use split='val' in the config.")
    extract_triplets(config, client=client)
    graph_path = _build_graph_cache(config)
    prediction_path = config.cache_path("predictions")
    seen_qids = read_seen_values(prediction_path, "qid")
    examples, subtitles, visual_concepts = load_tvqa_bundle(config)
    _ = subtitles, visual_concepts
    evidence_rows = _filter_evidence_to_examples(_load_evidence_rows(config.cache_path("evidence")), examples)
    evidence_by_doc_id = {chunk.doc_id: chunk for chunk in evidence_rows}
    evidence_rows_by_qid = _evidence_by_qid(evidence_rows)
    edges_by_qid = _graph_edges_from_rows(graph_path)
    llm_client = client or LLMClient(config.llm)
    prompts = PromptRepository(config.prompts)

    try:
        for example in examples:
            if example.qid in seen_qids:
                continue
            question_edges = edges_by_qid.get(example.qid, [])
            selected_edges, edge_scores = retrieve_question_edges(example, question_edges, config.retrieval)
            audit_result = audit_and_repair(example, selected_edges, question_edges, edge_scores, config.audit)
            selected_subtitles = retrieve_subtitle_context(
                example,
                evidence_rows_by_qid.get(example.qid, []),
                config.retrieval,
                limit=config.answering.max_subtitle_context_units,
            )
            prediction = predict_answer(
                example,
                audit_result,
                evidence_by_doc_id,
                selected_subtitles,
                llm_client,
                prompts,
                config.answering,
            )
            append_jsonl(prediction_path, [prediction.to_dict()])
            seen_qids.add(example.qid)
    except Exception:
        try:
            _refresh_prediction_outputs(config, examples)
        except Exception:
            pass
        raise
    _refresh_prediction_outputs(config, examples)
    return prediction_path


def report_val(config_or_path: PipelineConfig | str | Path | None = None) -> Path:
    config = _config(config_or_path)
    prediction_path = config.cache_path("predictions")
    predictions = _load_prediction_records(prediction_path)
    payload = build_report_payload(config.stage_prefix, predictions)
    dump_json(config.report_json_path(), payload)
    examples = _load_examples_only(config)
    _write_prediction_outputs(config, examples, predictions)
    config.report_markdown_path().write_text(render_markdown_report(payload), encoding="utf-8")
    return config.report_markdown_path()


def write_run_config(
    base_config_path: str | Path,
    run_id: str,
    out_path: str | Path | None = None,
) -> Path:
    """Copy a base JSON config and set ``run_id`` for artifact namespacing."""
    rid = normalize_run_id(run_id)
    if rid is None:
        raise ValueError("write_run_config requires a non-empty --run-id")

    root = project_root()
    base = resolve_path(base_config_path, root)
    data = json.loads(base.read_text(encoding="utf-8"))
    data["run_id"] = rid
    data["artifacts"] = {
        "cache_dir": f"artifacts/{rid}/cache",
        "reports_dir": f"artifacts/{rid}/reports",
        "runs_dir": f"artifacts/{rid}",
    }
    if out_path is None:
        out = root / "configs" / f"run_{rid}.json"
    else:
        out = resolve_path(out_path, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


def _load_pipeline_config(
    config_path: str,
    limit_override: int | None,
    run_id_override: str | None,
) -> PipelineConfig:
    loaded = read_pipeline_config(config_path)
    if limit_override is not None:
        if limit_override < 1:
            raise ValueError("--limit must be a positive integer")
        loaded = replace(loaded, limit=limit_override)
    if run_id_override is not None:
        normalized_rid = normalize_run_id(run_id_override)
        if normalized_rid is None:
            raise ValueError("--run-id must be a non-empty string (letters, digits, _, -).")
        loaded = replace(loaded, run_id=normalized_rid)
    if loaded.run_id is None:
        raise ValueError(
            'run_id is required for pipeline stages: pass --run-id <id> or set "run_id" in the config JSON.'
        )
    return loaded.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="TVQA+ MKG-RAG pipeline runner")
    parser.add_argument(
        "stage",
        choices=[
            "build_evidence_cache",
            "extract_triplets",
            "predict_val",
            "report_val",
            "write_run_config",
        ],
    )
    parser.add_argument("--config", default="configs/val_default.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max QA examples (first N rows in the split). Overrides config 'limit'. Omit for full split.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Required unless JSON sets run_id: 1-64 chars (letters, digits, _, -). "
        "Outputs go under artifacts/<run_id>/. Overrides config run_id when set.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="For write_run_config only: output JSON path (default: configs/run_<run_id>.json).",
    )
    args = parser.parse_args()

    if args.stage == "write_run_config":
        if not args.run_id:
            parser.error("write_run_config requires --run-id")
        written = write_run_config(args.config, args.run_id, args.out)
        print(written)
        return

    try:
        config = _load_pipeline_config(args.config, args.limit, args.run_id)
    except ValueError as exc:
        parser.error(str(exc))

    stage_map = {
        "build_evidence_cache": build_evidence_cache,
        "extract_triplets": extract_triplets,
        "predict_val": predict_val,
        "report_val": report_val,
    }
    stage_map[args.stage](config)


if __name__ == "__main__":
    main()
