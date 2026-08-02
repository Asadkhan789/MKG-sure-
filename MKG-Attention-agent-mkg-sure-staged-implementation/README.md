# TVQA+ MKG-RAG Validation Pipeline

This repository contains a validation-first, RAG-style pipeline for TVQA+ that approximates the MKG-Attention paper without training a model.

The pipeline does four main things:

1. Build localized evidence chunks from subtitles and visual annotations.
2. Extract question-conditioned triplets with an LLM.
3. Build a per-question evidence graph, retrieve the best evidence, and run a lightweight audit/repair step.
4. Predict the multiple-choice answer on the validation split and write a report.

## What You Need

- Python 3.10+ recommended
- The existing data files already present under `data/`
- An API key in `.env` or your shell for:
  - `AIGCBEST_API_KEY`

The implementation uses the Python standard library except for an optional **`tqdm`** dependency: install it (`pip install tqdm`) to show a progress bar when running `extract_triplets`.

## Default Inputs

The default config is:

- `configs/val_default.json`

Optional **`limit`** (in the JSON config and/or `--limit` on the CLI) caps how many QA rows you load from the start of the split (**first N examples**). When `limit` is set, cache and report filenames include a `_n{N}` segment in the stem (for example `val_default_val_subtitles_bbox_n50_evidence.jsonl`).

**`run_id` is required** for every pipeline stage (`build_evidence_cache`, `extract_triplets`, `predict_val`, `report_val`). It must be 1–64 characters: letters, digits, underscore, hyphen. All outputs for that run live under **`artifacts/<run_id>/`** (`cache/` for JSONL, `reports/` for the report). Use a new `run_id` when you want a fresh artifact tree; reuse the same `run_id` to resume append-only stages in that folder.

**Ways to set `run_id`:**

1. **CLI on every stage** (overrides the value in the JSON config): `--run-id exp1`
2. **Write a run config** (copies your base JSON, sets `run_id`, and writes the canonical `artifacts/<run_id>/…` paths into the JSON for reference):

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main write_run_config \
  --config configs/val_default.json \
  --run-id exp1
```

That prints the path to the new file (default `configs/run_exp1.json`). Use `--out path/to/my_run.json` to choose the path. Then run stages 1–4 with `--config` pointing at that file so every stage shares the same `run_id` and `limit` without repeating `--run-id`.

By default it uses:

- Validation annotations: `data/tvqa_plus_annotations_with_test/tvqa_plus_valid_preprocessed.json`
- Subtitles: `data/tvqa_plus_subtitles.json`
- Visual concepts: `data/det_visual_concepts_hq.pickle`
- Evidence mode: `subtitles + bbox`
- Visual concepts: disabled by default

## How To Run

Because the code lives under `src/`, run commands with `PYTHONPATH=src`.

### 1. Build evidence cache

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main build_evidence_cache \
  --config configs/val_default.json --run-id myrun
```

### 2. Extract triplets with the LLM

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main extract_triplets \
  --config configs/val_default.json --run-id myrun
```

Process only the first 50 validation questions (CLI overrides `"limit"` in the config file):

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main extract_triplets \
  --config configs/val_default.json --run-id myrun --limit 50
```

If you used a **sample `limit`** for triplets, use the **same `limit`** (same CLI flag or the same `"limit"` in the JSON) for **`predict_val`** and **`report_val`**. Otherwise the run points at the full-split filenames (`…_subtitles_bbox_…`) instead of the sample ones (`…_subtitles_bbox_n50_…`). Keep the same **`run_id`** and the same filename-driving settings (`run_name`, `split`, `limit`, and evidence mode) across the sequence.

### 3. Predict answers on validation

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main predict_val \
  --config configs/val_default.json --run-id myrun
```

After a `--limit 50` triplet run, predict on those same 50 questions:

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main predict_val \
  --config configs/val_default.json --run-id myrun --limit 50
```

### 4. Generate the validation report

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main report_val \
  --config configs/val_default.json --run-id myrun
```

Matching sample report:

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.pipelines.main report_val \
  --config configs/val_default.json --run-id myrun --limit 50
```

## Outputs

Artifacts for a run are written under **`artifacts/<run_id>/`**.

- JSONL caches: `artifacts/<run_id>/cache/`
- Reports: `artifacts/<run_id>/reports/`

Filename stems are `{run_name}_{split}_{evidence_mode}` (and `_n{N}` when `limit` is set), for example with `run_name` `val_default` and `--run-id myrun`:

- `artifacts/myrun/cache/val_default_val_subtitles_bbox_evidence.jsonl`
- `artifacts/myrun/cache/val_default_val_subtitles_bbox_triplets.jsonl`
- `artifacts/myrun/cache/val_default_val_subtitles_bbox_graphs.jsonl`
- `artifacts/myrun/cache/val_default_val_subtitles_bbox_predictions.jsonl`
- `artifacts/myrun/reports/val_default_val_subtitles_bbox_report.json`
- `artifacts/myrun/reports/val_default_val_subtitles_bbox_report.md`
- `artifacts/myrun/reports/val_default_val_subtitles_bbox_predicted_outputs.json`

The `predicted_outputs.json` report is a readable JSON view of the predictions with the question text,
predicted option index, predicted answer text, confidence, and correctness fields. It is refreshed by both
`predict_val` and `report_val`.

With `"limit": 50` or `--limit 50`, the stem includes `_n50` (for example `…_val_subtitles_bbox_n50_predictions.jsonl` and `…_n50_report.md`).

For Python callers, `read_pipeline_config(path)` loads JSON without resolving paths; merge CLI overrides with `dataclasses.replace`, then call `.resolve()` so `run_id` is set before paths are finalized.

## Resume Behavior

The pipeline is append-only and resumable when you rerun with the same `run_id` and the same effective `run_name`, `split`, `limit`, and evidence settings. LLM retries are controlled by `llm.max_retries` and `llm.sleep_seconds` in the JSON config.

- Re-running `build_evidence_cache` skips already-written `doc_id`s.
- Re-running `extract_triplets` skips already-written `source_hash` values.
- Re-running `predict_val` skips already-predicted `qid`s.

For LLM-backed stages, each successful chunk/question is written immediately. If an LLM request still fails after all retries, the stage stops without writing a failure marker; rerun the same command to retry the first unfinished unit.

## Changing The Setup

Edit `configs/val_default.json` to change:

- `limit`: run a smaller slice first
- `evidence.include_bbox`
- `evidence.include_visual_concepts`
- retrieval, audit, and answering budgets

If you want a quick small run first, set:

```json
"limit": 25
```

## Important Notes

- This is validation-first. `predict_val` only supports `split = "val"`.
- The code can read the preprocessed test file later, but there is no local test accuracy because the checked-in test file has no `answer_idx`.
- The graph is question-conditioned per `qid`; this is not a global KB build.
- Visual concepts are off by default because the local pickle contains raw tag lists, not natural-language captions.

## Main Code Locations

- Pipeline entrypoints: `src/tvqa_mkg_rag/pipelines/main.py`
- Config loading: `src/tvqa_mkg_rag/config.py`
- Data loading: `src/tvqa_mkg_rag/datasets/loaders.py`
- Evidence building: `src/tvqa_mkg_rag/evidence/builder.py`
- Triplet extraction: `src/tvqa_mkg_rag/triplets/extractor.py`
- Retrieval and audit: `src/tvqa_mkg_rag/retrieval/` and `src/tvqa_mkg_rag/audit/`
- Answering and reporting: `src/tvqa_mkg_rag/answering/` and `src/tvqa_mkg_rag/evaluation/`
