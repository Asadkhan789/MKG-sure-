# MKG-Sure staged implementation

This branch adds MKG-Sure without changing the accepted MKG-Attention pipeline. Expensive frozen models are executed in separate offline stages; lightweight modules are trained later from cached outputs.

## Supported input modes

### Raw-video mode

Set `data.raw_video_dir` to the TVQA clip directory. The source manifest records the matching clip path so later frozen Qwen3-VL and Grounding DINO stages can decode frames or temporal windows.

### No-video / precomputed mode

When raw clips are absent, the same manifest is constructed from:

- TVQA+ subtitles;
- TVQA+ frame/box annotations;
- existing visual-concept features;
- optional cached `.pt`, `.npy`, `.npz`, or `.safetensors` files.

`input_mode: "auto"` selects raw-video mode only when clips are actually present. Otherwise it uses the precomputed path. Box annotations are marked with `oracle_annotation: true`; they must not be treated as predicted test-time regions in the final main experiment.

## RTX 4090 configuration

The provided config uses frozen Qwen3-VL-4B for the reader/extractor, Qwen3-VL-Embedding-2B for retrieval embeddings, and Grounding DINO-B for offline region proposals. Trainable modules are restricted to the path encoder, utility head, source projector, cross-attention/gate, and sufficiency head.

## First commands

```bash
PYTHONPATH=src python -m tvqa_mkg_rag.mkg_sure.pipeline write_stage_plan \
  --config configs/mkg_sure_tvqaplus_4b.json

PYTHONPATH=src python -m tvqa_mkg_rag.mkg_sure.pipeline prepare_sources \
  --config configs/mkg_sure_tvqaplus_4b.json \
  --split val --limit 25

PYTHONPATH=src python -m tvqa_mkg_rag.mkg_sure.pipeline inspect_sources \
  --config configs/mkg_sure_tvqaplus_4b.json \
  --split val
```

Outputs are written under `artifacts/mkg_sure/<run_id>/`.

## Stage contract

1. `01_sources`: query-independent source manifest.
2. `02_regions`: frozen Grounding DINO predictions or imported region features.
3. `03_facts`: frozen Qwen3-VL structured fact extraction.
4. `04_embeddings`: frozen multimodal embedding cache.
5. `05_graphs`: provenance-aware video graphs.
6. `06_candidates`: temporal localization and one/two-hop paths.
7. `07_utility_labels`: train-only frozen-reader margin labels.
8. `08_stage1`: path encoder, anchor projection, utility head.
9. `09_selection`: budget-aware complementary path selection.
10. `10_sufficiency_data`: support-preserving and corrupted evidence sets.
11. `11_stage2`: source projection, two-stream injection, sufficiency head.
12. `12_calibration`: temperature and decision thresholds on validation.
13. `13_evaluation`: QA, grounding, path retrieval, calibration, robustness, recovery, and efficiency.

Every stage should be resumable and should consume saved outputs from earlier stages rather than keeping all frozen models in GPU memory.

## Implemented in this first slice

- dual raw/precomputed input manifest;
- staged config and artifact layout;
- temporal localization interface supporting raw or cached visual similarity;
- seed-edge scoring with semantic, confidence, and provenance signals;
- bounded one/two-hop path generation;
- utility/grounding/coverage/redundancy budgeted selection;
- short-path encoder and utility loss;
- source projector and two-stream injection module;
- four-feature sufficiency head.

## Remaining implementation work

- frozen Qwen3-VL and embedding adapters;
- Grounding DINO runner;
- structured fact JSON extraction and token likelihoods;
- video-level canonicalization and noisy-or graph builder;
- reader option log-likelihood label generator;
- Stage-1 and Stage-2 training loops;
- controlled sufficiency corruption and calibration;
- official TVQA+ grounding and full paper metrics.

The no-video mode is suitable for development and ablations. The final paper's main multimodal experiment should use raw clips or equivalent cached clip/frame embeddings and predicted regions.
