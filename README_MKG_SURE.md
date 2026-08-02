# MKG-Sure staged implementation

This package implements the MKG-Sure paper as separate offline, training, calibration, and evaluation stages so Qwen3-VL, the embedding model, and Grounding DINO do not need to remain in GPU memory simultaneously.

## Implemented

1. TVQA+ source manifests with raw-video, precomputed-feature, no-video, and dummy modes.
2. Predicted region generation through Grounding DINO or a visual-concept fallback.
3. Query-independent structured fact extraction through frozen Qwen3-VL or a deterministic fallback.
4. Frozen embedding cache.
5. Canonicalized provenance-aware graph with distinct-source noisy-or confidence.
6. Temporal localization, seed retrieval, and bounded one-/two-hop candidate paths.
7. Frozen-reader correct-answer-margin utility labels.
8. Stage-1 path encoder and utility-head training.
9. Utility, grounding, anchor coverage, redundancy, and marginal-cost selection.
10. Sufficiency examples with controlled path, relation, and source corruptions.
11. Stage-2 source projection, two-stream injection, and sufficiency-head training.
12. Temperature calibration with answer/recover/insufficient thresholds.
13. Selective QA, temporal grounding, path retrieval, recovery, and efficiency reporting.

## Install

```bash
pip install -e '.[test]'
```

For the real backbones:

```bash
pip install -e '.[real,test]'
```

## Dummy smoke test

```bash
mkg-sure smoke-test --work-dir artifacts/smoke_demo
pytest -q
```

The smoke test downloads no models or videos. It creates synthetic TVQA-style examples and executes all 13 stages, including both lightweight training stages.

## Real TVQA+ modes

No raw videos:

```json
{"data":{"input_mode":"precomputed","raw_video_dir":null,"feature_dir":"data/features"}}
```

Raw clips:

```json
{"data":{"input_mode":"raw_video","raw_video_dir":"data/tvqa_clips"}}
```

Gold TVQA+ boxes remain marked `oracle_annotation=true` and are excluded from main retrieval unless `allow_oracle_visual_input` is explicitly enabled.

## Run stages separately

```bash
mkg-sure run prepare_sources --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run regions --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run facts --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run embeddings --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run graphs --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run candidates --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run utility_labels --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run stage1 --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run selection --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run sufficiency_data --config configs/mkg_sure_tvqaplus_4b.json --split train
mkg-sure run stage2 --config configs/mkg_sure_tvqaplus_4b.json --split train
```

Repeat data, candidate, selection, and sufficiency stages for validation, then run calibration and evaluation. A complete sequential run is also available:

```bash
mkg-sure run-all --config configs/mkg_sure_tvqaplus_4b.json
```

## Validation boundary

The dummy path is fully executed and tested. The optional Qwen3-VL and Grounding-DINO adapters are implemented but were not executed in the assistant's CPU-only environment. The smoke Stage-2 path uses a frozen differentiable toy answer head to validate gradients through the trainable projector and injection modules without downloading the 4B checkpoint.
