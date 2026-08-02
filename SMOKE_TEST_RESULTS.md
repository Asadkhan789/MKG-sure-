# Dummy smoke test results

Executed locally in a CPU-only environment:

```bash
PYTHONPATH=src python -m mkg_sure.pipeline smoke-test --work-dir /tmp/mkg_smoke_modular
PYTHONPATH=src pytest -q
```

Result:

```text
pytest: 1 passed
train questions: 8
validation questions: 4
train candidate paths: 128
validation candidate paths: 64
utility labels: 64 train / 32 validation
helpful utility labels: 40 train / 21 validation
Stage-1 helpfulness accuracy: 0.734375
Stage-2 sufficiency accuracy: 0.8125
forced validation accuracy: 0.75
selective validation accuracy: 0.666667
coverage: 0.75
temporal mIoU: 0.75
path Hit@selected: 1.0
path Recall@selected: 0.25
path NDCG@selected: 1.0
mean candidate paths: 16.0
mean selected paths: 4.0
```

These are synthetic execution checks, not TVQA+ research results.
