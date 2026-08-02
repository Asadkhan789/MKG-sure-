from pathlib import Path

from mkg_sure.pipeline import smoke_test


def test_dummy_smoke(tmp_path: Path) -> None:
    report = smoke_test(tmp_path)
    metrics = report["evaluation"]
    assert metrics["examples"] > 0
    assert 0.0 <= metrics["forced_accuracy"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["mean_candidate_paths"] > 0
    assert (Path(report["smoke_artifacts"]) / "13_evaluation" / "val_metrics.json").exists()
