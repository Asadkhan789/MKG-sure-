from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tvqa_mkg_rag.config import AnswerConfig, PipelineConfig, load_config, normalize_run_id, read_pipeline_config


class LoadConfigTests(unittest.TestCase):
    def test_stage_prefix_includes_limit_suffix(self) -> None:
        cfg = PipelineConfig(project_root="/tmp", run_name="r", run_id="x", split="val", limit=50)
        self.assertEqual(cfg.stage_prefix, "r_val_subtitles_bbox_n50")

    def test_stage_prefix_omits_suffix_when_limit_none(self) -> None:
        cfg = PipelineConfig(project_root="/tmp", run_name="r", run_id="x", split="val", limit=None)
        self.assertEqual(cfg.stage_prefix, "r_val_subtitles_bbox")
        self.assertNotIn("_n", cfg.stage_prefix)

    def test_stage_prefix_omits_run_id(self) -> None:
        """run_id only affects artifact directories, not filenames."""
        cfg = PipelineConfig(
            project_root="/tmp",
            run_name="val_default",
            run_id="exp1",
            split="val",
            limit=50,
        )
        self.assertEqual(cfg.stage_prefix, "val_default_val_subtitles_bbox_n50")

    def test_normalize_run_id_strips_and_rejects_invalid(self) -> None:
        self.assertIsNone(normalize_run_id(None))
        self.assertIsNone(normalize_run_id("   "))
        self.assertEqual(normalize_run_id("  ab-12  "), "ab-12")
        with self.assertRaises(ValueError):
            normalize_run_id("has space")
        with self.assertRaises(ValueError):
            normalize_run_id("../evil")

    def test_load_config_accepts_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cfg.json"
            path.write_text(json.dumps({"run_id": "trial_a"}), encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.run_id, "trial_a")
            self.assertIn("trial_a", str(cfg.artifacts.cache_dir))

    def test_load_config_requires_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cfg.json"
            path.write_text(json.dumps({"run_name": "x"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_read_pipeline_config_allows_null_run_id_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cfg.json"
            path.write_text(json.dumps({"run_name": "x"}), encoding="utf-8")
            cfg = read_pipeline_config(path)
            self.assertIsNone(cfg.run_id)

    def test_load_config_rejects_invalid_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps({"run_id": "bad id"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_load_config_rejects_non_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps({"limit": 0, "run_id": "r"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_load_config_rejects_bool_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps({"limit": True, "run_id": "r"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_answer_config_uses_legacy_graph_budget_alias(self) -> None:
        config = AnswerConfig(max_evidence_units=7)
        self.assertEqual(config.max_graph_evidence_units, 7)


if __name__ == "__main__":
    unittest.main()
