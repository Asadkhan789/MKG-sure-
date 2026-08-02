from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from tvqa_mkg_rag.config import (
    AnswerConfig,
    AuditConfig,
    DataPaths,
    EvidenceConfig,
    LLMConfig,
    PipelineConfig,
    PromptPaths,
    RetrievalConfig,
)
from tvqa_mkg_rag.pipelines.main import build_evidence_cache, extract_triplets, predict_val, report_val
from tvqa_mkg_rag.utils import iter_jsonl


class FakeLLMClient:
    def query(self, prompt: str, model_name: str | None = None) -> str:
        del model_name
        if "Triplets:" in prompt:
            return '{"triplets": [{"subject": "Raj", "relation": "hugged", "object": "Penny"}, {"subject": "Raj", "relation": "emotion", "object": "excited"}]}'
        if "Draft answer index" in prompt:
            return '{"answer_idx": 3, "option_scores": {"0": 0.1, "3": 0.9}, "confidence": 0.95}'
        return '{"answer_idx": 3, "supporting_subtitle_doc_ids": ["101:subtitle:0:39b93c5763d4"], "supporting_triplets": ["(Raj, hugged, Penny)"], "confidence": 0.92}'


class PipelineEndToEndTests(unittest.TestCase):
    def test_pipeline_creates_cache_prediction_and_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            prompts_dir = root / "prompts"
            data_dir.mkdir(parents=True)
            prompts_dir.mkdir(parents=True)

            annotations_path = data_dir / "annotations.json"
            subtitles_path = data_dir / "subtitles.json"
            visuals_path = data_dir / "visuals.pkl"

            annotations_path.write_text(
                json.dumps(
                    [
                        {
                            "qid": 101,
                            "q": "Why did Raj step back?",
                            "a0": "He was tired.",
                            "a1": "He was scared.",
                            "a2": "He was hungry.",
                            "a3": "He was excited.",
                            "a4": "He was bored.",
                            "answer_idx": "3",
                            "ts": [0.0, 5.0],
                            "vid_name": "clip_101",
                            "bbox": {"1": [{"img_id": 1, "label": "Raj"}, {"img_id": 1, "label": "Penny"}]},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            subtitles_path.write_text(
                json.dumps(
                    {
                        "clip_101": {
                            "sub_text": "Penny hugs Raj <eos> Raj becomes flustered <eos> Raj steps back",
                            "sub_time": [0.5, 1.5, 2.5],
                        }
                    }
                ),
                encoding="utf-8",
            )
            with visuals_path.open("wb") as handle:
                pickle.dump({"clip_101": ["raj, penny, couch"]}, handle, protocol=0)

            (prompts_dir / "triplet_text_prompt.txt").write_text("Question:\n{question}\nOptions:\n{options}\nEvidence:\n{evidence}\nTriplets:", encoding="utf-8")
            (prompts_dir / "triplet_visual_prompt.txt").write_text("Question:\n{question}\nOptions:\n{options}\nVisual:\n{evidence}\nTriplets:", encoding="utf-8")
            (prompts_dir / "answer_multiple_choice_prompt.txt").write_text(
                "Question:\n{question}\nOptions:\n{options}\nSubtitle:\n{subtitle_context}\nGraph:\n{graph_evidence}",
                encoding="utf-8",
            )
            (prompts_dir / "verify_answer_prompt.txt").write_text(
                "Draft answer index:\n{draft_answer}\nRisk:\n{risk_flags}\nSubtitle:\n{subtitle_context}\nGraph:\n{graph_evidence}",
                encoding="utf-8",
            )

            config = PipelineConfig(
                project_root=str(root),
                run_name="fixture_run",
                run_id="fixture",
                split="val",
                data=DataPaths(
                    annotations_val=str(annotations_path),
                    annotations_test=str(annotations_path),
                    subtitles=str(subtitles_path),
                    visual_concepts=str(visuals_path),
                ),
                llm=LLMConfig(env_file=str(root / ".env")),
                evidence=EvidenceConfig(include_subtitles=True, include_bbox=True, include_visual_concepts=False),
                retrieval=RetrievalConfig(),
                audit=AuditConfig(),
                answering=AnswerConfig(),
                prompts=PromptPaths(
                    triplet_text=str(prompts_dir / "triplet_text_prompt.txt"),
                    triplet_visual=str(prompts_dir / "triplet_visual_prompt.txt"),
                    answer_mc=str(prompts_dir / "answer_multiple_choice_prompt.txt"),
                    verify_answer=str(prompts_dir / "verify_answer_prompt.txt"),
                ),
            ).resolve()

            build_evidence_cache(config)
            extract_triplets(config, client=FakeLLMClient())
            prediction_path = predict_val(config, client=FakeLLMClient())
            predicted_outputs_path = config.predicted_outputs_json_path()
            report_path = report_val(config)

            predictions = list(iter_jsonl(prediction_path))
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["predicted_idx"], 3)
            self.assertTrue(predictions[0]["selected_subtitle_doc_ids"])
            self.assertIn("confidence", predictions[0])
            self.assertTrue(report_path.exists())
            self.assertTrue(predicted_outputs_path.exists())
            predicted_outputs = json.loads(predicted_outputs_path.read_text(encoding="utf-8"))
            self.assertEqual(predicted_outputs["total_predictions"], 1)
            self.assertEqual(predicted_outputs["predictions"][0]["predicted_answer"], "He was excited.")


if __name__ == "__main__":
    unittest.main()
