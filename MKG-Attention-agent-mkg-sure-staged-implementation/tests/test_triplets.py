from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tvqa_mkg_rag.config import PromptPaths
from tvqa_mkg_rag.llm.prompts import PromptRepository
from tvqa_mkg_rag.triplets.extractor import build_triplet_records, extract_triplets_from_response
from tvqa_mkg_rag.types import EvidenceChunk, QAExample


class TripletExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = QAExample(
            qid=11,
            split="val",
            vid_name="clip_01",
            question="What happened?",
            options=["A", "B", "C", "D", "E"],
        )
        self.chunk = EvidenceChunk(
            qid=11,
            doc_id="11:subtitle:0:abc123",
            modality="subtitle",
            text="Raj hugged Penny and stepped back.",
            vid_name="clip_01",
            start_time=0.0,
            end_time=1.0,
        )

    def test_triplet_parser_handles_parentheses_and_pipe_format(self) -> None:
        raw = "(Raj, hugged, Penny)\n2. Raj | stepped away from | Penny\nNoise"
        triplets = extract_triplets_from_response(raw)

        self.assertEqual(len(triplets), 2)
        self.assertEqual(triplets[1].relation, "stepped away from")

    def test_triplet_parser_handles_json_and_thought_wrappers(self) -> None:
        raw = """```json
{"triplets": [{"subject": "Leslie", "relation": "feels", "object": "relieved"}]}
```
> *Thought for a second* NO_TRIPLETS
"""
        triplets = extract_triplets_from_response(raw)

        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0].subject, "Leslie")
        self.assertEqual(triplets[0].object, "relieved")

    def test_triplet_records_capture_empty_and_non_empty_outputs(self) -> None:
        empty_records = build_triplet_records(self.example, self.chunk, "NO_TRIPLETS")
        parsed_records = build_triplet_records(self.example, self.chunk, "(Raj, hugged, Penny)")

        self.assertEqual(empty_records[0].parse_status, "empty")
        self.assertEqual(parsed_records[0].relation_norm, "hugged")
        self.assertEqual(empty_records[0].source_hash, parsed_records[0].source_hash)

    def test_prompt_repository_preserves_multiline_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "triplet_text.txt").write_text("Question:\n{question}\n\nOptions:\n{options}\n\nEvidence:\n{evidence}", encoding="utf-8")
            (root / "triplet_visual.txt").write_text("{evidence}", encoding="utf-8")
            (root / "answer.txt").write_text("{evidence}", encoding="utf-8")
            (root / "verify.txt").write_text("{draft_answer}", encoding="utf-8")
            prompts = PromptRepository(
                PromptPaths(
                    triplet_text=str(root / "triplet_text.txt"),
                    triplet_visual=str(root / "triplet_visual.txt"),
                    answer_mc=str(root / "answer.txt"),
                    verify_answer=str(root / "verify.txt"),
                )
            )
            rendered = prompts.render(
                "triplet_text",
                question="Why?",
                options="0. A\n1. B",
                evidence="Line one.\nLine two.",
            )

        self.assertIn("0. A\n1. B", rendered)
        self.assertIn("Line one.\nLine two.", rendered)


if __name__ == "__main__":
    unittest.main()
