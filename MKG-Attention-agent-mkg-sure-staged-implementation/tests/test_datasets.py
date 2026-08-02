from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tvqa_mkg_rag.datasets.loaders import load_annotations, load_subtitles


class DatasetLoaderTests(unittest.TestCase):
    def test_load_annotations_supports_optional_fields(self) -> None:
        rows = [
            {
                "qid": 1,
                "q": "Why did Raj move away?",
                "a0": "Because he was tired.",
                "a1": "Because he was scared.",
                "a2": "Because he was hungry.",
                "a3": "Because he was excited.",
                "a4": "Because he was bored.",
                "answer_idx": "3",
                "ts": [0.0, 5.4],
                "vid_name": "clip_01",
                "bbox": {"1": [{"img_id": 1, "label": "Raj"}]},
            },
            {
                "qid": 2,
                "q": "What is Sheldon doing?",
                "a0": "Cleaning.",
                "a1": "Sleeping.",
                "a2": "Cooking.",
                "a3": "Reading.",
                "a4": "Running.",
                "vid_name": "clip_02",
                "bbox": {"2": []},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            examples = load_annotations(path, split="val")

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].answer_idx, 3)
        self.assertEqual(examples[0].ts, (0.0, 5.4))
        self.assertEqual(examples[1].answer_idx, None)
        self.assertEqual(examples[1].ts, None)
        self.assertEqual(examples[0].bbox[1][0].label, "Raj")

    def test_load_annotations_respects_limit(self) -> None:
        rows = [
            {"qid": 1, "q": "Q1", "a0": "", "a1": "", "a2": "", "a3": "", "a4": "", "vid_name": "c1", "bbox": {}},
            {"qid": 2, "q": "Q2", "a0": "", "a1": "", "a2": "", "a3": "", "a4": "", "vid_name": "c2", "bbox": {}},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            examples = load_annotations(path, split="val", limit=1)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].qid, 1)

    def test_load_subtitles_aligns_turns_with_times(self) -> None:
        payload = {
            "clip_01": {
                "sub_text": "Raj: Hi there <eos> Penny: Hello <eos> Sheldon: Bye",
                "sub_time": [0.0, 1.5, 3.0],
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "subtitles.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            subtitles = load_subtitles(path)

        turns = subtitles["clip_01"]
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0].end_time, 1.5)
        self.assertEqual(turns[2].end_time, None)
        self.assertEqual(turns[1].text, "Penny: Hello")


if __name__ == "__main__":
    unittest.main()
