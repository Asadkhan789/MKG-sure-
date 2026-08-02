from __future__ import annotations

import unittest

from tvqa_mkg_rag.config import EvidenceConfig
from tvqa_mkg_rag.evidence.builder import (
    build_bbox_chunks,
    build_subtitle_chunks,
    build_visual_concept_chunks,
    normalize_visual_tags,
)
from tvqa_mkg_rag.types import BBoxRegion, QAExample, SubtitleTurn


class EvidenceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = QAExample(
            qid=7,
            split="val",
            vid_name="clip_01",
            question="Why did Raj move away?",
            options=[
                "He was tired.",
                "He was scared.",
                "He was hungry.",
                "He was excited.",
                "He was bored.",
            ],
            answer_idx=3,
            ts=(0.0, 5.4),
            bbox={
                1: [BBoxRegion(frame_id=1, label="Raj"), BBoxRegion(frame_id=1, label="Penny")],
                7: [BBoxRegion(frame_id=7, label="Raj")],
            },
        )
        self.turns = [
            SubtitleTurn(index=0, text="Raj says he cannot stay calm.", start_time=0.5, end_time=1.1),
            SubtitleTurn(index=1, text="Penny hugs Raj tightly.", start_time=2.0, end_time=2.4),
            SubtitleTurn(index=2, text="Raj panics and steps back.", start_time=4.0, end_time=4.5),
            SubtitleTurn(index=3, text="Howard changes the subject.", start_time=9.0, end_time=9.5),
        ]
        self.config = EvidenceConfig(
            include_subtitles=True,
            include_bbox=True,
            include_visual_concepts=True,
            subtitle_window_padding_seconds=2.0,
            max_subtitle_turns_per_chunk=2,
            max_visual_frames_per_chunk=2,
            max_subtitle_chunks_per_question=4,
        )

    def test_subtitle_chunking_respects_window_and_stable_doc_ids(self) -> None:
        first = build_subtitle_chunks(self.example, self.turns, self.config)
        second = build_subtitle_chunks(self.example, self.turns, self.config)

        self.assertEqual(len(first), 2)
        self.assertEqual(first[0].doc_id, second[0].doc_id)
        self.assertIn("Raj panics and steps back.", " ".join(chunk.text for chunk in first))

    def test_reasoning_questions_expand_subtitle_padding(self) -> None:
        turns = self.turns + [
            SubtitleTurn(index=4, text="Penny says she likes hanging out with Raj and his friends.", start_time=12.0, end_time=12.5)
        ]
        chunks = build_subtitle_chunks(self.example, turns, self.config)

        self.assertIn("likes hanging out", " ".join(chunk.text for chunk in chunks))

    def test_bbox_and_visual_chunks_use_grouped_frames(self) -> None:
        bbox_chunks = build_bbox_chunks(self.example, self.config)
        visual_chunks = build_visual_concept_chunks(
            self.example,
            [
                "raj, penny, couch, logo",
                "raj, smiling, penny",
                "sheldon, wall",
                "table, cup",
                "chair, lamp",
                "floor, lamp",
                "raj, doorway, penny",
            ],
            self.config,
        )

        self.assertEqual(len(bbox_chunks), 1)
        self.assertIn("Frame 1: Raj, Penny", bbox_chunks[0].text)
        self.assertEqual(visual_chunks[0].frame_ids, [1, 7])
        self.assertEqual(normalize_visual_tags("raj, logo, raj, penny"), ["raj", "penny"])


if __name__ == "__main__":
    unittest.main()
