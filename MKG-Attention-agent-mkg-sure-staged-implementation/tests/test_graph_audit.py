from __future__ import annotations

import unittest

from tvqa_mkg_rag.audit.repair import audit_and_repair
from tvqa_mkg_rag.config import AuditConfig, RetrievalConfig
from tvqa_mkg_rag.graph.builder import build_question_graph
from tvqa_mkg_rag.retrieval.scorer import retrieve_question_edges
from tvqa_mkg_rag.types import EvidenceChunk, QAExample, Triplet, TripletRecord


class GraphAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = QAExample(
            qid=21,
            split="val",
            vid_name="clip_21",
            question="Why did Raj step back?",
            options=["He was tired.", "He was scared.", "He was hungry.", "He was excited.", "He was bored."],
            answer_idx=3,
            ts=(0.0, 5.0),
        )
        self.evidence_by_doc_id = {
            "doc_1": EvidenceChunk(qid=21, doc_id="doc_1", modality="subtitle", text="Raj hugged Penny.", vid_name="clip_21", start_time=1.0, end_time=1.5),
            "doc_2": EvidenceChunk(qid=21, doc_id="doc_2", modality="subtitle", text="Raj stepped back.", vid_name="clip_21", start_time=2.0, end_time=2.4),
            "doc_3": EvidenceChunk(qid=21, doc_id="doc_3", modality="bbox", text="Frame 1: Raj, Penny", vid_name="clip_21", frame_ids=[1]),
        }
        self.triplet_records = [
            TripletRecord(
                qid=21,
                doc_id="doc_1",
                triplet=Triplet("Raj", "hugged", "Penny"),
                relation_norm="hugged",
                raw_llm_output="(Raj, hugged, Penny)",
                source_hash="hash_1",
                parse_status="ok",
                provenance={"modality": "subtitle", "start_time": 1.0, "end_time": 1.5, "frame_ids": []},
            ),
            TripletRecord(
                qid=21,
                doc_id="doc_2",
                triplet=Triplet("Raj", "emotion", "excited"),
                relation_norm="emotion",
                raw_llm_output="(Raj, emotion, excited)",
                source_hash="hash_2",
                parse_status="ok",
                provenance={"modality": "subtitle", "start_time": 2.0, "end_time": 2.4, "frame_ids": []},
            ),
            TripletRecord(
                qid=21,
                doc_id="doc_3",
                triplet=Triplet("Raj", "emotion", "nervous"),
                relation_norm="emotion",
                raw_llm_output="(Raj, emotion, nervous)",
                source_hash="hash_3",
                parse_status="ok",
                provenance={"modality": "bbox", "start_time": None, "end_time": None, "frame_ids": [1]},
            ),
        ]

    def test_graph_builder_merges_support_and_audit_detects_conflicts(self) -> None:
        edges = build_question_graph(self.example.qid, self.triplet_records, self.evidence_by_doc_id)
        selected_edges, edge_scores = retrieve_question_edges(self.example, edges, RetrievalConfig())
        audit_result = audit_and_repair(self.example, selected_edges, edges, edge_scores, AuditConfig())

        self.assertEqual(len(edges), 3)
        self.assertTrue(any("Raj|emotion" == conflict for conflict in audit_result.diagnostics.conflicting_edges))
        self.assertGreaterEqual(audit_result.quality_score, 0.0)


if __name__ == "__main__":
    unittest.main()
