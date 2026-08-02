from __future__ import annotations

import json
import re

from ..types import EvidenceChunk, QAExample, Triplet, TripletRecord
from ..utils import normalize_whitespace, stable_hash


_TRIPLET_RE = re.compile(r"^\((?P<subject>.+?),\s*(?P<relation>.+?),\s*(?P<object>.+?)\)$")


def normalize_relation(value: str) -> str:
    value = normalize_whitespace(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def build_triplet_prompt_name(chunk: EvidenceChunk) -> str:
    return "triplet_text" if chunk.modality == "subtitle" else "triplet_visual"


def _triplet_from_mapping(value: object) -> Triplet | None:
    if not isinstance(value, dict):
        return None
    subject = normalize_whitespace(str(value.get("subject", value.get("head", ""))))
    relation = normalize_whitespace(str(value.get("relation", value.get("predicate", ""))))
    obj = normalize_whitespace(str(value.get("object", value.get("tail", ""))))
    if subject and relation and obj:
        return Triplet(subject=subject, relation=relation, object=obj)
    return None


def _extract_json_triplets(raw_response: str) -> list[Triplet]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_response):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items = parsed.get("triplets")
            if not isinstance(items, list):
                items = [parsed]
        elif isinstance(parsed, list):
            items = parsed
        else:
            continue
        triplets = [triplet for triplet in (_triplet_from_mapping(item) for item in items) if triplet is not None]
        if triplets:
            return triplets
    return []


def extract_triplets_from_response(raw_response: str) -> list[Triplet]:
    json_triplets = _extract_json_triplets(raw_response)
    if json_triplets:
        return json_triplets

    triplets: list[Triplet] = []
    for line in raw_response.splitlines():
        line = re.sub(r"^\d+\.\s*", "", line.strip())
        line = re.sub(r"^```(?:json)?\s*", "", line)
        line = re.sub(r"\s*```$", "", line)
        line = re.sub(r"^>\s*\*Thought.*?\*\s*", "", line)
        if line.startswith("(") and line.endswith(")") and line.strip().lower() == "(no_triplets)":
            continue
        if "|" in line and not line.startswith("("):
            parts = [normalize_whitespace(part) for part in line.split("|")]
            if len(parts) == 3:
                line = f"({parts[0]}, {parts[1]}, {parts[2]})"
        match = _TRIPLET_RE.match(line)
        if not match:
            continue
        subject = normalize_whitespace(match.group("subject"))
        relation = normalize_whitespace(match.group("relation"))
        obj = normalize_whitespace(match.group("object"))
        if subject and relation and obj:
            triplets.append(Triplet(subject=subject, relation=relation, object=obj))
    return triplets


def build_triplet_records(
    example: QAExample,
    chunk: EvidenceChunk,
    raw_response: str,
) -> list[TripletRecord]:
    source_hash = stable_hash(example.question, chunk.text, limit=32)
    triplets = extract_triplets_from_response(raw_response)
    if not triplets:
        return [
            TripletRecord(
                qid=example.qid,
                doc_id=chunk.doc_id,
                triplet=None,
                relation_norm=None,
                raw_llm_output=raw_response,
                source_hash=source_hash,
                parse_status="empty",
                provenance={
                    "vid_name": example.vid_name,
                    "modality": chunk.modality,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "frame_ids": chunk.frame_ids,
                },
            )
        ]

    records: list[TripletRecord] = []
    for triplet in triplets:
        records.append(
            TripletRecord(
                qid=example.qid,
                doc_id=chunk.doc_id,
                triplet=triplet,
                relation_norm=normalize_relation(triplet.relation),
                raw_llm_output=raw_response,
                source_hash=source_hash,
                parse_status="ok",
                provenance={
                    "vid_name": example.vid_name,
                    "modality": chunk.modality,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "frame_ids": chunk.frame_ids,
                },
            )
        )
    return records
