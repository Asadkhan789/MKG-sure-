from __future__ import annotations

from .embeddings import Embedder
from .packing import pack_paths
from .reader import FrozenReader, correct_margin
from .schemas import GraphPath, QAExample, SourceUnit, UtilityLabel


def generate_utility_labels(example: QAExample, candidates: list[GraphPath], localized_context: str, sources: dict[str, SourceUnit], reader: FrozenReader, embedder: Embedder, top_h: int, helpful_margin: float) -> list[UtilityLabel]:
    if example.answer_idx is None:
        return []
    baseline_scores = reader.score_options(example.question, example.options, localized_context)
    baseline_margin = correct_margin(baseline_scores, example.answer_idx)
    question_vector = embedder.encode_texts([example.question])[0].tolist()
    labels: list[UtilityLabel] = []
    for path in candidates[:top_h]:
        package = pack_paths([path], sources)
        context = localized_context + "\n" + package.text
        scores = reader.score_options(example.question, example.options, context, package.media_paths)
        path_margin = correct_margin(scores, example.answer_idx)
        utility = path_margin - baseline_margin
        labels.append(UtilityLabel(qid=example.qid, path_id=path.path_id, baseline_margin=baseline_margin, path_margin=path_margin, utility=utility, helpful=utility > helpful_margin, question_vector=question_vector, path_vector=embedder.encode_texts([path.serialized_text])[0].tolist()))
    return labels
