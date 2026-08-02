from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import SelectionConfig
from .schemas import GraphPath, SelectionRecord


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _redundancy(path: GraphPath, selected: list[GraphPath], vectors: dict[str, torch.Tensor], semantic_weight: float) -> float:
    if not selected:
        return 0.0
    values = []
    for other in selected:
        semantic = (float(F.cosine_similarity(vectors[path.path_id].unsqueeze(0), vectors[other.path_id].unsqueeze(0)).item()) + 1.0) / 2.0
        structural = _jaccard(set(path.edge_ids), set(other.edge_ids))
        values.append(semantic_weight * semantic + (1.0 - semantic_weight) * structural)
    return max(values)


def select_paths(paths: list[GraphPath], path_vectors: dict[str, torch.Tensor], anchors: list[str], config: SelectionConfig) -> tuple[list[GraphPath], list[SelectionRecord]]:
    selected: list[GraphPath] = []
    records: list[SelectionRecord] = []
    used_text: set[str] = set()
    used_visual: set[str] = set()
    covered: set[str] = set()
    remaining = list(paths)
    while remaining and len(selected) < config.max_selected_paths:
        best = None
        for path in remaining:
            new_anchors = set(path.matched_anchors) - covered
            coverage = len(new_anchors) / max(1, len(anchors))
            redundancy = _redundancy(path, selected, path_vectors, config.semantic_redundancy_weight)
            marginal_text = set(path.text_token_ids) - used_text
            marginal_visual = set(path.visual_token_ids) - used_visual
            cost = len(marginal_text) + 32 * len(marginal_visual)
            gain = path.utility + config.grounding_weight * path.grounding + config.coverage_weight * coverage - config.redundancy_weight * redundancy
            normalized = gain / max(1, cost)
            candidate = (normalized, gain, cost, coverage, redundancy, path)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _, gain, cost, coverage, redundancy, path = best
        current_cost = len(used_text) + 32 * len(used_visual)
        if gain < config.minimum_gain or current_cost + cost > config.evidence_budget:
            break
        selected.append(path)
        covered.update(path.matched_anchors)
        used_text.update(path.text_token_ids)
        used_visual.update(path.visual_token_ids)
        records.append(SelectionRecord(qid="", path_id=path.path_id, utility=path.utility, grounding=path.grounding, coverage_gain=coverage, redundancy=redundancy, marginal_cost=cost, marginal_gain=gain, rank=len(selected)))
        remaining = [item for item in remaining if item.path_id != path.path_id]
    return selected, records
