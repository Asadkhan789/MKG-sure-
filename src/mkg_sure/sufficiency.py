from __future__ import annotations

from collections import defaultdict, deque

import torch

from .schemas import GraphEdge, GraphPath, SufficiencyExample


def _connectivity(paths: list[GraphPath], edges: dict[str, GraphEdge], max_distance: int) -> float:
    nodes = {node_id for path in paths for node_id in path.node_ids}
    if not nodes:
        return 0.0
    if len(nodes) == 1:
        return 1.0
    adjacency: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        for edge_id in path.edge_ids:
            edge = edges[edge_id]
            adjacency[edge.head_id].add(edge.tail_id)
            adjacency[edge.tail_id].add(edge.head_id)
    pairs, connected = 0, 0
    ordered = sorted(nodes)
    for i, start in enumerate(ordered):
        for target in ordered[i + 1:]:
            pairs += 1
            queue = deque([(start, 0)])
            seen = {start}
            found = False
            while queue:
                node, distance = queue.popleft()
                if node == target:
                    found = True
                    break
                if distance >= max_distance:
                    continue
                for neighbor in adjacency[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, distance + 1))
            connected += int(found)
    return connected / max(1, pairs)


def sufficiency_features(paths: list[GraphPath], anchors: list[str], edge_map: dict[str, GraphEdge], max_distance: int) -> list[float]:
    if not paths:
        return [0.0, 0.0, 0.0, 0.0]
    covered = {anchor for path in paths for anchor in path.matched_anchors}
    coverage = len(covered) / max(1, len(anchors))
    connectivity = _connectivity(paths, edge_map, max_distance)
    utility_sum = sum(max(0.0, path.utility) for path in paths)
    grounding = sum(max(0.0, path.utility) * path.grounding for path in paths) / max(1e-6, utility_sum) if utility_sum > 0 else sum(path.grounding for path in paths) / len(paths)
    utility = sum(max(0.0, min(1.0, path.utility)) for path in paths) / len(paths)
    return [coverage, connectivity, grounding, utility]


def make_sufficiency_examples(qid: str, paths: list[GraphPath], anchors: list[str], edge_map: dict[str, GraphEdge], max_distance: int, positive: bool) -> list[SufficiencyExample]:
    rows = [SufficiencyExample(qid=qid, features=sufficiency_features(paths, anchors, edge_map, max_distance), label=int(positive), corruption="none")]
    if paths:
        rows.append(SufficiencyExample(qid=qid, features=sufficiency_features(paths[1:], anchors, edge_map, max_distance), label=0, corruption="remove_critical_path"))
        corrupted = [path for path in paths if path.hop_count == 1] or paths[:-1]
        rows.append(SufficiencyExample(qid=qid, features=sufficiency_features(corrupted, anchors, edge_map, max_distance), label=0, corruption="delete_relation"))
        masked = [GraphPath(**{**path.to_dict(), "grounding": path.grounding * 0.2}) for path in paths]
        rows.append(SufficiencyExample(qid=qid, features=sufficiency_features(masked, anchors, edge_map, max_distance), label=0, corruption="mask_source"))
    return rows


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid(logits / max(1e-6, temperature))
