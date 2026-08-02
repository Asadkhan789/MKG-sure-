from __future__ import annotations

import math


def accuracy(rows: list[dict]) -> float:
    labeled = [row for row in rows if row.get("gold_idx") is not None and row.get("predicted_idx") is not None]
    return sum(row["gold_idx"] == row["predicted_idx"] for row in labeled) / max(1, len(labeled))


def brier(labels: list[int], probabilities: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / max(1, len(labels))


def ece(labels: list[int], probabilities: list[float], bins: int = 10) -> float:
    total = max(1, len(labels)); score = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(y, p) for y, p in zip(labels, probabilities) if low <= p < high or (index == bins - 1 and p == 1.0)]
        if members:
            score += len(members) / total * abs(sum(y for y, _ in members) / len(members) - sum(p for _, p in members) / len(members))
    return score


def auroc(labels: list[int], probabilities: list[float]) -> float:
    positives = [p for y, p in zip(labels, probabilities) if y == 1]; negatives = [p for y, p in zip(labels, probabilities) if y == 0]
    if not positives or not negatives: return 0.5
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores = []
    for cls in (0, 1):
        tp = sum(y == cls and p == cls for y, p in zip(labels, predictions)); fp = sum(y != cls and p == cls for y, p in zip(labels, predictions)); fn = sum(y == cls and p != cls for y, p in zip(labels, predictions))
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn); scores.append(2 * precision * recall / max(1e-8, precision + recall))
    return sum(scores) / 2


def aurc(correct: list[int], confidences: list[float]) -> float:
    ordered = sorted(zip(confidences, correct), reverse=True)
    if not ordered: return 0.0
    area = 0.0; errors = 0
    for index, (_, is_correct) in enumerate(ordered, start=1): errors += 1 - is_correct; area += (errors / index) / len(ordered)
    return area


def interval_iou(predicted: tuple[float, float] | None, gold: tuple[float, float] | None) -> float:
    if predicted is None or gold is None: return 0.0
    intersection = max(0.0, min(predicted[1], gold[1]) - max(predicted[0], gold[0])); union = max(predicted[1], gold[1]) - min(predicted[0], gold[0])
    return intersection / union if union > 0 else 0.0


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1]); x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1); area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]); area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]); union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def ndcg_at_k(relevances: list[int], k: int) -> float:
    values = relevances[:k]; dcg = sum((2 ** rel - 1) / math.log2(index + 2) for index, rel in enumerate(values)); ideal = sorted(relevances, reverse=True)[:k]; idcg = sum((2 ** rel - 1) / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_iou(predictions: list[tuple[float, str, tuple[float, float, float, float]]], gold: list[tuple[str, tuple[float, float, float, float]]], threshold: float = 0.5) -> float | None:
    if not gold: return None
    matched = set(); ranked = sorted(predictions, key=lambda row: -row[0]); precision_sum = 0.0; true_positives = 0
    for rank, (_, label, box) in enumerate(ranked, start=1):
        best_index, best_iou = None, 0.0
        for index, (gold_label, gold_box) in enumerate(gold):
            if index in matched or label.lower() != gold_label.lower(): continue
            value = box_iou(box, gold_box)
            if value > best_iou: best_index, best_iou = index, value
        if best_index is not None and best_iou >= threshold:
            matched.add(best_index); true_positives += 1; precision_sum += true_positives / rank
    return precision_sum / max(1, len(gold))
