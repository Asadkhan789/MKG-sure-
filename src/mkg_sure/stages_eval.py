from __future__ import annotations
import math, time
import torch
from .io_utils import iter_jsonl, read_json, write_json, write_jsonl
from .metrics import accuracy, interval_iou, ndcg_at_k
from .packing import pack_paths
from .schemas import GraphPath, Prediction
from .stages_data import load_rows, localized_context, make_reader
from .sufficiency import sufficiency_features
from .training import load_sufficiency_head


def run_evaluation(config, split='val'):
    examples = {item.qid: item for item in load_rows(config, split, 'examples')}
    sources = {item.source_id: item for item in load_rows(config, split, 'sources') if config.data.allow_oracle_visual_input or not item.oracle_annotation}
    edge_map = {item.edge_id: item for item in load_rows(config, split, 'edges')}
    reader = make_reader(config)
    head = load_sufficiency_head(config.stage_dir(11, 'stage2') / 'stage2.pt')
    calibration = read_json(config.stage_dir(12, 'calibration') / 'calibration.json')
    predictions, forced, temporal, hits, recalls, ndcgs = [], [], [], [], [], []
    for row in iter_jsonl(config.stage_dir(9, 'selection') / f'{split}_selection.jsonl'):
        started = time.perf_counter()
        example = examples[row['qid']]
        selected = [GraphPath(**path) for path in row['selected_paths']]
        candidates = [GraphPath(**path) for path in row['candidate_paths']]
        features = sufficiency_features(selected, row['anchors'], edge_map, config.sufficiency.max_connecting_distance)
        with torch.no_grad():
            logit = float(head(torch.tensor([features])).item())
        probability = 1.0 / (1.0 + math.exp(-logit / calibration['temperature']))
        decision = 'answer' if probability >= calibration['high_threshold'] else 'recover' if probability >= calibration['low_threshold'] else 'insufficient'
        recovery = decision == 'recover'
        localized = localized_context(row['localized'], sources)
        before, _ = reader.predict(example.question, example.options, localized + '\n' + pack_paths(selected, sources).text)
        if recovery:
            selected_ids = {path.path_id for path in selected}
            selected += [path for path in candidates if path.path_id not in selected_ids][:config.sufficiency.recovery_paths]
            decision = 'answer' if probability >= calibration['high_threshold'] else 'insufficient'
        package = pack_paths(selected, sources)
        forced_idx, scores = reader.predict(example.question, example.options, localized + '\n' + package.text, package.media_paths)
        predicted_idx = forced_idx if decision == 'answer' else None
        forced.append(int(example.answer_idx == forced_idx))
        windows = row['localized']['windows']
        span = (min(window[0] for window in windows), max(window[1] for window in windows)) if windows else None
        temporal.append(interval_iou(span, example.gold_temporal_span))
        gold_paths = {path.path_id for path in candidates if path.matched_anchors}
        relevance = [int(path.path_id in gold_paths) for path in selected]
        hits.append(int(any(relevance)))
        recalls.append(sum(relevance) / max(1, len(gold_paths)))
        ndcgs.append(ndcg_at_k(relevance, len(selected)))
        predictions.append(Prediction(example.qid, predicted_idx, example.answer_idx, scores, [path.path_id for path in selected], package.source_ids, probability, probability, decision, recovery, before, (time.perf_counter() - started) * 1000, len(candidates), len(selected), package.text_tokens, package.visual_tokens, 2 if recovery else 1).to_dict())
    answered = [row for row in predictions if row['predicted_idx'] is not None]
    result = {
        'split': split,
        'examples': len(predictions),
        'selective_accuracy': accuracy(predictions),
        'forced_accuracy': sum(forced) / max(1, len(forced)),
        'coverage': len(answered) / max(1, len(predictions)),
        'temporal_miou': sum(temporal) / max(1, len(temporal)),
        'path_hit_at_selected': sum(hits) / max(1, len(hits)),
        'path_recall_at_selected': sum(recalls) / max(1, len(recalls)),
        'path_ndcg_at_selected': sum(ndcgs) / max(1, len(ndcgs)),
        'mean_candidate_paths': sum(row['candidate_path_count'] for row in predictions) / max(1, len(predictions)),
        'mean_selected_paths': sum(row['selected_path_count'] for row in predictions) / max(1, len(predictions)),
        'mean_latency_ms': sum(row['latency_ms'] for row in predictions) / max(1, len(predictions)),
    }
    stage = config.stage_dir(13, 'evaluation')
    write_jsonl(stage / f'{split}_predictions.jsonl', predictions)
    write_json(stage / f'{split}_metrics.json', result)
    return result
