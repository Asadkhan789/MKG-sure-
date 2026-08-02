from __future__ import annotations
import torch
from .io_utils import iter_jsonl, write_json, write_jsonl
from .packing import pack_paths
from .reader import correct_margin
from .schemas import GraphPath, SufficiencyExample, UtilityLabel
from .selection import select_paths
from .stages_data import load_rows, localized_context, make_model_embedder, make_reader
from .sufficiency import make_sufficiency_examples
from .training import load_stage1, load_sufficiency_head, train_stage1, train_stage2
from .metrics import auroc, brier, ece, macro_f1


def run_stage1(config, split='train'):
    labels = [UtilityLabel(**row) for row in iter_jsonl(config.stage_dir(7, 'utility_labels') / f'{split}_utility_labels.jsonl')]
    metrics = train_stage1(labels, config, config.stage_dir(8, 'stage1') / 'stage1.pt')[1]
    write_json(config.stage_dir(8, 'stage1') / 'metrics.json', metrics)
    return metrics


def run_selection(config, split):
    examples = {item.qid: item for item in load_rows(config, split, 'examples')}
    model, embedder = load_stage1(config.stage_dir(8, 'stage1') / 'stage1.pt'), make_model_embedder(config)
    output = []
    for row in iter_jsonl(config.stage_dir(6, 'candidates') / f'{split}_candidates.jsonl'):
        paths = [GraphPath(**path) for path in row['paths']]
        chosen, records = [], []
        if paths:
            q = embedder.encode_texts([examples[row['qid']].question])
            path_vectors = embedder.encode_texts([path.serialized_text for path in paths])
            with torch.no_grad():
                logits, _ = model(q.expand(len(paths), -1), path_vectors)
            for path, utility in zip(paths, torch.sigmoid(logits).tolist()):
                path.utility = float(utility)
            chosen, records = select_paths(paths, {path.path_id: vector for path, vector in zip(paths, path_vectors)}, row['anchors'], config.selection)
        for record in records:
            record.qid = row['qid']
        output.append({'qid': row['qid'], 'localized': row['localized'], 'anchors': row['anchors'], 'candidate_paths': [path.to_dict() for path in paths], 'selected_paths': [path.to_dict() for path in chosen], 'selection_records': [record.to_dict() for record in records]})
    write_jsonl(config.stage_dir(9, 'selection') / f'{split}_selection.jsonl', output)
    return {'questions': len(output), 'selected_paths': sum(len(row['selected_paths']) for row in output)}


def run_sufficiency_data(config, split):
    examples = {item.qid: item for item in load_rows(config, split, 'examples')}
    sources = {item.source_id: item for item in load_rows(config, split, 'sources') if config.data.allow_oracle_visual_input or not item.oracle_annotation}
    edge_map = {item.edge_id: item for item in load_rows(config, split, 'edges')}
    reader = make_reader(config)
    output = []
    for row in iter_jsonl(config.stage_dir(9, 'selection') / f'{split}_selection.jsonl'):
        example = examples[row['qid']]
        paths = [GraphPath(**path) for path in row['selected_paths']]
        package = pack_paths(paths, sources)
        positive = False
        if example.answer_idx is not None:
            scores = reader.score_options(example.question, example.options, localized_context(row['localized'], sources) + '\n' + package.text, package.media_paths)
            coverage = len({anchor for path in paths for anchor in path.matched_anchors}) / max(1, len(row['anchors']))
            positive = coverage >= config.sufficiency.support_threshold and correct_margin(scores, example.answer_idx) > 0
        output += make_sufficiency_examples(row['qid'], paths, row['anchors'], edge_map, config.sufficiency.max_connecting_distance, positive)
    write_jsonl(config.stage_dir(10, 'sufficiency_data') / f'{split}_sufficiency.jsonl', (item.to_dict() for item in output))
    return {'examples': len(output), 'positive': sum(item.label for item in output)}


def run_stage2(config, split='train'):
    examples = [SufficiencyExample(**row) for row in iter_jsonl(config.stage_dir(10, 'sufficiency_data') / f'{split}_sufficiency.jsonl')]
    metrics = train_stage2(examples, config, config.stage_dir(11, 'stage2') / 'stage2.pt')
    write_json(config.stage_dir(11, 'stage2') / 'metrics.json', metrics)
    return metrics


def run_calibration(config, split='val'):
    rows = [SufficiencyExample(**row) for row in iter_jsonl(config.stage_dir(10, 'sufficiency_data') / f'{split}_sufficiency.jsonl')]
    head = load_sufficiency_head(config.stage_dir(11, 'stage2') / 'stage2.pt')
    features = torch.tensor([row.features for row in rows])
    labels = [row.label for row in rows]
    with torch.no_grad():
        logits = head(features)
    best = (1.0, float('inf'))
    for temperature in [0.5 + 0.1 * index for index in range(26)]:
        score = brier(labels, torch.sigmoid(logits / temperature).tolist())
        if score < best[1]:
            best = (temperature, score)
    probabilities = torch.sigmoid(logits / best[0]).tolist()
    positives = [probability for label, probability in zip(labels, probabilities) if label]
    negatives = [probability for label, probability in zip(labels, probabilities) if not label]
    low, high = sum(negatives) / max(1, len(negatives)), sum(positives) / max(1, len(positives))
    if high <= low:
        low, high = min(low, 0.45), max(high, 0.55)
    result = {'temperature': best[0], 'low_threshold': low, 'high_threshold': high, 'brier': best[1], 'auroc': auroc(labels, probabilities), 'macro_f1': macro_f1(labels, [int(probability >= 0.5) for probability in probabilities]), 'ece': ece(labels, probabilities)}
    write_json(config.stage_dir(12, 'calibration') / 'calibration.json', result)
    return result
