from __future__ import annotations
from typing import Any
from .config import MKGSureConfig
from .embeddings import make_embedder
from .facts import HeuristicFactExtractor, Qwen3VLFactExtractor
from .graph import build_graph
from .io_utils import iter_jsonl, write_jsonl
from .localization import temporal_localize
from .paths import generate_paths, question_anchors
from .reader import LexicalFrozenReader, Qwen3VLFrozenReader
from .regions import GroundingDINORunner, dummy_regions, extract_raw_video_regions
from .schemas import Fact, GraphEdge, GraphNode, GraphPath, QAExample, SourceUnit
from .sources import build_sources, load_tvqa_examples, make_dummy_dataset, write_source_stage
from .utility import generate_utility_labels


def load_rows(config: MKGSureConfig, split: str, kind: str):
    table = {
        'examples': (1, 'sources', f'{split}_qa.jsonl', QAExample),
        'sources': (2, 'regions', f'{split}_sources.jsonl', SourceUnit),
        'facts': (3, 'facts', f'{split}_facts.jsonl', Fact),
        'nodes': (5, 'graphs', f'{split}_nodes.jsonl', GraphNode),
        'edges': (5, 'graphs', f'{split}_edges.jsonl', GraphEdge),
    }
    number, name, filename, cls = table[kind]
    return [cls(**row) for row in iter_jsonl(config.stage_dir(number, name) / filename)]


def make_reader(config: MKGSureConfig):
    if config.models.backend == 'dummy':
        return LexicalFrozenReader()
    return Qwen3VLFrozenReader(config.models.reader_name, config.models.device, config.models.dtype, config.models.flash_attention_2)


def make_model_embedder(config: MKGSureConfig):
    name = 'hash' if config.models.backend == 'dummy' else config.models.embedding_name
    return make_embedder(name, config.models.embedding_dim, config.models.device)


def localized_context(localized: dict[str, Any], sources: dict[str, SourceUnit]) -> str:
    rows = [sources[source_id] for source_id in localized.get('source_ids', []) if source_id in sources]
    rows.sort(key=lambda item: (item.start_time is None, item.start_time or 0.0, item.source_id))
    return '\n'.join(f'[{item.source_type}] {item.text}' for item in rows)


def prepare_sources(config: MKGSureConfig, split: str, limit: int | None = None):
    if config.data.input_mode == 'dummy':
        examples, all_sources = make_dummy_dataset(config.runtime.dummy_examples)
        examples = [item for item in examples if item.split == split]
        video_ids = {item.video_id for item in examples}
        sources = [item for item in all_sources if item.video_id in video_ids]
    else:
        examples = load_tvqa_examples(config, split, limit or config.runtime.max_examples)
        sources = build_sources(config, examples)
    write_source_stage(config, examples, sources, split)
    return {'examples': len(examples), 'sources': len(sources), 'videos': len({item.video_id for item in examples})}


def run_regions(config: MKGSureConfig, split: str):
    path = config.stage_dir(1, 'sources') / f'{split}_sources.jsonl'
    sources = [SourceUnit(**row) for row in iter_jsonl(path)]
    merged = dummy_regions(sources)
    clips = [item for item in sources if item.source_type == 'clip' and item.media_path]
    if config.models.backend != 'dummy' and clips:
        labels: dict[str, list[str]] = {}
        for source in sources:
            if source.source_type == 'subtitle':
                labels.setdefault(source.video_id, []).extend(token for token in source.text.split() if token[:1].isupper())
        detector = GroundingDINORunner(config.models.detector_name, config.models.device)
        merged += extract_raw_video_regions(clips, {key: list(dict.fromkeys(value))[:20] for key, value in labels.items()}, detector, config.stage_dir(2, 'regions') / 'frames', config.runtime.frames_per_window)
    write_jsonl(config.stage_dir(2, 'regions') / f'{split}_sources.jsonl', (item.to_dict() for item in merged))
    return {'sources': len(merged), 'predicted_regions': sum(item.source_type == 'region' and not item.oracle_annotation for item in merged)}


def run_facts(config: MKGSureConfig, split: str):
    sources = [item for item in load_rows(config, split, 'sources') if config.data.allow_oracle_visual_input or not item.oracle_annotation]
    extractor = HeuristicFactExtractor() if config.models.backend == 'dummy' else Qwen3VLFactExtractor(config.models.extractor_name, config.models.device, config.models.dtype)
    facts = [fact for source in sources for fact in extractor.extract(source)]
    write_jsonl(config.stage_dir(3, 'facts') / f'{split}_facts.jsonl', (fact.to_dict() for fact in facts))
    return {'facts': len(facts)}


def run_embeddings(config: MKGSureConfig, split: str):
    rows = [('question', item.qid, item.question) for item in load_rows(config, split, 'examples')]
    rows += [('source', item.source_id, item.text) for item in load_rows(config, split, 'sources') if config.data.allow_oracle_visual_input or not item.oracle_annotation]
    rows += [('fact', item.fact_id, f'{item.head} {item.relation} {item.tail}') for item in load_rows(config, split, 'facts')]
    model = make_model_embedder(config)
    vectors = model.encode_texts([item[2] for item in rows])
    write_jsonl(config.stage_dir(4, 'embeddings') / f'{split}_embeddings.jsonl', ({'kind': kind, 'id': item_id, 'text': text, 'vector': vector.tolist()} for (kind, item_id, text), vector in zip(rows, vectors)))
    return {'vectors': len(rows), 'dimension': model.dim}


def run_graphs(config: MKGSureConfig, split: str):
    nodes, edges = build_graph(load_rows(config, split, 'facts'), config.graph)
    stage = config.stage_dir(5, 'graphs')
    write_jsonl(stage / f'{split}_nodes.jsonl', (item.to_dict() for item in nodes))
    write_jsonl(stage / f'{split}_edges.jsonl', (item.to_dict() for item in edges))
    return {'nodes': len(nodes), 'edges': len(edges)}


def run_candidates(config: MKGSureConfig, split: str):
    examples = load_rows(config, split, 'examples')
    sources = [item for item in load_rows(config, split, 'sources') if config.data.allow_oracle_visual_input or not item.oracle_annotation]
    nodes, edges = load_rows(config, split, 'nodes'), load_rows(config, split, 'edges')
    source_map = {item.source_id: item for item in sources}
    model = make_model_embedder(config)
    output = []
    for example in examples:
        localized = temporal_localize(example.question, [item for item in sources if item.video_id == example.video_id], model, config.localization)
        paths = generate_paths(example.question, [item for item in nodes if item.video_id == example.video_id], [item for item in edges if item.video_id == example.video_id], localized, source_map, model, config.graph, config.selection)
        output.append({'qid': example.qid, 'localized': {'source_ids': localized.source_ids, 'windows': localized.windows, 'scores': localized.scores}, 'anchors': question_anchors(example.question), 'paths': [path.to_dict() for path in paths]})
    write_jsonl(config.stage_dir(6, 'candidates') / f'{split}_candidates.jsonl', output)
    return {'questions': len(output), 'candidate_paths': sum(len(row['paths']) for row in output)}


def run_utility_labels(config: MKGSureConfig, split: str):
    examples = {item.qid: item for item in load_rows(config, split, 'examples')}
    sources = {item.source_id: item for item in load_rows(config, split, 'sources') if config.data.allow_oracle_visual_input or not item.oracle_annotation}
    reader, model = make_reader(config), make_model_embedder(config)
    labels = []
    for row in iter_jsonl(config.stage_dir(6, 'candidates') / f'{split}_candidates.jsonl'):
        example = examples[row['qid']]
        if example.answer_idx is not None:
            labels += generate_utility_labels(example, [GraphPath(**path) for path in row['paths']], localized_context(row['localized'], sources), sources, reader, model, config.graph.utility_label_candidates, config.training.utility_margin)
    write_jsonl(config.stage_dir(7, 'utility_labels') / f'{split}_utility_labels.jsonl', (label.to_dict() for label in labels))
    return {'labels': len(labels), 'helpful': sum(label.helpful for label in labels)}
