from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
from .config import MKGSureConfig, load_config
from .io_utils import set_seed, write_json
from .stages_data import prepare_sources, run_candidates, run_embeddings, run_facts, run_graphs, run_regions, run_utility_labels
from .stages_train import run_calibration, run_selection, run_stage1, run_stage2, run_sufficiency_data
from .stages_eval import run_evaluation

STAGES = [(1,'sources'),(2,'regions'),(3,'facts'),(4,'embeddings'),(5,'graphs'),(6,'candidates'),(7,'utility_labels'),(8,'stage1'),(9,'selection'),(10,'sufficiency_data'),(11,'stage2'),(12,'calibration'),(13,'evaluation')]


def run_all(config):
    set_seed(config.training.seed)
    report = {'train': {}, 'val': {}}
    common = [('sources', prepare_sources), ('regions', run_regions), ('facts', run_facts), ('embeddings', run_embeddings), ('graphs', run_graphs), ('candidates', run_candidates), ('utility_labels', run_utility_labels)]
    for split in ('train', 'val'):
        for name, function in common:
            report[split][name] = function(config, split)
        if split == 'train':
            report['stage1'] = run_stage1(config)
        report[split]['selection'] = run_selection(config, split)
        report[split]['sufficiency_data'] = run_sufficiency_data(config, split)
        if split == 'train':
            report['stage2'] = run_stage2(config)
    report['calibration'] = run_calibration(config)
    report['evaluation'] = run_evaluation(config)
    write_json(config.run_dir / 'run_report.json', report)
    return report


def smoke_test(work_dir=None):
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix='mkg_sure_smoke_'))
    config = MKGSureConfig()
    config.data.input_mode = 'dummy'; config.models.backend = 'dummy'; config.models.embedding_name = 'hash'; config.models.embedding_dim = 64
    config.runtime.run_id = 'smoke'; config.runtime.artifacts_dir = str(root / 'artifacts'); config.runtime.dummy_examples = 12
    config.training.stage1_epochs = 4; config.training.stage2_epochs = 4; config.training.hidden_dim = 64; config.selection.evidence_budget = 400
    report = run_all(config); report['smoke_artifacts'] = str(config.run_dir); return report


def write_stage_plan(config):
    return write_json(config.run_dir / 'stage_plan.json', {'run_id': config.runtime.run_id, 'stages': [{'number': number, 'name': name, 'directory': str(config.stage_dir(number, name))} for number, name in STAGES], 'frozen': [config.models.reader_name, config.models.extractor_name, config.models.embedding_name, config.models.detector_name], 'trained': ['short_path_encoder', 'utility_head', 'source_projector', 'two_stream_injection', 'sufficiency_head']})


def main():
    parser = argparse.ArgumentParser(description='MKG-Sure staged implementation')
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('smoke-test'); smoke.add_argument('--work-dir')
    run = sub.add_parser('run'); run.add_argument('stage'); run.add_argument('--config', required=True); run.add_argument('--split', default='val')
    all_parser = sub.add_parser('run-all'); all_parser.add_argument('--config', required=True)
    args = parser.parse_args()
    if args.command == 'smoke-test': result = smoke_test(args.work_dir)
    elif args.command == 'run-all': result = run_all(load_config(args.config))
    else:
        config = load_config(args.config)
        functions = {'prepare_sources':prepare_sources,'regions':run_regions,'facts':run_facts,'embeddings':run_embeddings,'graphs':run_graphs,'candidates':run_candidates,'utility_labels':run_utility_labels,'stage1':run_stage1,'selection':run_selection,'sufficiency_data':run_sufficiency_data,'stage2':run_stage2,'calibration':run_calibration,'evaluation':run_evaluation,'write_stage_plan':lambda c,s:{'path':str(write_stage_plan(c))}}
        result = functions[args.stage](config, args.split)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
