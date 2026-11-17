"""Frozen protocol: deterministic Easy, 100 shared evaluation seeds per checkpoint."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_tracking import sha256, atomic_json, git_metadata, evaluation_fingerprint, runtime_versions
from src.evaluation_results import validate_results, EVALUATION_PROTOCOL_VERSION
from src.evaluation_metrics import REFERENCE_COMMIT


def discover(directory):
    import torch
    paths = list((Path(directory) / 'checkpoints').glob('*.pt'))
    if not paths:
        raise ValueError('No checkpoints found')
    rows = []
    for path in paths:
        state = torch.load(path, map_location='cpu', weights_only=True)
        rows.append(dict(path=path, global_step=int(state['global_step']), sha256=sha256(path)))
    rows.sort(key=lambda row: row['global_step'])
    if len({r['global_step'] for r in rows}) != len(rows):
        raise ValueError('Duplicate checkpoint steps: use one experiment directory per training seed')
    index_path = Path(directory) / 'checkpoints/index.json'
    if index_path.exists():
        import json
        index = json.loads(index_path.read_text())
        if {r['global_step'] for r in index} != {r['global_step'] for r in rows}:
            raise ValueError('Checkpoint index exposes missing/unindexed checkpoint')
        for row in rows:
            expected = next(r for r in index if r['global_step'] == row['global_step'])
            if expected['sha256'] != row['sha256']:
                raise ValueError('Checkpoint SHA256 mismatch with index')
    return rows


def expected_environment(max_steps):
    root = Path(__file__).resolve().parents[1]
    from sumolib import checkBinary
    return dict(sumo_binary_sha256=sha256(checkBinary('sumo')), scenario='easy', agent_ids=['ego_o3', 'ego_e74'], requested_speed_mode=96,
                step_length_s=0.1, max_steps=max_steps, stopped_speed_threshold_mps=0.5,
                net_sha256=sha256(root / 'sumo_new/Antwerp.net.xml'),
                route_sha256=sha256(root / 'sumo_new/Antwerp_easy_o3_d3_dual.rou.xml'))


def compatible(report, checkpoint, max_steps):
    return (report['episodes'] == 100 and report['seed_start'] == 200001
            and [r['seed'] for r in report['records']] == list(range(200001, 200101))
            and report['stochastic'] is False
            and report['checkpoint_sha256'] == checkpoint['sha256']
            and report['training_step'] == checkpoint['global_step']
            and report['reference_commit'] == REFERENCE_COMMIT
            and report['metric_version'] == 'dual-ego-v1'
            and report['evaluation_protocol_version'] == EVALUATION_PROTOCOL_VERSION
            and report['git']['commit'] == git_metadata()['commit']
            and report.get('evaluation_code_sha256') == evaluation_fingerprint()
            and report.get('runtime_versions') == runtime_versions()
            and all(report['environment'].get(k) == v for k, v in expected_environment(max_steps).items())
            and all(report['environment']['actual_speed_modes'][ego] for ego in ('ego_o3', 'ego_e74')))


def run(directory, rerun=False, max_steps=4500, evaluator=None):
    if evaluator is None:
        from evaluate_masac import evaluate
        evaluator = evaluate
    directory = Path(directory)
    checkpoints = discover(directory)
    manifest = [dict(global_step=r['global_step'], checkpoint=str(r['path']),
                     sha256=r['sha256'], status='pending') for r in checkpoints]
    manifest_path = directory / 'eval/index.json'
    atomic_json(manifest_path, manifest)
    for checkpoint, item in zip(checkpoints, manifest):
        path = directory / 'eval' / f"step_{checkpoint['global_step']}" / 'summary.json'
        try:
            if (path.exists() or path.with_name('episodes.csv').exists()) and not rerun:
                report = validate_results(path)
                if not compatible(report, checkpoint, max_steps):
                    raise ValueError('Existing evaluation protocol mismatch; explicitly use --rerun')
                item['status'] = 'skipped_complete'
            else:
                evaluator(SimpleNamespace(checkpoint=str(checkpoint['path']), episodes=100,
                    seed_start=200001, stochastic=False, trace=False,
                    max_steps=max_steps, output=str(path)))
                report = validate_results(path)
                if not compatible(report, checkpoint, max_steps):
                    raise ValueError('Evaluator produced incompatible results')
                item['status'] = 'complete'
            item['summary'] = str(path)
            atomic_json(manifest_path, manifest)
        except BaseException as exc:
            item.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            atomic_json(manifest_path, manifest)
            raise
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_dir')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--max-steps', type=int, default=4500)
    args = parser.parse_args()
    run(args.experiment_dir, args.rerun, args.max_steps)
