"""Run B1-B3 serially and compare their results with the existing A1 baseline."""

import argparse
import csv
import json
from pathlib import Path
import statistics

from stable_baselines3 import SAC

from . import basic_reward_ablation as runner
from . import exit_e76_d3_config as cfg
from evaluate_sac_delay import validate_checkpoint_setup


BASELINE_SHA = 'b603f13aff5ab7dd65d1811f556d5bbf084fd4a4'
DEFAULT_BASELINE = Path('/home/cyl/project/DAROM-TPR-basic-reward-ablation/source/models_experiments/basic_reward_ablation')
runner.OUTPUT = runner.ROOT / 'models_experiments' / 'a1_component_ablation'
runner.PROFILES = tuple(cfg.COMPONENT_PROFILES)
runner.SOURCE_BRANCH_SHA = BASELINE_SHA


def verify_baseline(root):
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    records = {(entry['profile'], entry['seed']): entry for entry in manifest['runs']}
    for seed in runner.SEEDS:
        directory = root / f'A1_seed{seed}'
        path = directory / f'A1_seed{seed}.zip'
        report = json.loads((directory / 'evaluation.json').read_text())
        record = records[('A1', seed)]
        digest = runner.sha256(path)
        model = SAC.load(path, device='cpu')
        validate_checkpoint_setup(model, 'full', False, cfg.TASK_ID, 'A1')
        episodes = report['runs'][0]['episode_results']
        if (record['git_sha'] != BASELINE_SHA or model.seed != seed or
                record['checkpoint_sha256'] != digest or
                report['checkpoint_sha256'] != digest or
                report.get('total_episodes') != 50 or
                [item['seed'] for item in episodes] != runner.EVAL_SEEDS or
                report.get('scenario') != 'easy' or report.get('delay_mode') != 'uniform' or
                report.get('evaluation_safety') is not False or
                report.get('deterministic') is not True):
            raise ValueError(f'A1 baseline provenance or evaluation mismatch for seed {seed}')


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_comparisons(baseline):
    verify_baseline(baseline)
    for name in ('evaluation_summary.csv', 'lane_behavior_summary.csv', 'reward_components.csv'):
        baseline_rows = [row for row in read_rows(baseline / name) if row['profile'] == 'A1']
        if {row['seed'] for row in baseline_rows} != {'42', '43'} or len(baseline_rows) != 2:
            raise ValueError(f'Invalid A1 baseline rows in {name}')
        rows = baseline_rows + read_rows(runner.OUTPUT / name)
        runner.write_csv(runner.OUTPUT / f'comparison_{name}', rows)
    rows = read_rows(runner.OUTPUT / 'comparison_evaluation_summary.csv')
    groups = []
    for profile in ('A1', *runner.PROFILES):
        selected = [row for row in rows if row['profile'] == profile]
        if len(selected) != 2:
            raise ValueError(f'Expected two training seeds for {profile}')
        groups.append({'profile': profile, **{
            f'{key}_{stat}': (statistics.mean(values) if stat == 'mean' else statistics.stdev(values))
            for key in selected[0] if key not in ('profile', 'seed')
            for values in [[float(row[key]) for row in selected]]
            for stat in ('mean', 'std')}})
    runner.write_csv(runner.OUTPUT / 'comparison_group_summary.csv', groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'evaluate', 'summarize', 'all'), default='all')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--baseline-dir', type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    # Verify the historical comparison before consuming the training budget.
    verify_baseline(args.baseline_dir)
    if args.stage in ('train', 'all'):
        runner.train_all(args.device)
    if args.stage in ('evaluate', 'all'):
        runner.evaluate_all(args.device)
    if args.stage in ('summarize', 'all'):
        runner.summarize()
        write_comparisons(args.baseline_dir)


if __name__ == '__main__':
    main()
