"""Offline paper figures. Never starts SUMO; input labels identify algorithm and training seed."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.evaluation_results import validate_results
from src.experiment_tracking import atomic_json

# Figures use long-form source data, with each sampled global_step preserved.
PANELS = {
    '01_success': ['success_rate'],
    '02_failures': ['collision_rate', 'missed_exit_rate'],
    '03_critic': ['critic_loss', 'mean_q', 'mean_target'],
    '04_actor': ['actor_loss', 'alpha'],
    '05_comfort': ['mean_speed_mps', 'mean_abs_jerk_mps3', 'hard_braking_episode_rate'],
    '06_lane_changes': ['mean_lane_change_requests', 'mean_physical_lane_changes'],
    '07_e77_lanes': ['first_e77_lane_percent'],
}


def metric_denominator(metrics, metric):
    stats = metrics.get(metric + '_stats', {})
    if 'denominator' in stats:
        return stats['denominator']
    explicit = metrics.get(metric + '_valid_episodes')
    if explicit is not None:
        return explicit
    if metric.startswith('mean_'):
        explicit = metrics.get(metric.removeprefix('mean_') + '_valid_episodes')
        if explicit is not None:
            return explicit
    sample_key = {'hard_braking_steps': 'hard_braking_samples',
                  'sumo_emergency_stopping_steps': 'event_samples',
                  'physical_lane_changes': 'physical_lane_change_valid_pairs'}.get(metric)
    return metrics.get(sample_key, metrics['episodes']) if sample_key else metrics['episodes']


def load(experiments):
    data, final, missing = [], [], []
    lane_entries = []
    def add(algorithm, seed, step, source, entity, metric, value, n=None, ci=None):
        if value is not None:
            data.append(dict(algorithm=algorithm, training_seed=int(seed), global_step=int(step),
                source=source, entity=entity, metric=metric, value=float(value), valid_n=n,
                ci95_low=ci[0] if ci else None, ci95_high=ci[1] if ci else None))
    for algorithm, seed, directory in experiments:
        directory = Path(directory)
        logfile = directory / 'train/metrics.jsonl'
        if logfile.exists():
            for line in logfile.read_text().splitlines():
                row = json.loads(line)
                step = row['global_step']
                if row['event'] == 'update':
                    for metric in ('critic_loss', 'mean_q', 'mean_target'):
                        add(algorithm, seed, step, 'training_update', 'joint', metric, row.get(metric))
                    for entity in ('ego_o3', 'ego_e74'):
                        for metric in ('actor_loss', 'alpha'):
                            add(algorithm, seed, step, 'training_update', entity, metric, row.get(entity + '_' + metric))
                if row['event'] == 'episode':
                    for entity in ('ego_o3', 'ego_e74', 'joint'):
                        for metric in ('success_rate', 'collision_rate', 'missed_exit_rate'):
                            add(algorithm, seed, step, 'exploration_training_rolling100', entity, metric,
                                row.get(entity + '_' + metric), row['rolling_episodes'])
        paths = sorted((directory / 'eval').glob('step_*/summary.json'))
        index = directory / 'eval/index.json'
        if index.exists():
            manifest = json.loads(index.read_text())
            allowed_steps = {row['global_step'] for row in manifest if row['status'] in ('complete', 'skipped_complete')}
            paths = [path for path in paths if int(path.parent.name.removeprefix('step_')) in allowed_steps]
            for row in manifest:
                if row['status'] not in ('complete', 'skipped_complete'):
                    missing.append(dict(algorithm=algorithm, training_seed=seed, **row))
        checkpoints = directory / 'checkpoints/index.json'
        if checkpoints.exists():
            evaluated = {int(p.parent.name.removeprefix('step_')) for p in paths}
            for row in json.loads(checkpoints.read_text()):
                if row['global_step'] not in evaluated:
                    missing.append(dict(algorithm=algorithm, training_seed=seed,
                                        global_step=row['global_step'], status='missing_evaluation'))
        reports = []
        for path in paths:
            report = validate_results(path)
            if (report['episodes'] != 100 or [r['seed'] for r in report['records']] != list(range(200001, 200101))):
                raise ValueError(f'Paper evaluation requires exactly the 100 fixed seeds: {path}')
            if report.get('stochastic') is not False:
                raise ValueError(f'Paper evaluation requires deterministic results: {path}')
            if report.get('training_seed') is not None and int(report['training_seed']) != int(seed):
                raise ValueError('Training seed label mismatch')
            reports.append(report)
            step = report['training_step']
            for entity, metrics in {**report['per_agent'], 'joint': report['joint']}.items():
                for metric, value in metrics.items():
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        stats = metrics.get(metric + '_stats', {})
                        valid_n = metric_denominator(metrics, metric)
                        add(algorithm, seed, step, 'checkpoint_evaluation', entity, metric, value, valid_n, stats.get('ci95'))
                if 'first_e77_lane_distribution' in metrics:
                    lane_entries.append((algorithm, seed, step, entity, metrics))
        if reports:
            last = max(reports, key=lambda r: r['training_step'])
            for entity, metrics in {**last['per_agent'], 'joint': last['joint']}.items():
                for metric, value in metrics.items():
                    if (isinstance(value, (float, int)) and not isinstance(value, bool)) or (value is None and metric.startswith('mean_')):
                        stats = metrics.get(metric + '_stats', {})
                        n = metric_denominator(metrics, metric)
                        final.append(dict(algorithm=algorithm, training_seed=int(seed), global_step=last['training_step'],
                                          entity=entity, metric=metric, value=value, valid_n=n))
    from src.evaluation_metrics import wilson
    lanes = {lane for _, _, _, _, metrics in lane_entries for lane in metrics['first_e77_lane_distribution']}
    for algorithm, seed, step, entity, metrics in lane_entries:
        n = metrics['first_e77_lane_valid_episodes']
        for lane in sorted(lanes):
            stats = metrics['first_e77_lane_distribution'].get(lane, wilson(0, n))
            add(algorithm, seed, step, 'checkpoint_evaluation', entity + '/lane_' + lane,
                'first_e77_lane_percent', stats['value'], n, stats['ci95'])
    return pd.DataFrame(data), pd.DataFrame(final), missing


def plot(experiments, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    identities = [(label, int(seed)) for label, seed, _ in experiments]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate algorithm/training-seed input")
    data, final, missing = load(experiments)
    atomic_json(output / 'missing_checkpoints.json', missing)
    if data.empty:
        raise ValueError('No plot data')
    data.to_csv(output / 'plot_source.csv', index=False)
    final.to_csv(output / 'final_metrics_per_seed.csv', index=False)
    if not final.empty:
        # Do not compare different final training budgets in the same table row.
        table = final.groupby(['algorithm', 'global_step', 'entity', 'metric']).agg(
            mean=('value', 'mean'), std=('value', 'std'), training_seeds=('value', 'count'),
            input_training_seeds=('training_seed', 'nunique'),
            valid_samples_sum=('valid_n', 'sum')).reset_index()
        table['mean_plus_minus_std'] = [f'{r["mean"]:.6g} ± {r["std"]:.6g}' if r['training_seeds'] > 1
            else (f'{r["mean"]:.6g}' if r['training_seeds'] else 'NA') for _, r in table.iterrows()]
        table.to_csv(output / 'final_metrics.csv', index=False)
    for name, metrics in PANELS.items():
        selected = data[data.metric.isin(metrics)].copy()
        selected.to_csv(output / (name + '_source.csv'), index=False)
        fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.5 * len(metrics)), squeeze=False)
        for ax, metric in zip(axes[:, 0], metrics):
            subset = selected[selected.metric == metric]
            if subset.empty:
                ax.text(.5, .5, 'No valid data', ha='center', transform=ax.transAxes)
            else:
                for (algorithm, source, entity), group in subset.groupby(['algorithm', 'source', 'entity']):
                    expected_seeds = {int(seed) for label, seed, _ in experiments if label == algorithm}
                    # Union of observed/missing checkpoint steps: NaN breaks the line, never interpolate.
                    steps = set(group.global_step)
                    if source == 'checkpoint_evaluation':
                        steps.update(int(r['global_step']) for r in missing if r['algorithm'] == algorithm)
                    pivot = group.pivot(index='global_step', columns='training_seed', values='value').reindex(sorted(steps))
                    pivot = pivot.reindex(columns=sorted(expected_seeds))
                    valid = pivot.count(axis=1) == len(expected_seeds)
                    average = pivot.mean(axis=1).where(valid)
                    std = pivot.std(axis=1, ddof=1).where(valid)
                    label = f'{algorithm} {entity} {source}'
                    ax.plot(pivot.index, average, marker='o' if source == 'checkpoint_evaluation' else None,
                            linestyle='-' if source == 'checkpoint_evaluation' else '--', label=label)
                    if len(expected_seeds) > 1:
                        ax.fill_between(pivot.index, average - std, average + std, alpha=.15)
            ylabel = metric + (' (%)' if metric.endswith('_rate') or metric == 'first_e77_lane_percent' else '')
            ax.set(xlabel='Training global_step (shared SUMO control steps)', ylabel=ylabel)
            if metric.endswith('_rate') or metric == 'first_e77_lane_percent':
                ax.set_ylim(0, 100)
            if not subset.empty: ax.legend(fontsize=6)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / (name + '.png'), dpi=180)
        fig.savefig(output / (name + '.pdf'))
        plt.close(fig)
    return data, final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', nargs=3, action='append', required=True,
                        metavar=('ALGORITHM', 'TRAINING_SEED', 'DIRECTORY'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(args.experiment, args.output)
