"""Summarise recorder episode summaries without changing trajectories or labels.

Example:
  python3 -m scripts.analyze_trajectory_collection --output-dir OUT \
      --group B0=/path/to/b0 --group B1=/path/to/b1
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


FIELDS = ('group', 'episodes', 'arrived', 'success_rate', 'terminal_counts_json',
          'mean_physical_lane_changes', 'exit_reachable_fraction',
          'mean_hard_braking_steps', 'mean_sumo_emergency_stopping_steps',
          'min_leader_ttc_s', 'min_follower_ttc_s', 'mean_efficiency_raw',
          'mean_efficiency_weighted_sum', 'mean_route_raw', 'mean_route_weighted_sum')


def number(row, key):
    value = row.get(key, '')
    if value == '':
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean(values):
    return sum(values) / len(values) if values else None


def summarize_group(name, directory):
    with (Path(directory) / 'episode_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'{name}: episode_summary.csv has no episode rows')
    outcomes = Counter(row['terminal_reason'] for row in rows)
    reachable_known = sum(int(number(row, 'exit_reachable_known_steps') or 0) for row in rows)
    reachable_true = sum(int(number(row, 'exit_reachable_true_steps') or 0) for row in rows)

    def values(key):
        return [value for row in rows if (value := number(row, key)) is not None]

    leader, follower = values('leader_ttc_min_s'), values('follower_ttc_min_s')
    return {
        'group': name, 'episodes': len(rows), 'arrived': outcomes['Arrived'],
        'success_rate': outcomes['Arrived'] / len(rows),
        'terminal_counts_json': json.dumps(dict(sorted(outcomes.items())), sort_keys=True),
        'mean_physical_lane_changes': mean(values('physical_lane_changes')),
        'exit_reachable_fraction': reachable_true / reachable_known if reachable_known else None,
        'mean_hard_braking_steps': mean(values('hard_braking_steps')),
        'mean_sumo_emergency_stopping_steps': mean(values('sumo_emergency_stopping_steps')),
        'min_leader_ttc_s': min(leader) if leader else None,
        'min_follower_ttc_s': min(follower) if follower else None,
        'mean_efficiency_raw': mean(values('efficiency_raw_mean')),
        'mean_efficiency_weighted_sum': mean(values('efficiency_weighted_sum')),
        'mean_route_raw': mean(values('route_raw_mean')),
        'mean_route_weighted_sum': mean(values('route_weighted_sum')),
    }


def parse_group(value):
    name, separator, directory = value.partition('=')
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError('--group must be NAME=DIRECTORY')
    return name, Path(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=parse_group, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for group_comparison.csv and .json')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f'output directory already exists: {args.output_dir}')
    names = [name for name, _ in args.group]
    if len(names) != len(set(names)):
        parser.error('group names must be unique')
    rows = [summarize_group(name, directory) for name, directory in args.group]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'group_comparison.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / 'group_comparison.json').open('x') as handle:
        json.dump({'groups': rows, 'notes': [
            'Each row is one episode; evaluation episodes are not independent training seeds.',
            'No preference labels or learned reward-model outputs are produced.',
            'Reachability is the fraction of diagnostically known step states marked reachable.',
        ]}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
