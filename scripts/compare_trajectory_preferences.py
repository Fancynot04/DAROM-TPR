"""Read-only lexicographic trajectory comparisons for the P1-B audit.

Pairs trajectories sharing an evaluation seed only when their recorded initial
SUMO state is comparable.  This produces rule decisions, never labels or a
learned reward model.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

RULE_VERSION = 'p1b-lexicographic-v1'
SAFETY_STEP_MARGIN = 5
TIME_STEP_MARGIN = 50


def number(row, key):
    try:
        value = float(row.get(key, ''))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass
class Trajectory:
    group: str
    summary: dict
    rows: list

    @property
    def id(self): return self.summary['trajectory_id']
    @property
    def seed(self): return int(self.summary['eval_seed'])
    @property
    def success(self): return self.summary['terminal_reason'] == 'Arrived'

    def metric(self, key): return number(self.summary, key)


def physical_directions(rows):
    result = Counter()
    for row in rows:
        if row.get('lane_change_physical') != '1':
            continue
        before, after = number(row, 'ego_lane_index'), number(row, 'ego_lane_index_after')
        if before is None or after is None or before == after:
            continue
        result['toward' if after < before else 'away'] += 1
    return result


def route_features(rows):
    """Conservative route evidence: no inferred lane-change intent."""
    settled = [(i, number(row, 'min_lane_changes_required'),
                number(row, 'route_remaining_to_d3_start_m'))
               for i, row in enumerate(rows)
               if row.get('route_state_settled') == '1'
               and number(row, 'min_lane_changes_required') is not None]
    first_zero = next((distance for _, need, distance in settled if need == 0 and distance is not None), None)
    return {'directions': physical_directions(rows), 'first_settled_zero_distance_m': first_zero}


def safety_features(t):
    return {'collision': t.summary['terminal_reason'] == 'Collision',
            'danger_steps': sum(1 for row in t.rows
                                if (number(row, 'ttc_leader_s') is not None and number(row, 'ttc_leader_s') < 2)
                                or (number(row, 'ttc_follower_s') is not None and number(row, 'ttc_follower_s') < 2)),
            'hard_braking': int(t.metric('hard_braking_steps') or 0)}


def comfort_features(t):
    jerk = [abs(number(row, 'jerk_measured_mps3')) for row in t.rows
            if number(row, 'jerk_measured_mps3') is not None]
    stopped_clear_route = sum(1 for row in t.rows if number(row, 'ego_speed_mps') is not None
                              and number(row, 'ego_speed_mps') < .5
                              and number(row, 'min_lane_changes_required') == 0)
    return {'steps': len(t.rows), 'mean_abs_jerk': sum(jerk) / len(jerk) if jerk else None,
            'stopped_clear_route_steps': stopped_clear_route}


def comparable(a, b):
    if a.seed != b.seed or a.summary.get('scenario') != b.summary.get('scenario') or not a.rows or not b.rows:
        return False, 'different seed, scenario, or missing initial row'
    left, right = a.rows[0], b.rows[0]
    for key in ('ego_road_id', 'ego_lane_index', 'ego_lane_pos_m', 'ego_speed_mps',
                'route_remaining_to_d3_end_m'):
        x, y = left.get(key, ''), right.get(key, '')
        if key == 'ego_road_id':
            if x != y: return False, f'initial {key} differs'
        elif number(left, key) is None or number(right, key) is None or abs(number(left, key)-number(right, key)) > 1e-3:
            return False, f'initial {key} differs or is missing'
    return True, ''


def choose(a, b):
    ok, reason = comparable(a, b)
    if not ok: return False, 'unresolved', 'none', reason
    if a.success != b.success:
        return True, 'A' if a.success else 'B', 'P1', 'Arrived is preferred to non-Arrived terminal outcome'
    sa, sb = safety_features(a), safety_features(b)
    if sa['collision'] != sb['collision']:
        return True, 'A' if not sa['collision'] else 'B', 'P2', 'non-collision is preferred when P1 ties'
    if (sa['danger_steps'] + SAFETY_STEP_MARGIN <= sb['danger_steps'] and sa['hard_braking'] <= sb['hard_braking']) or (sb['danger_steps'] + SAFETY_STEP_MARGIN <= sa['danger_steps'] and sb['hard_braking'] <= sa['hard_braking']):
        winner = 'A' if sa['danger_steps'] < sb['danger_steps'] else 'B'
        return True, winner, 'P2', 'at least 5 fewer TTC<2s steps with no more hard-braking steps'
    ra, rb = route_features(a.rows), route_features(b.rows)
    if ra['directions']['away'] != rb['directions']['away']:
        return True, 'A' if ra['directions']['away'] < rb['directions']['away'] else 'B', 'P3', 'fewer physical lane changes away from the exit side'
    if (ra['first_settled_zero_distance_m'] is None) != (rb['first_settled_zero_distance_m'] is None):
        return True, 'A' if ra['first_settled_zero_distance_m'] is not None else 'B', 'P3', 'reached settled zero required lane changes while the other trajectory did not'
    ca, cb = comfort_features(a), comfort_features(b)
    if abs(ca['steps'] - cb['steps']) >= TIME_STEP_MARGIN:
        return True, 'A' if ca['steps'] < cb['steps'] else 'B', 'P4', 'at least 5 seconds shorter recorded travel time'
    return True, 'similar', 'none', 'no reliable lexicographic difference under P1-P4'


def load_group(group, directory):
    directory = Path(directory)
    with (directory / 'episode_summary.csv').open() as handle: summary = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    with (directory / 'trajectory_steps.csv').open() as handle:
        for row in csv.DictReader(handle): grouped[row['trajectory_id']].append(row)
    return [Trajectory(group, row, grouped[row['trajectory_id']]) for row in summary]


def parse_group(value):
    name, sep, directory = value.partition('=')
    if not sep or not name or not directory: raise argparse.ArgumentTypeError('NAME=DIRECTORY required')
    return name, Path(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', type=parse_group, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): parser.error(f'output directory exists: {args.output_dir}')
    trajectories = [t for group, directory in args.group for t in load_group(group, directory)]
    by_seed = defaultdict(list)
    for t in trajectories: by_seed[t.seed].append(t)
    rows, counts = [], Counter()
    for seed, items in sorted(by_seed.items()):
        for index, a in enumerate(items):
            for b in items[index + 1:]:
                ok, pref, priority, reason = choose(a, b)
                row = {'pair_id': f'{seed}_{a.group}_vs_{b.group}', 'eval_seed': seed,
                       'trajectory_a': a.id, 'trajectory_b': b.id, 'comparable': int(ok),
                       'preference': pref, 'deciding_priority': priority, 'reason': reason,
                       'rule_version': RULE_VERSION}
                rows.append(row); counts[pref] += 1
                if priority != 'none': counts[f'priority_{priority}'] += 1
    args.output_dir.mkdir(parents=True)
    fields = ('pair_id', 'eval_seed', 'trajectory_a', 'trajectory_b', 'comparable', 'preference', 'deciding_priority', 'reason', 'rule_version')
    with (args.output_dir / 'trajectory_preferences.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / 'trajectory_preferences_summary.json').open('x') as handle:
        json.dump({'rule_version': RULE_VERSION, 'total_pairs': len(rows), 'A_preferred': counts['A'],
                   'B_preferred': counts['B'], 'similar': counts['similar'], 'unresolved': counts['unresolved'],
                   'decided_by_priority': {p: counts[f'priority_{p}'] for p in ('P1', 'P2', 'P3', 'P4')},
                   'limitations': ['P3 only treats completion-versus-noncompletion as decisive: no reliable threshold for small early/late distance differences exists.', 'Stopped-clear-route is descriptive only; stop necessity is unobserved.']}, handle, indent=2); handle.write('\n')


if __name__ == '__main__': main()
