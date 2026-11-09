import unittest

from scripts.audit_trajectory_preferences import (
    choose_samples,
    invert_preference,
    pair_type,
    reason_branch,
    three_cycles,
)
from scripts.compare_trajectory_preferences import Trajectory


def trajectory(name, outcome='Arrived'):
    return Trajectory('B0', {'trajectory_id': name, 'eval_seed': '1', 'scenario': 'easy',
                             'terminal_reason': outcome}, [])


class TrajectoryPreferenceAuditTest(unittest.TestCase):
    def test_pair_types_and_inversion(self):
        self.assertEqual(pair_type(trajectory('a'), trajectory('b')), 'success_vs_success')
        self.assertEqual(pair_type(trajectory('a'), trajectory('b', 'Collision')), 'success_vs_failure')
        self.assertEqual(pair_type(trajectory('a', 'Collision'), trajectory('b', 'Missed Exit')),
                         'failure_vs_failure')
        self.assertEqual(invert_preference('A'), 'B')
        self.assertEqual(invert_preference('similar'), 'similar')

    def test_reason_branch_uses_exact_executed_reason(self):
        self.assertEqual(reason_branch(
            'P3', 'A', 'fewer physical lane changes away from the exit side'),
            'fewer_away_lane_changes')
        self.assertEqual(reason_branch('P3', 'A', 'unexpected'), 'unknown')

    def test_three_cycle_detection(self):
        rows = [
            {'eval_seed': '1', 'trajectory_a': 'a', 'trajectory_b': 'b', 'preference': 'A'},
            {'eval_seed': '1', 'trajectory_a': 'b', 'trajectory_b': 'c', 'preference': 'A'},
            {'eval_seed': '1', 'trajectory_a': 'a', 'trajectory_b': 'c', 'preference': 'B'},
        ]
        self.assertEqual(three_cycles(rows), [('1', 'a', 'b', 'c')])

    def test_sample_selection_covers_actual_branches_first(self):
        rows = []
        trajectories = {}
        for index in range(6):
            a_id, b_id = f'a{index}', f'b{index}'
            a, b = trajectory(a_id), trajectory(b_id)
            a.group, b.group = f'B{index % 4}', f'B{(index + 1) % 4}'
            trajectories[a_id], trajectories[b_id] = a, b
            rows.append({'pair_id': str(index), 'eval_seed': str(index),
                         'trajectory_a': a_id, 'trajectory_b': b_id,
                         'preference': 'A', 'deciding_priority': 'P2',
                         'trigger_branch': 'non_collision' if index == 5 else 'ttc'})
        # Add the remaining categories so the fixed quota selector can complete.
        for bucket, count in (('P1', 4), ('P3', 7), ('P4', 3), ('similar', 2)):
            for index in range(count):
                suffix = f'{bucket}{index}'
                a, b = trajectory('a' + suffix), trajectory('b' + suffix)
                a.group, b.group = 'B0', 'B1'
                trajectories[a.id], trajectories[b.id] = a, b
                rows.append({'pair_id': suffix, 'eval_seed': str(100 + index),
                             'trajectory_a': a.id, 'trajectory_b': b.id,
                             'preference': 'similar' if bucket == 'similar' else 'A',
                             'deciding_priority': 'none' if bucket == 'similar' else bucket,
                             'trigger_branch': bucket})
        selected = choose_samples(rows, trajectories)
        p2_branches = {row['trigger_branch'] for row in selected
                       if row['deciding_priority'] == 'P2'}
        self.assertEqual(p2_branches, {'ttc', 'non_collision'})


if __name__ == '__main__':
    unittest.main()
