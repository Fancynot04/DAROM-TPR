import unittest

from scripts.audit_trajectory_preferences import (
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


if __name__ == '__main__':
    unittest.main()
