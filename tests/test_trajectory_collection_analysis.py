import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_trajectory_collection import FIELDS, summarize_group


class TrajectoryCollectionAnalysisTest(unittest.TestCase):
    def test_summarize_group_keeps_missing_efficiency_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with (directory / 'episode_summary.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'terminal_reason', 'exit_reachable_known_steps', 'exit_reachable_true_steps',
                    'physical_lane_changes', 'hard_braking_steps', 'sumo_emergency_stopping_steps',
                    'leader_ttc_min_s', 'follower_ttc_min_s', 'efficiency_raw_mean',
                    'efficiency_weighted_sum', 'route_raw_mean', 'route_weighted_sum'])
                writer.writeheader()
                writer.writerow({'terminal_reason': 'Arrived', 'exit_reachable_known_steps': '5',
                                 'exit_reachable_true_steps': '5', 'physical_lane_changes': '2',
                                 'hard_braking_steps': '1', 'sumo_emergency_stopping_steps': '0',
                                 'leader_ttc_min_s': '1.5', 'follower_ttc_min_s': '',
                                 'efficiency_raw_mean': '', 'efficiency_weighted_sum': '0',
                                 'route_raw_mean': '', 'route_weighted_sum': '0'})
                writer.writerow({'terminal_reason': 'Collision', 'exit_reachable_known_steps': '2',
                                 'exit_reachable_true_steps': '1', 'physical_lane_changes': '1',
                                 'hard_braking_steps': '3', 'sumo_emergency_stopping_steps': '1',
                                 'leader_ttc_min_s': '0.8', 'follower_ttc_min_s': '2.0',
                                 'efficiency_raw_mean': '', 'efficiency_weighted_sum': '0',
                                 'route_raw_mean': '', 'route_weighted_sum': '0'})
            result = summarize_group('B0', directory)
            self.assertEqual(result['episodes'], 2)
            self.assertEqual(result['arrived'], 1)
            self.assertEqual(result['mean_physical_lane_changes'], 1.5)
            self.assertAlmostEqual(result['exit_reachable_fraction'], 6 / 7)
            self.assertEqual(result['mean_efficiency_raw'], None)
            self.assertEqual(result['min_leader_ttc_s'], 0.8)
            self.assertEqual(set(result), set(FIELDS))
