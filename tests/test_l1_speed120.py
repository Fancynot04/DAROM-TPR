"""120 km/h road-limit variant of the frozen L1 reward configuration."""

import unittest
from unittest.mock import Mock

from src import exit_e76_d3_config as exits
from src import exit_progress_reward
from src.merging import Merging


class L1Speed120Tests(unittest.TestCase):
    def test_phase_changes_only_budget_and_road_limit(self):
        base = exit_progress_reward.load_experiment(
            phase='secrm_efficiency_directional_l1')
        variant = exit_progress_reward.load_experiment(
            phase='secrm_efficiency_directional_l1_speed120_50k')
        self.assertEqual(variant['reward'], base['reward'])
        self.assertEqual(variant['seed'], base['seed'])
        self.assertEqual(variant['scenario'], base['scenario'])
        self.assertEqual(variant['timeout_seconds'], base['timeout_seconds'])
        self.assertEqual(variant['total_timesteps'], 50_000)
        self.assertEqual(variant['checkpoint_freq'], 50_000)
        phase = exits.PHASES['secrm_efficiency_directional_l1_speed120_50k']
        self.assertAlmostEqual(phase['road_speed_limit_mps'], 120.0 / 3.6)
        self.assertFalse(phase['nav_features'])

    def test_corridor_override_includes_route_and_connectors(self):
        config = exit_progress_reward.load_experiment(
            phase='secrm_efficiency_directional_l1_speed120_50k')
        env = Merging(
            seed=42, scenario='easy', task=exits.TASK_ID,
            exit_reward_profile=exits.secrm_reward.ADAPTIVE_PROFILE,
            progress_experiment=config,
            road_speed_limit_mps=exits.SPEED_LIMIT_120_KPH_MPS,
        )
        env._connection = Mock()
        lanes = ('E76_0', 'E76_4', ':J105_0_0', 'E77_0', 'E77_5',
                 ':J106_0_0', 'D3_0', 'E75_0')
        env._connection.lane.getIDList.return_value = lanes
        selected = env._apply_exit_road_speed_limit()
        self.assertEqual(set(selected), set(lanes) - {'E75_0'})
        calls = env._connection.lane.setMaxSpeed.call_args_list
        self.assertEqual(len(calls), len(selected))
        self.assertTrue(all(call.args[1] == exits.SPEED_LIMIT_120_KPH_MPS
                            for call in calls))

    def test_missing_corridor_fails_closed(self):
        config = exit_progress_reward.load_experiment(
            phase='secrm_efficiency_directional_l1_speed120_50k')
        env = Merging(
            seed=42, scenario='easy', task=exits.TASK_ID,
            exit_reward_profile=exits.secrm_reward.ADAPTIVE_PROFILE,
            progress_experiment=config,
            road_speed_limit_mps=exits.SPEED_LIMIT_120_KPH_MPS,
        )
        env._connection = Mock()
        env._connection.lane.getIDList.return_value = ('E76_0', 'E77_0', 'D3_0')
        with self.assertRaisesRegex(RuntimeError, 'corridor is incomplete'):
            env._apply_exit_road_speed_limit()


if __name__ == '__main__':
    unittest.main()
