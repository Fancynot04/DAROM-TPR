"""120 km/h road-limit variant of the frozen L1 reward configuration."""

import unittest
from unittest.mock import Mock, patch

from src import exit_e76_d3_config as exits
from src import exit_progress_reward
from src.merging import Merging
from train_sac_delay import parse_args


class L1Speed120Tests(unittest.TestCase):
    def test_e10_changes_controller_and_checkpoint_schedule_not_reward(self):
        baseline_name = 'secrm_efficiency_directional_l1_speed120_500k'
        e10_name = 'secrm_efficiency_directional_l1_speed120_no_pending_500k'
        baseline = exit_progress_reward.load_experiment(phase=baseline_name)
        e10 = exit_progress_reward.load_experiment(phase=e10_name)
        for field in ('seed', 'scenario', 'timeout_seconds', 'total_timesteps', 'reward'):
            self.assertEqual(e10[field], baseline[field], field)
        self.assertEqual(e10['checkpoint_freq'], 250_000)
        baseline_phase, e10_phase = exits.PHASES[baseline_name], exits.PHASES[e10_name]
        for field in ('scenario', 'total_timesteps', 'gamma', 'nav_features',
                      'reward_profile', 'road_speed_limit_mps'):
            self.assertEqual(e10_phase[field], baseline_phase[field], field)
        self.assertEqual(e10_phase['e09_pending_mode'], 'off')
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', e10_name,
                                '--wandb_mode', 'online']):
            args = parse_args()
        self.assertEqual((args.seed, args.total_timesteps, args.checkpoint_freq),
                         (42, 500_000, 250_000))
        self.assertIsNone(args.load_path)
        self.assertIsNone(args.replay_buffer_path)
        self.assertEqual(args.wandb_mode, 'online')

    def test_phase_changes_only_budget_and_road_limit(self):
        base = exit_progress_reward.load_experiment(
            phase='secrm_efficiency_directional_l1')
        variant = exit_progress_reward.load_experiment(
            phase='secrm_efficiency_directional_l1_speed120_500k')
        self.assertEqual(variant['reward'], base['reward'])
        self.assertEqual(variant['seed'], base['seed'])
        self.assertEqual(variant['scenario'], base['scenario'])
        self.assertEqual(variant['timeout_seconds'], base['timeout_seconds'])
        self.assertEqual(variant['total_timesteps'], 500_000)
        self.assertEqual(variant['checkpoint_freq'], 250_000)
        phase = exits.PHASES['secrm_efficiency_directional_l1_speed120_500k']
        self.assertAlmostEqual(phase['road_speed_limit_mps'], 120.0 / 3.6)
        self.assertFalse(phase['nav_features'])

    def test_continuation_phases_require_their_frozen_parent(self):
        cases = (
            ('secrm_efficiency_directional_l1_speed120_500k',
             'secrm_efficiency_directional_l1_speed120_1m', 1_000_000),
            ('secrm_efficiency_directional_l1_speed120_1m',
             'secrm_efficiency_directional_l1_speed120_2m', 2_000_000),
        )
        for parent_phase, continuation_phase, target_steps in cases:
            with self.subTest(phase=continuation_phase):
                parent = exit_progress_reward.load_experiment(phase=parent_phase)
                continuation = exit_progress_reward.load_experiment(
                    phase=continuation_phase)
                for field in ('reward', 'seed', 'scenario', 'timeout_seconds'):
                    self.assertEqual(continuation[field], parent[field], field)
                self.assertEqual(
                    (continuation['total_timesteps'], continuation['checkpoint_freq']),
                    (target_steps, 250_000))
                phase = exits.PHASES[continuation_phase]
                self.assertAlmostEqual(phase['road_speed_limit_mps'], 120.0 / 3.6)

                with patch('sys.argv', ['train_sac_delay.py', '--exit_phase',
                                        continuation_phase]):
                    with self.assertRaises(SystemExit):
                        parse_args()
                with patch('sys.argv', ['train_sac_delay.py', '--exit_phase',
                                        continuation_phase,
                                        '--load_path', 'parent.zip',
                                        '--replay_buffer_path', 'parent.pkl']):
                    args = parse_args()
                self.assertEqual((args.total_timesteps, args.checkpoint_freq),
                                 (target_steps, 250_000))
                self.assertEqual((args.load_path, args.replay_buffer_path),
                                 ('parent.zip', 'parent.pkl'))

    def test_corridor_override_includes_route_and_connectors(self):
        config = exit_progress_reward.load_experiment(
            phase='secrm_efficiency_directional_l1_speed120_500k')
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
            phase='secrm_efficiency_directional_l1_speed120_500k')
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
