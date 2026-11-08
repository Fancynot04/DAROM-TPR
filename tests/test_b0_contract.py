import json
import sys
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces

from src import exit_e76_d3_config as cfg
from src import exit_progress_reward
from src.rewards.registry import RewardContext, settle
from src.merging import Merging, control_step_timeout, select_terminal_event


class B0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reward = exit_progress_reward.load_experiment('configs/rewards/events_only_v1.json')

    def test_nonterminal_and_terminal_rewards(self):
        self.assertEqual(settle('events_only_v1', RewardContext(None, self.reward))[0], 0.0)
        self.assertEqual(settle('events_only_v1', RewardContext('Arrived', self.reward))[0], 50.0)
        for event in ('Collision', 'Missed Exit', 'Ego Teleported', 'Timeout'):
            self.assertEqual(settle('events_only_v1', RewardContext(event, self.reward))[0], -100.0)

    def test_unknown_reward_configuration_fails(self):
        with self.assertRaises(ValueError):
            exit_progress_reward.validate_experiment({'version': 'old', 'events': {}})
        with self.assertRaises(ValueError):
            settle('old_profile', RewardContext(None, self.reward))

    def test_contract_time_and_speed(self):
        self.assertEqual(cfg.TIMEOUT_STEPS, 4000)
        self.assertEqual(cfg.SUMO_STEP_LENGTH, 0.1)
        self.assertEqual(cfg.EGO_DEPART_TIME, 120.0)
        self.assertAlmostEqual(cfg.ROAD_SPEED_LIMIT_MPS, 120 / 3.6)
        self.assertEqual(json.loads(Path('configs/rewards/events_only_v1.json').read_text())['events']['success'], 50)
        self.assertIn('<end value="600"', Path('sumo_new/Antwerp.sumocfg').read_text())

    def test_replay_buffer_preserves_timeout_as_terminal(self):
        obs = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        action = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        buffer = ReplayBuffer(4, obs, action, device='cpu', n_envs=1)
        buffer.add(np.zeros((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32),
                   np.zeros((1, 1), dtype=np.float32), np.array([-100.0], dtype=np.float32),
                   np.array([True]), [{}])
        self.assertTrue(bool(buffer.dones[0]))

    def test_timeout_and_emergency_contract_is_encoded_in_environment(self):
        source = inspect.getsource(Merging.collect_step)
        self.assertEqual(select_terminal_event(
            collision=False, ego_teleported=False, missed_exit=False,
            arrived=False, timeout=True), 'Timeout')
        self.assertTrue(control_step_timeout(4000, 4000))
        self.assertIn('terminated = True', source)
        self.assertIn('self.emergency_stopping_steps += 1', source)
        self.assertNotIn("info['message'] = 'Emergency Stopping'", source)
        self.assertNotIn("'--end'", inspect.getsource(Merging.reset))

    def test_training_cli_values_are_not_phase_overridden(self):
        from train_sac_delay import parse_args
        with patch.object(sys, 'argv', ['train_sac_delay.py', '--seed', '43',
                                        '--total_timesteps', '17', '--checkpoint_freq', '9',
                                        '--gamma', '0.97', '--scenario', 'hard',
                                        '--wandb_mode', 'disabled']):
            args = parse_args()
        self.assertEqual((args.seed, args.total_timesteps, args.checkpoint_freq, args.gamma, args.scenario),
                         (43, 17, 9, 0.97, 'hard'))


if __name__ == '__main__':
    unittest.main()
