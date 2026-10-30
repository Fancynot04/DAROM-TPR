"""Regression tests for the three single-component changes to A1."""

import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src import exit_e76_d3_config as cfg
from src.merging import Merging
from train_sac_delay import parse_args
from evaluate_sac_delay import validate_checkpoint_setup


class ComponentRewardTests(unittest.TestCase):
    def make_env(self, profile):
        env = Merging(seed=42, task=cfg.TASK_ID, reward_profile=profile)
        env.reset_episode_state()
        env.ego_id = 'ego'
        env.ego_on_road = True
        env.state = np.zeros((31, 3), dtype=np.float32)
        env.observation = env.state.copy()
        env.observation[0, 0] = 2
        env.parsed_action = [2, -1]
        env._ahead_state = [5, 0, 0]
        env._behind_state = [-3, 0, 0]
        return env

    def test_only_target_component_changes_and_event_priority_is_preserved(self):
        fake = Mock()
        fake.simulation.getCollidingVehiclesIDList.return_value = ()
        fake.simulation.getArrivedIDList.return_value = ()
        fake.simulation.getEmergencyStoppingVehiclesIDList.return_value = ()
        with patch('src.merging.traci', fake), patch('src.merging.safe_gap', return_value=(False, False)):
            environments = {name: self.make_env(name) for name in ('A1', 'B1', 'B2', 'B3')}
            for env in environments.values():
                reward = env.getReward()
                self.assertAlmostEqual(reward, sum(env.reward_components.values()))
            baseline = environments['A1'].reward_components
            for profile, removed in (('B1', 'behind_gap'), ('B2', 'action_change'), ('B3', 'stage')):
                actual = environments[profile].reward_components
                self.assertEqual(actual[removed], 0)
                for key in baseline:
                    if key != removed:
                        self.assertAlmostEqual(actual[key], baseline[key], msg=f'{profile}: {key}')
            self.assertLess(baseline['behind_gap'], 0)
            self.assertLess(baseline['action_change'], 0)

            for env in environments.values():
                env.stage_reached = True
                reward = env.getReward()
                self.assertTrue(env.given_merge_reward)
                self.assertAlmostEqual(reward, sum(env.reward_components.values()))
                self.assertEqual(env.reward_components['stage'],
                                 0 if env.reward_profile == 'B3' else 50)
                self.assertEqual(env.reward_components['lane_command'], 0)
                env.getReward()
                self.assertEqual(env.reward_components['stage'], 0)
                env.terminal_reason = 'Arrived'
                env.getReward()
                self.assertEqual(env.reward_components['arrival'], 50)

    def test_six_runs_are_fresh_and_isolated(self):
        paths = set()
        for profile in cfg.COMPONENT_PROFILES:
            for seed in (42, 43):
                with patch.object(sys, 'argv', ['train_sac_delay.py', '--exit_phase',
                                               cfg.ABLATION_PHASE, '--reward_profile', profile,
                                               '--seed', str(seed), '--wandb_mode', 'disabled']):
                    args = parse_args()
                self.assertEqual(args.total_timesteps, 200_000)
                self.assertEqual(args.checkpoint_freq, 50_000)
                self.assertIsNone(args.load_path)
                self.assertIsNone(args.replay_buffer_path)
                self.assertIn('a1_component_ablation', args.save_dir)
                paths.add(args.save_dir)
                model = Mock()
                model.antwerp_env_version = 2
                model.antwerp_task = cfg.TASK_ID
                model.ablation = 'full'
                model.exit_potential_beta = 0.0
                model.exit_reward_profile = profile
                model.num_timesteps = 200_000
                model.observation_space.shape = (134,)
                model.exit_training_config = {
                    'seed': seed, 'total_timesteps': 200_000, 'task': cfg.TASK_ID,
                    'scenario': 'easy', 'delay_mode': 'uniform', 'max_delay': 20,
                    'use_safety': False, 'encoder_mode': 'GRU', 'gru_input_mode': 'paper_raw',
                    'ablation': 'full', 'reward_profile': profile, 'potential_beta': 0.0,
                    'sac': {'learning_rate': 3e-5, 'batch_size': 512,
                            'buffer_size': 1_000_000, 'learning_starts': 100, 'gamma': 0.99}}
                validate_checkpoint_setup(model, 'full', False, cfg.TASK_ID, profile)
                with self.assertRaises(ValueError):
                    validate_checkpoint_setup(model, 'full', False, cfg.TASK_ID, 'A1')
        self.assertEqual(len(paths), 6)

    def test_same_sumo_actions_produce_same_trajectories(self):
        traces = {}
        for profile in ('A1', 'B1', 'B2', 'B3'):
            env = Merging(seed=42, task=cfg.TASK_ID, scenario='easy', reward_profile=profile)
            try:
                _, reset = env.reset(seed=42)
                self.assertEqual(reset['ego_lane_change_mode'], 0)
                trace = []
                for _ in range(130):
                    _, reward, terminated, truncated, info = env.step([0, -5])
                    self.assertAlmostEqual(reward, sum(info['reward_components'].values()))
                    self.assertEqual(info['potential_reward'], 0)
                    trace.append((info['road'], info['lane_index'], info['ego_distance'],
                                  info['ego_speed'], info['message'], terminated, truncated))
                    if terminated or truncated:
                        break
                self.assertTrue(any(row[1] == 0 for row in trace))
                traces[profile] = trace
            finally:
                env.close()
        for profile in ('B1', 'B2', 'B3'):
            self.assertEqual(traces['A1'], traces[profile])


if __name__ == '__main__':
    unittest.main()
