"""Reward factorial and fixed-action SUMO checks for E76 to D3."""

import unittest
from unittest.mock import Mock, patch
import sys

import numpy as np

from src import exit_e76_d3_config as cfg
from src.merging import Merging
from train_sac_delay import parse_args
from evaluate_sac_delay import evaluate_runs, validate_checkpoint_setup
from src.basic_reward_ablation import EVAL_SEEDS


class RewardFormulaTests(unittest.TestCase):
    def test_profiles_and_event_priority(self):
        expected = {'A0': (-0.5, 0.15), 'A1': (0.0, 0.15),
                    'A2': (-0.5, 0.0375), 'A3': (0.0, 0.0375)}
        fake = Mock()
        fake.simulation.getCollidingVehiclesIDList.return_value = ()
        fake.simulation.getArrivedIDList.return_value = ()
        fake.simulation.getEmergencyStoppingVehiclesIDList.return_value = ()
        with patch('src.merging.traci', fake), patch('src.merging.safe_gap', return_value=(True, True)):
            for name, (lane_penalty, progress_rate) in expected.items():
                with self.subTest(profile=name):
                    self.assertEqual(cfg.ABLATION_PROFILES[name], {
                        'lane_command': lane_penalty, 'progress_per_meter': progress_rate})
                    env = Merging(seed=1, task=cfg.TASK_ID, reward_profile=name)
                    env.ego_id = 'ego'
                    env.ego_on_road = True
                    env.given_merge_reward = False
                    env.state = np.zeros((31, 3), dtype=np.float32)
                    env.observation = env.state.copy()
                    env.observation[0, 0] = 2.0
                    env.parsed_action = [0, -1]
                    reward = env.getReward()
                    self.assertAlmostEqual(env.reward_components['progress'], 2 * progress_rate)
                    self.assertAlmostEqual(env.reward_components['lane_command'], lane_penalty)
                    self.assertAlmostEqual(reward, sum(env.reward_components.values()))
                    self.assertEqual(env.reward_components['base'], -0.03)
                    env.stage_reached = True
                    stage_reward = env.getReward()
                    self.assertAlmostEqual(env.reward_components['stage'], 50)
                    self.assertEqual(env.reward_components['lane_command'], 0)
                    self.assertAlmostEqual(stage_reward, sum(env.reward_components.values()))
                    env.terminal_reason = 'Collision'
                    terminal_reward = env.getReward()
                    self.assertEqual(env.reward_components['collision'], -100)
                    self.assertEqual(env.reward_components['stage'], 0)
                    self.assertEqual(env.reward_components['lane_command'], 0)
                    self.assertAlmostEqual(terminal_reward, sum(env.reward_components.values()))

    def test_terminal_priority_and_single_settlement(self):
        events = {
            'Collision': {'collision': True, 'teleport': True, 'arrived': True},
            'Ego Teleported': {'teleport': True, 'arrived': True},
            'Missed Exit': {'road': 'E78', 'arrived': True},
            'Emergency Stopping': {'emergency': True, 'arrived': True},
            'Arrived': {'arrived': True},
            'Max Steps': {'timeout': True},
        }
        component = {'Collision': 'collision', 'Ego Teleported': 'teleport',
                     'Missed Exit': 'missed_exit', 'Emergency Stopping': 'emergency',
                     'Arrived': 'arrival', 'Max Steps': 'timeout'}
        for outcome, flags in events.items():
            with self.subTest(outcome=outcome):
                env = Merging(seed=1, task=cfg.TASK_ID, reward_profile='A0')
                env.reset_episode_state()
                env.ego_id = 'ego'
                env.ego_on_road = True
                env.parsed_action = [0, -1]
                env.counter = 6000 if flags.get('timeout') else 1
                fake = Mock()
                fake.vehicle.getIDList.return_value = ('ego',)
                fake.vehicle.getRoadID.return_value = flags.get('road', 'E76')
                fake.vehicle.getLaneID.return_value = flags.get('road', 'E76') + '_1'
                fake.vehicle.getLaneIndex.return_value = 1
                fake.vehicle.getDistance.return_value = 10.0
                fake.vehicle.getSpeed.return_value = 10.0
                fake.vehicle.getAcceleration.return_value = 0.0
                fake.vehicle.getDrivingDistance.return_value = 100.0
                fake.simulation.getStartingTeleportIDList.return_value = ('ego',) if flags.get('teleport') else ()
                fake.simulation.getEndingTeleportIDList.return_value = ()
                fake.simulation.getCollidingVehiclesIDList.return_value = ('ego',) if flags.get('collision') else ()
                fake.simulation.getArrivedIDList.return_value = ('ego',) if flags.get('arrived') else ()
                fake.simulation.getEmergencyStoppingVehiclesIDList.return_value = ('ego',) if flags.get('emergency') else ()
                with patch('src.merging.traci', fake), patch('src.merging.safe_gap', return_value=(True, True)), \
                        patch.object(env, '_get_obs', return_value=env.state.copy()):
                    _, reward, terminated, truncated, info = env.collect_step(True)
                self.assertEqual(info['message'], outcome)
                self.assertTrue(terminated or truncated)
                self.assertEqual(info['reward_components'][component[outcome]],
                                 50 if outcome == 'Arrived' else -100)
                self.assertAlmostEqual(reward, sum(info['reward_components'].values()))
                self.assertEqual(info['reward_components']['lane_command'], 0)


class ConfigurationTests(unittest.TestCase):
    def test_eight_fresh_200k_configs_and_checkpoint_validation(self):
        for profile in cfg.ABLATION_PROFILES:
            for seed in (42, 43):
                with self.subTest(profile=profile, seed=seed):
                    argv = ['train_sac_delay.py', '--exit_phase', cfg.ABLATION_PHASE,
                            '--reward_profile', profile, '--seed', str(seed),
                            '--wandb_mode', 'disabled']
                    with patch.object(sys, 'argv', argv):
                        args = parse_args()
                    self.assertEqual(args.total_timesteps, 200_000)
                    self.assertEqual(args.checkpoint_freq, 50_000)
                    self.assertEqual(args.task, cfg.TASK_ID)
                    self.assertEqual(args.scenario, 'easy')
                    self.assertIsNone(args.load_path)
                    self.assertIsNone(args.replay_buffer_path)
                    self.assertIn(f'{profile}_seed{seed}', args.save_dir)
                    model = Mock()
                    model.antwerp_env_version = 2
                    model.antwerp_task = cfg.TASK_ID
                    model.ablation = 'full'
                    model.exit_potential_beta = 0.0
                    model.exit_reward_profile = profile
                    model.num_timesteps = 200_000
                    model.observation_space.shape = (134,)
                    model.exit_training_config = {
                        'seed': seed, 'total_timesteps': 200_000,
                        'task': cfg.TASK_ID, 'scenario': 'easy',
                        'delay_mode': 'uniform', 'max_delay': 20,
                        'use_safety': False, 'encoder_mode': 'GRU',
                        'gru_input_mode': 'paper_raw', 'ablation': 'full',
                        'reward_profile': profile, 'potential_beta': 0.0,
                        'sac': {'learning_rate': 3e-5, 'batch_size': 512,
                                'buffer_size': 1_000_000, 'learning_starts': 100,
                                'gamma': 0.99}}
                    validate_checkpoint_setup(model, 'full', False, cfg.TASK_ID, profile)
                    self.assertEqual(EVAL_SEEDS, list(range(200001, 200051)))
                    with self.assertRaises(ValueError):
                        evaluate_runs(model, 'easy', 'uniform', False, 'full',
                                      49, [200001], cfg.TASK_ID, profile)
                    model.exit_reward_profile = 'A3' if profile != 'A3' else 'A0'
                    with self.assertRaises(ValueError):
                        validate_checkpoint_setup(model, 'full', False, cfg.TASK_ID, profile)


class FixedActionPairTests(unittest.TestCase):
    def test_matched_sumo_trajectory_and_original_control(self):
        traces = {}
        for profile in cfg.ABLATION_PROFILES:
            env = Merging(seed=42, task=cfg.TASK_ID, scenario='easy',
                          reward_profile=profile)
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
                self.assertTrue(any(item[1] == 0 for item in trace))
                traces[profile] = trace
            finally:
                env.close()
        for profile in ('A1', 'A2', 'A3'):
            self.assertEqual(traces[profile], traces['A0'])


if __name__ == '__main__':
    unittest.main()
