"""Focused checks for route-end progress, terminal accounting and checkpoint provenance."""

import copy
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

from evaluate_sac_delay import evaluate, validate_checkpoint_setup
from train_sac_delay import parse_args as parse_train_args
from src import exit_e76_d3_config as exit_config
from src import exit_progress_reward as progress
from src.merging import Merging


class ProgressRewardTests(unittest.TestCase):
    def setUp(self):
        self.config = progress.load_experiment()
        self.reward = self.config['reward']

    def test_progress_continues_inside_d3_and_stops_when_parked(self):
        early = progress.progress_reward(1000, 990, 0, 10, 1000, self.reward)
        on_d3 = progress.progress_reward(150, 140, 850, 860, 1000, self.reward)
        parked = progress.progress_reward(140, 140, 860, 860, 1000, self.reward)
        self.assertGreater(on_d3, early)
        self.assertEqual(parked, 0.0)
        self.assertEqual(progress.progress_reward(150, 0, 850, 851, 1000, self.reward),
                         self.reward['progress_per_meter'] * (1 + 0.85) * 1)
        self.assertEqual(progress.progress_reward(150, -1, 850, 860, 1000, self.reward), 0.0)

    def test_exactly_one_event_and_no_stage_payment(self):
        env = Merging(seed=42, task=exit_config.TASK_ID,
                      exit_reward_profile=exit_config.PROGRESS_REWARD_PROFILE,
                      progress_experiment=self.config)
        env.progress_initial_remaining = 1000.0
        env.progress_before_remaining = 150.0
        env.progress_after_remaining = 140.0
        env.progress_before_odometer = 850.0
        env.progress_after_odometer = 860.0
        env.stage_reached = True
        for outcome, key in progress.EVENTS.items():
            with self.subTest(outcome=outcome):
                env.terminal_reason = outcome
                result = env.getReward()
                self.assertAlmostEqual(result, env.reward_components['progress'] + self.reward[key])
                self.assertEqual(env.reward_components[key], self.reward[key])
                self.assertEqual(sum(value for name, value in env.reward_components.items()
                                     if name != 'progress'), self.reward[key])
                self.assertEqual(env.last_stage_reward, 0.0)
        env.terminal_reason = ''
        self.assertGreater(env.getReward(), 0.0)
        self.assertEqual(sum(value for name, value in env.reward_components.items()
                             if name != 'progress'), 0.0)
        env.teleport_transition = True
        self.assertEqual(env.getReward(), 0.0)

    def test_timeout_is_400_seconds_only_for_new_profile(self):
        new = Merging(seed=42, task=exit_config.TASK_ID,
                      exit_reward_profile=exit_config.PROGRESS_REWARD_PROFILE,
                      progress_experiment=self.config)
        old = Merging(seed=42, task=exit_config.TASK_ID)
        self.assertEqual(new.episode_limit_steps, 4000)
        self.assertEqual(old.episode_limit_steps, 6000)
        invalid = copy.deepcopy(self.config)
        invalid['timeout_seconds'] = 400.05
        with self.assertRaises(ValueError):
            Merging(seed=42, task=exit_config.TASK_ID,
                    exit_reward_profile=exit_config.PROGRESS_REWARD_PROFILE,
                    progress_experiment=invalid)

    def test_d3_remaining_distance_is_to_road_end(self):
        env = Merging(seed=42, task=exit_config.TASK_ID,
                      exit_reward_profile=exit_config.PROGRESS_REWARD_PROFILE,
                      progress_experiment=self.config)
        env.ego_id = 'ego'
        env.ego_on_road = True
        fake = Mock()
        fake.vehicle.getRoadID.return_value = 'D3'
        fake.vehicle.getLanePosition.return_value = 100.0
        fake.lane.getLength.return_value = 205.79
        env._connection = fake
        self.assertAlmostEqual(env._remaining_to_d3_end(), 105.79)
        fake.vehicle.getDrivingDistance.assert_not_called()

    def test_run_configuration_controls_seed_scenario_budget_and_interval(self):
        config = copy.deepcopy(self.config)
        config.update(seed=57, scenario='hard', total_timesteps=1234,
                      checkpoint_freq=100)
        with patch.object(sys, 'argv', ['train_sac_delay.py', '--exit_phase', 'progress_events',
                                        '--exit_config', 'config.json']), \
                patch('train_sac_delay.exit_progress_reward.load_experiment', return_value=config):
            args = parse_train_args()
        self.assertEqual((args.seed, args.scenario, args.total_timesteps,
                          args.checkpoint_freq), (57, 'hard', 1234, 100))
        self.assertEqual(args.progress_experiment, config)

    def test_teleport_terminates_before_arrival_and_timeout(self):
        env = Merging(seed=42, task=exit_config.TASK_ID,
                      exit_reward_profile=exit_config.PROGRESS_REWARD_PROFILE,
                      progress_experiment=self.config)
        env.ego_id = 'ego'
        env.given_merge_reward = False
        env.ego_on_road = True
        env.counter = 4000
        env.state = np.zeros((31, 3), dtype=np.float32)
        env.observation = env.state.copy()
        env.parsed_action = [0, 0]
        env.action = [0, 0]
        env.progress_initial_remaining = 1000.0
        env.progress_before_remaining = 200.0
        env.progress_before_odometer = 800.0
        fake = Mock()
        fake.simulation.getStartingTeleportIDList.return_value = ('ego',)
        fake.simulation.getEndingTeleportIDList.return_value = ()
        fake.simulation.getCollidingVehiclesIDList.return_value = ()
        fake.simulation.getArrivedIDList.return_value = ('ego',)
        fake.simulation.getEmergencyStoppingVehiclesIDList.return_value = ()
        fake.vehicle.getIDList.return_value = ()
        with patch('src.merging.traci', fake):
            _, reward, terminated, truncated, info = env.collect_step(True, close_on_done=False)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['message'], 'Ego Teleported')
        self.assertFalse(info['is_success'])
        self.assertEqual(reward, self.reward['teleport'])
        self.assertEqual(info['reward_components']['arrival'], 0.0)

    def test_timeout_boundary_is_4000_takeover_steps(self):
        env = Merging(seed=42, task=exit_config.TASK_ID,
                      exit_reward_profile=exit_config.PROGRESS_REWARD_PROFILE,
                      progress_experiment=self.config)
        env.ego_id = 'ego'
        env.ego_on_road = True
        env.given_merge_reward = False
        env.lane_name = 'D3'
        env.state = np.zeros((31, 3), dtype=np.float32)
        env.observation = env.state.copy()
        env.parsed_action = [0, 0]
        env.action = [0, 0]
        env.progress_initial_remaining = 1000.0
        env.progress_before_remaining = 155.79
        env.progress_before_odometer = 100.0
        fake = Mock()
        fake.vehicle.getIDList.return_value = ('ego',)
        fake.vehicle.getRoadID.return_value = 'D3'
        fake.vehicle.getLaneID.return_value = 'D3_0'
        fake.vehicle.getLaneIndex.return_value = 0
        fake.vehicle.getLanePosition.return_value = 50.0
        fake.vehicle.getDistance.return_value = 100.0
        fake.lane.getLength.return_value = 205.79
        for name in ('getStartingTeleportIDList', 'getEndingTeleportIDList',
                     'getCollidingVehiclesIDList', 'getArrivedIDList',
                     'getEmergencyStoppingVehiclesIDList'):
            getattr(fake.simulation, name).return_value = ()
        with patch('src.merging.traci', fake), \
                patch.object(env, '_update_exit_maneuver'), \
                patch.object(env, '_get_obs', return_value=env.state.copy()):
            env.counter = 3999
            _, _, terminated, truncated, _ = env.collect_step(True, close_on_done=False)
            self.assertFalse(terminated or truncated)
            env.counter = 4000
            _, reward, terminated, truncated, info = env.collect_step(True, close_on_done=False)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['message'], 'Max Steps')
        self.assertEqual(reward, self.reward['timeout'])

    def test_checkpoint_reward_identity_rejects_change(self):
        model = SimpleNamespace(
            antwerp_env_version=3, antwerp_task=exit_config.TASK_ID,
            exit_reward_profile=exit_config.PROGRESS_REWARD_PROFILE,
            exit_potential_beta=0.0,
            exit_control_profile=exit_config.CONTROL_PROFILE,
            exit_observation_profile=None,
            exit_potential_gamma=0.99, gamma=0.99,
            observation_space=SimpleNamespace(shape=(134,)),
            policy_kwargs={'features_extractor_kwargs': {'nav_dim': 0}},
            exit_progress_experiment=self.config,
            exit_progress_reward_identity=progress.reward_identity(self.reward),
            exit_progress_experiment_identity=progress.experiment_identity(self.config),
        )
        setup = validate_checkpoint_setup(model, 'full', False, exit_config.TASK_ID)
        self.assertEqual(setup['progress_experiment'], self.config)
        model.exit_progress_experiment = copy.deepcopy(self.config)
        model.exit_progress_experiment['reward']['arrival'] = 60.0
        with self.assertRaises(ValueError):
            validate_checkpoint_setup(model, 'full', False, exit_config.TASK_ID)

    def test_sac_checkpoint_serializes_full_reward_configuration(self):
        env = gym.make('Pendulum-v1')
        try:
            model = SAC('MlpPolicy', env, device='cpu', buffer_size=8, learning_starts=1)
            model.exit_reward_profile = exit_config.PROGRESS_REWARD_PROFILE
            model.exit_progress_experiment = self.config
            model.exit_progress_reward_identity = progress.reward_identity(self.reward)
            model.exit_progress_experiment_identity = progress.experiment_identity(self.config)
            stream = io.BytesIO()
            model.save(stream)
            stream.seek(0)
            loaded = SAC.load(stream, device='cpu')
            self.assertEqual(loaded.exit_progress_experiment, self.config)
            self.assertEqual(loaded.exit_progress_reward_identity,
                             progress.reward_identity(self.reward))
            self.assertEqual(loaded.exit_progress_experiment_identity,
                             progress.experiment_identity(self.config))
        finally:
            env.close()

    def test_evaluation_counts_teleport_once_as_failure(self):
        model = SimpleNamespace(predict=lambda observation, deterministic:
                                (np.zeros(2, dtype=np.float32), None))
        environment = Mock()
        environment.unwrapped = SimpleNamespace(ego_on_road=False)
        environment.reset.return_value = (np.zeros(134, dtype=np.float32), {})
        environment.step.return_value = (
            np.zeros(134, dtype=np.float32), -100.0, True, False,
            {'message': 'Ego Teleported', 'ego_teleported': True,
             'stage_reached': False, 'stage_reward_paid': False})
        result = evaluate(model, environment, 1, 123, task=exit_config.TASK_ID)
        self.assertEqual(result['Ego Teleported'], 1)
        self.assertEqual(result['ego_teleport_rate'], 100)
        self.assertEqual(result['success_rate'], 0)


if __name__ == '__main__':
    unittest.main()
