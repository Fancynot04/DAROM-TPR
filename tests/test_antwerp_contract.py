"""Regression checks for Antwerp geometry, accounting and evaluation protocols."""

import unittest
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import torch

from evaluate_sac_delay import (evaluate, evaluate_runs, parse_args as parse_eval_args,
                                summarize_runs, validate_checkpoint_setup, validate_seed_ranges,
                                checkpoint_gru_input_mode)
from train_sac_delay import EpisodeRewardCallback, parse_args as parse_train_args
from src.delay_encoder import DelayAwareGRUEncoder
from src import exit_e76_d3_config as exit_config
from src.geometry import TrafficSnapshot, align_delayed_neighbors
from src.merging import ANTWERP_ENV_VERSION, ANTWERP_TASKS, Merging
from src.safety_controller import safetyCheck
from src.wrapper import ABLATION_MODES, DelayWrapper, SafetyWrapper


class GRUInputContractTests(unittest.TestCase):
    def test_navigation_split_keeps_vehicle_history_and_delay_positions(self):
        class CaptureGRU(torch.nn.Module):
            def forward(self, actions):
                self.actions = actions.detach().clone()
                return None, torch.zeros(1, actions.shape[0], 64)

        sample = torch.arange(136, dtype=torch.float32).unsqueeze(0)
        space = gym.spaces.Box(-5000, 5000, shape=(136,), dtype=np.float32)
        encoder = DelayAwareGRUEncoder(space, act_dim=2, hist_len=20,
                                       delay_dim=1, input_mode='paper_raw', nav_dim=2)
        encoder.obs_encoder = torch.nn.Identity()
        encoder.gru = CaptureGRU()
        encoder.fusion = torch.nn.Identity()
        output = encoder(sample)
        torch.testing.assert_close(output[:, :95], sample[:, :95])
        torch.testing.assert_close(encoder.gru.actions, sample[:, 95:135].reshape(1, 20, 2))
        torch.testing.assert_close(output[:, -1:], sample[:, -1:])
        with self.assertRaisesRegex(ValueError, '3 features per vehicle'):
            DelayAwareGRUEncoder(space, act_dim=2, hist_len=20, delay_dim=1)

    def test_raw_and_legacy_modes_preserve_their_distinct_input_contracts(self):
        class CaptureGRU(torch.nn.Module):
            def forward(self, actions):
                self.actions = actions.detach().clone()
                return None, torch.zeros(1, actions.shape[0], 64)

        sample = torch.arange(134, dtype=torch.float32).unsqueeze(0)
        space = gym.spaces.Box(-5000, 5000, shape=(134,), dtype=np.float32)
        for mode in DelayAwareGRUEncoder.INPUT_MODES:
            with self.subTest(mode=mode):
                encoder = DelayAwareGRUEncoder(space, act_dim=2, hist_len=20,
                                               delay_dim=1, input_mode=mode)
                encoder.obs_encoder = torch.nn.Identity()
                encoder.gru = CaptureGRU()
                encoder.fusion = torch.nn.Identity()
                output = encoder(sample)
                expected_actions = sample[:, 93:133].reshape(1, 20, 2)
                if mode == 'antwerp_scaled_reversed':
                    torch.testing.assert_close(output[:, :93],
                                               sample[:, :93] / encoder.observation_scale)
                    torch.testing.assert_close(encoder.gru.actions,
                                               expected_actions.flip(1) / 5.0)
                    torch.testing.assert_close(output[:, -1:], sample[:, -1:] / 20)
                else:
                    torch.testing.assert_close(output[:, :93], sample[:, :93])
                    torch.testing.assert_close(encoder.gru.actions, expected_actions)
                    torch.testing.assert_close(output[:, -1:], sample[:, -1:])

    def test_checkpoints_without_mode_keep_legacy_processing(self):
        old = SimpleNamespace(policy_kwargs={
            'features_extractor_class': DelayAwareGRUEncoder,
            'features_extractor_kwargs': {'act_dim': 2, 'hist_len': 20},
        })
        self.assertEqual(checkpoint_gru_input_mode(old), 'antwerp_scaled_reversed')
        old.policy_kwargs['features_extractor_kwargs']['input_mode'] = 'paper_raw'
        self.assertEqual(checkpoint_gru_input_mode(old), 'paper_raw')


class SeparateEgoTaskTests(unittest.TestCase):
    def test_navigation_uses_live_network_lane_and_rejects_bad_distance(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        env.ego_on_road = True
        fake = Mock()
        fake.vehicle.getRoadID.return_value = ':J105_0'
        fake.vehicle.getLaneID.return_value = ':J105_0_3'
        fake.vehicle.getDrivingDistance.return_value = 630.37
        with patch('src.merging.traci', fake):
            np.testing.assert_allclose(env.exit_navigation(), [0.5, 0.6], atol=1e-6)
            fake.vehicle.getLaneID.return_value = 'E77_2'
            fake.vehicle.getRoadID.return_value = 'E77'
            fake.vehicle.getLaneIndex.return_value = 2
            np.testing.assert_allclose(env.exit_navigation(), [0.5, 0.4], atol=1e-6)
            fake.vehicle.getDrivingDistance.return_value = -1
            with self.assertRaisesRegex(ValueError, 'Invalid nonterminal'):
                env.exit_navigation()
            fake.vehicle.getDrivingDistance.return_value = float('nan')
            with self.assertRaisesRegex(ValueError, 'Invalid nonterminal'):
                env.exit_navigation()
            fake.vehicle.getRoadID.return_value = 'D3'
            np.testing.assert_array_equal(env.exit_navigation(), [0, 0])

    def test_navigation_checkpoint_profile_and_shape_must_agree(self):
        model = SimpleNamespace(antwerp_env_version=ANTWERP_ENV_VERSION,
                                antwerp_task=exit_config.TASK_ID,
                                gamma=exit_config.POTENTIAL_GAMMA,
                                exit_potential_beta=exit_config.POTENTIAL_BETA,
                                exit_potential_gamma=exit_config.POTENTIAL_GAMMA,
                                exit_reward_profile=exit_config.REWARD_PROFILE,
                                exit_control_profile=exit_config.CONTROL_PROFILE,
                                exit_observation_profile=exit_config.NAV_OBSERVATION_PROFILE,
                                observation_space=gym.spaces.Box(-5000, 5000, (136,), np.float32),
                                policy_kwargs={'features_extractor_kwargs': {'nav_dim': 2}})
        self.assertTrue(validate_checkpoint_setup(model, 'full', False,
                         task=exit_config.TASK_ID)['nav_features'])
        model.observation_space = gym.spaces.Box(-5000, 5000, (134,), np.float32)
        with self.assertRaisesRegex(ValueError, 'profile and shape disagree'):
            validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)

    def test_forge_request_window_omits_fraction_without_requests(self):
        callback = EpisodeRewardCallback()
        callback.num_timesteps = 1000
        callback.model = SimpleNamespace(_n_updates=900, logger=SimpleNamespace(name_to_value={}))
        callback.locals = {'rewards': [0.0], 'dones': [False],
                           'infos': [{'stage_name': 'exit', 'exit_actor_request_direction': 0}]}
        with patch('train_sac_delay.wandb.log') as log:
            callback._on_step()
        record = log.call_args[0][0]
        self.assertEqual(record['train/toward_request_count_1k'], 0)
        self.assertNotIn('train/away_request_fraction_1k', record)

    def test_antwerp_environment_version_pins_completion_tolerance_contract(self):
        self.assertEqual(ANTWERP_ENV_VERSION, 3)

    def test_exit_forge_metrics_share_one_environment_step_axis(self):
        callback = EpisodeRewardCallback()
        callback.num_timesteps = 100
        callback.model = SimpleNamespace(logger=SimpleNamespace(name_to_value={
            'train/actor_loss': -3.0, 'train/critic_loss': 1.0,
            'train/ent_coef': 0.8, 'train/ent_coef_loss': -0.2,
        }))
        callback.locals = {
            'rewards': np.array([1.0]), 'dones': np.array([True]),
            'infos': [{'message': 'Arrived', 'stage_name': 'exit',
                       'stage_reached': True, 'stage_reward_paid': True,
                       'merged': False, 'ego_teleported': False,
                       'is_success': True, 'ego_start_lane': 2,
                       'initial_potential': 0.9, 'potential_reward': -0.1}],
        }
        with patch('train_sac_delay.wandb.log') as log:
            self.assertTrue(callback._on_step())
        records = [call.args[0] for call in log.call_args_list]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record['env_steps'] == 100 for record in records))
        self.assertTrue(all(not any(key.startswith('monitor/') for key in record)
                            for record in records))
        self.assertEqual(records[0]['train/success'], 1)
        self.assertEqual(records[1]['train/actor_loss'], -3.0)

    def test_e76_phase_selects_designated_forge_and_d_group_profile(self):
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', 'easy_1m']):
            args = parse_train_args()
        self.assertEqual((args.task, args.scenario, args.total_timesteps),
                         (exit_config.TASK_ID, 'easy', 1_000_000))
        self.assertEqual((args.project, args.entity),
                         (exit_config.WANDB_PROJECT, exit_config.WANDB_ENTITY))
        self.assertEqual(args.project, 'DAROM-OUT')
        self.assertEqual((args.checkpoint_freq, args.device, args.encoder_mode),
                         (250_000, 'cuda', 'GRU'))
        self.assertTrue(args.save_replay_buffer)

    def test_exp005_phase_changes_only_discount_and_schedule(self):
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase',
                                'exp005_gamma0999_200k', '--run_name',
                                'E76D3-GAMMA0999-S42-200K']):
            args = parse_train_args()
        self.assertEqual(args.run_name, 'E76D3-GAMMA0999-S42-200K')
        self.assertEqual((args.task, args.scenario, args.seed),
                         (exit_config.TASK_ID, 'easy', 42))
        self.assertEqual((args.total_timesteps, args.checkpoint_freq),
                         (200_000, 100_000))
        self.assertEqual(args.gamma, 0.999)
        self.assertFalse(args.use_safety)
        self.assertEqual((args.encoder_mode, args.gru_input_mode, args.delay_mode),
                         ('GRU', 'paper_raw', 'uniform'))

    def test_nav_phase_is_200k_fresh_gamma099_profile(self):
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', 'nav_200k']):
            args = parse_train_args()
        self.assertTrue(args.nav_features)
        self.assertEqual((args.total_timesteps, args.checkpoint_freq, args.gamma),
                         (200_000, 100_000, 0.99))
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', 'nav_200k',
                                '--load_path', 'old.zip', '--replay_buffer_path', 'old.pkl']):
            with self.assertRaises(SystemExit):
                parse_train_args()

    def test_control_134_phase_matches_nav_schedule_except_navigation(self):
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', 'nav_200k']):
            nav = parse_train_args()
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase', 'control_134_200k']):
            control = parse_train_args()
        for field in ('task', 'scenario', 'seed', 'radius', 'total_timesteps',
                      'checkpoint_freq', 'gamma', 'use_safety', 'encoder_mode',
                      'gru_input_mode', 'delay_mode', 'ablation', 'torch_threads'):
            self.assertEqual(getattr(nav, field), getattr(control, field), field)
        self.assertTrue(nav.nav_features)
        self.assertFalse(control.nav_features)
        env = DelayWrapper(Merging(seed=42, task=exit_config.TASK_ID,
                                   scenario='easy'), max_delay=20, mode='all',
                           nav_features=control.nav_features)
        self.assertEqual(env.observation_space.shape, (134,))
        with patch('sys.argv', ['train_sac_delay.py', '--exit_phase',
                                'control_134_200k', '--load_path', 'old.zip',
                                '--replay_buffer_path', 'old.pkl']):
            with self.assertRaises(SystemExit):
                parse_train_args()

    def test_each_route_changes_only_the_ego_trip(self):
        routes = Path(__file__).resolve().parents[1] / 'sumo_new'
        for scenario in ('easy', 'medium', 'hard'):
            original = ET.parse(routes / f'Antwerp_{scenario}.rou.xml').getroot()
            background = [(node.tag, node.attrib) for node in original if node.get('id') != 'ego']
            for task, (origin, destination, _) in ANTWERP_TASKS.items():
                if task == 'legacy_o3_d1':
                    continue
                with self.subTest(scenario=scenario, task=task):
                    variant = ET.parse(routes / f'Antwerp_{scenario}_{task}.rou.xml').getroot()
                    self.assertEqual([(node.tag, node.attrib) for node in variant
                                      if node.get('id') != 'ego'], background)
                    ego = variant.find("trip[@id='ego']")
                    self.assertEqual((ego.get('from'), ego.get('to')), (origin, destination))
                    if task == exit_config.TASK_ID:
                        self.assertEqual(ego.get('departLane'), 'random')

    def test_e76_exit_potential_distinguishes_start_lanes(self):
        scale = 1260.74
        distance = 1062.95
        values = [exit_config.potential(distance, scale, lane) for lane in range(5)]
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))
        self.assertAlmostEqual(values[0], 10 * (1 - distance / scale))
        self.assertAlmostEqual(values[4], values[0] / 5)
        self.assertEqual(exit_config.potential(-1, scale, 0), 0)
        self.assertEqual(exit_config.potential(float('nan'), scale, 0), 0)
        self.assertEqual(exit_config.potential(3.53, float('nan'), 0), 0)

    def test_e76_exit_invalid_route_distance_has_no_potential(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        env.ego_on_road = True
        env.potential_distance_scale = 1260.74
        fake = Mock()
        fake.vehicle.getRoadID.return_value = 'E77'
        fake.vehicle.getLaneIndex.return_value = 1
        with patch('src.merging.traci', fake):
            for invalid in (-1.0, float('nan'), float('inf')):
                fake.vehicle.getDrivingDistance.return_value = invalid
                self.assertEqual(env._exit_potential(), 0.0)
            fake.vehicle.getDrivingDistance.return_value = 3.53
            self.assertAlmostEqual(env._exit_potential(),
                                   exit_config.potential(3.53, 1260.74, 1))
            fake.vehicle.getRoadID.return_value = 'D3'
            self.assertEqual(env._exit_potential(), exit_config.potential(0, 1260.74, 0))

    def test_j105_straight_lane_geometry_exposes_raw_index_change(self):
        network = ET.parse(Path(__file__).resolve().parents[1]
                           / 'sumo_new/Antwerp.net.xml').getroot()
        lanes = {lane.get('id'): lane for lane in network.iter('lane')}
        for lane in range(5):
            e76_y = float(lanes[f'E76_{lane}'].get('shape').split()[0].split(',')[1])
            e77_y = float(lanes[f'E77_{lane + 1}'].get('shape').split()[0].split(',')[1])
            self.assertAlmostEqual(e76_y, e77_y)

    def test_j105_navigation_targets_match_network_connections(self):
        network = ET.parse(Path(__file__).resolve().parents[1]
                           / 'sumo_new/Antwerp.net.xml').getroot()
        lanes = {lane.get('id'): lane for lane in network.iter('lane')}
        distance_scale = sum(float(lanes[f'{edge}_0'].get('length'))
                             for edge in exit_config.DISTANCE_SCALE_EDGES)
        self.assertAlmostEqual(distance_scale, exit_config.NAV_DISTANCE_SCALE_METERS)
        connections = {node.get('via'): int(node.get('toLane'))
                       for node in network.iter('connection')
                       if node.get('from') == 'E76' and node.get('to') == 'E77'}
        self.assertEqual(connections, exit_config.J105_TARGET_LANES)

    def test_e76_exit_checkpoint_requires_potential_profile(self):
        model = SimpleNamespace(antwerp_env_version=ANTWERP_ENV_VERSION,
                                antwerp_task=exit_config.TASK_ID,
                                gamma=exit_config.POTENTIAL_GAMMA,
                                exit_potential_beta=exit_config.POTENTIAL_BETA,
                                exit_potential_gamma=exit_config.POTENTIAL_GAMMA,
                                exit_reward_profile=exit_config.REWARD_PROFILE,
                                exit_control_profile=exit_config.CONTROL_PROFILE,
                                observation_space=gym.spaces.Box(-5000, 5000, (134,), np.float32),
                                policy_kwargs={'features_extractor_kwargs': {}})
        validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)
        model.exit_potential_beta = 5
        with self.assertRaisesRegex(ValueError, 'potential configuration'):
            validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)
        model.exit_potential_beta = exit_config.POTENTIAL_BETA
        model.exit_reward_profile = None
        with self.assertRaisesRegex(ValueError, 'reward profile'):
            validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)
        model.exit_reward_profile = exit_config.REWARD_PROFILE
        del model.exit_control_profile
        with self.assertRaisesRegex(ValueError, 'control profile'):
            validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)
        model.exit_control_profile = exit_config.CONTROL_PROFILE
        model.exit_potential_gamma = 0.999
        with self.assertRaisesRegex(ValueError, 'gamma do not match'):
            validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)

    def test_e77_lane_change_requires_room_before_junction(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        env.lane_name = 'E77'
        env.lane_idx = 2
        fake = Mock()
        fake.edge.getLaneNumber.return_value = 6
        fake.vehicle.getLaneID.return_value = 'E77_2'
        fake.lane.getLength.return_value = 172.35
        fake.vehicle.getLength.return_value = 5.0
        fake.vehicle.getSpeed.return_value = 10.0
        fake.simulation.getTime.return_value = 100.0
        with patch('src.merging.traci', fake):
            fake.vehicle.getLanePosition.return_value = 100.0
            env._apply_lane_change(-1)
            fake.vehicle.changeLane.assert_called_once_with(
                'ego', 1, exit_config.LANE_CHANGE_TIME_SECONDS)
            fake.vehicle.changeLane.reset_mock()
            env._apply_lane_change(1)
            fake.vehicle.changeLane.assert_not_called()
            env.exit_maneuver = None
            fake.vehicle.getLanePosition.return_value = 140.0
            env._apply_lane_change(-1)
            fake.vehicle.changeLane.assert_not_called()

    def test_exit_maneuver_stays_latched_across_j105_until_stable(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        env.exit_maneuver = {'edge': 'E76', 'target': 2, 'start_time': 100.0,
                             'last_progress_time': 100.0, 'stable_steps': 0}
        fake = Mock()
        fake.vehicle.getLaneID.return_value = ':J105_0_3'
        fake.vehicle.getWidth.return_value = 1.8
        fake.lane.getWidth.return_value = 3.2
        fake.vehicle.getLateralLanePosition.return_value = 0.0
        fake.vehicle.getLateralSpeed.return_value = 0.0
        with patch('src.merging.traci', fake):
            env.lane_name, env.lane_idx = ':J105_0', 3
            fake.simulation.getTime.return_value = 102.0
            env._update_exit_maneuver()
            self.assertIsNotNone(env.exit_maneuver)
            self.assertEqual(env.exit_maneuver['stable_steps'], 1)
            env.lane_name = 'E77'
            fake.vehicle.getLaneID.return_value = 'E77_3'
            for time in (102.1, 102.2):
                fake.simulation.getTime.return_value = time
                env._update_exit_maneuver()
            self.assertIsNone(env.exit_maneuver)

    def test_exit_lateral_settling_uses_only_millimetre_boundary_tolerance(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        fake = Mock()
        fake.vehicle.getLaneID.return_value = 'E76_3'
        fake.vehicle.getWidth.return_value = 1.8
        fake.lane.getWidth.return_value = 3.2
        fake.vehicle.getLateralSpeed.return_value = 0.0
        with patch('src.merging.traci', fake):
            fake.vehicle.getLateralLanePosition.return_value = 0.7005
            self.assertTrue(env._exit_laterally_settled())
            fake.vehicle.getLateralLanePosition.return_value = 0.7011
            self.assertFalse(env._exit_laterally_settled())
            fake.vehicle.getLateralLanePosition.return_value = 0.7005
            fake.vehicle.getLateralSpeed.return_value = 0.051
            self.assertFalse(env._exit_laterally_settled())

    def test_exit_maneuver_tolerance_still_requires_target_lane(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        env.lane_name, env.lane_idx = 'E76', 2
        env.exit_maneuver = {'edge': 'E76', 'target': 1, 'start_time': 100.0,
                             'last_progress_time': 100.0, 'stable_steps': 0}
        fake = Mock()
        fake.vehicle.getLaneID.return_value = 'E76_2'
        fake.vehicle.getWidth.return_value = 1.8
        fake.lane.getWidth.return_value = 3.2
        fake.vehicle.getLateralLanePosition.return_value = 0.7005
        fake.vehicle.getLateralSpeed.return_value = 0.0
        fake.simulation.getTime.return_value = 100.1
        with patch('src.merging.traci', fake):
            env._update_exit_maneuver()
        self.assertEqual(env.exit_maneuver['stable_steps'], 0)

    def test_e77_legal_maneuver_gets_bounded_completion_window(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.exit_maneuver = {'edge': 'E77', 'target': 0, 'start_time': 100.0,
                             'last_progress_time': 102.0, 'stable_steps': 0}
        fake = Mock()
        with patch('src.merging.traci', fake):
            fake.simulation.getTime.return_value = 103.0
            self.assertTrue(env._e77_maneuver_can_finish())
            fake.simulation.getTime.return_value = 104.0
            self.assertFalse(env._e77_maneuver_can_finish())
            env.exit_maneuver['last_progress_time'] = 105.0
            fake.simulation.getTime.return_value = 106.0
            self.assertFalse(env._e77_maneuver_can_finish())

    def test_exit_maneuver_stays_latched_across_j106(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        env.exit_maneuver = {'edge': 'E77', 'target': 0, 'start_time': 100.0,
                             'last_progress_time': 100.0, 'stable_steps': 0}
        fake = Mock()
        fake.vehicle.getLaneID.return_value = ':J106_0_0'
        fake.vehicle.getWidth.return_value = 1.8
        fake.vehicle.getLateralLanePosition.return_value = 1.0
        fake.vehicle.getLateralSpeed.return_value = -1.0
        fake.lane.getWidth.return_value = 3.2
        with patch('src.merging.traci', fake):
            env.lane_name, env.lane_idx = ':J106_0', 0
            fake.simulation.getTime.return_value = 102.0
            env._update_exit_maneuver()
            self.assertIsNotNone(env.exit_maneuver)
            env.lane_name = 'D3'
            fake.vehicle.getLaneID.return_value = 'D3_0'
            fake.vehicle.getLateralLanePosition.return_value = 0.0
            fake.vehicle.getLateralSpeed.return_value = 0.0
            for time in (102.1, 102.2, 102.3):
                fake.simulation.getTime.return_value = time
                env._update_exit_maneuver()
            self.assertIsNone(env.exit_maneuver)
            fake.vehicle.getSpeed.return_value = 0.0
            fake.vehicle.getLanePosition.return_value = 172.35
            env._apply_lane_change(-1)
            fake.vehicle.changeLane.assert_not_called()

    def test_e77_lane_end_misses_exit_even_after_lane_index_changes_to_zero(self):
        env = Merging(seed=1, task=exit_config.TASK_ID)
        env.ego_id = 'ego'
        env.ego_on_road = True
        env.potential_distance_scale = 1260.74
        env.potential_previous = 5.0
        env.given_merge_reward = False
        env.parsed_action = [0, -1]
        env.action = [0, -5]
        env.counter = 1
        env.state = np.zeros((31, 3), np.float32)
        env.observation = env.state.copy()
        fake = Mock()
        fake.vehicle.getIDList.return_value = ('ego',)
        fake.vehicle.getRoadID.return_value = 'E77'
        fake.vehicle.getLaneID.return_value = 'E77_0'
        fake.vehicle.getLaneIndex.return_value = 0
        fake.vehicle.getLanePosition.return_value = 172.35
        fake.vehicle.getWidth.return_value = 1.8
        fake.vehicle.getLateralLanePosition.return_value = 1.55
        fake.vehicle.getLateralSpeed.return_value = -1.0
        fake.lane.getLength.return_value = 172.35
        fake.lane.getWidth.return_value = 3.2
        for name in ('getStartingTeleportIDList', 'getEndingTeleportIDList',
                     'getCollidingVehiclesIDList', 'getArrivedIDList',
                     'getEmergencyStoppingVehiclesIDList'):
            getattr(fake.simulation, name).return_value = ()
        with patch('src.merging.traci', fake), patch('src.merging.safe_gap', return_value=(True, True)), \
                patch.object(env, '_get_obs', return_value=env.state.copy()):
            _, reward, terminated, truncated, info = env.collect_step(True, close_on_done=False)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['message'], 'Missed Exit')
        self.assertFalse(info['stage_reached'])
        self.assertFalse(info['is_success'])
        self.assertEqual(info['reward_components']['missed_exit'], -100.0)
        self.assertEqual(info['reward_components']['arrival'], 0.0)
        self.assertEqual(info['reward_components']['stage'], 0.0)
        self.assertEqual(info['potential_reward'], -5.0)
        self.assertLess(reward, -100.0)

    def test_e76_exit_does_not_penalize_lane_command(self):
        fake = Mock()
        fake.simulation.getCollidingVehiclesIDList.return_value = ()
        fake.simulation.getArrivedIDList.return_value = ()
        fake.simulation.getEmergencyStoppingVehiclesIDList.return_value = ()
        with patch('src.merging.traci', fake), patch('src.merging.safe_gap', return_value=(True, True)):
            for task, expected in ((exit_config.TASK_ID, 0.0), ('entry', -0.5)):
                with self.subTest(task=task):
                    env = Merging(seed=1, task=task)
                    env.ego_id = 'ego'
                    env.ego_on_road = True
                    env.counter = 0
                    env.last_acc = 0
                    env.state = np.zeros((31, 3), dtype=np.float32)
                    env.observation = env.state.copy()
                    env.parsed_action = [0, 0]
                    neutral = env.getReward()
                    env.parsed_action = [0, -1]
                    changing = env.getReward()
                    self.assertAlmostEqual(changing - neutral, expected)
                    self.assertAlmostEqual(env.reward_components['lane_command'], expected)

    def test_exit_stage_bonus_paid_once_and_arrival_remains_separate(self):
        env = Merging(seed=1, task='exit')
        env.ego_id = 'ego'
        env.ego_on_road = True
        env.parsed_action = [0, 0]
        env.last_acc = 0
        env.counter = 0
        env.state = np.zeros((31, 3), dtype=np.float32)
        env.observation = env.state.copy()
        env.stage_reached = True
        env.given_merge_reward = False
        fake = Mock()
        fake.simulation.getCollidingVehiclesIDList.return_value = ()
        fake.simulation.getArrivedIDList.return_value = ()
        fake.simulation.getEmergencyStoppingVehiclesIDList.return_value = ()
        with patch('src.merging.traci', fake), patch('src.merging.safe_gap', return_value=(True, True)):
            first = env.getReward()
            self.assertAlmostEqual(env.last_stage_reward, 50)
            second = env.getReward()
            self.assertAlmostEqual(first - second, 50)
            self.assertEqual(env.last_stage_reward, 0)
            fake.simulation.getArrivedIDList.return_value = ('ego',)
            arrived = env.getReward()
            self.assertAlmostEqual(arrived - second, 50)

    def test_exit_stage_requires_entering_d3_during_step(self):
        env = Merging(seed=1, task='exit')
        env.ego_id = 'ego'
        env.ego_on_road = True
        env.counter = 0
        env.last_acc = 0
        env.given_merge_reward = False
        env.state = np.zeros((31, 3), dtype=np.float32)
        env.observation = env.state.copy()
        fake = Mock()
        fake.vehicle.getIDList.return_value = ('ego',)
        fake.vehicle.getSpeed.return_value = 10
        fake.vehicle.getRoadID.side_effect = ('E77', 'E77', 'E77', 'D3', 'D3', 'D3')
        fake.vehicle.getLaneID.side_effect = ('E77_0', 'E77_0', 'E77_0', 'D3_0',
                                              'D3_0', 'D3_0')
        for name in ('getStartingTeleportIDList', 'getEndingTeleportIDList',
                     'getCollidingVehiclesIDList', 'getArrivedIDList',
                     'getEmergencyStoppingVehiclesIDList'):
            getattr(fake.simulation, name).return_value = ()
        with patch('src.merging.traci', fake), patch('src.merging.safe_gap', return_value=(True, True)), \
                patch.object(env, '_get_obs', return_value=env.state.copy()):
            before = env.step([0, 0])[4]
            at_exit = env.step([0, 0])[4]
            later = env.step([0, 0])[4]
        self.assertFalse(before['stage_reached'])
        self.assertEqual(before['stage_reward'], 0)
        self.assertTrue(at_exit['stage_reached'])
        self.assertFalse(at_exit['merged'])
        self.assertEqual(at_exit['stage_reward'], 50)
        self.assertEqual(later['stage_reward'], 0)

    def test_exit_checkpoint_cannot_be_evaluated_as_entry(self):
        model = SimpleNamespace(antwerp_env_version=ANTWERP_ENV_VERSION,
                                antwerp_task='exit')
        self.assertEqual(parse_eval_args(['model.zip', '--task', 'exit']).task, 'exit')
        with self.assertRaisesRegex(ValueError, 'Checkpoint task'):
            validate_checkpoint_setup(model, 'full', False, task='entry')
        self.assertFalse(validate_checkpoint_setup(model, 'full', False, task='exit')
                         ['evaluation_safety'])

    def test_exit_timeout_is_reported_as_exit_not_merge(self):
        model = Mock()
        model.predict.return_value = (np.zeros(2), None)
        env = Mock()
        env.reset.return_value = (np.zeros(134), {'task_id': 'exit'})
        env.step.return_value = (None, -10, False, True,
                                 {'message': 'Max Steps', 'ego_teleported': False,
                                  'stage_reached': True, 'merged': False})
        result = evaluate(model, env, 1, 1, task='exit')
        self.assertEqual(result['Exit Reached Timeout'], 1)
        self.assertNotIn('Merged Timeout', result)
        self.assertNotIn('no_merge_rate', result)
        self.assertEqual(result['timeout_rate'], 100)

    def test_e76_exit_timeout_uses_exit_labels(self):
        model = Mock()
        model.predict.return_value = (np.zeros(2), None)
        env = Mock()
        env.reset.return_value = (np.zeros(134), {'task_id': exit_config.TASK_ID})
        env.step.return_value = (None, -1, False, True,
                                 {'message': 'Max Steps', 'ego_teleported': False,
                                  'stage_reached': False, 'merged': False})
        result = evaluate(model, env, 1, 1, task=exit_config.TASK_ID)
        self.assertEqual(result['No Exit Timeout'], 1)
        self.assertNotIn('no_merge_rate', result)


def snapshot(time=0.0, ego=(0, 0), angle=90, positions=((5, 3.2),),
             velocities=((0, 0),)):
    points = np.zeros((30, 2))
    vectors = np.zeros((30, 2))
    valid = np.zeros(30, dtype=bool)
    points[:len(positions)] = positions
    vectors[:len(positions)] = velocities
    valid[:len(positions)] = True
    return TrafficSnapshot(time, np.asarray(ego), angle, points, vectors, valid)


class GeometryTests(unittest.TestCase):
    def test_route_guidance_includes_all_d1_connected_lanes(self):
        from src.diagnostics import d1_reachable_lanes
        reachable = d1_reachable_lanes()
        self.assertEqual(reachable['E78'], [2, 3, 4])
        self.assertEqual(reachable['E82'], [3, 4, 5])
        self.assertEqual(reachable['E77'], [3, 4, 5])

    def test_velocity_projection_for_crossing_and_opposing_traffic(self):
        env = Merging(seed=1)
        env.ego_id = 'ego'
        fake = Mock()
        fake.vehicle.getIDList.return_value = ('ego', 'same', 'crossing', 'opposing')
        fake.vehicle.getPosition.side_effect = lambda vehicle: {
            'ego': (0, 0), 'same': (5, 0), 'crossing': (10, 0), 'opposing': (15, 0),
        }[vehicle]
        fake.vehicle.getSpeed.return_value = 10
        fake.vehicle.getAngle.side_effect = lambda vehicle: {
            'ego': 90, 'same': 90, 'crossing': 0, 'opposing': 270,
        }[vehicle]
        fake.vehicle.getDistance.return_value = 0
        fake.vehicle.getLaneIndex.return_value = 0
        fake.simulation.getTime.return_value = 5
        with patch('src.merging.traci', fake):
            observation = env._get_obs()
        neighbors = observation[1:4]
        np.testing.assert_allclose(neighbors[np.argsort(neighbors[:, 0]), 2], [0, -10, -20])
        self.assertEqual(np.count_nonzero(env.observation_snapshot.valid), 3)

    def test_lateral_move_reveals_close_leader(self):
        captured = snapshot()
        current = snapshot(time=1, ego=(0, 3.2))
        aligned = align_delayed_neighbors(captured, current, [0, 1, 0])
        np.testing.assert_allclose(aligned[1], [5, 0, 0], atol=1e-6)
        action, details = safetyCheck([0, 0], aligned, False, 'all', (0, 0))
        self.assertFalse(details['safe'])
        self.assertEqual(action, [-5, 0])
        np.testing.assert_array_equal(aligned[2:], 0)

    def test_rotation_translation_and_prediction_use_only_captured_neighbors(self):
        captured = snapshot(positions=((10, 5),), velocities=((0, 2),))
        # Current neighbor data is deliberately unrelated; only ego pose is usable.
        current = snapshot(time=2, ego=(10, 1), angle=0,
                           positions=((999, 999),), velocities=((999, 999),))
        aligned = align_delayed_neighbors(captured, current, [20, 2, 3])
        np.testing.assert_allclose(aligned[1], [8, 0, -1], atol=1e-6)
        np.testing.assert_allclose(captured.positions[0], [10, 5])

    def test_predict_all_neighbors_before_selecting_closest(self):
        captured = snapshot(positions=((10, 0), (20, 0)), velocities=((0, 0), (-15, 0)))
        aligned = align_delayed_neighbors(captured, snapshot(time=1), [0, 0, 0])
        _, details = safetyCheck([0, 0], aligned, False, 'all', (0, 0))
        self.assertEqual(details['front_gap'], 5)
        self.assertEqual(details['front_approach_speed'], 15)


class ManagedDiagnosticTests(unittest.TestCase):
    def setUp(self):
        from src.diagnostics import ManagedController
        self.controller = ManagedController()
        self.base = SimpleNamespace(ego_id='ego', ego_on_road=True)
        self.fake = Mock()
        vehicle = self.fake.vehicle
        vehicle.getRoadID.return_value = 'E75'
        vehicle.getLaneIndex.return_value = 0
        vehicle.getLaneID.return_value = 'E75_0'
        vehicle.getWidth.return_value = 1.8
        self.fake.lane.getWidth.return_value = 3.2
        vehicle.getSpeed.return_value = 10
        vehicle.getLateralLanePosition.return_value = 0
        vehicle.getLateralSpeed.return_value = 0
        vehicle.getNeighbors.return_value = ()
        vehicle.getDecel.return_value = 4.5
        vehicle.getSecureGap.return_value = 8
        vehicle.getFollowSpeed.return_value = 10

    def test_overlap_or_fast_follower_prevents_start(self):
        for speed, gap in [(10, -0.5), (20, 20)]:
            with self.subTest(speed=speed, gap=gap), patch('src.diagnostics.traci', self.fake):
                self.fake.vehicle.getNeighbors.side_effect = lambda ego, mode: (('rear', gap),) if mode == 0 else ()
                self.fake.vehicle.getSpeed.side_effect = lambda name: speed if name == 'rear' else 10
                action, details = self.controller(self.base)
                self.assertEqual(action[1], 0)
                self.assertEqual(details['phase'], 'waiting_gap')
                self.assertEqual(details['block_reason'], 'rear')
                self.assertIsNone(self.controller.target)

    def test_lane_index_crossing_does_not_start_second_maneuver(self):
        with patch('src.diagnostics.traci', self.fake):
            action, _ = self.controller(self.base)
            self.assertEqual(action[1], 5)
            self.fake.vehicle.getLaneIndex.return_value = 1
            self.fake.vehicle.getLateralLanePosition.return_value = -1.4
            self.fake.vehicle.getLateralSpeed.return_value = 0.8
            action, details = self.controller(self.base)
            self.assertEqual(action[1], 0)
            self.assertEqual(details['target_lane'], 1)
            self.assertEqual(details['phase'], 'maneuver')
            self.fake.vehicle.getLateralLanePosition.return_value = -0.6995
            self.fake.vehicle.getLateralSpeed.return_value = 0
            action, details = self.controller(self.base)
            self.assertEqual(action[1], 5)
            self.assertEqual(details['target_lane'], 2)

    def test_yield_changes_speed_without_relaxing_gap_guard(self):
        self.controller.gap_control = 'yield'
        self.fake.vehicle.getNeighbors.side_effect = lambda ego, mode: (('rear', 1),) if mode == 0 else ()
        with patch('src.diagnostics.traci', self.fake):
            action, details = self.controller(self.base)
        self.assertEqual(action[1], 0)
        self.assertEqual(action[0], -1.5)
        self.assertEqual(details['phase'], 'waiting_gap')
        self.assertIsNone(self.controller.target)

    def test_edge_transition_and_teleport_clear_latched_target(self):
        with patch('src.diagnostics.traci', self.fake):
            self.controller(self.base)
            self.fake.vehicle.getRoadID.return_value = 'E76'
            self.fake.vehicle.getLaneIndex.return_value = 2
            action, _ = self.controller(self.base)
            self.assertEqual(action[1], 0)
            self.assertIsNone(self.controller.target)
            self.base.ego_on_road = False
            action, _ = self.controller(self.base)
            np.testing.assert_array_equal(action, [0, 0])
            self.assertIsNone(self.controller.edge)


class EvaluationProtocolTests(unittest.TestCase):
    def model(self, training_safety=False):
        return SimpleNamespace(antwerp_env_version=ANTWERP_ENV_VERSION,
                               ablation='full', training_safety=training_safety,
                               seed=7, num_timesteps=20000,
                               predict=Mock(return_value=(np.zeros(2), None)))

    def test_safety_evaluation_is_independent_without_relabeling_training(self):
        for training in (False, True):
            for evaluation in (False, True):
                model = self.model(training)
                before = vars(model).copy()
                setup = validate_checkpoint_setup(model, 'full', evaluation)
                self.assertEqual(setup, {'training_safety': training, 'evaluation_safety': evaluation})
                self.assertEqual(vars(model), before)
        model = SimpleNamespace(antwerp_env_version=ANTWERP_ENV_VERSION)
        self.assertIsNone(validate_checkpoint_setup(model, 'full', True)['training_safety'])

    def test_incompatible_policies_rejected_before_creating_simulation(self):
        for attribute, value in [('antwerp_env_version', -1), ('ablation', 'delayed_state_only')]:
            for safety in (False, True):
                model = self.model()
                setattr(model, attribute, value)
                with patch('evaluate_sac_delay.make_env') as make, self.assertRaises(ValueError):
                    evaluate_runs(model, 'hard', 'uniform', safety, 'full', 1, [10])
                make.assert_not_called()

    def test_repeat_aggregation_keeps_episode_seeds_and_training_provenance(self):
        envs = [Mock(), Mock()]
        for index, env in enumerate(envs):
            env.reset.return_value = (np.zeros(134), {'sumo_seed': 100 + index,
                                                     'ego_speed_mode': 31,
                                                     'sumo_version': [20, 'SUMO 1.18.0']})
        envs[0].step.side_effect = [
            (None, 10, True, False, {'message': 'Arrived', 'ego_teleported': False})
        ] * 2
        envs[1].step.side_effect = [
            (None, -10, True, False, {'message': 'Collision', 'ego_teleported': False}),
            (None, -10, False, True, {'message': 'Max Steps', 'ego_teleported': False, 'merged': True}),
        ]
        with patch('evaluate_sac_delay.make_env', side_effect=envs) as make:
            report = evaluate_runs(self.model(), 'hard', 'uniform', True, 'full', 2, [10, 20])
        self.assertEqual(report['aggregate']['success_rate']['mean'], 50)
        self.assertAlmostEqual(report['aggregate']['success_rate']['std'], np.sqrt(5000))
        self.assertAlmostEqual(report['aggregate']['success_rate']['ci95_half_width'], 98)
        self.assertEqual(report['aggregate']['mean_return']['mean'], 0)
        self.assertAlmostEqual(report['aggregate']['mean_return']['ci95_half_width'], 19.6)
        single = summarize_runs(report['runs'][:1])
        self.assertIsNone(single['success_rate']['std'])
        self.assertIsNone(single['success_rate']['ci95_half_width'])
        self.assertFalse(report['training_safety'])
        self.assertTrue(report['evaluation_safety'])
        self.assertEqual(report['total_episodes'], 4)
        self.assertEqual(report['runs'][1]['Merged Timeout'], 1)
        self.assertEqual([entry['seed'] for run in report['runs'] for entry in run['episode_results']],
                         [10, 11, 20, 21])
        self.assertEqual([entry['reset_info']['sumo_seed']
                          for run in report['runs'] for entry in run['episode_results']],
                         [100, 100, 101, 101])
        self.assertEqual(report['runs'][1]['episode_results'][0]['reset_info']['ego_speed_mode'], 31)
        for env in envs:
            env.close.assert_called_once()
        self.assertEqual([call.args[3] for call in make.call_args_list], [True, True])
        json.dumps(report, allow_nan=False)

    def test_failed_evaluation_closes_environment(self):
        model, env = self.model(), Mock()
        env.reset.return_value = (np.zeros(134), {})
        model.predict.side_effect = RuntimeError('prediction failed')
        with patch('evaluate_sac_delay.make_env', return_value=env), self.assertRaises(RuntimeError):
            evaluate_runs(model, 'hard', 'uniform', True, 'full', 1, [10])
        env.close.assert_called_once()

    def test_run_seed_ranges_must_be_disjoint(self):
        validate_seed_ranges(500, [42, 542, 1042])
        for episodes, seeds in [(0, [1]), (1, []), (1, [-1]), (1, [2, 2]), (500, [42, 43])]:
            with self.subTest(episodes=episodes, seeds=seeds), self.assertRaises(ValueError):
                validate_seed_ranges(episodes, seeds)

    def test_cli_explicit_safety_and_repeated_runs(self):
        args = parse_eval_args(['model.zip', '--safety', '--seeds', '10', '20', '--episodes', '2'])
        self.assertTrue(args.safety)
        self.assertEqual(args.seeds, [10, 20])
        self.assertFalse(parse_eval_args(['model.zip', '--no-safety']).safety)
        for flags in [ ['--safety', '--no-safety'], ['--seeds', '10', '11', '--episodes', '2'],
                       ['--output', 'README.md'] ]:
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                parse_eval_args(['model.zip', *flags])

    def test_training_budget_matches_paper_and_can_be_overridden_for_smoke_runs(self):
        from train_sac_delay import parse_args
        with patch('sys.argv', ['train_sac_delay.py']):
            args = parse_args()
        self.assertEqual(args.total_timesteps, 7_000_000)
        self.assertEqual(args.gru_input_mode, 'paper_raw')
        self.assertEqual(args.checkpoint_freq, 2_000_000)
        self.assertFalse(args.use_safety)
        self.assertEqual(args.task, 'legacy_o3_d1')
        with patch('sys.argv', ['train_sac_delay.py', '--total_timesteps', '20000']):
            self.assertEqual(parse_args().total_timesteps, 20000)
        with patch('sys.argv', ['train_sac_delay.py', '--task', 'exit']):
            self.assertEqual(parse_args().task, 'exit')

    def test_unknown_outcome_is_not_silently_excluded_from_rates(self):
        env = Mock()
        env.reset.return_value = (np.zeros(134), {})
        env.step.return_value = (None, 0, True, False, {'message': 'Unknown'})
        with self.assertRaisesRegex(ValueError, 'Unexpected episode outcome'):
            evaluate(self.model(), env, 1, 10)

    def test_missed_exit_is_counted_separately_from_timeout(self):
        env = Mock()
        env.reset.return_value = (np.zeros(134), {})
        env.step.return_value = (None, -100, True, False,
                                 {'message': 'Missed Exit', 'ego_teleported': False,
                                  'stage_reached': False})
        report = evaluate(self.model(), env, 1, 10, task=exit_config.TASK_ID)
        self.assertEqual(report['Missed Exit'], 1)
        self.assertEqual(report['missed_exit_rate'], 100)
        self.assertEqual(report['timeout_rate'], 0)
        self.assertEqual(report['success_rate'], 0)


class PacketEnv(gym.Env):
    """A moving ego with deliberately changing live neighbor information."""
    observation_space = gym.spaces.Box(-5000, 5000, (31, 3), np.float32)
    action_space = gym.spaces.Box(-5, 5, (2,), np.float32)
    ego_id = 'ego'
    ego_on_road = True

    def reset(self, **kwargs):
        self.time = 0.0
        self.observation_snapshot = snapshot()
        self.observation = np.zeros((31, 3), np.float32)
        self.observation[0] = [0, 0, 0]
        self.observation[1] = [5, 3.2, 0]
        self.action = np.zeros(2, np.float32)
        return self.observation, {}

    def step(self, action):
        self.time = round(self.time + 0.1, 6)
        self.action = np.asarray(action)
        self.observation_snapshot = snapshot(self.time, ego=(0, 3.2),
                                             positions=((100, 100),))
        self.observation = np.zeros((31, 3), np.float32)
        self.observation[0] = [self.time, 1, 0]
        self.observation[1] = [100, 96.8, 0]
        return self.observation, 0.0, False, False, {}

    def exit_navigation(self):
        return np.array([self.time / 10, 0.4], np.float32)


class DelayedSafetyTests(unittest.TestCase):
    def test_navigation_is_live_even_when_neighbor_packet_is_delayed(self):
        delay = DelayWrapper(PacketEnv(), max_delay=20, mode='all', nav_features=True)
        initial, _ = delay.reset()
        self.assertEqual(initial.shape, (136,))
        with patch.object(delay, '_sample_delay', return_value=20):
            later, *_ = delay.step([0, 0])
        self.assertEqual(later.shape, (136,))
        self.assertAlmostEqual(later[93], 0.01)
        self.assertAlmostEqual(later[94], 0.4)
        self.assertEqual(later[4], 3.2)

    def test_masked_policy_still_gets_safety_from_matching_old_packet(self):
        for ablation in ABLATION_MODES:
            with self.subTest(ablation=ablation):
                base = PacketEnv()
                delay = DelayWrapper(base, max_delay=20, mode='all', ablation=ablation)
                initial, _ = delay.reset()
                captured = delay.last_snapshot
                with patch.object(delay, '_sample_delay', return_value=20):
                    observation, *_ = delay.step([0, 0])
                self.assertIs(delay.last_snapshot, captured)
                self.assertEqual(delay.delay_of_received_observation, 1)
                self.assertEqual(observation.shape, (134,))
                self.assertEqual(observation.dtype, np.float32)
                self.assertTrue(delay.observation_space.contains(observation))
                safe = SafetyWrapper(delay)
                with patch('src.wrapper.target_lane_line', return_value=(0, 0)):
                    self.assertEqual(safe.action([0, 0]), [-5, 0])
                # Current ego update has not overwritten the original reset observation.
                self.assertEqual(initial[0], 0)
                np.testing.assert_allclose(captured.ego_position, [0, 0])

    def test_out_of_order_packet_keeps_its_original_snapshot(self):
        delay = DelayWrapper(PacketEnv(), max_delay=20, mode='all')
        delay.reset()
        # The t=.1 packet arrives after a fresher t=.2 packet.
        with patch.object(delay, '_sample_delay', side_effect=[2, 0, 20]):
            delay.step([0, 0])
            delay.step([0, 0])
            selected = delay.last_snapshot
            delay.step([0, 0])
        self.assertAlmostEqual(selected.time, 0.2)
        self.assertIs(delay.last_snapshot, selected)
        self.assertEqual(delay.delay_of_received_observation, 1)

    def test_current_ego_update_does_not_mutate_env_state(self):
        base = PacketEnv()
        delay = DelayWrapper(base, max_delay=20, mode='all')
        delay.reset()
        with patch.object(delay, '_sample_delay', side_effect=[0, 20]):
            delay.step([0, 0])
            previous_state = base.observation
            saved = previous_state.copy()
            delay.step([0, 0])
        np.testing.assert_array_equal(previous_state, saved)

    def test_no_safety_queries_while_ego_is_off_road(self):
        base = PacketEnv()
        delay = DelayWrapper(base, max_delay=20, mode='all')
        delay.reset()
        base.ego_on_road = False
        with patch('src.wrapper.target_lane_line') as target:
            np.testing.assert_array_equal(SafetyWrapper(delay).action([1, 2]), [1, 2])
            target.assert_not_called()


class TeleportTests(unittest.TestCase):
    def test_training_log_separates_raw_arrival_from_success(self):
        from train_sac_delay import EpisodeRewardCallback
        callback = EpisodeRewardCallback()
        callback.log_file = io.StringIO()
        callback.num_timesteps = 1
        callback.locals = {'rewards': [-0.03], 'dones': [True], 'infos': [{
            'message': 'Arrived', 'merged': True, 'ego_teleported': True, 'is_success': False,
        }]}
        with patch('train_sac_delay.wandb.log'):
            self.assertTrue(callback._on_step())
        entry = json.loads(callback.log_file.getvalue())
        self.assertEqual(entry['train/arrival'], 1)
        self.assertEqual(entry['train/success'], 0)
        self.assertEqual(entry['train/teleported_arrival'], 1)

    def make_env(self):
        env = Merging(seed=1)
        env.ego_id = 'ego'
        env.ego_on_road = True
        env.state = np.zeros((31, 3), np.float32)
        env.observation = env.state.copy()
        env.counter = 0
        env.given_merge_reward = False
        return env

    def test_buffer_reinsertion_and_later_arrival_cannot_earn_success(self):
        env = self.make_env()
        fake = Mock()
        phase = [-1]
        fake.simulationStep.side_effect = lambda: phase.__setitem__(0, phase[0] + 1)
        fake.simulation.getDepartedIDList.return_value = ()
        fake.simulation.getStartingTeleportIDList.side_effect = lambda: ('ego',) if phase[0] == 0 else ()
        fake.simulation.getEndingTeleportIDList.side_effect = lambda: ('ego',) if phase[0] == 2 else ()
        fake.simulation.getArrivedIDList.side_effect = lambda: ('ego',) if phase[0] == 3 else ()
        fake.simulation.getCollidingVehiclesIDList.return_value = ()
        fake.simulation.getEmergencyStoppingVehiclesIDList.return_value = ()
        fake.vehicle.getIDList.side_effect = lambda: ('ego',) if phase[0] == 2 else ()
        fake.vehicle.getLaneID.return_value = 'E76_0'
        fake.vehicle.getRoadID.return_value = 'E76'
        fake.vehicle.getLaneIndex.return_value = 0
        fake.vehicle.getSpeed.return_value = 0
        jumped = env.state.copy()
        jumped[0, 0] = 2000
        with patch('src.merging.traci', fake), patch.object(env, '_get_obs', return_value=jumped):
            _, reward, done, _, info = env.step([0, 0])
            self.assertTrue(info['ego_teleported'])
            self.assertFalse(done)
            self.assertLessEqual(reward, 0)
            fake.vehicle.reset_mock()
            env.step([0, 0])
            fake.vehicle.setAcceleration.assert_not_called()
            fake.vehicle.getLaneID.assert_not_called()
            _, reward, _, _, info = env.step([0, 0])
            self.assertTrue(info['teleport_ended'])
            self.assertTrue(info['ego_on_road'])
            self.assertLessEqual(reward, 0)  # Neither route jump nor merge bonus.
            _, reward, done, _, info = env.step([0, 0])
            self.assertEqual(info['message'], 'Arrived')
            self.assertTrue(done)
            self.assertFalse(info['is_success'])
            self.assertLessEqual(reward, 0)

    def test_raw_and_uninterrupted_arrival_rates_are_separate(self):
        model = Mock()
        model.predict.return_value = (np.zeros(2), None)
        env = Mock()
        env.reset.return_value = (np.zeros(134), {})
        env.step.side_effect = [
            (None, 50, True, False, {'message': 'Arrived', 'ego_teleported': False}),
            (None, 0, True, False, {'message': 'Arrived', 'ego_teleported': True}),
            (None, -100, False, True, {'message': 'Max Steps', 'ego_teleported': True, 'merged': True}),
        ]
        result = evaluate(model, env, 3, 1)
        self.assertEqual(result['Arrived'], 2)
        self.assertEqual(result['Uninterrupted Arrived'], 1)
        self.assertEqual(result['Teleported Arrived'], 1)
        self.assertEqual(result['Ego Teleported'], 2)
        self.assertAlmostEqual(result['success_rate'], 100 / 3)
        self.assertAlmostEqual(result['raw_arrival_rate'], 200 / 3)
        self.assertEqual(result['Merged Timeout'], 1)

    def test_legacy_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'retrain'):
            validate_checkpoint_setup(SimpleNamespace(), 'full', False)
        validate_checkpoint_setup(SimpleNamespace(antwerp_env_version=ANTWERP_ENV_VERSION), 'full', False)


if __name__ == '__main__':
    unittest.main()
