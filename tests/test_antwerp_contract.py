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

    def test_e76_exit_checkpoint_requires_potential_profile(self):
        model = SimpleNamespace(antwerp_env_version=ANTWERP_ENV_VERSION,
                                antwerp_task=exit_config.TASK_ID,
                                exit_potential_beta=exit_config.POTENTIAL_BETA)
        validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)
        model.exit_potential_beta = 5
        with self.assertRaisesRegex(ValueError, 'potential configuration'):
            validate_checkpoint_setup(model, 'full', False, task=exit_config.TASK_ID)

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


class DelayedSafetyTests(unittest.TestCase):
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
