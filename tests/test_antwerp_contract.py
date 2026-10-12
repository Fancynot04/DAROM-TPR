"""Regression checks for teleport accounting and delayed traffic geometry."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

from evaluate_sac_delay import evaluate, validate_checkpoint_setup
from src.geometry import TrafficSnapshot, align_delayed_neighbors
from src.merging import ANTWERP_ENV_VERSION, Merging
from src.safety_controller import safetyCheck
from src.wrapper import ABLATION_MODES, DelayWrapper, SafetyWrapper


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
