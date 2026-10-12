"""Opt-in real SUMO checks: python -m tests.test_sumo_integration.

The arrival/teleport fixtures reposition ego and shorten the teleport timeout
only inside these tests. They validate accounting, not driving performance.
"""

import socket
import unittest
from unittest.mock import patch

import numpy as np

from src.geometry import align_delayed_neighbors
from src.merging import Merging, traci
from src.wrapper import DelayWrapper, SafetyWrapper


@unittest.skipUnless(__name__ == '__main__', 'run python -m tests.test_sumo_integration')
class SumoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fail promptly if local TraCI sockets are blocked by the sandbox.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))

    def tearDown(self):
        if traci.isLoaded():
            traci.close()

    def test_three_scenarios_and_delayed_safety(self):
        for scenario in ('easy', 'medium', 'hard'):
            with self.subTest(scenario=scenario):
                base = Merging(seed=579071, scenario=scenario)
                delay = DelayWrapper(base, max_delay=20, mode='all', seed=579071)
                env = SafetyWrapper(delay)
                observation, info = env.reset()
                self.assertAlmostEqual(info['ego_depart_time'], 120.1)
                self.assertEqual(traci.vehicle.getLaneChangeMode('ego'), 0)
                for _ in range(125):
                    observation, reward, terminated, truncated, info = env.step([0.5, 0])
                    self.assertFalse(terminated or truncated)
                    self.assertTrue(env.observation_space.contains(observation))
                    self.assertTrue(np.isfinite(reward))
                    self.assertFalse(info['ego_teleported'])
                    snapshot = base.observation_snapshot
                    aligned = align_delayed_neighbors(snapshot, snapshot, base.observation[0])
                    np.testing.assert_allclose(aligned, base.observation, atol=1e-5)
                    self.assertAlmostEqual(snapshot.time - delay.last_snapshot.time,
                                           0.1 * delay.delay_of_received_observation)
                    self.assertTrue(np.isfinite(delay.safety_observation()).all())
                env.close()

    def force_arrival(self, env):
        traci.vehicle.moveTo('ego', 'D1_0', traci.lane.getLength('D1_0') - 0.1)
        traci.vehicle.setSpeed('ego', 10)
        return env.step([0, 0])

    def test_clean_arrival_reward_and_flags(self):
        env = Merging(seed=579071, scenario='easy')
        env.reset()
        _, reward, terminated, truncated, info = self.force_arrival(env)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['message'], 'Arrived')
        self.assertTrue(info['is_success'])
        self.assertFalse(info['ego_teleported'])
        self.assertGreater(reward, 49)

    def test_real_wrong_lane_teleport_then_arrival_and_reset(self):
        env = Merging(seed=579071, scenario='easy')
        original_start = traci.start
        with patch('src.merging.traci.start', side_effect=lambda command: original_start(
                command + ['--time-to-teleport', '0.2'])):
            env.reset()
        traci.vehicle.moveTo('ego', 'E82_1', traci.lane.getLength('E82_1') - 0.1)
        traci.vehicle.setPreviousSpeed('ego', 0)
        traci.vehicle.setSpeed('ego', 0)
        # Repositioning is a fixture, so establish its physical state before driving.
        env.observation = env._get_obs()
        env.state = env.observation.copy()
        detected = False
        for _ in range(30):
            _, reward, terminated, truncated, info = env.step([0, 0])
            self.assertFalse(terminated or truncated)
            if info['teleport_started']:
                detected = True
                self.assertLessEqual(reward, 0)
            if detected and info['ego_on_road']:
                break
        self.assertTrue(detected, 'Fixture must produce an actual SUMO teleport')
        self.assertTrue(info['ego_on_road'])
        self.assertTrue(info['ego_teleported'])
        _, reward, terminated, _, info = self.force_arrival(env)
        self.assertTrue(terminated)
        self.assertEqual(info['message'], 'Arrived')
        self.assertFalse(info['is_success'])
        self.assertTrue(info['ego_teleported'])
        self.assertLessEqual(reward, 0)
        env.reset()
        self.assertFalse(env.ego_teleported)
        self.assertFalse(env.ego_teleporting)
        self.assertTrue(env.ego_on_road)


if __name__ == '__main__':
    unittest.main()
