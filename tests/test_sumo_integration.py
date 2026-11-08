"""Opt-in real SUMO checks: python -m tests.test_sumo_integration.

The arrival/teleport fixtures reposition ego and shorten the teleport timeout
only inside these tests. They validate accounting, not driving performance.
"""

import socket
import hashlib
import json
from pathlib import Path
import unittest

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
                self.assertGreaterEqual(info['ego_depart_time'], 120.1)
                self.assertEqual(traci.vehicle.getLaneChangeMode('ego'), 0)
                self.assertEqual(info['ego_lane_change_mode'], 0)
                self.assertEqual(info['ego_speed_mode'], traci.vehicle.getSpeedMode('ego'))
                self.assertEqual(info['sumo_version'], list(traci.getVersion()))
                self.assertEqual(info['step_length'], 0.1)
                self.assertEqual(info['episode_limit_steps'], 4000)
                self.assertEqual(info['ego_route'][0], 'E76')
                self.assertEqual(info['ego_route'][-1], 'D3')
                self.assertEqual(info['scene_files']['routes']['path'],
                                 f'sumo_new/Antwerp_{scenario}_exit_e76_d3.rou.xml')
                self.assertAlmostEqual(traci.lane.getMaxSpeed('E76_0'), 120.0 / 3.6, places=5)
                for scene_file in info['scene_files'].values():
                    path = Path(__file__).resolve().parents[1] / scene_file['path']
                    self.assertEqual(scene_file['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                command = info['sumo_command']
                self.assertEqual(int(command[command.index('--seed') + 1]), info['sumo_seed'])
                json.dumps(info, allow_nan=False)
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
        traci.vehicle.moveTo('ego', 'D3_0', traci.lane.getLength('D3_0') - 0.1)
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

if __name__ == '__main__':
    unittest.main()
