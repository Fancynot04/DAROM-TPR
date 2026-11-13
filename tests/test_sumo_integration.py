"""Opt-in real SUMO checks: python -m tests.test_sumo_integration.

Tests explicitly distinguish controlled TraCI terminal fixtures from the
natural fixed-action Missed Exit trajectory. They validate accounting, not
policy performance, and never alter the formal route files.
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
                # SUMO 1.20 reports 32 after setSpeedMode(96).
                self.assertIn(info['ego_speed_mode'], (32, 96))
                self.assertEqual(traci.vehicle.getTypeID('ego'), 'ego')
                self.assertAlmostEqual(traci.vehicle.getSpeed('ego'), 10.0)
                self.assertAlmostEqual(traci.vehicle.getMaxSpeed('ego'), 120.0 / 3.6)
                self.assertAlmostEqual(traci.vehicle.getAccel('ego'), 5.0)
                self.assertAlmostEqual(traci.vehicle.getDecel('ego'), 5.0)
                self.assertAlmostEqual(base.observation[0, 2], traci.vehicle.getSpeed('ego'), places=5)
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
                sumo_emergency_steps = 0
                hard_braking_steps = 0
                for _ in range(125):
                    observation, reward, terminated, truncated, info = env.step([0.5, 0])
                    self.assertFalse(terminated or truncated)
                    self.assertTrue(env.observation_space.contains(observation))
                    self.assertEqual(reward, 0.0)
                    self.assertEqual(info['reward_components']['emergency_stopping'], 0.0)
                    sumo_emergency_steps += int(info['sumo_emergency_stopping_this_step'])
                    hard_braking_steps += int(info['hard_braking_this_step'])
                    self.assertEqual(info['sumo_emergency_stopping_steps'],
                                     sumo_emergency_steps)
                    self.assertEqual(info['hard_braking_steps'], hard_braking_steps)
                    self.assertEqual(info['hard_braking_threshold_mps2'], -4.5)
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
        # Controlled SUMO integration: place Ego at the real D3 route endpoint.
        env = Merging(seed=579071, scenario='easy')
        env.reset()
        _, reward, terminated, truncated, info = self.force_arrival(env)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['message'], 'Arrived')
        self.assertTrue(info['is_success'])
        self.assertFalse(info['ego_teleported'])
        self.assertEqual(reward, 50.0)
        self.assertEqual(info['reward_episode_summary']['_meta']['weighted_total'], 50.0)

    def test_controlled_collision_reward_and_flags(self):
        # Controlled SUMO integration: overlap a background vehicle with Ego.
        env = Merging(seed=43, scenario='easy')
        env.reset(seed=43)
        ego_lane = traci.vehicle.getLaneID('ego')
        ego_position = traci.vehicle.getLanePosition('ego')
        blocker = next(vehicle_id for vehicle_id in traci.vehicle.getIDList()
                       if vehicle_id != 'ego')
        traci.vehicle.moveTo(blocker, ego_lane, ego_position + 0.1)
        traci.vehicle.setSpeed(blocker, 0.0)
        _, reward, terminated, truncated, info = env.step([0, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['message'], 'Collision')
        self.assertEqual(reward, -100.0)
        self.assertEqual(info['reward_episode_summary']['_meta']['weighted_total'], -100.0)

    def test_natural_fixed_action_missed_exit(self):
        # Natural driving trajectory: no lane-change requests, no repositioning.
        env = Merging(seed=43, scenario='easy')
        env.reset(seed=43)
        for _ in range(1000):
            _, reward, terminated, truncated, info = env.step([5.0, 0.0])
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['message'], 'Missed Exit')
        self.assertEqual(reward, -100.0)
        self.assertEqual(info['reward_episode_summary']['_meta']['weighted_total'], -100.0)

    def test_controlled_timeout_at_exact_control_step_and_tracker_reset(self):
        # Controlled SUMO integration: hold Ego in place without changing formal scenes.
        env = Merging(seed=43, scenario='easy')
        env.reset(seed=43)
        for control_step in range(1, 4001):
            was_on_road = env.prepare_step([0.0, 0.0])
            traci.vehicle.setSpeed('ego', 0.0)
            traci.simulationStep()
            _, reward, terminated, truncated, info = env.collect_step(
                was_on_road, close_on_done=False)
            if control_step == 3999:
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(reward, 0.0)
        self.assertEqual(control_step, 4000)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['message'], 'Timeout')
        self.assertEqual(reward, -100.0)
        self.assertEqual(info['reward_episode_summary']['_meta']['steps'], 4000)
        self.assertEqual(info['reward_episode_summary']['_meta']['weighted_total'], -100.0)
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            env.step([0, 0])
        env.reset(seed=43)
        self.assertEqual(env.reward_tracker.summary()['_meta']['steps'], 0)
        env.close()

if __name__ == '__main__':
    unittest.main()
