"""Mock-SUMO unit test: recorder mechanics only (no TraCI, no real route semantics)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.route_urgency import Lane, RouteNetwork
from src.trajectory_recorder import OBS_DIM, TrajectoryRecorder, ttc


class FakeVehicle:
    def __init__(self):
        self.t = 0
        self.fail_follower = False

    def getRoadID(self, e): return 'E76'
    def getLaneID(self, e): return 'E76_2'
    def getLaneIndex(self, e): return 2
    def getLanePosition(self, e): return 10.0 + self.t
    def getPosition(self, e): return (100.0 + self.t, 5.0)
    def getAngle(self, e): return 90.0
    def getLateralLanePosition(self, e): return 0.0
    def getLateralSpeed(self, e): return 0.0
    def getDistance(self, e): return 10.0 * self.t
    def getSpeed(self, e): return 20.0 if e == 'ego' else 15.0
    def getAcceleration(self, e): return 0.1 * self.t
    def getDrivingDistance(self, e, edge, pos): return 500.0
    def getMinGap(self, e): return 2.5
    def getLeader(self, e, d): return ('lead', 20.0)
    def getFollower(self, e, d):
        if self.fail_follower:
            raise RuntimeError('boom')
        return ('', -1.0)


class Fake:
    def __init__(self):
        self.vehicle = FakeVehicle()
        self.simulation = SimpleNamespace(getTime=lambda: 120.0 + 0.1 * self.vehicle.t)


class FakeBase:
    ego_id = 'ego'
    ego_teleporting = False
    exit_maneuver = None

    def __init__(self):
        self.traci = Fake()
        self.ego_on_road = True
        self.lane_name, self.observation = 'E76', np.zeros((31, 3), np.float32)

    def _remaining_to_d3_end(self): return 900.0


NET = RouteNetwork({'E76': {0: Lane(500, 33.3), 1: Lane(500, 33.3), 2: Lane(500, 33.3)}},
                   {})
REWARD = {'terms': {'success': {}, 'collision': {}, 'timeout': {}}}


def make(tmp):
    return TrajectoryRecorder(
        Path(tmp) / 'traj', model_label='m', checkpoint='c.zip', checkpoint_sha256='ab' * 32,
        training_seed=42, scenario='easy', reward_config=REWARD, route_network=NET,
        lane_change_edges={'E76'}, parse_action=lambda a: [float(a[0]), 0],
        config={'x': 1})


def info(terminal=False, on_road=True, message=''):
    return {'message': message, 'ego_on_road': on_road, 'action': np.array([1.0, 0.0], np.float32),
            'exit_actor_request_direction': 0, 'exit_applied_lane_change_direction': 0,
            'reward_terms': [{'name': 'success', 'raw': 50.0, 'weighted': 50.0, 'active': True}]
            if terminal else [], 'teleport_started': False, 'teleport_ended': False,
            'ego_teleported': False, 'hard_braking_this_step': False,
            'sumo_emergency_stopping_this_step': False, 'exit_maneuver_active': False}


class RecorderTest(unittest.TestCase):
    def test_arrived_terminal_row_kept_and_after_state_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, base = make(tmp), FakeBase()
            rec.begin_episode(run_seed=1, episode_index=0, eval_seed=1, reset_info={})
            obs = np.zeros(OBS_DIM, np.float32)
            for k in range(3):
                action = np.array([1.0, 0.0], np.float32)
                base.traci.vehicle.t = k
                pre = rec.capture_pre(base, obs, action)
                nxt = obs + 1
                last = k == 2
                if last:
                    base.ego_on_road = False
                rec.record_step(pre, base, nxt, 50.0 if last else 0.0, last, False,
                                info(last, not last, 'Arrived' if last else ''))
                obs = nxt
            rec.end_episode('Arrived', 3)
            rec.close()
            with (Path(tmp) / 'traj/trajectory_steps.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            with (Path(tmp) / 'traj/trajectory_transitions.csv').open() as handle:
                trans = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]['terminal_reason'], 'Arrived')
            self.assertEqual(rows[-1]['post_state_source'], 'unavailable')
            self.assertEqual(rows[-1]['ego_speed_after_mps'], '')
            self.assertEqual(rows[-1]['ego_removed_from_sim'], '1')
            self.assertEqual(rows[0]['jerk_measured_mps3'], '')       # step 0 has no jerk
            self.assertAlmostEqual(float(rows[1]['jerk_measured_mps3']), 0.0, places=6)
            self.assertEqual(rows[0]['ttc_leader_status'], 'ok')       # gap 22.5, closing 5
            self.assertAlmostEqual(float(rows[0]['ttc_leader_s']), 22.5 / 5, places=6)
            self.assertEqual(rows[0]['ttc_follower_status'], 'no_follower')
            self.assertEqual(trans[-1]['next_obs_valid'], '0')
            self.assertEqual(len([k for k in trans[0] if k.startswith('obs_')]), 134)
            manifest = json.loads((Path(tmp) / 'traj/trajectory_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'ok')
            self.assertEqual(manifest['episodes'][0]['chain_breaks'], 0)

    def test_diagnostic_failure_leaves_blank_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, base = make(tmp), FakeBase()
            base.traci.vehicle.fail_follower = True
            rec.begin_episode(run_seed=1, episode_index=0, eval_seed=1, reset_info={})
            pre = rec.capture_pre(base, np.zeros(OBS_DIM, np.float32), np.zeros(2, np.float32))
            self.assertEqual(pre['row']['ttc_follower_status'], 'unavailable')
            self.assertGreater(rec.errors['follower'], 0)
            rec.close()

    def test_refuses_existing_directory_and_wrong_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp).close()
            with self.assertRaises(FileExistsError):
                make(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            rec = make(tmp)
            rec.begin_episode(run_seed=1, episode_index=0, eval_seed=1, reset_info={})
            with self.assertRaises(ValueError):
                rec.capture_pre(FakeBase(), np.zeros(10, np.float32), np.zeros(2, np.float32))
            rec.close()

    def test_ttc_rules(self):
        self.assertEqual(ttc(10, 0.05), (None, 'not_closing'))
        self.assertEqual(ttc(-1, 5), (None, 'overlap'))
        self.assertEqual(ttc(None, 5), (None, 'unavailable'))
        self.assertEqual(ttc(10, 5), (2.0, 'ok'))


if __name__ == '__main__':
    unittest.main()
