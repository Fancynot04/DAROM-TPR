"""Contract tests for the single-agent E76-to-D3 environment."""

import inspect
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

from src import exit_e76_d3_config as exit_config
from src import exit_progress_reward
from src.merging import ANTWERP_TASKS, Merging, check_ego_speed, limit_ego_acceleration
from train_sac_delay import parse_args


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / 'sumo_new'


class E76D3ContractTests(unittest.TestCase):
    def test_ego_speed_limiter_preserves_braking_and_caps_acceleration(self):
        vmax = exit_config.ROAD_SPEED_LIMIT_MPS
        self.assertEqual(limit_ego_acceleration(5, 10, 0.1, vmax), 5)
        self.assertAlmostEqual(limit_ego_acceleration(5, vmax - 0.2, 0.1, vmax), 2)
        self.assertEqual(limit_ego_acceleration(-3, 20, 0.1, vmax), -3)
        self.assertEqual(limit_ego_acceleration(5, vmax + 1e-7, 0.1, vmax), 0)
        check_ego_speed(vmax + 1e-7, vmax, 20, 5, 0)
        with self.assertRaisesRegex(RuntimeError, 'control_step=20') as error:
            check_ego_speed(vmax + 0.01, vmax, 20, 5, 0)
        self.assertIn('sac_accel_command_mps2=5', str(error.exception))
        self.assertIn('executed_accel_mps2=0', str(error.exception))
        with self.assertRaisesRegex(RuntimeError, 'previous_executed_accel_mps2=-3'):
            check_ego_speed(vmax + 0.01, vmax, 21, 5, -3,
                            phase='before_step')

    def test_only_exit_task_is_registered(self):
        self.assertEqual(ANTWERP_TASKS, {
            exit_config.TASK_ID: ('E76', 'D3', 'exit'),
        })

    def test_training_entry_is_fresh_and_single_task(self):
        with patch.object(sys, 'argv', [
                'train_sac_delay.py', '--seed', '43', '--scenario', 'hard',
                '--total_timesteps', '17', '--checkpoint_freq', '9', '--wandb_mode', 'disabled']):
            args = parse_args()
        self.assertEqual(args.task, exit_config.TASK_ID)
        self.assertEqual(args.scenario, 'hard')
        with patch.object(sys, 'argv', [
                'train_sac_delay.py', '--load_path', 'old.zip']):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_current_routes_have_the_fixed_ego_contract(self):
        for scenario in ('easy', 'medium', 'hard'):
            route = ROUTES / f'Antwerp_{scenario}_exit_e76_d3.rou.xml'
            self.assertTrue(route.is_file())
            ego = [trip for trip in ET.parse(route).getroot().findall('trip')
                   if trip.attrib.get('id') == 'ego']
            self.assertEqual(len(ego), 1)
            self.assertEqual(ego[0].attrib['from'], 'E76')
            self.assertEqual(ego[0].attrib['to'], 'D3')
            self.assertEqual(ego[0].attrib['departLane'], 'random')
            self.assertEqual(ego[0].attrib['speedFactor'], '1.0')

    def test_net_contains_120_kmh_corridor_limit(self):
        root = ET.parse(ROUTES / 'Antwerp.net.xml').getroot()
        expected = exit_config.ROAD_SPEED_LIMIT_MPS
        ids = ('E76_0', 'E76_4', 'E77_0', 'E77_5', 'D3_0',
               ':J105_0_0', ':J105_0_5', ':J106_0_0', ':J106_1_4')
        lanes = {lane.attrib['id']: float(lane.attrib['speed'])
                 for lane in root.iter('lane')}
        for lane_id in ids:
            self.assertAlmostEqual(lanes[lane_id], expected, places=8)

    def test_merging_does_not_reset_speed_in_code(self):
        source = inspect.getsource(Merging)
        self.assertNotIn('setMaxSpeed', source)
        self.assertNotIn('vehicle.setSpeed(', source)

    def test_pending_off_submits_repeated_requests(self):
        env = Merging(seed=42, scenario='easy', task=exit_config.TASK_ID,
                      reward_config=exit_progress_reward.load_experiment())
        env.ego_on_road = True
        env.ego_id = 'ego'
        env.lane_name = 'E77'
        env.lane_idx = 1
        env._connection = Mock()
        env._connection.edge.getLaneNumber.return_value = 6
        env._connection.simulation.getTime.return_value = 123.0
        env._apply_lane_change(-1)
        env._apply_lane_change(-1)
        self.assertEqual(env._connection.vehicle.changeLane.call_count, 2)
        self.assertEqual(env._connection.vehicle.changeLane.call_args.args,
                         ('ego', 0, exit_config.LANE_CHANGE_TIME_SECONDS))

    def test_fixed_episode_contract(self):
        env = Merging(seed=42, scenario='medium', task=exit_config.TASK_ID)
        self.assertEqual(env.origin, 'E76')
        self.assertEqual(env.destination, 'D3')
        self.assertEqual(env.episode_limit_steps, 4000)

    def test_training_defaults_match_final_baseline(self):
        self.assertEqual(exit_config.DEFAULT_SCENARIO, 'easy')
        self.assertEqual(exit_config.DEFAULT_SEED, 42)
        self.assertEqual(exit_config.DEFAULT_GAMMA, 0.999)
        self.assertEqual(exit_config.DEFAULT_TOTAL_TIMESTEPS, 200_000)
        self.assertEqual(exit_config.DEFAULT_CHECKPOINT_FREQ, 200_000)


if __name__ == '__main__':
    unittest.main()
