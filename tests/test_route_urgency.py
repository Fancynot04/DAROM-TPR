"""Route urgency follows lane connections rather than fixed edge names."""

import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from evaluate_sac_delay import diagnostic_reward_without_potential, evaluate
from src import exit_e76_d3_config as exit_config
from src.merging import ANTWERP_DIR, LANE_CHANGE_EDGES, Merging
from src.route_urgency import (load_route_network, observe_route_urgency,
                               route_urgency)


class RouteUrgencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_route_network(ANTWERP_DIR / 'Antwerp.net.xml')

    def estimate(self, route, lane, position=0.0, speed=13.89,
                 allowed=LANE_CHANGE_EDGES, edge_index=0):
        return route_urgency(self.network, route, edge_index, lane, position,
                             speed, 4.5, allowed)

    def test_o1_origin_uses_future_changeable_edges(self):
        route = ('O1', 'E74', 'E75', 'E76', 'E77', 'D3')
        result = self.estimate(route, 2)
        self.assertEqual(result['status'], 'ok')
        self.assertGreater(result['urgency'], 0)
        self.assertEqual(result['minimum_lane_changes'], 4)
        self.assertEqual([segment['lane_changes'] for segment in result['plan'][:2]], [0, 0])
        self.assertTrue(all(segment['lane_changes'] == 0
                            or segment['edge'] in LANE_CHANGE_EDGES
                            for segment in result['plan']))
        for segment, next_edge in zip(result['plan'], route[1:]):
            self.assertIn(
                (segment['next_lane'], segment['connector']),
                self.network.connections[
                    (segment['edge'], next_edge, segment['exit_lane'])])

    def test_j105_mapping_is_a_connection_not_a_physical_change(self):
        result = self.estimate(('E76', 'E77', 'D3'), 3,
                               position=700.0)
        first = result['plan'][0]
        self.assertEqual(result['minimum_lane_changes'], 3)
        self.assertEqual(first['next_lane'], first['exit_lane'] + 1)
        self.assertEqual(result['selected_lane_changes'],
                         sum(segment['lane_changes'] for segment in result['plan']))

    def test_e77_distance_and_speed_change_urgency_without_nan(self):
        route = ('E77', 'D3')
        far = self.estimate(route, 2)
        near = self.estimate(route, 2, position=100)
        stopped = self.estimate(route, 2, position=100, speed=0)
        ready = self.estimate(route, 0, position=100)
        self.assertLess(far['urgency'], near['urgency'])
        self.assertGreater(near['urgency'], 1)
        self.assertGreater(stopped['urgency'], 0)
        self.assertEqual(stopped['urgency'],
                         self.estimate(route, 2, position=100, speed=1)['urgency'])
        self.assertEqual(ready['urgency'], 0)
        self.assertIsNone(ready['bottleneck_edge'])

    def test_unreachable_lane_is_reported_separately_from_zero_urgency(self):
        result = self.estimate(('O1', 'E74', 'E75', 'E76', 'E77', 'D3'),
                               2, allowed=())
        self.assertEqual(result['status'], 'unreachable')
        self.assertIsNone(result['urgency'])

    def test_internal_connector_projects_using_via_lane(self):
        vehicle = Mock()
        vehicle.getRoute.return_value = ('E76', 'E77', 'D3')
        vehicle.getRoadID.return_value = ':J105_0'
        vehicle.getLaneID.return_value = ':J105_0_2'
        vehicle.getRouteIndex.return_value = 0
        vehicle.getLateralSpeed.return_value = 0.0
        vehicle.getLateralLanePosition.return_value = 0.0
        vehicle.getWidth.return_value = 1.8
        vehicle.getLength.return_value = 4.5
        vehicle.getSpeed.return_value = 13.89
        lane = Mock()
        lane.getWidth.return_value = 3.2
        base = SimpleNamespace(ego_on_road=True, ego_id='ego',
                               traci=SimpleNamespace(vehicle=vehicle, lane=lane),
                               exit_maneuver=None)
        result = observe_route_urgency(base, self.network, LANE_CHANGE_EDGES)
        self.assertEqual(result['projected_edge'], 'E77')
        self.assertEqual(result['projected_lane'], 2)
        self.assertEqual(result['minimum_lane_changes'], 2)
        self.assertTrue(result['provisional'])
        self.assertFalse(result['valid_for_threshold'])

    def test_invalid_inputs_do_not_silently_become_safe(self):
        with self.assertRaises(ValueError):
            self.estimate(('E77', 'D3'), 2, speed=float('nan'))
        with self.assertRaises(ValueError):
            self.estimate(('E77', 'D3'), 2, position=-1)

    def test_reward_formula_is_bounded_and_rejects_invalid_values(self):
        for urgency, expected in ((0.0, 0.0), (0.15, 0.0),
                                  (0.5, -0.035), (1.0, -0.085),
                                  (1.65, -0.15), (5.0, -0.15)):
            with self.subTest(urgency=urgency):
                self.assertAlmostEqual(exit_config.urgency_penalty(urgency), expected)
        for invalid in (-1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                exit_config.urgency_penalty(invalid)

    def test_reward_waits_for_settled_lane_during_pending(self):
        env = Merging(seed=42, task=exit_config.TASK_ID,
                      exit_reward_profile=exit_config.URGENCY_REWARD_PROFILE)
        env.ego_on_road = True
        env.ego_id = 'ego'
        with patch('src.merging.observe_route_urgency', side_effect=[
                {'status': 'ok', 'urgency': 0.5, 'valid_for_threshold': True,
                 'projected_edge': 'E76', 'projected_lane': 2},
                {'status': 'ok', 'urgency': 0.0, 'valid_for_threshold': False,
                 'road': ':J105_0'},
                {'status': 'ok', 'urgency': 0.0, 'valid_for_threshold': True,
                 'projected_edge': 'E77', 'projected_lane': 0}]):
            self.assertEqual(env._reward_urgency(), (0.5, False))
            self.assertEqual(env._reward_urgency(), (0.5, True))
            self.assertEqual(env._reward_urgency(), (0.0, False))
        self.assertEqual(env.last_valid_urgency, 0.0)

    def test_pending_same_edge_uses_last_settled_lane_and_current_distance(self):
        env = Merging(seed=42, task=exit_config.TASK_ID,
                      exit_reward_profile=exit_config.URGENCY_REWARD_PROFILE)
        env.ego_on_road = True
        env.ego_id = 'ego'
        env.last_valid_urgency = 0.2
        env.last_settled_urgency_edge = 'E77'
        env.last_settled_urgency_lane = 2
        vehicle = Mock()
        vehicle.getRoute.return_value = ('E77', 'D3')
        vehicle.getRouteIndex.return_value = 0
        vehicle.getLanePosition.return_value = 100.0
        vehicle.getSpeed.return_value = 13.89
        vehicle.getLength.return_value = 4.5
        env._connection = SimpleNamespace(vehicle=vehicle)
        with patch('src.merging.observe_route_urgency', return_value={
                'status': 'ok', 'urgency': 0.0,
                'valid_for_threshold': False, 'road': 'E77'}):
            urgency, provisional = env._reward_urgency()
        expected = route_urgency(self.network, ('E77', 'D3'), 0, 2, 100.0,
                                 13.89, 4.5, LANE_CHANGE_EDGES)['urgency']
        self.assertTrue(provisional)
        self.assertAlmostEqual(urgency, expected)
        self.assertGreater(urgency, env.last_valid_urgency)

    def test_urgency_reward_profile_rejects_other_tasks(self):
        with self.assertRaises(ValueError):
            Merging(seed=42, task='exit',
                    exit_reward_profile=exit_config.URGENCY_REWARD_PROFILE)


class OptionalEvaluatorTraceTests(unittest.TestCase):
    def test_shaping_is_subtracted_only_from_diagnostic_reward(self):
        info = {'potential_reward': -0.4,
                'reward_components': {'potential': -0.4}}
        potential, unshaped = diagnostic_reward_without_potential(
            0.6, info, task='exit_e76_d3')
        self.assertAlmostEqual(potential, -0.4)
        self.assertAlmostEqual(unshaped, 1.0)
        self.assertAlmostEqual(0.6, unshaped + potential)
        with self.assertRaises(ValueError):
            diagnostic_reward_without_potential(
                0.6, {'potential_reward': -0.4,
                      'reward_components': {'potential': 0.4}},
                task='exit_e76_d3')

    def test_trace_does_not_change_episode_result(self):
        class Env:
            unwrapped = None

            def __init__(self):
                self.unwrapped = self

            def reset(self, seed):
                self.steps = 0
                return np.zeros(134, np.float32), {}

            def step(self, action):
                self.steps += 1
                return np.zeros(134, np.float32), 1.0, self.steps == 2, False, {
                    'message': 'Arrived' if self.steps == 2 else '',
                    'ego_teleported': False, 'merged': True,
                    'stage_reached': True, 'lane': 'D3_0',
                }

        model = Mock()
        model.predict.return_value = (np.zeros(2, np.float32), None)
        baseline = evaluate(model, Env(), 1, 200004, task='exit')
        trace = io.StringIO()
        with patch('evaluate_sac_delay.observe_route_urgency',
                   return_value={'status': 'ok', 'urgency': 0.5}):
            observed = evaluate(model, Env(), 1, 200004, task='exit',
                                urgency_trace=trace, route_network=object())
        self.assertEqual(observed['mean_return'], baseline['mean_return'])
        self.assertEqual(observed['success_rate'], baseline['success_rate'])
        self.assertEqual(observed['episode_results'][0]['return'],
                         baseline['episode_results'][0]['return'])
        self.assertEqual(observed['episode_results'][0]['steps'],
                         baseline['episode_results'][0]['steps'])
        rows = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]['outcome'], 'Arrived')
        self.assertEqual(rows[-1]['reward_without_potential'], 1.0)
        self.assertEqual(observed['episode_results'][0]['return_without_potential'], 2.0)


if __name__ == '__main__':
    unittest.main()
