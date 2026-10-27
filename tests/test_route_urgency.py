"""Route urgency follows lane connections rather than fixed edge names."""

import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from evaluate_sac_delay import diagnostic_reward_without_potential, evaluate
from src.merging import ANTWERP_DIR, LANE_CHANGE_EDGES
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
