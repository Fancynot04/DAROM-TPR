import math
import unittest

from src.merging import ANTWERP_DIR, LANE_CHANGE_EDGES
from src.rewards.config import load_hydra_config, load_reward_file, validate_reward_config
from src.rewards.registry import RewardContext, settle_terms
from src.rewards.secrm_efficiency import efficiency
from src.rewards.secrm_route import executable_distance, route_penalty
from src.rewards.terms import RewardTracker
from src.route_urgency import load_route_network


class DenseRewardProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_route_network(ANTWERP_DIR / 'Antwerp.net.xml')
        cls.route = ('E76', 'E77', 'D3')

    def test_hydra_profiles_and_enabled_combinations(self):
        b0 = load_reward_file('conf/reward/e76_d3.yaml')[0]
        self.assertEqual(b0['name'], 'events_only_v1')
        for name, expected in (('e76_d3_b1_efficiency', (True, False)),
                               ('e76_d3_b2_route', (False, True)),
                               ('e76_d3_b3_combined', (True, True))):
            _, config, _, _ = load_hydra_config(overrides=[f'reward={name}'])
            self.assertEqual(config['name'], 'e76_d3_dense_v1')
            self.assertEqual((config['terms']['efficiency']['enabled'],
                              config['terms']['route']['enabled']), expected)

    def test_invalid_dense_configuration_is_rejected(self):
        _, config, _, _ = load_hydra_config(overrides=['reward=e76_d3_b1_efficiency'])
        config['terms']['route']['distance_scale_m'] = 0
        with self.assertRaises(ValueError):
            validate_reward_config(config)

    def test_b0_terms_remain_identical(self):
        b0 = load_reward_file('conf/reward/e76_d3.yaml')[0]
        self.assertEqual(sum(term.weighted for term in settle_terms('events_only_v1', RewardContext(None, b0))), 0.0)
        self.assertEqual(sum(term.weighted for term in settle_terms('events_only_v1', RewardContext('Arrived', b0))), 50.0)

    def test_efficiency_formula_edge_cases(self):
        raw, info = efficiency(20, 33.333333, 0.1, epsilon=2.0)
        self.assertEqual(info['constraint_status'], 'no_leader')
        self.assertLess(raw, 0)
        normal, _ = efficiency(20, 33.333333, 0.1, gap=30, leader_speed=18,
                               ego_decel=5, leader_decel=5, stopped_gap=2, epsilon=2)
        low, _ = efficiency(1, 33.333333, 0.1, gap=5, leader_speed=0,
                            ego_decel=5, leader_decel=5, stopped_gap=2, epsilon=2)
        self.assertTrue(math.isfinite(normal))
        self.assertTrue(math.isfinite(low))
        infeasible, details = efficiency(30, 33.333333, 0.1, gap=0, leader_speed=0,
                                         ego_decel=5, leader_decel=5, stopped_gap=2, epsilon=2)
        self.assertEqual(details['target_speed'], 0.0)
        self.assertLessEqual(infeasible, 0.0)
        with self.assertRaises(ValueError):
            efficiency(float('nan'), 33.333333, 0.1, epsilon=2)

    def test_dense_clipping_and_terminal_exclusion(self):
        _, config, _, _ = load_hydra_config(overrides=['reward=e76_d3_b3_combined'])
        terms = {term.name: term for term in settle_terms(
            'e76_d3_dense_v1', RewardContext(None, config, efficiency_raw=-7, route_raw=-0.2))}
        self.assertEqual(terms['efficiency'].raw, -7.0)
        self.assertEqual(terms['efficiency'].clipped_value, -1.0)
        self.assertTrue(terms['efficiency'].clipped)
        self.assertAlmostEqual(terms['efficiency'].weighted, -0.005)
        terminal = {term.name: term for term in settle_terms(
            'e76_d3_dense_v1', RewardContext('Collision', config, efficiency_raw=-1, route_raw=-1))}
        self.assertNotIn('efficiency', terminal)
        self.assertNotIn('route', terminal)
        self.assertEqual(sum(term.weighted for term in terminal.values()), -100.0)

    def test_route_zero_unknown_and_late_window_are_distinct(self):
        required, distance = executable_distance(self.network, self.route, 0, 0, 0.0, LANE_CHANGE_EDGES)
        self.assertEqual((required, distance), (0, 0.0))
        required, far = executable_distance(self.network, self.route, 0, 2, 0.0, LANE_CHANGE_EDGES)
        self.assertEqual(required, 2)
        _, late = executable_distance(self.network, self.route, 1, 2, 150.0, LANE_CHANGE_EDGES)
        self.assertLess(late, far)
        self.assertLessEqual(route_penalty(required, late, 200.0), route_penalty(required, far, 200.0))
        _, config, _, _ = load_hydra_config(overrides=['reward=e76_d3_b2_route'])
        route = {term.name: term for term in settle_terms(
            'e76_d3_dense_v1', RewardContext(None, config, efficiency_raw=-0.1, route_raw=None))}['route']
        self.assertFalse(route.active)
        self.assertEqual(route.weighted, 0.0)

    def test_j105_connector_preserves_the_zero_deficit_lane_mapping(self):
        # Dense payment is unavailable on internal lanes, but topology must
        # still map E76_0 through J105 to E77_0 rather than inventing a jump.
        connector = self.network.connections[('E76', 'E77', 0)]
        self.assertIn((0, ':J105_0_0'), connector)
        required, _ = executable_distance(self.network, self.route, 0, 0, 10.0,
                                          LANE_CHANGE_EDGES)
        self.assertEqual(required, 0)

    def test_tracker_accepts_event_only_terminal_after_dense_steps(self):
        _, config, _, _ = load_hydra_config(overrides=['reward=e76_d3_b3_combined'])
        tracker = RewardTracker(0.999)
        dense = settle_terms('e76_d3_dense_v1', RewardContext(None, config,
                                                               efficiency_raw=-0.5, route_raw=-0.2))
        tracker.add(dense, sum(term.weighted for term in dense))
        terminal = settle_terms('e76_d3_dense_v1', RewardContext('Missed Exit', config))
        tracker.add(terminal, sum(term.weighted for term in terminal))
        summary = tracker.summary(terminated=True)
        self.assertAlmostEqual(summary['_meta']['weighted_total'], -100.0035)
        self.assertEqual(summary['efficiency']['active_ratio'], 0.5)
        self.assertEqual(summary['route']['active_ratio'], 0.5)
        required, _ = executable_distance(self.network, self.route, 1, 0, 10.0,
                                          LANE_CHANGE_EDGES)
        self.assertEqual(required, 0)


if __name__ == '__main__':
    unittest.main()
