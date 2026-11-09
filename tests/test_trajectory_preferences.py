import unittest
from scripts.compare_trajectory_preferences import Trajectory, choose

def make(name, seed=1, outcome='Arrived', danger=0, hard=0, away=0, steps=100):
    rows = [{'ego_road_id': 'E76', 'ego_lane_index': '2', 'ego_lane_pos_m': '1', 'ego_speed_mps': '20', 'route_remaining_to_d3_end_m': '900', 'ttc_leader_s': '', 'ttc_follower_s': '', 'jerk_measured_mps3': '', 'min_lane_changes_required': '1', 'route_state_settled': '1', 'lane_change_physical': '0'} for _ in range(steps)]
    for i in range(danger): rows[i]['ttc_leader_s'] = '1'
    for i in range(away): rows[i].update(lane_change_physical='1', ego_lane_index_after='3')
    return Trajectory(name, {'trajectory_id': name, 'eval_seed': str(seed), 'scenario': 'easy', 'terminal_reason': outcome, 'hard_braking_steps': str(hard)}, rows)

class PreferenceRuleTest(unittest.TestCase):
    def test_priority_success_then_safety_then_route_then_time(self):
        self.assertEqual(choose(make('a'), make('b', outcome='Collision'))[1:3], ('A', 'P1'))
        self.assertEqual(choose(make('a', danger=0), make('b', danger=6))[1:3], ('A', 'P2'))
        self.assertEqual(choose(make('a', away=0), make('b', away=1))[1:3], ('A', 'P3'))
        self.assertEqual(choose(make('a', steps=100), make('b', steps=160))[1:3], ('A', 'P4'))
    def test_unresolved_initial_mismatch(self):
        a, b = make('a'), make('b'); b.rows[0]['ego_lane_index'] = '1'
        self.assertEqual(choose(a, b)[1], 'unresolved')
