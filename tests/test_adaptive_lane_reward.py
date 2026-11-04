"""L1/L2 distance reward, event timing and replay compatibility."""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import exit_progress_reward as config, exit_e76_d3_config as exits, secrm_reward
from src.merging import Merging
from src.rewards.directional_lane import adaptive_lane_reward


class AdaptiveLaneRewardTests(unittest.TestCase):
    def test_formulas_and_boundaries(self):
        for d in (0., 100., 176., 500., 1067., 1500.):
            f = min(d, 1067.) / 1067.
            for direction, expected in [('toward', 3+2*f), ('away', -9+3*f)]:
                self.assertAlmostEqual(adaptive_lane_reward(direction,d,'l1')[0], expected)
            c=min(d,1067.)
            expected_t=(3+c/176 if c<=176 else 4+(c-176)/891)
            expected_a=(-9+c/176 if c<=176 else -8+2*(c-176)/891)
            self.assertAlmostEqual(adaptive_lane_reward('toward',d,'l2')[0],expected_t)
            self.assertAlmostEqual(adaptive_lane_reward('away',d,'l2')[0],expected_a)
        for direction in ('toward','away'):
            self.assertAlmostEqual(adaptive_lane_reward(direction,176-1e-7,'l2')[0],
                                   adaptive_lane_reward(direction,176+1e-7,'l2')[0],places=8)
        for d in (None, float('nan'), float('inf'), -1., -1073741824.):
            self.assertEqual(adaptive_lane_reward('toward',d,'l1'),(0.,'invalid_distance'))

    def make_env(self, mode='l1'):
        c=config.load_experiment(phase='secrm_efficiency_directional_'+mode)
        e=Merging(seed=42,scenario='easy',task=exits.TASK_ID,
                  exit_reward_profile=secrm_reward.ADAPTIVE_PROFILE,progress_experiment=c)
        e.ego_on_road=True;e.ego_id='ego';e.lane_name='E77';e.lane_idx=1
        e._connection=Mock();e._connection.edge.getLaneNumber.return_value=6
        e._connection.vehicle.getLateralSpeed.return_value=0.0
        e._connection.vehicle.getDrivingDistance.return_value=100.0
        e._connection.simulation.getTime.return_value=123.0
        return e,c

    def estimate(self,lane,n,pending=False,road='E77'):
        return {'status':'ok','valid_for_threshold':not pending,'pending':pending,
                'road':road,'projected_lane':lane,'minimum_lane_changes':n}

    def test_latched_request_distance_pending_and_once(self):
        e,c=self.make_env()
        with patch('src.merging.observe_route_urgency') as observe, patch.object(e,'_exit_change_has_room',return_value=True):
            observe.return_value=self.estimate(1,1);e._update_directional_lane_reward()
            e._apply_lane_change(-1);self.assertEqual(e.directional_request_distance,100.)
            e._connection.vehicle.getDrivingDistance.return_value=10.
            e._apply_lane_change(-1)
            self.assertEqual(e._connection.vehicle.changeLane.call_count,1)
            observe.return_value=self.estimate(0,0,pending=True);e.directional_seen_lateral_motion=True
            e._update_directional_lane_reward();self.assertEqual(e.directional_lane_reward,0.)
            e.exit_maneuver=None;observe.return_value=self.estimate(0,0)
            e._update_directional_lane_reward()
            self.assertAlmostEqual(e.directional_lane_reward,3+2*100/1067)
            self.assertEqual(e.directional_lane_event['request_distance_to_d3_start_m'],100.)
            self.assertEqual(e.directional_lane_event['distance_reward_status'],'ok')
            e._update_directional_lane_reward();self.assertEqual(e.directional_lane_reward,0.)
            self.assertIsNone(e.directional_lane_event)

    def test_rejected_and_failed_request_do_not_pay(self):
        e,c=self.make_env()
        with patch('src.merging.observe_route_urgency') as observe, patch.object(e,'_exit_change_has_room',return_value=False):
            observe.return_value=self.estimate(1,1);e._update_directional_lane_reward();e._apply_lane_change(-1)
            e._connection.vehicle.changeLane.assert_not_called();self.assertIsNone(e.directional_pending_request)
        with patch('src.merging.observe_route_urgency') as observe, patch.object(e,'_exit_change_has_room',return_value=True):
            observe.return_value=self.estimate(1,1);e._apply_lane_change(-1)
            e.exit_maneuver=None;e._update_directional_lane_reward()
            self.assertEqual(e.directional_lane_reward,0.)

    def test_invalid_distance_is_not_clamped_to_exit(self):
        e,c=self.make_env('l2');e._connection.vehicle.getDrivingDistance.return_value=-1073741824.
        with patch('src.merging.observe_route_urgency') as observe, patch.object(e,'_exit_change_has_room',return_value=True):
            observe.return_value=self.estimate(1,1);e._update_directional_lane_reward();e._apply_lane_change(-1)
            self.assertIsNone(e.directional_request_distance)
            e.directional_seen_lateral_motion=True;e.exit_maneuver=None;observe.return_value=self.estimate(0,0)
            e._update_directional_lane_reward();self.assertEqual(e.directional_lane_reward,0.)
            self.assertEqual(e.directional_lane_event['distance_reward_status'],'invalid_distance')
        e.lane_name='D3';self.assertIsNone(e._remaining_to_d3_start())

    def test_identity_and_cross_profile_replay(self):
        l1=config.load_experiment(phase='secrm_efficiency_directional_l1')
        l2=config.load_experiment(phase='secrm_efficiency_directional_l2')
        self.assertNotEqual(config.reward_identity(l1['reward']),config.reward_identity(l2['reward']))
        buf=SimpleNamespace();secrm_reward.stamp_replay(buf,l1);secrm_reward.validate_replay(buf,l1)
        with self.assertRaises(ValueError):secrm_reward.validate_replay(buf,l2)
        bad=copy.deepcopy(l1);bad['reward']['lane_reward_mode']='unknown'
        with self.assertRaises(ValueError):config.validate_experiment(bad)

    def test_cross_section_distance_consistency(self):
        e,c=self.make_env()
        for lane,dist in [('E76',184.11),(':J105_0',184.01),('E77',175.88)]:
            e.lane_name=lane;e._connection.vehicle.getDrivingDistance.return_value=dist
            self.assertEqual(e._remaining_to_d3_start(),dist)
        self.assertAlmostEqual(adaptive_lane_reward('toward',184.11,'l2')[0]-
                               adaptive_lane_reward('toward',184.01,'l2')[0],.1/891)

if __name__=='__main__':unittest.main()
