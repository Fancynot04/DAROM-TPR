"""SECRM reward contracts, topology, numerical failures and boundary coverage."""
import copy
import math
import pickle
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import secrm_reward as s, exit_progress_reward as config
from src import exit_e76_d3_config as exits
from src.merging import Merging, ANTWERP_DIR, LANE_CHANGE_EDGES
from src.route_urgency import load_route_network
from src.road_boundary import coverage, ccw, bbox, footprint, road_tiles
from evaluate_sac_delay import validate_checkpoint_setup, evaluate
from train_sac_delay import parse_args


class SecrmTests(unittest.TestCase):
    def make_env(self, route=True):
        c=config.load_experiment(phase='secrm_efficiency_route' if route else 'secrm_efficiency')
        return Merging(seed=42,task=exits.TASK_ID,exit_reward_profile=config.reward_profile(c),progress_experiment=c)

    def test_free_road_and_overspeed(self):
        for speed,expected in [(0,-1),(15,-0.5),(30,0),(36,-0.2)]:
            value,d=s.efficiency(speed,30,.1)
            self.assertAlmostEqual(value,expected)
            self.assertEqual(d['target_speed'],30)

    def test_safe_speed_satisfies_paper_stopping_gap_identity(self):
        _,d=s.efficiency(15,30,.1,gap=25,leader_speed=10,ego_decel=4,leader_decel=6,stopped_gap=2)
        v=d['safe_speed']
        self.assertAlmostEqual((15+v)*.1/2+v*v/(2*4)-100/(2*6)+2,25)

    def test_effective_braking_and_stopped_target(self):
        _,d=s.efficiency(0,30,.1,gap=2,leader_speed=0,ego_decel=9,leader_decel=3,stopped_gap=2)
        self.assertEqual(d['effective_decel'],3)
        self.assertAlmostEqual(d['target_speed'],0)
        self.assertEqual(s.efficiency(0,30,.1,gap=2,leader_speed=0,ego_decel=3,leader_decel=3,stopped_gap=2)[0],0)

    def test_stopped_leader_denominator_floor_is_explicit_unclipped(self):
        value,d=s.efficiency(10,30,.1,gap=2.5,leader_speed=0,
                             ego_decel=4,leader_decel=4,stopped_gap=2)
        self.assertAlmostEqual(d['target_speed'],0)
        self.assertAlmostEqual(value,-100)

    def test_invalid_safe_state_is_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError,'radicand'):
            s.efficiency(30,30,.1,gap=0,leader_speed=0,ego_decel=4,leader_decel=4,stopped_gap=2)
        for kw in [{'epsilon':0},{'gap':math.nan},{'gap':1,'leader_speed':-1}]:
            with self.assertRaises(ValueError):s.efficiency(10,30,.1,**kw)

    def test_route_penalty_has_no_clipping(self):
        self.assertEqual(s.route_penalty(4,0),-4)
        self.assertAlmostEqual(s.route_penalty(2,99),-.02)
        for args in [(None,10),(-1,10),(1,-1),(1,math.nan)]:
            with self.assertRaises(ValueError):s.route_penalty(*args)

    def test_multiedge_lane_topology(self):
        n=load_route_network(ANTWERP_DIR/'Antwerp.net.xml')
        route=('E76','E77','D3')
        self.assertEqual(s.minimum_changes(n,route,0,0,LANE_CHANGE_EDGES),0)
        for lane in range(5):self.assertEqual(s.minimum_changes(n,route,0,lane,LANE_CHANGE_EDGES),lane)
        for lane in range(6):self.assertEqual(s.minimum_changes(n,route,1,lane,LANE_CHANGE_EDGES),lane)
        self.assertEqual(s.minimum_changes(n,route,2,0,LANE_CHANGE_EDGES),0)
        with self.assertRaises(ValueError):s.minimum_changes(n,route,0,100,LANE_CHANGE_EDGES)

    def test_configuration_identity_and_no_legacy_fields(self):
        a=config.load_experiment(phase='secrm_efficiency');b=config.load_experiment(phase='secrm_efficiency_route')
        self.assertNotEqual(config.reward_identity(a['reward']),config.reward_identity(b['reward']))
        self.assertNotIn('alpha',b['reward']);self.assertNotIn('progress_per_meter',b['reward'])
        for key in ['efficiency_weight','route_weight','speed_epsilon']:
            bad=copy.deepcopy(b);bad['reward'][key]=-1
            with self.assertRaises(ValueError):config.validate_experiment(bad)
        with self.assertRaises(ValueError):config.validate_experiment({**a,'reward':{**a['reward'],'alpha':3}})

    def test_replay_configuration_cannot_cross_groups_or_versions(self):
        a=config.load_experiment(phase='secrm_efficiency')
        b=config.load_experiment(phase='secrm_efficiency_route')
        replay=SimpleNamespace()
        with self.assertRaises(ValueError):s.validate_replay(replay,a)
        s.stamp_replay(replay,a);replay=pickle.loads(pickle.dumps(replay));s.validate_replay(replay,a)
        with self.assertRaises(ValueError):s.validate_replay(replay,b)
        replay.secrm_formula_version='other'
        with self.assertRaises(ValueError):s.validate_replay(replay,a)

    def test_training_defaults(self):
        for phase in config.SECRM_DEFAULTS:
            with patch.object(sys,'argv',['train','--exit_phase',phase]):args=parse_args()
            self.assertEqual((args.seed,args.total_timesteps,args.checkpoint_freq,args.wandb_mode),(42,200000,50000,'disabled'))
            self.assertTrue(args.save_replay_buffer)
            self.assertFalse(args.nav_features)

    def test_only_requested_rewards_and_event_exclusivity(self):
        for route in [True,False]:
            e=self.make_env(route);e.secrm_before={'efficiency':-.2,'route_reward':-.03}
            for event,key in config.EVENTS.items():
                e.terminal_reason=event
                reward=e.getReward()
                self.assertAlmostEqual(reward,e.progress_experiment['reward'][key]-.2-(.03 if route else 0))
                self.assertAlmostEqual(reward,sum(e.reward_components.values()))
                self.assertEqual(e.reward_components['progress'],0)
                self.assertEqual(e.reward_components['lane_change'],0)
                self.assertEqual(sum(e.reward_components[k]!=0 for k in config.EVENTS.values()),1)
            e.ego_teleported=True;e.terminal_reason='Ego Teleported'
            self.assertEqual(e.getReward(),-100)
            self.assertEqual(e.episode_limit_steps,4000)

    def fake_snapshot(self,e):
        e.ego_id='ego';e.ego_on_road=True
        t=Mock();e._connection=t
        v=t.vehicle;v.getLeader.return_value=('lead',8)
        v.getMinGap.return_value=2;v.getSpeed.side_effect=lambda x:10 if x=='ego' else 12
        v.getAllowedSpeed.return_value=30;v.getDecel.return_value=4
        v.getRoadID.return_value='E76';v.getRoute.return_value=('E76','E77','D3')
        v.getRouteIndex.return_value=0;v.getLaneIndex.return_value=2
        v.getLaneID.return_value='E76_2';v.getLanePosition.return_value=100
        t.lane.getLength.return_value=1000
        return t

    def test_snapshot_leader_gap_and_pending_cache(self):
        e=self.make_env();t=self.fake_snapshot(e)
        with patch.object(e,'_exit_laterally_settled',return_value=True):d=e._secrm_snapshot()
        self.assertEqual(d['target']['gap'],10)
        self.assertEqual(d['route']['required_changes'],2)
        t.vehicle.getLaneIndex.return_value=1;e.exit_maneuver={'target':1}
        d=e._secrm_snapshot();self.assertEqual(d['route']['required_changes'],2)
        self.assertEqual(d['route']['status'],'pending_cached')
        e.exit_maneuver=None
        with patch.object(e,'_exit_laterally_settled',return_value=True):d=e._secrm_snapshot()
        self.assertEqual(d['route']['required_changes'],1)

    def test_connector_and_destination(self):
        e=self.make_env();t=self.fake_snapshot(e)
        t.vehicle.getRoadID.return_value=':J105_0'
        d=e._secrm_snapshot();self.assertEqual(d['route_reward'],0)
        self.assertIsNone(d['route']['required_changes'])
        t.vehicle.getRoadID.return_value='D3'
        with patch.object(e,'_remaining_to_d3_end',return_value=100):d=e._secrm_snapshot()
        self.assertEqual(d['route_reward'],0);self.assertLess(d['efficiency'],0)

    def test_evaluator_does_not_certify_missing_terminal_geometry(self):
        base=self.make_env();base.ego_on_road=False
        components={'efficiency': -1., 'route_following': -.1, 'arrival': 50.,
                    'progress': 0., 'lane_change': 0.}
        info={'message':'Arrived', 'ego_teleported':False, 'stage_reached':True,
              'reward_components':components, 'road_boundary':{'status':'inside'}}
        env=SimpleNamespace(unwrapped=base,reset=lambda **kw:([],{}),
                            step=lambda action:([],48.9,True,False,info))
        model=SimpleNamespace(gamma=.99,predict=lambda *a,**kw:([0,0],None))
        result=evaluate(model,env,1,42,task=exits.TASK_ID)
        self.assertEqual(result['success_rate'],100)
        self.assertEqual(result['boundary_diagnostics']['observed_body_safe_arrivals'],1)
        self.assertIsNone(result['boundary_diagnostics']['full_body_exit_rate'])
        self.assertEqual(result['episode_results'][0]['road_boundary']['full_body_exit_status'],'unknown')
        info['message']='Collision'
        result=evaluate(model,env,1,42,task=exits.TASK_ID)
        self.assertEqual(result['episode_results'][0]['road_boundary']['full_body_exit_status'],'failed')

    def test_checkpoint_rejects_modified_reward_or_profile(self):
        c=config.load_experiment(phase='secrm_efficiency_route')
        m=SimpleNamespace(antwerp_env_version=3,antwerp_task=exits.TASK_ID,
            exit_reward_profile=s.ROUTE_PROFILE,secrm_formula_version=s.FORMULA_VERSION,exit_potential_beta=0.,exit_control_profile=exits.CONTROL_PROFILE,
            exit_observation_profile=None,exit_potential_gamma=.99,gamma=.99,
            observation_space=SimpleNamespace(shape=(134,)),policy_kwargs={},
            exit_progress_experiment=c,exit_progress_reward_identity=config.reward_identity(c['reward']),
            exit_progress_experiment_identity=config.experiment_identity(c))
        validate_checkpoint_setup(m,'full',False,exits.TASK_ID)
        m.exit_progress_experiment=copy.deepcopy(c);m.exit_progress_experiment['reward']['route_weight']=2
        with self.assertRaises(ValueError):validate_checkpoint_setup(m,'full',False,exits.TASK_ID)
        m.exit_progress_experiment=c;m.exit_reward_profile=s.EFFICIENCY_PROFILE
        with self.assertRaises(ValueError):validate_checkpoint_setup(m,'full',False,exits.TASK_ID)


class BoundaryTests(unittest.TestCase):
    def tiles(self,polys):return [(ccw(p),bbox(p)) for p in polys]
    def test_whole_body_and_divider_crossing(self):
        tiles=self.tiles([[(0,0),(10,0),(10,3),(0,3)],[(0,3),(10,3),(10,6),(0,6)]])
        body=footprint((8,3),90,4,2)
        self.assertEqual(coverage(body,tiles)['status'],'inside')
        outside=footprint((8,0),90,4,2)
        self.assertEqual(coverage(outside,tiles)['status'],'outside')
        self.assertAlmostEqual(coverage(outside,tiles)['uncovered_area_m2'],4)
    def test_splitter_gap_is_not_drivable(self):
        tiles=self.tiles([[(0,0),(10,0),(10,2),(0,2)],[(0,4),(10,4),(10,6),(0,6)]])
        self.assertEqual(coverage(footprint((8,3),90,4,1),tiles)['status'],'outside')
    def test_invalid_body_is_unknown_capable_not_nan_geometry(self):
        with self.assertRaises(ValueError):footprint((math.nan,0),90,4,2)
        with self.assertRaises(ValueError):footprint((0,0),90,4,0)

    def test_actual_network_geometry_loads(self):
        self.assertGreater(len(road_tiles(str(ANTWERP_DIR/'Antwerp.net.xml'))),50)

if __name__=='__main__':unittest.main()
