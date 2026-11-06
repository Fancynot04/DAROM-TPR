"""SECRM reward contracts, topology, numerical failures and boundary coverage."""
import copy
import heapq
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
from train_sac_delay import parse_args, IntermediateCheckpointCallback


class SecrmTests(unittest.TestCase):
    def make_env(self, route=True):
        c=config.load_experiment(phase='secrm_efficiency_route' if route else 'secrm_efficiency')
        return Merging(seed=42,task=exits.TASK_ID,exit_reward_profile=config.reward_profile(c),progress_experiment=c)

    def test_free_road_and_overspeed(self):
        for speed,expected in [(0,-1),(15,-0.5),(30,0),(36,-0.2)]:
            value,d=s.efficiency(speed,30,.1)
            self.assertAlmostEqual(value,expected)
            self.assertEqual(d['target_speed'],30)
        self.assertEqual(s.efficiency(0,0,.1)[0],0)
        self.assertEqual(s.efficiency(1,0,.1)[0],-.5)

    def test_safe_speed_satisfies_paper_stopping_gap_identity(self):
        _,d=s.efficiency(15,30,.1,gap=25,leader_speed=10,ego_decel=4,leader_decel=6,stopped_gap=2)
        v=d['safe_speed']
        self.assertAlmostEqual((15+v)*.1/2+v*v/(2*4)-100/(2*6)+2,25)

    def test_independent_quadratic_oracle_and_braking_parameters(self):
        for gap in (8, 20, 40):
            for ego_brake, leader_brake in ((4, 9), (5, 6), (5, 9)):
                _, d = s.efficiency(12, 30, .1, gap=gap, leader_speed=8,
                                    ego_decel=ego_brake, leader_decel=leader_brake,
                                    stopped_gap=2.5)
                if d['constraint_status'] == 'feasible':
                    candidate = d['safe_speed']
                    stopping_gap = ((12+candidate)*.05 + candidate**2/(2*ego_brake)
                                    -8**2/(2*leader_brake)+2.5)
                    self.assertAlmostEqual(stopping_gap, gap, places=7)
        _, comfortable=s.efficiency(20,30,.1,gap=30,leader_speed=15,
                                    ego_decel=5,leader_decel=4.5,stopped_gap=2.5)
        _, emergency=s.efficiency(20,30,.1,gap=30,leader_speed=15,
                                  ego_decel=5,leader_decel=9,stopped_gap=2.5)
        self.assertLess(emergency['target_speed'],comfortable['target_speed'])

    def test_effective_braking_and_stopped_target(self):
        _,d=s.efficiency(0,30,.1,gap=2,leader_speed=0,ego_decel=9,leader_decel=3,stopped_gap=2)
        self.assertEqual(d['effective_decel'],9)
        self.assertFalse(d['paper_braking_assumption'])
        self.assertAlmostEqual(d['target_speed'],0)
        self.assertEqual(s.efficiency(0,30,.1,gap=2,leader_speed=0,ego_decel=3,leader_decel=3,stopped_gap=2)[0],0)

    def test_stopped_leader_denominator_floor_is_explicit_unclipped(self):
        value,d=s.efficiency(10,30,.1,gap=2.5,leader_speed=0,
                             ego_decel=4,leader_decel=4,stopped_gap=2)
        self.assertAlmostEqual(d['target_speed'],0)
        self.assertAlmostEqual(value,-5)

    def test_invalid_safe_state_is_diagnosed_and_finite(self):
        value, d = s.efficiency(30,30,.1,gap=0,leader_speed=0,
                                ego_decel=4,leader_decel=4,stopped_gap=2)
        self.assertEqual(d['constraint_status'], 'infeasible')
        self.assertIsNone(d['safe_speed'])
        self.assertEqual(d['target_speed'], 0)
        self.assertTrue(math.isfinite(value))
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

        # Independent state graph: legal road connections cost zero, each
        # adjacent active lane change costs one.
        for start_edge in range(len(route)):
            for start_lane in n.lanes[route[start_edge]]:
                queue=[(0,start_edge,start_lane)];seen={}
                while queue:
                    cost,edge,lane=heapq.heappop(queue)
                    if (edge,lane) in seen:continue
                    seen[edge,lane]=cost
                    if edge==len(route)-1:break
                    for adjacent in (lane-1,lane+1):
                        if adjacent in n.lanes[route[edge]] and route[edge] in LANE_CHANGE_EDGES:
                            heapq.heappush(queue,(cost+1,edge,adjacent))
                    for next_lane,_ in n.connections.get((route[edge],route[edge+1],lane),()):
                        heapq.heappush(queue,(cost,edge+1,next_lane))
                self.assertEqual(s.minimum_changes(n,route,start_edge,start_lane,LANE_CHANGE_EDGES),cost)

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
        old=copy.deepcopy(a);old['reward']['speed_epsilon']=0.1
        self.assertEqual(a['reward']['speed_epsilon'],2.0)
        self.assertNotEqual(config.reward_identity(a['reward']),config.reward_identity(old['reward']))
        replay=SimpleNamespace()
        with self.assertRaises(ValueError):s.validate_replay(replay,a)
        s.stamp_replay(replay,a);replay=pickle.loads(pickle.dumps(replay));s.validate_replay(replay,a)
        with self.assertRaises(ValueError):s.validate_replay(replay,old)
        with self.assertRaises(ValueError):s.validate_replay(replay,b)
        replay.secrm_formula_version='other'
        with self.assertRaises(ValueError):s.validate_replay(replay,a)

    def test_no_event_group_keeps_dense_reward_for_every_terminal_event(self):
        c=config.load_experiment(phase='secrm_efficiency_route_no_events')
        b=config.load_experiment(phase='secrm_efficiency_route')
        self.assertEqual(config.reward_profile(c),s.ROUTE_PROFILE)
        self.assertNotEqual(config.reward_identity(c['reward']),config.reward_identity(b['reward']))
        env=Merging(seed=42,task=exits.TASK_ID,exit_reward_profile=s.ROUTE_PROFILE,progress_experiment=c)
        env.secrm_before={'efficiency':-.2,'route_reward':-.03}
        for event,key in config.EVENTS.items():
            env.terminal_reason=event
            self.assertAlmostEqual(env.getReward(),-.23)
            self.assertTrue(all(env.reward_components[k]==0 for k in config.EVENTS.values()))
        env.ego_teleported=True
        self.assertEqual(env.getReward(),0.)
        replay=SimpleNamespace();s.stamp_replay(replay,b)
        with self.assertRaises(ValueError):s.validate_replay(replay,c)
        self.assertEqual((c['seed'],c['total_timesteps'],c['checkpoint_freq'],env.episode_limit_steps),
                         (42,200000,50000,4000))

    def test_training_defaults(self):
        for phase in config.SECRM_DEFAULTS:
            with patch.object(sys,'argv',['train','--exit_phase',phase]):args=parse_args()
            budget = ((50000, 50000)
                      if phase == 'secrm_efficiency_directional_l1_speed120_50k'
                      else (500000, 250000)
                      if phase.startswith('secrm_efficiency_directional')
                      else (200000, 50000))
            self.assertEqual((args.seed,args.total_timesteps,args.checkpoint_freq,args.wandb_mode),(42,*budget,'disabled'))
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

    def test_directional_profile_tracker_and_settlement(self):
        c = config.load_experiment(phase='secrm_efficiency_directional')
        env = Merging(seed=42, task=exits.TASK_ID, exit_reward_profile=s.DIRECTIONAL_PROFILE,
                      progress_experiment=c)
        env.ego_on_road = True
        env.lane_name = 'E77'
        env._connection = Mock()
        env._connection.vehicle.getLateralSpeed.return_value = 0.0
        def estimate(lane, n, pending=False, road='E77'):
            return {'status': 'ok', 'valid_for_threshold': not pending, 'pending': pending,
                    'road': road, 'projected_lane': lane, 'minimum_lane_changes': n}
        env.secrm_before = {'efficiency': -.2, 'route_reward': -4.0}
        with patch('src.merging.observe_route_urgency') as observe:
            observe.return_value = estimate(1, 1)
            env._update_directional_lane_reward()
            env.directional_pending_request = -1
            observe.return_value = estimate(1, 1, pending=True)
            env._update_directional_lane_reward()
            self.assertEqual(env.directional_lane_reward, 0)
            env.directional_seen_lateral_motion = True
            observe.return_value = estimate(0, 0)
            env._update_directional_lane_reward()
            self.assertEqual(env.directional_lane_reward, 3)
            self.assertAlmostEqual(env.getReward(), 2.8)
            self.assertEqual(env.reward_components['route_following'], 0)
            env._update_directional_lane_reward()
            self.assertEqual(env.directional_lane_reward, 0)
            env.directional_pending_request = 1
            env.directional_seen_lateral_motion = True
            observe.return_value = estimate(1, 1)
            env._update_directional_lane_reward()
            self.assertEqual(env.directional_lane_reward, -6)
            env.terminal_reason = 'Arrived'
            self.assertAlmostEqual(env.getReward(), 43.8)
            env.teleport_transition = True
            env._update_directional_lane_reward()
            self.assertEqual(env.directional_lane_reward, 0)

    def test_intermediate_checkpoint_skips_only_final_step(self):
        callback = IntermediateCheckpointCallback(final_step=500000, save_freq=250000,
                                                   save_path='unused', name_prefix='test')
        callback.num_timesteps = 500000
        with patch('stable_baselines3.common.callbacks.CheckpointCallback._on_step', return_value=True) as save:
            self.assertTrue(callback._on_step())
            save.assert_not_called()
            callback.num_timesteps = 250000
            self.assertTrue(callback._on_step())
            save.assert_called_once()

    def test_directional_profile_replay_and_checkpoint_identity(self):
        c = config.load_experiment(phase='secrm_efficiency_directional')
        buffer = SimpleNamespace()
        s.stamp_replay(buffer, c)
        s.validate_replay(buffer, c)
        changed = copy.deepcopy(c)
        changed['reward']['beta'] = 7.0
        with self.assertRaises(ValueError):
            s.validate_replay(buffer, changed)
        with self.assertRaises(ValueError):
            s.validate_replay(buffer, config.load_experiment(phase='secrm_efficiency'))

    def fake_snapshot(self,e):
        e.ego_id='ego';e.ego_on_road=True
        t=Mock();e._connection=t
        v=t.vehicle;v.getLeader.return_value=('lead',8)
        v.getMinGap.return_value=2;v.getSpeed.side_effect=lambda x:10 if x=='ego' else 12
        v.getAllowedSpeed.return_value=30;v.getDecel.return_value=4
        v.getApparentDecel.return_value=4;v.getEmergencyDecel.return_value=9
        v.getPosition.side_effect=lambda x:(100,0) if x=='ego' else (120,0)
        v.getAngle.return_value=90;v.getWidth.return_value=2
        v.getLength.return_value=4.2
        v.getLateralSpeed.return_value=0
        v.getIDList.return_value=('ego',)
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
        t.vehicle.getLaneIndex.return_value=1;t.vehicle.getLaneID.return_value='E76_1';e.exit_maneuver={'target':1}
        d=e._secrm_snapshot();self.assertEqual(d['route']['required_changes'],1)
        self.assertEqual(d['route']['status'],'pending_current_lane')
        e.exit_maneuver=None
        with patch.object(e,'_exit_laterally_settled',return_value=True):d=e._secrm_snapshot()
        self.assertEqual(d['route']['required_changes'],1)
        t.vehicle.getLaneIndex.return_value=0;t.vehicle.getLaneID.return_value='E76_0'
        with patch.object(e,'_exit_laterally_settled',return_value=True):d=e._secrm_snapshot()
        self.assertEqual(d['route']['required_changes'],0)

    def test_seed43_negative_gap_laterally_separate_regression(self):
        e=self.make_env();t=self.fake_snapshot(e);v=t.vehicle
        v.getLeader.return_value=('lead',-5.237229346061554)
        v.getMinGap.return_value=2.5
        v.getSpeed.side_effect=lambda x:9.946592761466428 if x=='ego' else 14.66490589856147
        v.getPosition.side_effect=lambda x:(594.064,0) if x=='ego' else (590,2.63)
        v.getLanePosition.side_effect=lambda x:594.064 if x=='ego' else 590
        v.getWidth.return_value=1.8
        v.getLateralSpeed.side_effect=lambda x:.4 if x=='ego' else 0
        v.getIDList.return_value=('ego','lead')
        with patch.object(e,'_exit_laterally_settled',return_value=False):d=e._secrm_snapshot()
        self.assertEqual(d['target']['leader_selection']['status'],'laterally_separated')
        self.assertEqual(d['target']['constraint_status'],'no_leader')
        self.assertTrue(math.isfinite(d['efficiency']))
        self.assertAlmostEqual(d['target']['target_speed'],30)
        v.getPosition.side_effect=lambda x:(594.064,0) if x=='ego' else (590,.5)
        with patch.object(e,'_exit_laterally_settled',return_value=False):d=e._secrm_snapshot()
        self.assertEqual(d['target']['leader_selection']['status'],'traci_leader')
        self.assertEqual(d['target']['leader_decel'],9)
        self.assertTrue(math.isfinite(d['efficiency']))

    def test_separated_leader_falls_back_to_occupied_path(self):
        e=self.make_env();t=self.fake_snapshot(e);v=t.vehicle
        v.getLeader.return_value=('side',-2)
        v.getPosition.side_effect=lambda x:{'ego':(100,0),'side':(105,3),
                                            'front':(114,.4)}[x]
        v.getLanePosition.side_effect=lambda x:{'ego':100,'side':105,'front':114}[x]
        v.getIDList.return_value=('ego','side','front')
        with patch.object(e,'_exit_laterally_settled',return_value=True):d=e._secrm_snapshot()
        self.assertEqual(d['target']['leader_selection']['id'],'front')
        self.assertEqual(d['target']['leader_selection']['status'],'lateral_path')
        self.assertAlmostEqual(d['target']['gap'],9.8)
        self.assertTrue(math.isfinite(d['efficiency']))

    def test_pending_crosses_e76_e77_and_nonfinite_distance_rejected(self):
        e=self.make_env();t=self.fake_snapshot(e);v=t.vehicle
        v.getLeader.return_value=None
        e.exit_maneuver={'target':0}
        v.getRoadID.return_value='E77';v.getRouteIndex.return_value=1
        v.getLaneIndex.return_value=1;v.getLaneID.return_value='E77_1'
        t.lane.getLength.return_value=172.35
        d=e._secrm_snapshot()
        self.assertEqual((d['route']['road'],d['route']['lane'],d['route']['required_changes']),('E77',1,1))
        self.assertEqual(d['route']['status'],'pending_current_lane')
        self.assertAlmostEqual(d['route_reward'],-1/73.35)
        for bad in (math.nan,math.inf,-math.inf):
            t.lane.getLength.return_value=bad
            with self.assertRaisesRegex(ValueError,'section distance'):e._secrm_snapshot()
        t.lane.getLength.return_value=50
        d=e._secrm_snapshot()
        self.assertEqual(d['route']['remaining_section_m'],0)

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
