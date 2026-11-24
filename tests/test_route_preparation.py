import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from traci.exceptions import TraCIException

from src.agents.masac import MultiAgentGaussianActor
from src.envs.multi_ego_env import MultiEgoEnv
from src.envs.route_lane_plan import RouteLaneNetwork
from src.envs.route_preparation import (
    DEFAULT_LINEAR_PREPARATION_SETTINGS, DEFAULT_REFERENCE_STEP_M, DEFAULT_ROUTE_PREPARATION_STAGES,
    DEFAULT_ROUTE_PREPARATION_WEIGHTS,
    PREPARATION_BONUS, PREPARATION_LINEAR_V1, PREPARATION_PROGRESS_BONUS, PREPARATION_V1,
    V1_REFERENCE_STEP_M,
    RoutePreparation,
)
from src.evaluation_trajectory import EvaluationTrajectory
from src.experiment_tracking import sha256
from src.rewards.masac import LaneChange, MASACRewardConfig, MASACRewards
from train_masac import AGENT_IDS, initialize_actor_weights


NETWORK = Path(__file__).resolve().parents[1] / 'sumo_new/Antwerp.net.xml'
O3_D3 = ('O3', 'E75', 'E76', 'E77', 'D3')


def settings(exit_weight=0.002):
    return {'weights': {**DEFAULT_ROUTE_PREPARATION_WEIGHTS, 'exit': exit_weight},
            'stages': DEFAULT_ROUTE_PREPARATION_STAGES}


def setup(route=O3_D3, *, exit_weight=0.002):
    network = RouteLaneNetwork(NETWORK)
    tracker = RoutePreparation(settings(exit_weight))
    tracker.reset(AGENT_IDS)
    tracker.configure('ego_1', network, network.plan(route, 'passenger'))
    return tracker


def test_stage_local_demand_urgency_and_normalization():
    tracker = setup(exit_weight=0.005)
    phases = tracker.ego_phases['ego_1']
    assert set(phases) == {2, 3}
    assert phases[2].demand == dict(enumerate(range(5)))
    assert phases[3].demand == dict(enumerate(range(6)))
    assert (phases[2].normalization, phases[3].normalization) == (4, 5)
    exit_phase = phases[3]
    assert exit_phase.penalty(0, 1, True, False) == pytest.approx(-0.001)
    assert exit_phase.penalty(0, 5, True, False) == pytest.approx(-0.005)
    assert exit_phase.penalty(exit_phase.distance_m / 2, 5, True, False) == pytest.approx(-0.0025)
    assert exit_phase.penalty(exit_phase.distance_m, 5, True, False) == 0
    assert exit_phase.penalty(0, 0, True, False) == 0
    assert exit_phase.penalty(0, None, True, False) == 0
    assert exit_phase.penalty(0, 5, False, False) == 0
    assert exit_phase.penalty(0, 5, True, None) == 0


def test_default_zones_cover_both_weaving_connectors_and_full_edges():
    network = RouteLaneNetwork(NETWORK)
    plan = network.plan(O3_D3, 'passenger')
    tracker = setup()
    state = SimpleNamespace()
    vehicle = SimpleNamespace(
        getRoadID=lambda ego: state.road, getRouteIndex=lambda ego: state.index,
        getLaneID=lambda ego: state.lane_id, getLaneIndex=lambda ego: state.lane,
        getLanePosition=lambda ego: state.position, getLateralSpeed=lambda ego: 0.0,
        getSpeed=lambda ego: 10.0, getRightLeaders=lambda ego: (), getRightFollowers=lambda ego: (),
    )
    lengths = {lane['id']: float(lane['length']) for lanes in network.lanes.values() for lane in lanes.values()}
    conn = SimpleNamespace(vehicle=vehicle, lane=SimpleNamespace(getLength=lengths.__getitem__),
                           simulation=SimpleNamespace(getTime=lambda: 10.0))

    def record(zone_tracker, index, road, lane, position):
        state.index, state.road, state.lane, state.position = index, road, lane, position
        state.lane_id = f'{road}_{lane}'
        sample = zone_tracker.before_step(conn, 'ego_1', plan.position(vehicle, 'ego_1'), None)
        penalty = zone_tracker.after_step(conn, 'ego_1', sample, True, None) if sample is not None else 0.0
        return sample, penalty, zone_tracker.step_info.get('ego_1')

    assert record(tracker, 1, 'E75', 2, 50)[2]['phase'] == 'preparation'
    incoming, reward, row = record(tracker, 1, ':J104_0', 2, 1)
    assert row['phase'] == 'weaving' and incoming.demand == 1 and reward == 0
    weaving, reward, row = record(tracker, 2, 'E76', 1, 100)
    assert row['phase'] == 'weaving' and weaving.phase.distance_m == pytest.approx(882.60)
    assert weaving.distance_m > 300 and reward < 0
    outgoing, reward, row = record(tracker, 2, ':J105_0', 0, 1)
    assert row['phase'] == 'weaving' and outgoing.demand == 0 and reward == 0
    wrong, reward, row = record(tracker, 2, ':J105_0', 1, 1)
    assert row['phase'] == 'weaving' and wrong.demand is None and reward == 0
    assert record(tracker, 3, 'E77', 1, 0)[2]['phase'] == 'exit'
    assert record(tracker, 3, ':J106_0', 0, 1)[2] is None
    assert tracker.diagnostics['ego_1']['preparation_edge_steps'] == 1
    assert tracker.diagnostics['ego_1']['phases']['weaving']['preparation_zone_steps'] == 4

    config = settings()
    config['stages'] = [{'route': list(O3_D3), 'phases': [
        {'kind': 'weaving', 'critical_connection': ['E76', 'E77'],
         'accepted_entry_lanes': ['E77_0'], 'preparation_distance_m': 300},
    ]}]
    legacy = RoutePreparation(config)
    legacy.configure('ego_1', network, plan)
    assert record(legacy, 1, 'E75', 2, 50)[2] is None
    assert record(legacy, 1, ':J104_0', 2, 1)[2] is None
    assert record(legacy, 2, 'E76', 1, 100)[2] is None
    assert record(legacy, 2, ':J105_0', 0, 1)[2] is None
    assert record(legacy, 2, 'E76', 1, 750)[1] < 0


def test_zero_weight_unconfigured_route_and_custom_d2_phase():
    tracker = setup(exit_weight=0)
    stats = tracker.diagnostics['ego_1']
    assert stats['enabled'] and not stats['phases']['exit']['enabled']
    assert tracker.ego_phases['ego_1'][3].penalty(0, 5, True, False) == 0
    route = ('O3', 'E75', 'E76', 'E77', 'E78', 'E82', 'D2')
    assert setup(route).diagnostics['ego_1']['reason'] == 'unconfigured_route'
    cfg = settings()
    cfg['stages'] = [{'route': list(route), 'phases': [
        {'kind': 'exit', 'critical_connection': ['E82', 'D2'],
         'accepted_entry_lanes': ['D2_0', 'D2_1', 'D2_2'], 'preparation_distance_m': 100},
    ]}]
    tracker = RoutePreparation(cfg)
    tracker.reset(AGENT_IDS)
    network = RouteLaneNetwork(NETWORK)
    tracker.configure('ego_1', network, network.plan(route, 'passenger'))
    assert tracker.ego_phases['ego_1'][5].demand == dict(enumerate([0, 0, 0, 1, 2, 3]))


def test_pre_action_gaps_and_physical_motion_gate():
    tracker = setup(exit_weight=0.005)
    state = SimpleNamespace(speed=0.0, lane=1, lateral_speed=0.0, neighbors=[])
    vehicle = SimpleNamespace(
        getRoadID=lambda ego: 'E77', getLaneID=lambda ego: f'E77_{state.lane}',
        getLaneIndex=lambda ego: state.lane, getLanePosition=lambda ego: 171.35,
        getLateralSpeed=lambda ego: state.lateral_speed, getSpeed=lambda ego: state.speed,
        getMinGap=lambda ego: 2.5, getRightLeaders=lambda ego: (),
        getRightFollowers=lambda ego: state.neighbors,
    )
    conn = SimpleNamespace(vehicle=vehicle, lane=SimpleNamespace(getLength=lambda lane: 172.35),
                           simulation=SimpleNamespace(getTime=lambda: 1.0))
    # A request alone leaves the true motion samples at zero.
    request = SimpleNamespace(road='E77', source_lane=1, target_lane=0, direction=-1, last_motion=None)
    before = tracker.before_step(conn, 'ego_1', (3, 1), request)
    assert before.safe_opportunity
    assert tracker.after_step(conn, 'ego_1', before, True, None) < 0
    assert tracker.diagnostics['ego_1']['phases']['exit']['penalty_steps'] == 1

    before = tracker.before_step(conn, 'ego_1', (3, 1), request)
    state.lateral_speed = -0.1
    assert tracker.after_step(conn, 'ego_1', before, True, None) == 0
    assert tracker.step_info['ego_1']['executing_necessary_change']
    before = tracker.before_step(conn, 'ego_1', (3, 1), request)
    state.lateral_speed = 0.0
    assert tracker.after_step(conn, 'ego_1', before, True, None) == 0

    state.neighbors = [('rear', -2.4)]
    before = tracker.before_step(conn, 'ego_1', (3, 1), request)
    assert before.demand == 1 and not before.safe_opportunity
    assert tracker.after_step(conn, 'ego_1', before, True, None) == 0
    assert tracker.before_step(conn, 'ego_1', (1, 1), None) is None

    state.neighbors = []
    before = tracker.before_step(conn, 'ego_1', (3, 1), None)
    state.lane = 0
    assert tracker.after_step(conn, 'ego_1', before, True, {
        'road': 'E77', 'source_lane': 1, 'target_lane': 0, 'time': 1.1,
    }) == 0
    assert tracker.diagnostics['ego_1']['phases']['exit']['last_necessary_completion'] == {
        'time_s': 1.1, 'distance_m': 1.0,
    }
    state.lane = 1

    def missing(ego):
        raise TraCIException('missing follower query')
    vehicle.getRightFollowers = missing
    assert not tracker.before_step(conn, 'ego_1', (3, 1), None).safe_opportunity
    vehicle.getRightFollowers = lambda ego: None
    assert not tracker.before_step(conn, 'ego_1', (3, 1), None).safe_opportunity


@pytest.fixture
def bonus_step(request):
    network = RouteLaneNetwork(NETWORK)
    mode = getattr(request, 'param', PREPARATION_PROGRESS_BONUS)
    settings = (DEFAULT_LINEAR_PREPARATION_SETTINGS if mode == PREPARATION_LINEAR_V1
                else {'mode': mode, 'stages': DEFAULT_ROUTE_PREPARATION_STAGES})
    tracker = RoutePreparation(settings)
    tracker.configure('ego_1', network, network.plan(O3_D3, 'passenger'))
    rewards = MASACRewards(MASACRewardConfig())
    state = SimpleNamespace(index=2, road='E76', lane=1, along=50.0, speed=10.0, lateral=0.0,
                            time=1.0, leader=None, follower=('', -1), adjacent=(), offset=0.0)
    vehicle = SimpleNamespace(
        getRoute=lambda _: O3_D3, getRouteIndex=lambda _: state.index, getRoadID=lambda _: state.road,
        getLaneID=lambda _: f'{state.road}_{state.lane}', getLaneIndex=lambda _: state.lane,
        getLanePosition=lambda ego: state.along if ego == 'ego_1' else state.along + 100,
        getSpeed=lambda _: state.speed, getLateralSpeed=lambda _: state.lateral,
        getLeader=lambda *_: state.leader, getFollower=lambda *_: state.follower,
        getRightLeaders=lambda _: state.adjacent, getRightFollowers=lambda _: (),
        getLeftLeaders=lambda _: state.adjacent, getLeftFollowers=lambda _: (),
        getLateralLanePosition=lambda _: state.offset, getWidth=lambda _: 2,
        getMinGap=lambda _: 2.5, getLength=lambda _: 4,
    )
    lengths = {lane['id']: float(lane['length']) for lanes in network.lanes.values() for lane in lanes.values()}
    conn = SimpleNamespace(vehicle=vehicle, simulation=SimpleNamespace(getTime=lambda: state.time),
                           lane=SimpleNamespace(getWidth=lambda _: 3.2, getLength=lengths.__getitem__,
                                                getLastStepVehicleIDs=lambda _: ('ego_1',)))

    def tick(change=None, complete=None, unavailable_reason=None, **updates):
        if tracker.linear:
            sample = tracker.before_linear(conn, 'ego_1')
        else:
            sample = tracker.before_bonus(conn, 'ego_1', rewards)
            candidate = tracker.bonus_change('ego_1', change)
            target = tracker.target_risk(conn, 'ego_1', sample.motion, candidate, rewards)
        state.time += 0.1
        state.along += state.speed * 0.1
        if 'started' in updates:
            change.last_motion = state.time
            del updates['started']
        vars(state).update(updates)
        if tracker.linear:
            reward = tracker.after_linear(conn, 'ego_1', sample, unavailable_reason)
        else:
            reward = tracker.after_bonus(conn, 'ego_1', sample, rewards, candidate, target,
                                         complete, unavailable_reason)
        return reward, tracker.step_info['ego_1']

    return state, tracker, tick


@pytest.mark.parametrize('bonus_step', [PREPARATION_LINEAR_V1], indirect=True)
@pytest.mark.parametrize('index,road,along,distance', [
    (2, 'E76', 0.0, 1054.95),
    (2, 'E76', 441.30, 613.65),
    (3, 'E77', 0.0, 172.35),
    (3, 'E77', 86.175, 86.175),
])
@pytest.mark.parametrize('lane,readiness', [(0, 1.0), (1, 0.5), (2, 0.0)])
def test_linear_route_preparation_matches_distance_and_lane_factors(
    bonus_step, index, road, along, distance, lane, readiness,
):
    state, _, tick = bonus_step
    state.index, state.road, state.along, state.lane = index, road, along, lane
    reward, row = tick()
    assert row['remaining_distance_m'] == pytest.approx(distance)
    assert row['local_lane_changes'] == lane
    assert row['progress_factor'] == pytest.approx(1)
    assert reward == pytest.approx(0.02 * readiness * min(1, distance / 1000))
    assert 'event_bonus' not in row


@pytest.mark.parametrize('bonus_step', [PREPARATION_LINEAR_V1], indirect=True)
@pytest.mark.parametrize('speed,forward,factor', [
    (0, 0, 0), (10, -0.5, 0), (5, 0.5, 0.5), (10, 1, 1), (20, 2, 1),
])
def test_linear_route_preparation_saturates_at_one_meter(bonus_step, speed, forward, factor):
    state, _, tick = bonus_step
    state.lane, state.along, state.speed = 0, 50, speed
    reward, row = tick(along=state.along + forward)
    assert row['progress_factor'] == pytest.approx(factor)
    assert reward == pytest.approx(0.02 * factor)


@pytest.mark.parametrize('bonus_step', [PREPARATION_LINEAR_V1], indirect=True)
def test_linear_route_preparation_crosses_internal_lanes_and_stops_at_exit(bonus_step):
    state, _, tick = bonus_step
    state.lane, state.along = 0, 882.35
    reward, row = tick(road=':J105_0', along=0.25)
    assert row['progress_m'] == pytest.approx(0.5)
    assert row['remaining_distance_m'] == pytest.approx(172.6)
    assert reward == pytest.approx(0.02 * 0.1726 * 0.5)
    state.along = 7.88
    reward, row = tick(road='E77', index=3, along=0.25)
    assert row['progress_m'] == pytest.approx(0.5)
    assert row['remaining_distance_m'] == pytest.approx(172.35)
    assert reward == pytest.approx(0.02 * 0.17235 * 0.5)
    state.road, state.along = ':J106_0', 0.0
    reward, row = tick()
    assert row['progress_m'] == pytest.approx(1.0)
    assert row['remaining_distance_m'] == 0
    assert reward == 0


@pytest.mark.parametrize('bonus_step', [PREPARATION_LINEAR_V1], indirect=True)
def test_linear_route_preparation_skips_unverified_progress(bonus_step):
    state, _, tick = bonus_step
    state.lane, state.along = 0, 100
    reward, row = tick(unavailable_reason='teleport')
    assert reward == 0 and row['unavailable_reason'] == 'teleport'


@pytest.mark.parametrize('speed,position', [(1.0, 1.0), (25.0, 865.0)])
def test_bonus_scales_with_distance_at_equal_readiness(bonus_step, speed, position):
    state, _, tick = bonus_step
    state.speed, state.along = speed, position
    bonus, row = tick()
    assert bonus == pytest.approx(0.00375 * speed * 0.1 / DEFAULT_REFERENCE_STEP_M)
    assert (row['local_lane_changes'], row['normalization'], row['readiness']) == (1, 4, 0.75)
    assert row['progress_m'] == pytest.approx(speed * 0.1)
    assert row['first_progress_m'] == pytest.approx(row['progress_m'])


@pytest.mark.parametrize('bonus_step', [PREPARATION_BONUS], indirect=True)
def test_saved_continuous_mode_still_pays_the_fixed_bonus(bonus_step):
    state, _, tick = bonus_step
    state.speed = 2.0
    reward, row = tick()
    assert reward == pytest.approx(0.00375)
    assert 'first_progress_m' not in row


def test_only_first_progress_earns_bonus(bonus_step):
    state, _, tick = bonus_step
    state.lane = 0
    assert tick()[0] == pytest.approx(0.0015)
    assert tick(along=50.25)[0] == 0
    reward, row = tick()
    assert row['progress_m'] == pytest.approx(1.0)
    assert row['first_progress_m'] == pytest.approx(0.25)
    assert reward == pytest.approx(0.000375)
    state.leader = ('leader', -2.4)
    assert tick()[0] == 0
    state.leader = None
    assert tick(along=51.25)[0] == 0
    assert tick()[0] == 0
    assert tick()[0] == pytest.approx(0.0015)


def test_parallel_connector_cannot_reset_progress(bonus_step):
    state, _, tick = bonus_step
    state.index, state.road, state.lane, state.along = 2, ':J105_0', 1, 1.0
    assert tick()[0] == 0
    state.lane, state.along = 0, 1.0
    assert tick()[1]['first_progress_m'] == pytest.approx(0)
    assert tick()[0] == pytest.approx(0.0015)


def test_recrossing_junction_only_pays_for_new_distance(bonus_step):
    state, _, tick = bonus_step
    state.lane, state.along = 0, 882.1
    assert tick(road=':J105_0', along=0.5)[0] == pytest.approx(0.0015)
    assert tick(road='E76', along=882.1)[0] == 0
    assert tick(road=':J105_0', along=0.25)[0] == 0
    assert tick()[1]['first_progress_m'] == pytest.approx(0.75)


def test_progress_bonus_respects_step_cap(bonus_step):
    state, _, tick = bonus_step
    state.lane, state.speed = 0, 120 / 3.6
    reward, row = tick(along=state.along + DEFAULT_REFERENCE_STEP_M + 0.03)
    assert row['first_progress_m'] > DEFAULT_REFERENCE_STEP_M
    assert reward == pytest.approx(0.005)


@pytest.mark.parametrize('bonus_step', [PREPARATION_V1], indirect=True)
def test_v1_first_demand_reduction_is_independent_of_cruise_safety(bonus_step):
    state, tracker, tick = bonus_step
    state.index, state.road, state.lane = 1, 'E75', 5
    phase = tracker.ego_phases['ego_1'][1]
    assert list(phase.demand[i] for i in range(6)) == [0, 0, 1, 2, 3, 4]
    assert phase.normalization == 4
    state.leader = ('leader', -2.4)
    for source, target, expected in ((5, 4, 0.1), (4, 5, 0), (5, 4, 0), (4, 3, 0.1)):
        change = LaneChange('E75', source, target, target - source, deadline=5)
        completion = dict(road='E75', source_lane=source, target_lane=target, time=state.time + 0.1)
        reward, row = tick(change, completion, lane=target, started=True)
        assert row['safety'] == 'unsafe'
        assert row['cruise_bonus'] == 0
        assert row['event_bonus'] == pytest.approx(expected)
        assert reward == pytest.approx(expected)
    stats = tracker.diagnostics['ego_1']['phases']['preparation']
    assert stats['cumulative_event_bonus'] == pytest.approx(0.2)
    assert stats['rewarded_progress_m'] == 0


@pytest.mark.parametrize('bonus_step', [PREPARATION_V1], indirect=True)
def test_v1_reset_and_stage_transition_do_not_carry_minimum_demand(bonus_step):
    state, tracker, tick = bonus_step
    state.index, state.road, state.lane, state.along = 2, ':J105_0', 0, 1.0
    assert tick()[1]['minimum_demand_after'] == 0
    state.index, state.road, state.lane, state.along = 3, 'E77', 1, 10.0
    _, row = tick()
    assert row['minimum_demand_before'] == 1
    assert row['event_bonus'] == 0
    change = LaneChange('E77', 1, 0, -1, deadline=5)
    complete = dict(road='E77', source_lane=1, target_lane=0, time=state.time + 0.1)
    _, row = tick(change, complete, lane=0, started=True)
    assert row['event_bonus'] == pytest.approx(0.1)
    assert row['cruise_bonus'] == pytest.approx(0.005 / V1_REFERENCE_STEP_M)
    tracker.reset(('ego_1',))
    assert not tracker.minimum_demands


@pytest.mark.parametrize('bonus_step', [PREPARATION_V1], indirect=True)
def test_v1_cruise_caps_new_distance_and_does_not_reward_repeated_distance(bonus_step):
    state, _, tick = bonus_step
    state.lane, state.speed = 0, 120 / 3.6
    start = state.along
    _, row = tick(along=start + 3.35)
    assert row['first_progress_m'] > V1_REFERENCE_STEP_M
    assert row['cruise_bonus'] == pytest.approx(0.005)
    tick(along=start)
    _, row = tick(along=start + 3.2)
    assert row['first_progress_m'] == 0
    assert row['cruise_bonus'] == 0


def test_bonus_safety_progress_and_missing_data(bonus_step):
    state, _, tick = bonus_step
    state.lane, state.adjacent = 0, [('neighbor', -2.4)]
    assert tick()[0] == pytest.approx(0.0015)
    state.lane = 4
    bonus, row = tick()
    assert bonus == 0 and row['readiness'] == 0
    state.lane = 0
    state.speed = 0
    bonus, row = tick()
    assert bonus == 0 and row['progress_m'] == 0 and not row['advancing']
    state.speed = 10
    assert tick(along=500)[1]['progress_m'] is None
    assert tick(along=499.5)[1]['progress_m'] is None
    bonus, row = tick(follower=None)
    assert bonus == 0 and row['safety'] == 'unknown'
    state.follower = ('rear', -2.4)
    bonus, row = tick()
    assert bonus == 0 and row['safety'] == 'unsafe'
    state.follower = ('', -1)
    state.leader = ('leader', -2.4)
    assert tick()[1]['safety'] == 'unsafe'
    state.leader = None
    for reason in ('Collision', 'teleport'):
        bonus, row = tick(unavailable_reason=reason)
        assert bonus == 0 and row['unavailable_reason'] == reason


def test_bonus_uses_source_until_stable_and_checks_live_target_risk(bonus_step):
    state, _, tick = bonus_step
    request = LaneChange('E76', 1, 0, -1, deadline=5, max_risk=1)
    state.adjacent = [('neighbor', -2.4)]
    bonus, row = tick(request)
    assert bonus == pytest.approx(0.001125)
    assert row['actual_lateral_motion'] is False
    bonus, row = tick(request, lateral=-0.2, started=True)
    assert bonus == 0 and row['target_lane_risk_after'] > 0
    assert request.max_risk == 1
    state.adjacent = ()
    bonus, row = tick(request, lane=0, lateral=-0.2, offset=1.0)
    assert bonus == pytest.approx(0.001125) and row['local_lane_changes'] == 1
    bonus, row = tick(lateral=0.0, offset=0.8)
    assert bonus == pytest.approx(0.001125) and row['local_lane_changes'] == 1
    completion = {'road': 'E76', 'source_lane': 1, 'target_lane': 0, 'time': 1.5}
    bonus, row = tick(complete=completion, offset=0.0)
    assert bonus == pytest.approx(0.0015) and row['local_lane_changes'] == 0


def test_bonus_connector_demand_and_all_lanes_ready(bonus_step):
    state, tracker, tick = bonus_step
    state.index, state.road, state.lane, state.along = 1, ':J104_0', 2, 1.0
    bonus, row = tick()
    assert bonus == pytest.approx(0.001125)
    assert row['connection'] == ('E75_2', 'E76_1')
    state.index, state.road, state.lane, state.along = 2, ':J105_0', 1, 1.0
    bonus, row = tick()
    assert bonus == 0 and row['readiness'] is None
    assert row['connection'] == ('E76_0', 'E77_1')
    state.lane, state.along = 0, 2.0
    assert tick()[0] == pytest.approx(0.0015)
    network = RouteLaneNetwork(NETWORK)
    all_ready = {'route': list(O3_D3), 'phases': [
        {'kind': 'weaving', 'critical_connection': ['E75', 'E76'],
         'accepted_entry_lanes': [f'E76_{i}' for i in range(5)]},
    ]}
    tracker.reset(('ego_1',))
    tracker.stages[O3_D3] = all_ready['phases']
    tracker.preparation_edges.clear()
    tracker.configure('ego_1', network, network.plan(O3_D3, 'passenger'))
    state.index, state.road, state.lane, state.along = 1, 'E75', 2, 50
    bonus, row = tick()
    assert bonus == pytest.approx(0.0015)
    assert (row['local_lane_changes'], row['normalization'], row['readiness']) == (0, 0, 1.0)


@pytest.mark.parametrize('enabled', [True, False])
def test_real_transition_keeps_speed_and_preparation_independent(tmp_path, enabled):
    cfg = MASACRewardConfig(route_preparation=enabled, lane_change=False)
    route_file = tmp_path / 'no-traffic.rou.xml'
    route_file.write_text("""<routes>
        <trip id="ego_1" from="O3" to="D3" depart="0" departLane="0" departSpeed="10"/>
        <trip id="ego_2" from="O3" to="D3" depart="10" departLane="0" departSpeed="10"/>
    </routes>""")
    env = MultiEgoEnv(reward_config=cfg, route_file=route_file)
    try:
        env.reset(seed=42)
        conn = env.world.conn
        lane = 'E77_1'
        conn.vehicle.moveTo('ego_1', lane, 100)
        _, rewards, _, _, infos = env.step({'ego_1': [0, 0]})
        step = infos['ego_1']['reward_components']
        assert step['route_preparation'] == pytest.approx(0.02 * 0.5 * 0.07235 if enabled else 0)
        assert step['efficiency'] == pytest.approx(0.004)
        assert step['route_guidance'] == 0.0
        assert rewards['ego_1'] == pytest.approx(sum(step.values()))
        diagnostics = env.reward_summary()['route_preparation_diagnostics']['ego_1']
        assert diagnostics['enabled'] is enabled
        if enabled:
            stats = diagnostics['phases']['preparation']
            assert stats['preparation_zone_steps'] == stats['bonus_steps'] == 1
            assert infos['ego_1']['route_preparation']['lane_id'] == lane

            vehicle = conn.vehicle
            vehicle.moveTo('ego_1', 'E76_0', conn.lane.getLength('E76_0') - 0.5)
            _, _, _, _, infos = env.step({'ego_1': [0, 0]})
            row = infos['ego_1']['route_preparation']
            assert row['lane_id'] == 'E76_0'
            assert row['post_lane_id'] == ':J105_0_0'
            assert row['progress_m'] == pytest.approx(1.0)
            assert row['reward'] == pytest.approx(0.02 * 0.17285)
    finally:
        env.close()


def test_real_weaving_boundaries_are_counted_once(tmp_path):
    route_file = tmp_path / 'boundary.rou.xml'
    route_file.write_text("""<routes>
        <trip id="ego_1" from="O3" to="D3" depart="0" departLane="0" departSpeed="10"/>
        <trip id="ego_2" from="O3" to="D3" depart="60" departLane="0" departSpeed="10"/>
    </routes>""")
    legacy_settings = {'mode': PREPARATION_V1, 'stages': DEFAULT_ROUTE_PREPARATION_STAGES}
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_preparation_settings=legacy_settings,
                      route_file=route_file, collect_control_trace=True)
    trace = EvaluationTrajectory(tmp_path / 'trajectory.jsonl.gz', AGENT_IDS)
    try:
        env.reset(seed=42)
        conn = env.world.conn
        trace.reset(42, conn.getVersion()[1])
        rows = []
        for lane, finish in (('E75_2', 'E76'), ('E76_0', 'E77')):
            conn.vehicle.moveTo('ego_1', lane, conn.lane.getLength(lane) - 0.5)
            for _ in range(100):
                lane_before = conn.vehicle.getLaneID('ego_1')
                actions = {'ego_1': [0, 0]}
                before = trace.positions(conn.vehicle, actions)
                _, rewards, _, _, infos = env.step(actions)
                trace.step(env, len(rows), actions, before, rewards, infos, {})
                info = infos['ego_1']
                row = info['route_preparation']
                assert row is not None and row['lane_id'] == lane_before
                assert info['control_trace']['route_preparation'] == row
                assert info['reward_components']['route_preparation'] == row['reward']
                assert row['advancing'] and row['safety'] == 'safe', row
                rows.append(row)
                if info['road_id'] == finish and row['road'] == finish:
                    break
            else:
                pytest.fail(f'Did not complete passage to {finish}')
        assert rows[0]['phase'] == 'preparation'
        assert rows[0]['reward'] == pytest.approx(0.005 * 0.4 * 0.5 / V1_REFERENCE_STEP_M)
        incoming = [r for r in rows if r['road'].startswith(':J104')]
        outgoing = [r for r in rows if r['road'].startswith(':J105')]
        assert incoming and all(r['phase'] == 'weaving' and r['connection'] == ('E75_2', 'E76_1')
                                and r['reward'] == pytest.approx(0.005 * 0.5 / V1_REFERENCE_STEP_M) for r in incoming)
        assert outgoing and all(r['phase'] == 'weaving' and r['connection'] == ('E76_0', 'E77_0')
                                and r['reward'] == pytest.approx(0.005 / V1_REFERENCE_STEP_M) for r in outgoing)
        assert rows[-1]['phase'] == 'exit' and rows[-1]['reward'] == pytest.approx(0.005 / V1_REFERENCE_STEP_M)
        assert all(r['event_bonus'] == 0 for r in rows)
        assert len({r['time_s'] for r in rows}) == len(rows)
        stats = env.route_preparation.diagnostics['ego_1']
        assert stats['preparation_edge_steps'] == sum(r['phase'] == 'preparation' for r in rows)
        for phase in ('weaving', 'exit'):
            region = [r['reward'] for r in rows if r['phase'] == phase]
            assert stats['phases'][phase]['preparation_zone_steps'] == len(region)
            assert stats['phases'][phase]['cumulative_bonus'] == pytest.approx(sum(region))
    finally:
        trace.close()
        env.close()
    with gzip.open(trace.path, 'rt') as stream:
        recorded = [json.loads(line) for line in stream][1:]
    assert [event['egos']['ego_1']['route_preparation'] for event in recorded] == json.loads(json.dumps(rows))
    assert [event['egos']['ego_1']['components']['route_preparation'] for event in recorded] == [
        row['reward'] for row in rows
    ]


def test_real_lane_change_updates_bonus_on_stable_completion(tmp_path):
    route_file = tmp_path / 'lane-change.rou.xml'
    route_file.write_text("""<routes>
        <trip id="ego_1" from="O3" to="D3" depart="0" departLane="0" departSpeed="10"/>
        <trip id="ego_2" from="O3" to="D3" depart="60" departLane="0" departSpeed="10"/>
    </routes>""")
    legacy_settings = {'mode': PREPARATION_V1, 'stages': DEFAULT_ROUTE_PREPARATION_STAGES}
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_preparation_settings=legacy_settings,
                      route_file=route_file)
    try:
        env.reset(seed=42)
        env.world.conn.vehicle.moveTo('ego_1', 'E76_1', 100)
        saw_index_flip_before_completion = False
        for _ in range(100):
            _, _, _, _, infos = env.step({'ego_1': [0, -5]})
            info = infos['ego_1']
            row = info['route_preparation']
            if any(event['event'] == 'completed' for event in info['lane_change_events']):
                assert row['local_lane_changes'] == 0
                assert row['event_bonus'] == pytest.approx(0.1)
                assert row['reward'] == pytest.approx(0.1 + 0.005 / V1_REFERENCE_STEP_M)
                break
            assert row['local_lane_changes'] == 1
            assert row['event_bonus'] == 0 and row['reward'] == pytest.approx(0.005 * 0.5 / V1_REFERENCE_STEP_M), row
            saw_index_flip_before_completion |= row['post_lane_id'] == 'E76_0'
        else:
            pytest.fail('The actual SUMO lane change did not settle')
        assert saw_index_flip_before_completion
    finally:
        env.close()


@pytest.mark.parametrize('lane,speed,accel,weighted_readiness,base_net', [
    ('E75_2', 2, 2, 0.2, -0.00295),
    ('E75_2', 2, 0, 0.2, 0.0008),
    ('E75_2', 10, 2, 0.2, 0.00025),
    ('E76_4', 2, 2, 0, -0.00295),
    ('E76_3', 2, 2, 0, -0.00295),
    ('E76_0', 2, 2, 1.0, -0.00295),
])
def test_real_safe_progress_dense_net(tmp_path, lane, speed, accel, weighted_readiness, base_net):
    route_file = tmp_path / 'dense.rou.xml'
    route_file.write_text(f"""<routes>
        <trip id="ego_1" from="O3" to="D3" depart="0" departLane="0" departSpeed="{speed}"/>
        <trip id="ego_2" from="O3" to="D3" depart="60" departLane="0" departSpeed="2"/>
    </routes>""")
    legacy_settings = {'mode': PREPARATION_V1, 'stages': DEFAULT_ROUTE_PREPARATION_STAGES}
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_preparation_settings=legacy_settings,
                      route_file=route_file)
    try:
        env.reset(seed=42)
        env.world.conn.vehicle.moveTo('ego_1', lane, 100)
        _, _, _, _, infos = env.step({'ego_1': [accel, 0]})
        info = infos['ego_1']
        assert info['route_preparation']['advancing'] and info['route_preparation']['safety'] == 'safe'
        bonus = 0.005 * weighted_readiness * info['route_preparation']['progress_m'] / V1_REFERENCE_STEP_M
        parts = info['reward_components']
        assert parts['route_preparation'] == pytest.approx(bonus)
        assert parts['comfort'] == pytest.approx(-0.00375 if accel else 0)
        assert parts['efficiency'] == pytest.approx(0.01 * speed / 25)
        dense = ('following', 'rear_following', 'lane_change', 'comfort', 'efficiency', 'route_preparation')
        assert sum(parts[key] for key in dense) == pytest.approx(base_net + bonus)
    finally:
        env.close()


@pytest.mark.parametrize('speed,steps', [(2, 100), (10, 20)])
def test_real_safe_twenty_meters_pays_same_route_bonus(tmp_path, speed, steps):
    route_file = tmp_path / 'same-distance.rou.xml'
    route_file.write_text(f"""<routes>
        <trip id="ego_1" from="O3" to="D3" depart="0" departLane="0" departSpeed="{speed}"/>
        <trip id="ego_2" from="O3" to="D3" depart="60" departLane="0" departSpeed="2"/>
    </routes>""")
    legacy_settings = {'mode': PREPARATION_V1, 'stages': DEFAULT_ROUTE_PREPARATION_STAGES}
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_preparation_settings=legacy_settings,
                      route_file=route_file)
    try:
        env.reset(seed=42)
        vehicle = env.world.conn.vehicle
        vehicle.moveTo('ego_1', 'E76_0', 100)
        total = 0.0
        for _ in range(steps):
            _, _, _, _, infos = env.step({'ego_1': [0, 0]})
            row = infos['ego_1']['route_preparation']
            assert row['readiness'] == 1 and row['safety'] == 'safe'
            total += row['reward']
        assert vehicle.getLanePosition('ego_1') == pytest.approx(120)
        assert total == pytest.approx(0.005 * 20 / V1_REFERENCE_STEP_M)
    finally:
        env.close()


def test_actors_only_initialization_checks_hash_and_uses_identical_weights(tmp_path):
    source = MultiAgentGaussianActor(ego_ids=AGENT_IDS, nav_dim=2)
    path = tmp_path / 'source.pt'
    torch.save({'agent_ids': AGENT_IDS, 'global_step': 100, 'config': {'masac_env_version': 3},
                'actors': source.state_dict()}, path)
    actor = MultiAgentGaussianActor(ego_ids=AGENT_IDS, nav_dim=2)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        initialize_actor_weights(actor, path, '0' * 64, False)
    info = initialize_actor_weights(actor, path, sha256(path), False)
    assert info['source_global_step'] == 100
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in source.state_dict().items())
