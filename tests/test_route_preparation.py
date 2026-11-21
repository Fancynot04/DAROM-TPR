from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from traci.exceptions import TraCIException

from src.agents.masac import MultiAgentGaussianActor
from src.envs.multi_ego_env import MultiEgoEnv
from src.envs.route_lane_plan import RouteLaneNetwork
from src.envs.route_preparation import (
    DEFAULT_ROUTE_PREPARATION_STAGES, DEFAULT_ROUTE_PREPARATION_WEIGHTS, RoutePreparation,
)
from src.experiment_tracking import sha256
from src.rewards.masac import MASACRewardConfig
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


@pytest.mark.parametrize('enabled,expected_penalty', [(True, True), (False, False)])
def test_real_transition_keeps_speed_and_preparation_independent(tmp_path, enabled, expected_penalty):
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
        lane = 'E77_2'
        conn.vehicle.moveTo('ego_1', lane, 100)
        _, rewards, _, _, infos = env.step({'ego_1': [0, 0]})
        step = infos['ego_1']['reward_components']
        assert (step['route_preparation'] < 0) is expected_penalty
        assert step['efficiency'] == pytest.approx(0.005)
        assert step['route_guidance'] == 0.0
        assert rewards['ego_1'] == pytest.approx(sum(step.values()))
        diagnostics = env.reward_summary()['route_preparation_diagnostics']['ego_1']
        assert diagnostics['enabled'] is enabled
        if enabled:
            stats = diagnostics['phases']['exit']
            assert stats['preparation_zone_steps'] == stats['safe_opportunity_steps'] == stats['penalty_steps'] == 1
            assert stats['first_penalty']['distance_m'] == pytest.approx(72.35)
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
