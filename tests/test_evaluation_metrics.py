"""Metric contracts and read-only diagnostics against the frozen reference."""
import ast
import math
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.evaluation_metrics import (
    AgentMetrics, BrakingResponse, aggregate, joint_record, lane_end_arrival_fields, measured_jerk,
)
from src.evaluation_results import validate_results
from src.evaluation_warnings import read_sumo_warnings
from src.envs.multi_ego_env import MultiEgoEnv
from src.envs.sumo_world import SumoWorld
from src.agents.masac import MultiAgentGaussianActor
from train_masac import snapshot, AGENT_IDS


def reference_function(path, name, extra=None):
    source = subprocess.check_output(['git', 'show', f'913a73f:{path}'], text=True)
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'math': math, 'np': np, **(extra or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


def sample(acc=-4.5, pre=True, agent=True):
    return dict(pre=dict(speed=20., acceleration=1., time=1.) if pre else None,
                post_acceleration=acc, post_time=1.1, on_road=acc is not None,
                emergency_stopping=False, agent_step=agent)


def test_reference_jerk_and_braking():
    old = reference_function('evaluate_sac_delay.py', 'measured_jerk')
    for args in [(1., 2., 1., 1.2, 1, True), (1., 9., 0., .1, 0, True),
                 (1., None, 1., 1.1, 2, False), (1., 2., 1., 1., 2, True)]:
        if args[2] == args[3]:
            with pytest.raises(ValueError): old(*args)
            with pytest.raises(ValueError): measured_jerk(*args)
        else:
            assert old(*args) == measured_jerk(*args)
    classify = reference_function('src/merging.py', 'classify_braking_diagnostics',
                                 {'exit_config': SimpleNamespace(HARD_BRAKING_THRESHOLD_MPS2=-4.5)})
    for acc in (-4.50001, -4.5, -4.49999, None):
        m = AgentMetrics(); m.observe(sample(acc), True)
        assert bool(m.hard_braking_steps) == classify(False, acc)[1]
        assert not m.jerks


def test_sumo_emergency_braking_is_separate_terminal_outcome():
    from src.rewards.masac import MASACRewards, MASACRewardConfig
    from src.evaluation_metrics import KNOWN_OUTCOMES

    assert "Emergency Braking" not in KNOWN_OUTCOMES  # preserve older report category contract
    reward = MASACRewards(MASACRewardConfig()).terminal("Emergency Braking")
    assert reward == -400
    stopped = AgentMetrics().finish("Emergency Braking", reward)
    arrived = AgentMetrics().finish("Arrived", 50)
    assert stopped["outcome"] == "Emergency Braking"
    assert not stopped["success"]
    joint = joint_record({"ego_1": stopped, "ego_2": arrived})
    assert not joint["success"]
    stats = aggregate([stopped, arrived])
    assert stats["emergency_braking_rate"] == 50
    assert stats["outcome_counts"]["Emergency Braking"] == 1
    clean_joint = joint_record({"ego_1": arrived, "ego_2": arrived})
    assert joint.keys() == clean_joint.keys()
    assert aggregate([clean_joint, joint])["emergency_braking_rate"] == 50


def test_masks_counts_and_aggregation():
    a = AgentMetrics()
    a.observe(sample(), False)  # first transition: speed/braking, no jerk
    a.observe(sample(2.), False)
    a.observe(sample(-6.), True)  # terminal: braking, no jerk
    r = a.finish('Arrived', 50.)
    assert (r['speed_samples'], r['jerk_samples'], r['hard_braking_samples']) == (3, 1, 3)
    assert r['hard_braking_steps'] == 2
    assert r['mean_abs_jerk_mps3'] == pytest.approx(10.)
    b = AgentMetrics().finish('Timeout', -100.)
    j = joint_record({'a': r, 'b': b})
    assert j['mean_speed_mps'] == 20 and j['mean_speed_mps_valid_agents'] == 1
    assert not j['success'] and j['hard_braking_episode']
    assert aggregate([b])['mean_abs_jerk_mps3'] is None
    old_mean = reference_function('evaluate_sac_delay.py', 'mean_valid_episode_jerk')
    assert aggregate([r, b])['mean_abs_jerk_mps3'] == old_mean([r, b])
    for reason in (None, '', 'Unknown', 'UnknownRemoval'):
        with pytest.raises(RuntimeError): a.finish(reason, 0)
    assert a.finish('LegalNewEvent', 0)['outcome'] == 'Other'
    with pytest.raises(ValueError): AgentMetrics().observe(sample(float('nan')), False)


def rollout(diagnostics, stochastic=False, steps=35):
    random.seed(200001); np.random.seed(200001); torch.manual_seed(200001)
    actors = MultiAgentGaussianActor(); actors.eval()
    env = MultiEgoEnv(collect_metrics=diagnostics)
    trace = []
    try:
        obs, _ = env.reset(seed=200001)
        with torch.inference_mode():
            for _ in range(steps):
                policy, critic, mask = snapshot(env, obs)
                action, _ = actors(torch.from_numpy(policy[None]), torch.from_numpy(mask[None]),
                                   deterministic=not stochastic)
                actions = {key: action[0, i].tolist() for i, key in enumerate(AGENT_IDS) if mask[i] == 1}
                obs, reward, done, trunc, infos = env.step(actions)
                trace.append((action.numpy().copy(), np.stack(list(obs.values())), reward, done, trunc,
                              env.step_count, tuple(infos[k]['terminal_reason'] for k in AGENT_IDS)))
                if all(done.values()): break
    finally:
        env.close()
    return trace, torch.get_rng_state(), np.random.get_state(), random.getstate()


def assert_trace_equal(a, b):
    assert len(a[0]) == len(b[0])
    for x, y in zip(a[0], b[0]):
        np.testing.assert_array_equal(x[0], y[0]); np.testing.assert_array_equal(x[1], y[1])
        assert x[2:] == y[2:]
    assert torch.equal(a[1], b[1])
    np.testing.assert_equal(a[2], b[2]); assert a[3] == b[3]


def test_real_sumo_diagnostic_behavior_and_rng_parity():
    assert_trace_equal(rollout(False, steps=4500), rollout(True, steps=4500))
    first = rollout(True, stochastic=True)
    assert_trace_equal(first, rollout(True, stochastic=True))
    assert_trace_equal(first, rollout(False, stochastic=True))


def test_evaluator_files_and_incomplete_error():
    from pathlib import Path
    import uuid
    tmp_path = Path("tmp/ai/masac_metrics") / str(uuid.uuid4())
    tmp_path.mkdir(parents=True)
    import csv
    import gzip
    import json
    from evaluate_masac import evaluate
    torch.manual_seed(19)
    actors = MultiAgentGaussianActor()
    checkpoint = tmp_path / 'fixture.pt'
    torch.save({'agent_ids': AGENT_IDS, 'actors': actors.state_dict(), 'global_step': 0}, checkpoint)
    args = SimpleNamespace(checkpoint=checkpoint, episodes=1, seed_start=200001,
                           max_steps=4500, stochastic=False, output=tmp_path / 'summary.json')
    report = evaluate(args)
    assert json.loads(args.output.read_text()) == report
    with (tmp_path / 'episodes.csv').open() as stream:
        row, = list(csv.DictReader(stream))
    assert int(row['global_steps']) == report['global_steps']
    assert int(row['agent_steps_sum']) == report['agent_steps_sum']
    assert report['agent_steps_sum'] == sum(r['agent_steps'] for r in report['per_agent'].values())
    assert report['global_steps'] < report['agent_steps_sum']
    assert 'joint_lane_end_block_free_arrival' in row
    assert report['records'][0]['traffic_sumo_warnings'] is not None
    assert validate_results(args.output) == report
    trace = tmp_path / report['trajectory']['file']
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        samples = [json.loads(line) for line in stream]
    assert samples[0]['event'] == 'reset' and samples[0]['seed'] == 200001
    assert len(samples) - 1 == report['global_steps']
    assert samples[1]['egos']['ego_1']['pre_position'] is not None
    assert samples[1]['egos']['ego_1']['control'] is not None
    observed_terminals = {ego: sample['egos'][ego]['terminal']
                          for sample in samples[1:] for ego in AGENT_IDS
                          if sample['egos'][ego]['terminal'] is not None}
    assert observed_terminals == report['records'][0]['outcomes']
    args.max_steps = 1
    args.output = tmp_path / 'incomplete' / 'summary.json'
    with pytest.raises(RuntimeError, match='incomplete episode'): evaluate(args)
    assert not args.output.exists()
    assert not (args.output.parent / trace.name).exists()


def test_lane_end_warnings_are_grouped_and_exclude_only_blocked_arrivals(tmp_path):
    path = tmp_path / "warnings.log"
    path.write_text("\n".join([
        "Warning: Vehicle 'background' performs emergency braking on lane 'E77_0' "
        "with decel=9.00, wished=4.50, severity=1.00, time=304.80.",
        "Warning: Vehicle 'ego_2' performs emergency braking on lane ':J106_0_0' "
        "with decel=9.00, wished=4.50, severity=1.00, time=304.80.",
        *[
            "Warning: Vehicle 'ego_1' performs emergency stop at the end of lane 'E77_2' "
            f"because there is no connection to the next edge (decel=4.01, offset=0.04), time={time}."
            for time in ("304.80", "304.90", "305.00", "305.10", "305.20", "305.40")
        ],
    ]), encoding="utf-8")
    warnings = read_sumo_warnings(path, 0.1)
    assert not warnings["background"]["lane_end_blocks"]
    assert warnings["background"]["emergency_braking_warnings"][0]["severity"] == 1.0
    assert not warnings["ego_2"]["lane_end_blocks"]
    first, second = warnings["ego_1"]["lane_end_blocks"]
    assert first["lane_id"] == "E77_2"
    assert (first["first_time_s"], first["last_time_s"], first["warning_count"]) == (304.8, 305.2, 5)
    assert second["warning_count"] == 1

    agents = {ego_id: AgentMetrics().finish("Arrived", 0) for ego_id in AGENT_IDS}
    for ego_id, agent in agents.items():
        agent.update(lane_end_arrival_fields(True, bool(warnings.get(ego_id, {}).get("lane_end_blocks"))))
    joint = joint_record(agents)
    assert agents["ego_2"]["lane_end_block_free_arrival"]
    assert joint["raw_arrival"]
    assert joint["arrived_after_lane_end_block"]
    assert not joint["lane_end_block_free_arrival"]
    totals = aggregate([joint])
    assert totals["raw_arrival_rate"] == 100
    assert totals["lane_end_block_free_arrival_rate"] == 0
    assert totals["arrivals_after_lane_end_block"] == 1

    assert read_sumo_warnings(tmp_path / "missing.log", 0.1) is None
    agents["ego_1"].update(lane_end_arrival_fields(True, None))
    totals = aggregate([joint_record(agents)])
    assert totals["lane_end_block_free_arrival_rate"] is None
    assert totals["lane_end_block_free_arrival_unknown_episodes"] == 1


def test_real_sumo_writes_wrong_lane_stop_to_warning_log(tmp_path):
    route = tmp_path / "wrong-lane.rou.xml"
    route.write_text("""<routes>
        <route id="to-exit" edges="E77 D3"/>
        <vehicle id="ego_1" route="to-exit" depart="0" departLane="2" departSpeed="0"/>
    </routes>""", encoding="utf-8")
    warning_log = tmp_path / "sumo.log"
    world = SumoWorld(route_file=route, warning_log=warning_log)
    try:
        conn = world.start()
        world.step()
        vehicle = conn.vehicle
        lane = "E77_2"
        vehicle.setLaneChangeMode("ego_1", 0)
        vehicle.setSpeedMode("ego_1", 96)
        vehicle.moveTo("ego_1", lane, conn.lane.getLength(lane) - 0.04)
        vehicle.setSpeed("ego_1", 2.0)
        for _ in range(5):
            world.step()
    finally:
        world.close()
    blocks = read_sumo_warnings(warning_log, world.step_length)["ego_1"]["lane_end_blocks"]
    assert blocks[0]["lane_id"] == lane


def test_real_terminal_braking_before_removal():
    env = MultiEgoEnv(collect_metrics=True)
    try:
        env.reset(seed=200001)
        active = [k for k in AGENT_IDS if env.egos[k].state.status == 'active']
        for key in active:
            env.active_steps[key] = env.max_active_steps - 1
        _, rewards, done, _, infos = env.step({key: [-6., 0.] for key in active})
        for key in active:
            assert done[key]
            raw = infos[key]['evaluation_metrics']
            assert raw['post_acceleration'] is not None
            assert key not in env.world.conn.vehicle.getIDList()
            m = AgentMetrics(); m.observe(raw, True)
            record = m.finish(infos[key]['terminal_reason'], rewards[key])
            assert record['hard_braking_samples'] == 1
            assert record['jerk_samples'] == 0
            assert record['hard_braking_steps'] == int(raw['post_acceleration'] <= -4.5)
    finally:
        env.close()


def test_frozen_reference_speed_success_and_recorder_dt():
    source = subprocess.check_output(['git', 'show', '913a73f:evaluate_sac_delay.py'], text=True)
    tree = ast.parse(source)
    expressions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in ('mean_speed_mps', 'success'):
                    expressions[key.value] = value
    for speeds in ([], [10.], [10., 20., 30.]):
        m = AgentMetrics()
        for speed in speeds:
            raw = sample(); raw['pre']['speed'] = speed
            m.observe(raw, False)
        old = eval(compile(ast.Expression(expressions['mean_speed_mps']), '<reference>', 'eval'),
                   {'np': np, 'speed_samples': speeds})
        assert m.finish('Arrived', 0)['mean_speed_mps'] == old
    for outcome in ('Arrived', 'Collision', 'Timeout'):
        old = eval(compile(ast.Expression(expressions['success']), '<reference>', 'eval'),
                   {'outcome': outcome, 'teleported': False})
        assert AgentMetrics().finish(outcome, 0)['success'] == old
    recorder = subprocess.check_output(['git', 'show', '913a73f:src/trajectory_recorder.py'], text=True)
    assert "(after - before) / self.dt" in recorder



def test_braking_response_times_follow_first_ttc_encounter():
    response = BrakingResponse()

    def step(time, requested, measured, risk):
        response.observe(
            {"agent_step": True, "pre": {"time": 120 + time},
             "depart_time": 120, "post_time": 120 + time + 0.1, "post_acceleration": measured},
            [requested, 0], {"ttc_risk": risk},
        )

    step(1.0, 0, -0.3, 0)
    step(1.1, 0, 0.0, 0.5)
    step(1.2, -1.0, -0.7, 0.6)
    step(1.3, 0, 0.0, 0.0)
    step(1.4, -2.0, -2.0, 0.8)

    assert response.result() == {
        "first_requested_brake_since_departure_s": 1.2,
        "first_ttc_risk_since_departure_s": 1.1,
        "first_ttc_risk_to_requested_brake_s": 0.1,
        "first_ttc_risk_to_measured_brake_s": 0.2,
    }
