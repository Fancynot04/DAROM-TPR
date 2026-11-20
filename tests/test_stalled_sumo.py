"""Small deterministic SUMO fixtures for Stalled-v1 shadow mode."""

import pytest

import src.envs.multi_ego_env as env_module
from src.envs.stalled import StalledDetector
from src.envs.sumo_world import SumoWorld
from src.rewards.masac import MASACRewardConfig


EGO_TYPE = '<vType id="ego" length="4.2" minGap="2.5" accel="5" decel="5"/>'
ROUTE = "O3 E75 E76 E77 D3"


def observe_world(tmp_path, route, *, speed, first_pos, leader_speed=None, scheduled=False):
    fixture = tmp_path / "stalled.rou.xml"
    lead = ""
    if leader_speed is not None:
        lead = (f'<vehicle id="leader" type="ego" route="test" depart="0" '
                f'departLane="0" departPos="{first_pos+12}" departSpeed="{leader_speed}"/>')
    fixture.write_text(
        f'<routes>{EGO_TYPE}<route id="test" edges="{route}"/>{lead}'
        f'<vehicle id="ego_1" type="ego" route="test" depart="0" '
        f'departLane="0" departPos="{first_pos}" departSpeed="{speed}"/></routes>'
    )
    world = SumoWorld(route_file=fixture)
    detector = StalledDetector()
    samples = []
    try:
        conn = world.start()
        world.step()
        conn.vehicle.setLaneChangeMode("ego_1", 0)
        if scheduled:
            conn.vehicle.setStop("ego_1", route.split()[0], pos=first_pos+10, duration=20)
        else:
            conn.vehicle.setSpeedMode("ego_1", 0)
            conn.vehicle.setSpeed("ego_1", speed)
        if leader_speed is not None:
            conn.vehicle.setSpeedMode("leader", 0)
            conn.vehicle.setLaneChangeMode("leader", 0)
            conn.vehicle.setSpeed("leader", leader_speed)
        for _ in range(200 if scheduled else 101):
            samples.append(detector.sample(conn, "ego_1", conn.simulation.getTime()))
            world.step()
        return samples
    finally:
        world.close()


@pytest.mark.parametrize("speed, road, expected_progress", [(0, "O3", 0), (.6, "O3", 6), (0, "D3", 0)])
def test_unimpeded_stop_or_creep_in_sumo(tmp_path, speed, road, expected_progress):
    rows = observe_world(tmp_path, ROUTE if road == "O3" else "D3", speed=speed, first_pos=30)
    assert not any(r["would_stall"] for r in rows[:100])
    assert rows[-1]["road"] == road and rows[-1]["would_stall"]
    assert rows[-1]["valid_window_s"] == 10
    assert rows[-1]["window_progress_m"] == pytest.approx(expected_progress, abs=.01)
    assert rows[-1]["leader"] is None


@pytest.mark.parametrize("speed, reason", [(0, "leader_constrained"), (3, "progress_sufficient")])
def test_queue_or_normal_following_in_sumo(tmp_path, speed, reason):
    rows = observe_world(tmp_path, ROUTE, speed=speed, leader_speed=speed, first_pos=30)
    assert not any(r["would_stall"] for r in rows)
    assert rows[-1]["reason"] == reason
    assert rows[-1]["leader"]["id"] == "leader"


def test_sumo_scheduled_wait_resets_window(tmp_path):
    rows = observe_world(tmp_path, ROUTE, speed=2, first_pos=30, scheduled=True)
    assert sum(row["reason"] == "scheduled_stop" for row in rows) > 100
    assert not any(row["would_stall"] for row in rows)
    assert rows[-1]["valid_window_s"] == 0


def test_crossing_real_internal_edge_preserves_progress(tmp_path):
    rows = observe_world(tmp_path, "E76 E77 D3", speed=6, first_pos=860)
    roads = {row["road"] for row in rows}
    assert "E76" in roads and "E77" in roads and any(r.startswith(":J105") for r in roads)
    assert rows[-1]["valid_window_s"] == 10
    assert rows[-1]["window_progress_m"] == pytest.approx(60, abs=.1)
    assert not rows[-1]["would_stall"]


def test_dual_ego_uses_independent_windows_without_changing_outcomes(tmp_path, monkeypatch):
    fixture = tmp_path / "Antwerp_easy_o3_d3_homogeneous_dual.rou.xml"
    fixture.write_text(
        f'<routes>{EGO_TYPE}'
        '<trip id="ego_1" type="ego" from="D3" to="D3" depart="0" departPos="80" departSpeed="0"/>'
        '<trip id="ego_2" type="ego" from="O3" to="D3" depart="3" departPos="30" departSpeed="0"/>'
        '</routes>'
    )
    monkeypatch.setattr(env_module, "SUMO_DIR", tmp_path)
    baseline = []
    for shadow in (False, True):
        env = env_module.MultiEgoEnv(stalled_shadow=shadow)
        try:
            env.reset(seed=42)
            first = {}
            for step in range(145):
                actions = {key: [0, 0] for key, ego in env.egos.items() if ego.state.status == "active"}
                obs, rewards, done, trunc, infos = env.step(actions)
                trajectory = (rewards, done, trunc, {key: value.tobytes() for key, value in obs.items()})
                if not shadow:
                    baseline.append(trajectory)
                    assert all("stalled_shadow" not in row for row in infos.values())
                else:
                    assert trajectory == baseline[step]
                    for key, info in infos.items():
                        result = info["stalled_shadow"]
                        if result["would_stall"]:
                            first.setdefault(key, result)
            assert not any(env.terminations.values())
            if shadow:
                assert first["ego_1"]["road"] == "D3" and first["ego_2"]["road"] == "O3"
                assert first["ego_2"]["trigger_time_s"] - first["ego_1"]["trigger_time_s"] == pytest.approx(3)
        finally:
            env.close()


def test_red_wait_and_clear_passage_on_same_junction(tmp_path):
    import shutil
    import subprocess

    netconvert = shutil.which("netconvert")
    if netconvert is None:
        pytest.skip("SUMO netconvert is required for this junction fixture")

    nodes = tmp_path / "crossing.nod.xml"
    edges = tmp_path / "crossing.edg.xml"
    network = tmp_path / "crossing.net.xml"
    routes = tmp_path / "crossing.rou.xml"
    nodes.write_text(
        '<nodes><node id="W" x="-100" y="0"/><node id="E" x="100" y="0"/>'
        '<node id="N" x="0" y="100"/><node id="S" x="0" y="-100"/>'
        '<node id="center" x="0" y="0" type="traffic_light"/></nodes>'
    )
    edges.write_text(
        '<edges><edge id="west" from="W" to="center"/><edge id="east" from="center" to="E"/>'
        '<edge id="north" from="N" to="center"/><edge id="south" from="center" to="S"/></edges>'
    )
    routes.write_text(
        f'<routes>{EGO_TYPE}<route id="test" edges="west east"/>'
        '<vehicle id="ego_1" type="ego" route="test" depart="0" departPos="90" departSpeed="0"/>'
        '</routes>'
    )
    subprocess.run([netconvert, "--node-files", str(nodes), "--edge-files", str(edges),
                    "--output-file", str(network)], check=True, capture_output=True)

    world = SumoWorld(route_file=routes)
    world.net_file = network
    detector = StalledDetector()
    try:
        conn = world.start()
        world.step()
        conn.vehicle.setSpeedMode("ego_1", 0)
        conn.vehicle.setSpeed("ego_1", 0)
        count = len(conn.trafficlight.getRedYellowGreenState("center"))
        conn.trafficlight.setRedYellowGreenState("center", "r" * count)
        for _ in range(101):
            row = detector.sample(conn, "ego_1", conn.simulation.getTime())
            assert not row["would_stall"] and row["reason"] == "traffic_signal"
            world.step()

        conn.trafficlight.setRedYellowGreenState("center", "G" * count)
        for i in range(101):
            row = detector.sample(conn, "ego_1", conn.simulation.getTime())
            assert row["would_stall"] is (i == 100)
            world.step()
        assert row["valid_window_s"] == 10
    finally:
        world.close()


@pytest.fixture
def stalled_terminate_route(tmp_path, monkeypatch):
    fixture = tmp_path / "Antwerp_easy_o3_d3_homogeneous_dual.rou.xml"
    fixture.write_text(
        f'<routes>{EGO_TYPE}'
        '<trip id="ego_1" type="ego" from="D3" to="D3" depart="0" departPos="80" departSpeed="0"/>'
        '<trip id="ego_2" type="ego" from="O3" to="D3" depart="3" departPos="30" departSpeed="0"/>'
        '</routes>'
    )
    monkeypatch.setattr(env_module, "SUMO_DIR", tmp_path)


def assert_teleport_disabled(sim):
    for option in ("time-to-teleport", "time-to-teleport.highways",
                   "time-to-teleport.disconnected", "time-to-teleport.bidi"):
        assert float(sim.getOption(option)) == -1
    assert sim.getOption("collision.action") == "remove"


@pytest.mark.parametrize("reward_config", [
    None,
    MASACRewardConfig(task=False, merge=False, following=False, lane_change=False,
                      efficiency=False, comfort=False, team=False),
    MASACRewardConfig(),
], ids=["legacy-b0", "all-modules-off", "default-masac"])
def test_sumo_disables_teleport_without_stalled_termination(stalled_terminate_route, reward_config):
    env = env_module.MultiEgoEnv(reward_config=reward_config)
    try:
        env.reset(seed=42)
        assert not env.stalled_terminate
        assert_teleport_disabled(env.world.conn.simulation)
    finally:
        env.close()


def test_stalled_termination_is_independent_and_precedes_timeout(stalled_terminate_route):
    env = env_module.MultiEgoEnv(stalled_terminate=True, max_active_steps=101)
    try:
        env.reset(seed=42)
        sim = env.world.conn.simulation
        assert_teleport_disabled(sim)
        total_penalties = 0
        for index in range(115):
            actions = {k: [0, 0] if k == "ego_1" else [0.5, 0]
                       for k, ego in env.egos.items() if ego.state.status == "active"}
            _, rewards, done, _, infos = env.step(actions)
            if rewards["ego_1"]:
                total_penalties += 1
                assert rewards["ego_1"] == -400
                assert done == {"ego_1": True, "ego_2": False}
                assert infos["ego_1"]["terminal_reason"] == "Stalled"
                assert infos["ego_1"]["active_steps"] == 101
                assert not infos["ego_1"]["pre_step_terminal"]
                stalled_ego2_position = env.world.conn.vehicle.getLanePosition("ego_2")
            assert "stalled_shadow" not in infos["ego_1"]
            assert not sim.getStartingTeleportIDList()
            assert not sim.getEndingTeleportIDList()
        assert total_penalties == 1
        assert env.active_steps["ego_1"] == 101
        assert env.world.conn.vehicle.getLanePosition("ego_2") > stalled_ego2_position
    finally:
        env.close()


def test_sumo_arrival_precedes_stalled_and_timeout(stalled_terminate_route, monkeypatch):
    env = env_module.MultiEgoEnv(stalled_terminate=True, stalled_shadow=True, max_active_steps=2)
    try:
        env.reset(seed=42)
        env.step({"ego_1": [0, 0]})
        conn = env.world.conn
        conn.vehicle.moveTo("ego_1", "D3_0", conn.lane.getLength("D3_0") - 0.01)
        monkeypatch.setattr(env.stall_detectors["ego_1"], "sample", lambda *_: {"would_stall": True})
        _, rewards, done, _, infos = env.step({"ego_1": [5.0, 0]})
        assert done["ego_1"] and infos["ego_1"]["terminal_reason"] == "Arrived"
        assert infos["ego_1"]["stalled_shadow"]["would_stall"]
        assert rewards["ego_1"] == 50.0
    finally:
        env.close()


def test_stalled_termination_rejects_usp():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        env_module.MultiEgoEnv(stalled_terminate=True, usp_enabled=True)


def test_stalled_training_replay_and_checkpoint_evaluation(stalled_terminate_route, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import numpy as np
    import torch

    import evaluate_masac
    import train_masac
    from src.evaluation_results import validate_results

    captured = []

    class ObservedReplay(train_masac.JointReplayBuffer):
        def add(self, **transition):
            super().add(**transition)
            captured.append({key: value.tolist() for key, value in transition.items()})

    def fixed_actions(actors, policy_obs, active_mask, rng, random_exploration):
        joint = np.zeros((2, 2), dtype=np.float32)
        joint[1, 0] = 0.5 if active_mask[1] else 0
        return joint, {key: joint[i].copy() for i, key in enumerate(train_masac.AGENT_IDS)
                       if active_mask[i]}

    monkeypatch.setattr(train_masac, "JointReplayBuffer", ObservedReplay)
    monkeypatch.setattr(train_masac, "choose_actions", fixed_actions)
    output = tmp_path / "training"
    args = train_masac.build_parser().parse_args([
        "--total-steps", "163", "--device", "cpu", "--torch-threads", "1",
        "--learning-starts", "16", "--batch-size", "32", "--train-every", "20",
        "--buffer-capacity", "200", "--max-active-steps", "130", "--no-tensorboard",
        "--stalled-terminate", "--output-dir", str(output),
    ])
    train_masac.main(args)

    indices = [i for i, row in enumerate(captured) if row["rewards"][0] == -400]
    assert len(indices) == 1
    index = indices[0]
    event = captured[index]
    assert event["active_mask"] == event["acted_mask"] == [1.0, 1.0]
    assert event["dones"] == [1.0, 0.0]
    assert event["next_active_mask"] == [0.0, 1.0]
    assert event["team_active"] == event["team_done"] == 1.0
    assert event["team_reward"] == 0.0
    next_sample = captured[index+1]
    assert next_sample["acted_mask"] == next_sample["next_active_mask"] == [0.0, 1.0]
    assert next_sample["rewards"][0] == 0.0 and next_sample["team_active"] == 0.0
    keys = ("rewards", "dones", "active_mask", "acted_mask", "next_active_mask",
            "team_active", "team_done", "team_reward")
    (tmp_path / "stalled-transition.json").write_text(json.dumps(
        {key: event[key] for key in keys}, indent=2,
    ))

    logs = [json.loads(line) for line in (output / "train/metrics.jsonl").read_text().splitlines()]
    row = next(row for row in logs if row["event"] == "episode")
    assert row["ego_1_terminal_reason"] == "Stalled"
    assert row["ego_2_terminal_reason"] == "Timeout"
    assert row["ego_1_stalled_rate"] == row["joint_stalled_rate"] == 100.0
    assert not any(row["event"] == "stalled_shadow" for row in logs)

    class StopPolicy(evaluate_masac.MultiAgentGaussianActor):
        def forward(self, observations, active_mask, deterministic=False):
            return observations.new_zeros((1, 2, 2)), observations.new_zeros((1, 2))

    monkeypatch.setattr(evaluate_masac, "MultiAgentGaussianActor", StopPolicy)
    summary = tmp_path / "evaluation" / "summary.json"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        evaluate_masac.evaluate(SimpleNamespace(
            checkpoint=str(output / "checkpoints" / "masac_seed42_step_163.pt"),
            output=str(summary), max_steps=300, episodes=1, seed_start=200001,
            stochastic=False, trace=False,
        ))
    finally:
        torch.set_num_threads(threads)
    results = validate_results(summary)
    assert results["environment"]["disable_teleport"] is True
    assert results["environment"]["stalled_terminate"] is True
    assert results["joint"]["stalled_rate"] == 100.0
    for ego_id in env_module.MultiEgoEnv.AGENTS:
        assert results["per_agent"][ego_id]["outcome_counts"]["Stalled"] == 1
        assert results["per_agent"][ego_id]["outcome_counts"]["Other"] == 0
