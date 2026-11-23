"""Regression tests for MASAC safety rewards and Actor route observations."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.envs.multi_ego_env import MultiEgoEnv
from src.envs.observation import RouteGoal, remaining_goal_distance
from src.envs.route_lane_plan import DEFAULT_ROUTE_GUIDANCE_RULES, RouteLaneNetwork
from src.envs.sumo_world import SUMO_DIR
from src.rewards.masac import (
    LEGACY_MASAC_REWARD_PROTOCOL, MASACRewardConfig, MASACRewardProtocol, MASACRewards, required_gap,
)
from train_masac import snapshot


@pytest.mark.parametrize("delta, expected", [(0.4, -0.001), (1.5, -0.0025)])
def test_default_comfort_penalty_scales_and_clips(delta, expected):
    config = MASACRewardConfig(following=False, rear_following=False, lane_change=False,
                               efficiency=False, merge=False)
    tracker = MASACRewards(config)
    vehicle = Mock()
    vehicle.getAcceleration.side_effect = [0.0, delta]
    conn = SimpleNamespace(vehicle=vehicle, simulation=Mock())
    tracker.before_step(conn, "ego", SimpleNamespace(target_lane=None))
    rewards = tracker.after_step(conn, "ego", on_road=True, needs_merge=False)
    assert rewards["comfort"] == pytest.approx(expected)


def test_terminal_penalty_scaling_preserves_success_and_team_rewards():
    tracker = MASACRewards(MASACRewardConfig(), MASACRewardProtocol(terminal_penalty_scale=0.2))
    assert tracker.terminal("Collision") == -100
    for event in ("Missed Exit", "Stalled", "Timeout"):
        assert tracker.terminal(event) == -80
    assert tracker.terminal("Emergency Braking") == -400
    assert tracker.terminal("Arrived") == 50
    assert tracker.settle_team({"a": "Collision", "b": "Arrived"}) == 0
    assert tracker.settle_team({"a": "Arrived", "b": "Arrived"}) == 100


@pytest.mark.parametrize("scale,other_penalty", [(1.0, -400), (0.2, -80)])
def test_collision_penalty_override_is_independent(scale, other_penalty):
    protocol = MASACRewardProtocol(terminal_penalty_scale=scale, collision_penalty=1000)
    tracker = MASACRewards(MASACRewardConfig(), protocol)
    assert tracker.terminal("Collision") == -1000
    assert tracker.terminal("Arrived") == 50
    for event in ("Missed Exit", "Stalled", "Timeout"):
        assert tracker.terminal(event) == other_penalty
    task_disabled = MASACRewards(MASACRewardConfig(task=False), protocol)
    assert task_disabled.terminal("Collision") == 0
    assert task_disabled.terminal("Emergency Braking") == -400


def test_lane_change_plan_uses_all_natural_successors():
    network = RouteLaneNetwork(SUMO_DIR / "Antwerp.net.xml")
    corridor = ("O3", "E75", "E76", "E77")
    d1 = network.plan(corridor + ("E78", "E82", "D1"), "passenger")
    d2 = network.plan(corridor + ("E78", "E82", "D2"), "passenger")
    d3 = network.plan(corridor + ("D3",), "passenger")
    assert [plan.signed_changes[3, 0] for plan in (d1, d2, d3)] == [3, 1, 0]
    assert [plan.signed_changes[3, 3] for plan in (d1, d2, d3)] == [0, 0, -3]
    assert d3.signed_changes[1, 1] == 0
    assert d3.signed_changes[2, 2] == -2
    vehicle = Mock()
    vehicle.getRoadID.return_value = ":J106_0"
    vehicle.getLaneID.return_value = ":J106_0_0"
    vehicle.getRouteIndex.return_value = 3
    assert d1.position(vehicle, "ego") is None
    assert d3.position(vehicle, "ego") == (4, 0)


@pytest.mark.parametrize("via,destination,expected", [
    ("E76_0", "E77_0", 5),
    (":J105_0_0", "E77_0", 5),
    (":J105_0_1", "E77_1", 0),
    (":J105_0_1", "E77_0", 0),
])
def test_route_guidance_scores_traversed_link_once(via, destination, expected):
    rule = DEFAULT_ROUTE_GUIDANCE_RULES[0]
    route = tuple(rule["route"])
    plan = RouteLaneNetwork(SUMO_DIR / "Antwerp.net.xml").plan(route, "passenger")
    vehicle = Mock()
    vehicle.getRoute.return_value = route
    vehicle.getRoadID.return_value = "E77"
    vehicle.getLaneID.return_value = destination
    vehicle.getRouteIndex.return_value = 3
    critical = tuple(rule["critical_connection"])
    accepted = frozenset(map(tuple, rule["accepted_connections"]))
    tracker = MASACRewards(MASACRewardConfig(route_guidance=True))
    connection = plan.completed_connection((2, via), vehicle, "ego", critical)
    assert tracker.score_route_guidance("ego", connection, accepted) == expected
    assert tracker.score_route_guidance("ego", connection, accepted) == 0
    assert tracker.score_route_guidance("other", connection, accepted) == expected
    assert plan.completed_connection((3, destination), vehicle, "ego", critical) is None
    tracker.reset(("ego",))
    assert tracker.score_route_guidance("ego", connection, accepted) == expected


@pytest.mark.parametrize("event", ["Arrived", "Collision", "Missed Exit", "Stalled", "Timeout", "Emergency Braking"])
@pytest.mark.parametrize("protocol,immediate_bonus,deferred_bonus", [
    (MASACRewardProtocol(), 5, 0),
    (MASACRewardProtocol.from_checkpoint({"merge_on_success": True}), 0, 25),
    (LEGACY_MASAC_REWARD_PROTOCOL, 25, 0),
])
def test_merge_bonus_and_efficiency(event, protocol, immediate_bonus, deferred_bonus):
    config = MASACRewardConfig(following=False, rear_following=False, lane_change=False, comfort=False)
    tracker = MASACRewards(config, protocol)
    vehicle = Mock()
    vehicle.getRoadID.return_value = "E76"
    control = SimpleNamespace(target_lane=None)
    conn = SimpleNamespace(vehicle=vehicle, simulation=Mock())
    for key, speed, needs_merge in (("ramp", 20, True), ("mainline", 30, False), ("slow", 10, True)):
        vehicle.getSpeed.return_value = speed
        tracker.before_step(conn, key, control)
        values = tracker.after_step(conn, key, True, needs_merge)
        assert values["efficiency"] == pytest.approx(protocol.efficiency_weight * min(1, speed / 20))
        assert values["merge"] == (immediate_bonus if needs_merge else 0)
        assert values["task"] == 0
        assert values["route_guidance"] == 0
    assert tracker.terminal("Arrived") == 50
    assert tracker.settle_merge("mainline", "Arrived") == 0
    assert tracker.settle_merge("ramp", event) == (deferred_bonus if event == "Arrived" else 0)
    assert tracker.settle_merge("ramp", "Arrived") == 0
    tracker.reset(("ramp", "slow"))
    assert tracker.settle_merge("slow", "Arrived") == 0


@pytest.mark.parametrize("collect_metrics", [False, True])
def test_sumo_emergency_braking_is_mandatory_and_independent_per_ego(tmp_path, monkeypatch, collect_metrics):
    route = tmp_path / "emergency-braking.rou.xml"
    route.write_text("""<routes>
        <trip id="ego_1" depart="0" from="O3" to="D3" departSpeed="20"/>
        <trip id="ego_2" depart="3" from="E76" to="D3"/>
    </routes>""")
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_file=route, collect_metrics=collect_metrics)
    try:
        env.reset(seed=42)
        monkeypatch.setattr(env.world.conn.simulation,
                            "getEmergencyStoppingVehiclesIDList", lambda: ["ego_1", "traffic_1"])
        _, rewards, terminated, _, infos = env.step({"ego_1": [0, 0]})
        assert terminated == {"ego_1": True, "ego_2": False}
        assert infos["ego_1"]["terminal_reason"] == "Emergency Braking"
        assert infos["ego_1"]["reward_components"]["task"] == -400
        assert rewards["ego_1"] < -399
        assert "ego_1" not in env.world.conn.vehicle.getIDList()
        if collect_metrics:
            assert infos["ego_1"]["evaluation_metrics"]["emergency_stopping"]
    finally:
        env.close()


@pytest.mark.parametrize("old_checkpoint", [False, True])
def test_sumo_merge_payment_precedes_timeout_for_new_protocol(tmp_path, old_checkpoint):
    route = tmp_path / "timeout.rou.xml"
    route.write_text("""<routes>
        <trip id="ego_1" depart="0" from="O3" to="D3" departSpeed="20"/>
        <trip id="ego_2" depart="3" from="E76" to="D3"/>
    </routes>""")
    protocol = (MASACRewardProtocol.from_checkpoint({"merge_on_success": True})
                if old_checkpoint else MASACRewardProtocol())
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), reward_protocol=protocol,
                      route_file=route, max_active_steps=2)
    try:
        env.reset()
        env.world.conn.vehicle.moveTo("ego_1", "E76_0", 30)
        _, _, _, _, infos = env.step({"ego_1": [0, 0]})
        assert "ego_1" in env.reward_tracker.merged
        assert infos["ego_1"]["reward_components"]["merge"] == (0 if old_checkpoint else 5)
        _, _, terminated, _, infos = env.step({"ego_1": [0, 0]})
        assert terminated == {"ego_1": True, "ego_2": False}
        assert infos["ego_1"]["terminal_reason"] == "Timeout"
        values = infos["ego_1"]["reward_components"]
        assert (values["task"], values["merge"], values["route_guidance"]) == (-400, 0, 0)
        assert env.reward_summary()["reward_components"]["ego_1"]["merge"] == (0 if old_checkpoint else 5)
    finally:
        env.close()


@pytest.mark.parametrize(
    "edge,goal,wrong_lane,right_lane,changes",
    [("E82", "D1", 2, 4, 1), ("E82", "D2", 3, 1, -1), ("E77", "D3", 4, 0, -4)],
)
def test_lane_change_actor_and_missed_exit_with_sumo(tmp_path, edge, goal, wrong_lane, right_lane, changes):
    route = tmp_path / "exits.rou.xml"
    route.write_text(f"""<routes>
        <vType id="ego" maxSpeed="33.3333333333" speedFactor="1"/>
        <trip id="ego_1" type="ego" depart="0" from="{edge}" to="{goal}"
              departLane="{wrong_lane}" departSpeed="25"/>
        <trip id="ego_2" type="ego" depart="3" from="{edge}" to="{goal}"
              departLane="{right_lane}" departPos="-10" departSpeed="20" arrivalPos="20"/>
    </routes>""")
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_file=route)
    try:
        obs, infos = env.reset()
        actor, critic, _ = snapshot(env, obs)
        assert infos["ego_1"]["signed_min_lane_changes"] == changes
        assert actor[0, 1] == changes
        assert critic[0, 1] == wrong_lane

        env.step({"ego_1": [0, 0]})
        conn = env.world.conn
        lane_id = f"{edge}_{wrong_lane}"
        conn.vehicle.moveTo("ego_1", lane_id, conn.lane.getLength(lane_id) - 5)
        _, rewards, terminated, _, infos = env.step({"ego_1": [0, 0]})
        assert terminated == {"ego_1": True, "ego_2": False}
        assert infos["ego_1"]["terminal_reason"] == "Missed Exit"
        assert infos["ego_1"]["pre_step_terminal"]
        assert rewards["ego_1"] == -400

        for _ in range(100):
            actions = {"ego_2": [0, 0]} if infos["ego_2"]["status"] == "active" else {}
            _, _, terminated, _, infos = env.step(actions)
            if terminated["ego_2"]:
                break
            if infos["ego_2"]["status"] == "active":
                assert infos["ego_2"]["signed_min_lane_changes"] == 0
        assert infos["ego_2"]["terminal_reason"] == "Arrived"
    finally:
        env.close()


def test_route_deviation_settles_before_next_actor_observation(tmp_path, monkeypatch):
    route = tmp_path / "deviation.rou.xml"
    route.write_text("""<routes>
        <vType id="ego" maxSpeed="33.3333333333" speedFactor="1"/>
        <trip id="ego_1" type="ego" depart="0" from="E77" to="D2"
              departLane="0" departSpeed="20"/>
        <trip id="ego_2" type="ego" depart="3" from="E77" to="D3"/>
    </routes>""")
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_file=route)
    try:
        env.reset()
        step = env.world.step

        def step_with_deviation():
            time = step()
            env.world.conn.vehicle.setRoute("ego_1", ["E77", "D3"])
            return time

        monkeypatch.setattr(env.world, "step", step_with_deviation)
        _, _, terminated, _, infos = env.step({"ego_1": [0, 0]})
        ego = infos["ego_1"]
        assert terminated == {"ego_1": True, "ego_2": False}
        assert ego["terminal_reason"] == "Missed Exit"
        assert ego["active_steps"] == 1
        assert not ego["pre_step_terminal"]
        assert ego["reward_components"]["task"] == -400
    finally:
        env.close()


@pytest.mark.parametrize(
    "speed,front,gap,gap_risk,ttc_risk",
    [
        (20, 20, 30, 0, 0),
        (30, 0, 60, 0, 1 / 3),
        (20, 20, 3, 0.9, 0),
        (20, 25, 15, 0.5, 0),
        (0, 10, 2, 0, 0),
        (30, 10, 60, 0, 0),
        (30, None, 0, 0, 0),
        (30, 0, -1, 1, 1),
        (20, 0, 20, 1 / 3, 2 / 3),
    ],
)
@pytest.mark.parametrize("rear", [False, True])
def test_following_net_gap_and_ttc(speed, front, gap, gap_risk, ttc_risk, rear):
    tracker = MASACRewards(MASACRewardConfig(
        following=not rear, rear_following=rear, lane_change=False,
    ))
    vehicle = Mock()
    neighbor = ("neighbor", gap - (3.5 if rear else 2.5)) if front is not None else ("", -1)
    vehicle.getMinGap.side_effect = {"ego": 2.5, "neighbor": 3.5}.__getitem__
    if rear:
        vehicle.getFollower.return_value = neighbor
        vehicle.getSpeed.side_effect = {"ego": front if front is not None else 0, "neighbor": speed}.__getitem__
    else:
        vehicle.getLeader.return_value = neighbor
        vehicle.getSpeed.side_effect = {"ego": speed, "neighbor": front}.__getitem__
    tracker.before_step(SimpleNamespace(vehicle=vehicle), "ego", None)
    query = vehicle.getFollower if rear else vehicle.getLeader
    query.assert_called_once_with("ego", 120.0)
    disabled_query = vehicle.getLeader if rear else vehicle.getFollower
    disabled_query.assert_not_called()
    values = tracker.after_step(None, "ego", False, False)
    penalty = -0.1 * max(gap_risk, ttc_risk)
    assert tracker.following_risks["ego"] == pytest.approx({
        "gap_risk": 0 if rear else gap_risk,
        "ttc_risk": 0 if rear else ttc_risk,
        "following_reward": 0 if rear else penalty,
        "rear_gap_risk": gap_risk if rear else 0,
        "rear_ttc_risk": ttc_risk if rear else 0,
        "rear_following_reward": penalty if rear else 0,
    })
    assert values["following"] == pytest.approx(0 if rear else penalty)
    assert values["rear_following"] == pytest.approx(penalty if rear else 0)


@pytest.fixture
def lane_change():
    tracker = MASACRewards(MASACRewardConfig(
        following=False, rear_following=False, merge=False, comfort=False,
    ))
    lanes = {"a": 0, "b": 0}
    lateral_speed = dict.fromkeys(lanes, 0)
    vehicle = Mock()
    vehicle.getRoadID.return_value = "E76"
    vehicle.getSpeed.return_value = 20
    vehicle.getLaneIndex.side_effect = lanes.__getitem__
    vehicle.getLaneID.side_effect = lambda key: f"E76_{lanes[key]}"
    vehicle.getLateralSpeed.side_effect = lateral_speed.__getitem__
    vehicle.getLateralLanePosition.return_value = 0.0
    vehicle.getLength.return_value = 4.0
    vehicle.getWidth.return_value = 2.0
    vehicle.getMinGap.return_value = 2.5
    vehicle.getLeftFollowers.return_value = []
    vehicle.getRightLeaders.side_effect = AssertionError("wrong adjacent lane")
    lane = Mock()
    lane.getWidth.return_value = 3.2
    lane.getLastStepVehicleIDs.side_effect = {"E76_1": ["a", "b", "lead"]}.__getitem__
    time = 0.0
    conn = SimpleNamespace(vehicle=vehicle, lane=lane, simulation=SimpleNamespace(getTime=lambda: time))

    def tick(entries):
        nonlocal time
        for key, (target, _, _, risk, _) in entries.items():
            set_risk(key, risk)
            control = SimpleNamespace(target_lane=target, lane_duration=4)
            tracker.before_step(conn, key, control)
        time += 0.1
        for key, (_, post_lane, motion, _, _) in entries.items():
            lanes[key], lateral_speed[key] = post_lane, motion
        values = {}
        for key, (_, _, _, risk, terminal) in entries.items():
            set_risk(key, risk)
            values[key] = tracker.after_step(
                conn, key, terminal is None, False, terminal_event=terminal,
            )["lane_change"]
        return values

    def set_risk(ego, risk):
        gap = required_gap(20, 20) * (1 - risk)
        vehicle.getLeftLeaders.return_value = [("lead", gap - 2.5)]
        positions = {ego: 100, "lead": 104 + gap}
        vehicle.getLanePosition.side_effect = lambda key: positions.get(key, 500)

    return tracker, tick


def test_lane_change_risk_increments_and_stable_target(lane_change):
    tracker, tick = lane_change
    assert tick({"a": (1, 0, 0, 0.9, None)})["a"] == 0
    assert not tracker.lane_change_events
    steps = [
        (1, 0, 1, 0.2, None),
        (1, 0, 1, 0.4, None),
        (1, 1, 1, 0.8, None),
        (2, 1, 1, 0.5, None),
        (None, 1, 0, 0.3, None),
    ]
    rewards = [tick({"a": step})["a"] for step in steps]
    assert rewards == pytest.approx([-0.1, -0.1, -0.2, 0, 0])
    started, completed = tracker.lane_change_events
    assert started["event"] == "started"
    assert completed["event"] == "completed"
    assert completed["target_lane"] == 1
    assert completed["max_risk"] == pytest.approx(0.8)
    assert completed["cumulative_penalty"] == pytest.approx(sum(rewards))
    assert not tracker.pending_changes


def test_lane_change_collision_and_reversal_are_independent(lane_change):
    tracker, tick = lane_change
    assert tick({
        "a": (1, 0, 1, 0.2, None), "b": (1, 0, 1, 0.1, None),
    }) == pytest.approx({"a": -0.1, "b": -0.05})
    assert tick({
        "a": (1, 0, 0, 0.7, "Collision"), "b": (1, 0, 1, 0.4, None),
    }) == pytest.approx({"a": -0.25, "b": -0.15})
    assert set(tracker.pending_changes) == {"b"}
    assert tick({"b": (None, 0, -1, 0.25, None)}) == {"b": 0}
    collision, reversal = tracker.lane_change_events[-2:]
    assert collision["reason"] == "Collision"
    assert collision["cumulative_penalty"] == pytest.approx(-0.35)
    assert reversal["reason"] == "reversed_or_returned"
    assert reversal["cumulative_penalty"] == pytest.approx(-0.2)
    assert not tracker.pending_changes
    assert tracker.terminal("Collision") == -500


@pytest.mark.parametrize(
    "origin,goal,arrival",
    [("E76", "D3", 120), ("O1", "D3", 120), ("O1", "D1", 0)],
)
def test_route_goal_observations_with_sumo(tmp_path, origin, goal, arrival):
    route = tmp_path / "dual.rou.xml"
    route.write_text(f"""<routes>
        <vType id="ego" maxSpeed="33.3333333333" speedFactor="1"/>
        <trip id="ego_1" type="ego" depart="0" from="O3" to="D3" departSpeed="25"/>
        <trip id="ego_2" type="ego" depart="3" from="{origin}" to="{goal}"
              departLane="0" departSpeed="20" arrivalPos="{arrival}"/>
    </routes>""")
    env = MultiEgoEnv(reward_config=MASACRewardConfig(route_guidance=True), route_file=route)
    seen = {ego: set() for ego in env.AGENTS}
    previous = {}
    try:
        obs, infos = env.reset()
        assert infos["ego_2"]["status"] == "waiting"
        assert "remaining_goal_distance_m" not in infos["ego_2"]
        assert infos["ego_1"]["remaining_goal_distance_m"] > 1800
        with pytest.raises(RuntimeError, match="invalid remaining distance"):
            remaining_goal_distance(env.world.conn, "ego_1", RouteGoal("invalid", 1))
        with pytest.raises(RuntimeError, match="unable to query route goal"):
            remaining_goal_distance(env.world.conn, "missing", RouteGoal("D3", 1))

        for _ in range(2000):
            actor, critic, mask = snapshot(env, obs)
            assert actor.shape == (2, 136)
            assert critic.shape == (2, 95)
            actions = {}
            for i, ego in enumerate(env.AGENTS):
                if not mask[i]:
                    continue
                info = infos[ego]
                distance = info["remaining_goal_distance_m"]
                assert actor[i, 0] == pytest.approx(distance)
                assert critic[i, 0] == obs[ego][0, 0]
                if ego in previous:
                    old_road, old_distance = previous[ego]
                    # SUMO uses estimated lengths for downstream internal connectors.
                    if old_road == info["road_id"]:
                        assert 0 <= old_distance - distance <= 3.34 + 1e-6
                    else:
                        assert abs(old_distance - distance) <= 9
                previous[ego] = info["road_id"], distance
                seen[ego].add(info["road_id"])
                conn = env.world.conn
                lane = conn.vehicle.getLaneIndex(ego)
                lateral = 0
                if ego == "ego_2" and origin == "E76" and info["road_id"] == "E76":
                    # Repeat left across the index flip, then return to the exit lane.
                    completed = sum(
                        event["ego_id"] == ego and event["event"] == "completed"
                        for event in env.reward_tracker.lane_change_events
                    )
                    lateral = (5, -5, 0)[completed]
                    assert lane in (0, 1)
                if ego == "ego_2" and origin == "O1" and goal == "D3" and info["road_id"] == "E75":
                    lateral = -5 if lane > 1 else 0
                actions[ego] = [0, lateral]
            obs, rewards, terminated, _, infos = env.step(actions)
            assert np.isfinite(list(rewards.values())).all()
            for ego, info in infos.items():
                values = info["reward_components"]
                if values["merge"]:
                    assert ego == "ego_1"
                    assert info["terminal_reason"] is None
                    assert info["road_id"] in ("E75", "E76")
                    assert values["merge"] == 5
                    assert values["task"] == 0
                if values["route_guidance"]:
                    assert ego == "ego_1"
                    assert info["road_id"] == "E77"
            if all(terminated.values()):
                break

        assert env.terminal_reasons == dict.fromkeys(env.AGENTS, "Arrived")
        ego_1, ego_2 = env.reward_summary()["reward_components"].values()
        assert (ego_1["task"], ego_1["merge"], ego_1["route_guidance"]) == (50, 5, 5)
        assert (ego_2["task"], ego_2["merge"], ego_2["route_guidance"]) == (50, 0, 0)
        assert {"O3", "E75", "E76", "E77", "D3"} <= seen["ego_1"]
        assert origin in seen["ego_2"]
        if arrival > 0:
            assert goal in seen["ego_2"]
        else:
            assert goal not in seen["ego_2"]
        assert any(road.startswith(":") for road in seen["ego_1"])
        assert env.route_goals["ego_2"] == RouteGoal(goal, arrival)
        assert 0 < previous["ego_2"][1] < 3.34
        if origin == "E76":
            events = env.reward_summary()["lane_change_events"]
            assert [event["event"] for event in events] == [
                "started", "completed", "started", "completed",
            ]
            assert [event["target_lane"] for event in events] == [1, 1, 0, 0]
    finally:
        env.close()


@pytest.mark.parametrize("actor_leader_features", [False, True])
def test_real_leader_gap_and_ttc_beyond_old_lookahead(tmp_path, actor_leader_features):
    route = tmp_path / "following.rou.xml"
    route.write_text("""<routes>
        <vType id="ego" maxSpeed="33.3333333333" speedFactor="1" length="5" minGap="2.5"/>
        <trip id="ego_2" type="ego" depart="0" from="E76" to="D3"
              departLane="0" departPos="200" departSpeed="0"/>
        <trip id="ego_1" type="ego" depart="0.1" from="E76" to="D3"
              departLane="0" departPos="10" departSpeed="30"/>
    </routes>""")
    env = MultiEgoEnv(reward_config=MASACRewardConfig(), route_file=route,
                      actor_leader_features=actor_leader_features)
    try:
        observations, infos = env.reset()
        actor_obs, critic_obs, _ = snapshot(env, observations)
        assert actor_obs.shape == (2, 139 if actor_leader_features else 136)
        assert critic_obs.shape == (2, 95)
        if actor_leader_features:
            np.testing.assert_array_equal(actor_obs[1, 93:96], [120, 0, 10])
        vehicle = env.world.conn.vehicle
        vehicle.setSpeed("ego_2", 0)
        _, _, _, _, infos = env.step({"ego_2": [0, 0]})
        assert all(info["status"] == "active" for info in infos.values())
        vehicle.moveTo("ego_2", "E76_0", vehicle.getLanePosition("ego_1") + 65)
        leader, reported_gap = vehicle.getLeader("ego_1", 120)
        assert leader == "ego_2"
        assert reported_gap == pytest.approx(57.5)
        observations, _ = env._build_output()
        update = env.delay_buffer.update
        env.delay_buffer.update = lambda ego, observation: update(ego, observation, delay=0)
        env._update_policy_observations(observations, {
            ego: {} for ego in env.AGENTS
        })
        if actor_leader_features:
            actor_obs, critic_obs, _ = snapshot(env, observations)
            assert actor_obs[0, 93:96] == pytest.approx([60, 30, 2])
            np.testing.assert_array_equal(critic_obs[0, :93], observations["ego_1"].ravel())
        _, _, _, _, infos = env.step(dict.fromkeys(env.AGENTS, [0, 0]))
        assert infos["ego_1"]["following_risks"] == pytest.approx({
            "gap_risk": 0, "ttc_risk": 1 / 3, "following_reward": -1 / 30,
            "rear_gap_risk": 0, "rear_ttc_risk": 0, "rear_following_reward": 0,
        })
        assert infos["ego_2"]["following_risks"]["rear_ttc_risk"] == pytest.approx(1 / 3)
        assert infos["ego_2"]["reward_components"]["rear_following"] == pytest.approx(-1 / 30)
    finally:
        env.close()
