
"""阶段 0：E74_2 规则换道可达性测试。"""

from src.envs.multi_ego_env import MultiEgoEnv


def rule_action(env, state):
    """Ego X：在 E76 逐步向右进入 0 号车道。"""

    conn = env.world.conn
    ego_id = "ego_e74"
    vehicle = conn.vehicle

    if ego_id not in vehicle.getIDList():
        return [0.0, 0.0]

    road = vehicle.getRoadID(ego_id)

    # 第一版只在 E76 换道
    if road != "E76":
        return [0.0, 0.0]

    lane = vehicle.getLaneIndex(ego_id)

    if lane == 0:
        return [0.0, 0.0]

    # 接近路段末端时不再提出新的换道请求
    lane_id = vehicle.getLaneID(ego_id)
    remaining = (
        conn.lane.getLength(lane_id)
        - vehicle.getLanePosition(ego_id)
    )

    if remaining < 150.0:
        return [0.0, 0.0]

    now = float(conn.simulation.getTime())

    # 两次请求至少间隔 6 秒
    if now - state["last_request_time"] < 6.0:
        return [0.0, 0.0]

    # 当前车辆横向运动尚未稳定
    lateral_speed = vehicle.getLateralSpeed(ego_id)
    lateral_pos = vehicle.getLateralLanePosition(ego_id)

    if abs(lateral_speed) > 0.1:
        return [0.0, 0.0]

    if abs(lateral_pos) > 0.2:
        return [0.0, 0.0]

    # 提交向右换道请求
    state["last_request_time"] = now
    state["requests"].append(
        (round(now, 1), road, lane, lane - 1)
    )

    return [0.0, -5.0]


def test_e74_lane2_rule():
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    state = {
        "last_request_time": -1e9,
        "requests": [],
    }

    results = {}
    lane_changes = []
    previous = None
    departure_lane = None

    try:
        env.reset(seed=42)

        for _ in range(4500):
            actions = {}

            for ego_id in env.AGENTS:
                if (
                    env.egos[ego_id].state.status
                    != "active"
                    or env.terminations[ego_id]
                ):
                    continue

                if ego_id == "ego_o3":
                    actions[ego_id] = [0.0, 0.0]
                else:
                    actions[ego_id] = rule_action(
                        env, state
                    )

            _, rewards, terms, truncs, infos = (
                env.step(actions)
            )

            conn = env.world.conn
            vehicle = conn.vehicle

            # 记录真实车道索引变化
            if "ego_e74" in vehicle.getIDList():
                road = vehicle.getRoadID("ego_e74")

                if not road.startswith(":"):
                    lane = vehicle.getLaneIndex(
                        "ego_e74"
                    )

                    if road == "E74" and departure_lane is None:
                        departure_lane = lane

                    current = (road, lane)

                    if previous is not None:
                        old_road, old_lane = previous

                        if (
                            road == old_road
                            and lane != old_lane
                        ):
                            lane_changes.append(
                                (
                                    round(
                                        conn.simulation.getTime(), 1
                                    ),
                                    road,
                                    old_lane,
                                    lane,
                                )
                            )

                    previous = current

            # 两辆 Ego 的终止独立记录
            for ego_id in env.AGENTS:
                if ego_id in results:
                    continue

                if terms[ego_id] or truncs[ego_id]:
                    results[ego_id] = {
                        "reason": infos[ego_id][
                            "terminal_reason"
                        ],
                        "reward": rewards[ego_id],
                        "active_steps": env.active_steps[
                            ego_id
                        ],
                    }

            if len(results) == len(env.AGENTS):
                break

        print("\n===== E74 rule baseline =====")
        print("Departure lane:", departure_lane)

        print("\nLane change requests:")
        for item in state["requests"]:
            print(item)

        print("\nObserved lane index changes:")
        for item in lane_changes:
            print(item)

        print("\nFinal results:")
        for ego_id, result in results.items():
            print(ego_id, result)

        # 先确认程序完整运行，不预设策略一定成功
        assert departure_lane == 2
        assert len(results) == 2

    finally:
        env.close()
