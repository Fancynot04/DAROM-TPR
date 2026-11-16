
"""阶段 0：验证 E74_2 的固定动作驶离结果。"""

import xml.etree.ElementTree as ET
from pathlib import Path

from src.envs.multi_ego_env import MultiEgoEnv


ROUTE_FILE = (
    Path(__file__).resolve().parents[1]
    / "sumo_new"
    / "Antwerp_easy_o3_d3_dual.rou.xml"
)


def test_e74_lane2_hold():
    # 1. 检查路线文件
    root = ET.parse(ROUTE_FILE).getroot()

    trip = next(
        x for x in root.findall("trip")
        if x.get("id") == "ego_e74"
    )

    assert trip.get("from") == "E74"
    assert trip.get("to") == "D3"
    assert trip.get("departLane") == "2"

    # 2. 启动双 Ego 环境
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        env.reset(seed=42)

        departure_lane = None
        last_road = None
        road_trace = []

        results = {}

        # 3. 两辆 Ego 都不主动加减速或换道
        for _ in range(4500):
            actions = {
                ego_id: [0.0, 0.0]
                for ego_id in env.AGENTS
                if (
                    env.egos[ego_id].state.status == "active"
                    and not env.terminations[ego_id]
                )
            }
            

            _, rewards, terms, truncs, infos = (
                env.step(actions)
            )

            conn = env.world.conn

            # 4. 记录 Ego X 的实际车道
            if "ego_e74" in conn.vehicle.getIDList():
                road = conn.vehicle.getRoadID("ego_e74")

                if not road.startswith(":"):
                    lane = conn.vehicle.getLaneIndex(
                        "ego_e74"
                    )

                    if (
                        road == "E74"
                        and departure_lane is None
                    ):
                        departure_lane = lane

                    if road != last_road:
                        road_trace.append((road, lane))
                        last_road = road

            # 5. 每辆 Ego 只记录一次终止事件
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

        print("\n===== E74 lane 2 / hold baseline =====")
        print("Actual departure lane:", departure_lane)
        print("Ego X road trace:", road_trace)

        for ego_id, result in results.items():
            print(ego_id, result)

        # 本轮先验证实际场景和回合终止
        assert departure_lane == 2
        assert set(results) == set(env.AGENTS)

    finally:
        env.close()
