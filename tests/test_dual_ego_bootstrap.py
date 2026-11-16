
"""验证两个 Ego 共享一个 SUMO 世界。"""

import numpy as np

from src.envs.sumo_world import SumoWorld, SUMO_DIR
from src.envs.ego import Ego
from src.envs.control import EgoController
from src.envs.observation import build_observation


def test_dual_ego_bootstrap():
    route_file = (
        SUMO_DIR / "Antwerp_easy_o3_d3_dual.rou.xml"
    )

    world = SumoWorld(
        scenario="easy",
        seed=42,
        route_file=route_file,
    )

    egos = {
        "ego_o3": Ego("ego_o3"),
        "ego_e74": Ego("ego_e74"),
    }

    try:
        conn = world.start()

        controllers = {
            ego_id: EgoController(conn, ego_id)
            for ego_id in egos
        }

        # 等待两个 Ego 实际进入 SUMO
        while conn.simulation.getTime() < 150.0:
            world.step()

            for ego_id, ego in egos.items():
                was_waiting = (
                    ego.state.status == "waiting"
                )

                state = ego.update(conn)

                assert state.status != "finished", (
                    ego_id,
                    state.terminal_reason,
                )

                if was_waiting and state.status == "active":
                    controllers[ego_id].configure()

                    print(
                        f"{ego_id} departed "
                        f"at {state.depart_time:.1f}s, "
                        f"road={state.road_id}"
                    )

            if all(
                ego.state.status == "active"
                for ego in egos.values()
            ):
                break

        # 两个 Ego 都必须成功出发
        assert all(
            ego.state.status == "active"
            for ego in egos.values()
        )

        assert conn.vehicle.getRoadID("ego_o3") == "O3"
        assert conn.vehicle.getRoadID("ego_e74") == "E74"

        # 两条完整路线均以 D3 为终点
        for ego_id in egos:
            route = list(conn.vehicle.getRoute(ego_id))

            print(ego_id, "route:", route)

            assert route[-1] == "D3"

            obs = build_observation(
                conn,
                ego_id=ego_id,
                radius=50.0,
                num_entities=31,
            ).observation

            assert obs.shape == (31, 3)
            assert np.isfinite(obs).all()

            assert (
                conn.vehicle.getLaneChangeMode(ego_id) == 0
            )

        # 两辆 Ego 分别提交动作
        # EgoController 不允许推进 SUMO
        t0 = conn.simulation.getTime()

        for controller in controllers.values():
            controller.apply_action([0.0, 0.0])

        assert np.isclose(
            conn.simulation.getTime(),
            t0,
        )

        # 统一推进一次
        t1 = world.step()

        assert np.isclose(
            t1 - t0,
            0.1,
            atol=1e-6,
        )

        # 两辆 Ego 分别读取推进后的状态
        for ego_id, ego in egos.items():
            controllers[ego_id].observe_after_step()
            ego.update(conn)

            assert ego.state.status == "active"

        print("\nDual Ego bootstrap PASS")
        print("Shared simulation time:", t1)
        print("Active Ego count:", len(egos))

    finally:
        world.close()
