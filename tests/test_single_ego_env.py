
"""测试 O3 -> D3 单 Ego 环境初始化。"""

import numpy as np

from src.envs.single_ego_env import SingleEgoEnv


def test_single_ego_reset():
    env = SingleEgoEnv(
        scenario="easy",
        gui=False,
    )

    try:
        obs, info = env.reset(seed=42)

        # 1. 基础观测维度
        assert obs.shape == (31, 3)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()

        # 2. Ego 必须从 O3 出发
        assert info["ego_id"] == "ego"
        assert info["origin"] == "O3"
        assert info["destination"] == "D3"
        assert info["lane"] == "O3_0"

        # 3. Ego 应当处于运行状态
        assert env.ego.state.status == "active"
        assert env.step_count == 0
        assert not env.done

        # 4. SUMO 连接正常
        conn = env.world.conn
        assert conn is not None
        assert "ego" in conn.vehicle.getIDList()

        # 5. 自动车道控制已关闭
        assert conn.vehicle.getLaneChangeMode("ego") == 0

        # 6. 观测首行对应 Ego 当前状态
        assert np.isclose(
            obs[0, 2],
            conn.vehicle.getSpeed("ego"),
        )

        print("\nSingle Ego reset PASS")
        print("SUMO:", info["sumo_version"])
        print("Departure:", info["depart_time"])
        print("Route:", info["route"])
        print("Observation shape:", obs.shape)

    finally:
        env.close()



def test_b0_reward():
    """检查事件奖励是否与原 B0 一致。"""

    from src.envs.single_ego_env import b0_reward

    assert b0_reward(None) == 0.0
    assert b0_reward("Arrived") == 50.0
    assert b0_reward("Collision") == -100.0
    assert b0_reward("Ego Teleported") == -100.0
    assert b0_reward("Missed Exit") == -100.0
    assert b0_reward("Timeout") == -100.0


def test_single_ego_step():
    """验证动作执行、时间推进和观测更新。"""

    env = SingleEgoEnv(scenario="easy")

    try:
        obs, info = env.reset(seed=42)

        previous_time = info["simulation_time"]

        for i in range(10):
            obs, reward, terminated, truncated, info = (
                env.step([0.0, 0.0])
            )

            assert obs.shape == (31, 3)
            assert np.isfinite(obs).all()
            assert reward == 0.0

            assert not terminated
            assert not truncated

            assert np.isclose(
                info["simulation_time"] - previous_time,
                0.1,
                atol=1e-6,
            )

            previous_time = info["simulation_time"]

        assert env.step_count == 10

        print("\nSingle Ego step PASS")
        print("Final time:", previous_time)
        print("Step count:", env.step_count)
        print("Road:", info["road_id"])
        print("Speed:", info["speed"])

    finally:
        env.close()
