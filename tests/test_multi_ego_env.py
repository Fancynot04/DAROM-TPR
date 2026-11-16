
"""验证 MultiEgoEnv 的异步插入与共享时钟。"""

import numpy as np

from src.envs.multi_ego_env import MultiEgoEnv


def test_multi_ego_shared_clock():
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        observations, infos = env.reset(seed=42)

        assert set(observations) == {
            "ego_o3", "ego_e74"
        }

        print("\n===== After reset =====")

        for ego_id in env.AGENTS:
            print(
                ego_id,
                infos[ego_id]["status"],
                infos[ego_id]["depart_time"],
            )

        previous_time = (
            env.world.conn.simulation.getTime()
        )

        # 测试前 30 个共享仿真步
        for _ in range(30):
            active_ids = [
                ego_id
                for ego_id in env.AGENTS
                if env.egos[ego_id].state.status == "active"
            ]

            actions = {
                ego_id: [0.0, 0.0]
                for ego_id in active_ids
            }

            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            current_time = (
                env.world.conn.simulation.getTime()
            )

            # 不论活跃 Ego 数量是多少，时间只走 0.1s
            assert np.isclose(
                current_time - previous_time,
                0.1,
                atol=1e-6,
            )

            previous_time = current_time

            for ego_id in env.AGENTS:
                obs = observations[ego_id]
                status = infos[ego_id]["status"]

                assert obs.shape == (31, 3)
                assert np.isfinite(obs).all()

                if status == "waiting":
                    assert np.all(obs == 0.0)

                assert rewards[ego_id] == 0.0
                assert not terminations[ego_id]
                assert not truncations[ego_id]

        # 30 步后，两辆车应当都已出发
        assert all(
            ego.state.status == "active"
            for ego in env.egos.values()
        )

        assert env.step_count == 30

        print("\n===== After 30 steps =====")
        print("Simulation time:", previous_time)
        print("Shared steps:", env.step_count)

        for ego_id in env.AGENTS:
            print(
                ego_id,
                "status=", infos[ego_id]["status"],
                "active_steps=", infos[ego_id]["active_steps"],
            )

        print("\nMultiEgoEnv shared clock PASS")

    finally:
        env.close()


def test_one_ego_timeout_other_continues():
    """Ego A 超时，不应导致 Ego B 结束。"""

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        env.reset(seed=42)

        assert env.egos["ego_o3"].state.status == "active"
        assert env.egos["ego_e74"].state.status == "waiting"

        # 人为设置到最后一个控制步
        env.active_steps["ego_o3"] = 3999

        (
            obs,
            rewards,
            terminated,
            truncated,
            infos,
        ) = env.step({
            "ego_o3": [0.0, 0.0],
        })

        # Ego A 独立超时
        assert terminated["ego_o3"]
        assert not terminated["ego_e74"]

        assert rewards["ego_o3"] == -100.0
        assert rewards["ego_e74"] == 0.0

        assert infos["ego_o3"]["terminal_reason"] == "Timeout"
        assert infos["ego_o3"]["active_steps"] == 4000

        assert "ego_o3" not in (
            env.world.conn.vehicle.getIDList()
        )

        # Ego B 仍然处于 Waiting，可以继续推进世界
        for _ in range(30):
            active_ids = [
                ego_id for ego_id in env.AGENTS
                if env.egos[ego_id].state.status == "active"
                and not env.terminations[ego_id]
            ]

            actions = {
                ego_id: [0.0, 0.0]
                for ego_id in active_ids
            }

            obs, rewards, terminated, _, infos = (
                env.step(actions)
            )

            # Ego A 的终止奖励只结算一次
            assert terminated["ego_o3"]
            assert rewards["ego_o3"] == 0.0

            if env.egos["ego_e74"].state.status == "active":
                break

        assert env.egos["ego_e74"].state.status == "active"

        # Ego B 可以独立执行动作
        obs, rewards, terminated, _, infos = env.step({
            "ego_e74": [0.0, 0.0],
        })

        assert terminated["ego_o3"]
        assert not terminated["ego_e74"]

        assert rewards["ego_o3"] == 0.0
        assert rewards["ego_e74"] == 0.0

        print("\nIndependent Ego termination PASS")
        print("Ego A:", infos["ego_o3"]["terminal_reason"])
        print("Ego B:", infos["ego_e74"]["status"])

    finally:
        env.close()



def test_action_history_integration():
    """验证动作历史随真实 SUMO 控制步更新。"""

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        env.reset(seed=42)

        # 初始仅 O3 Ego 出发
        assert env.egos["ego_o3"].state.status == "active"
        assert env.egos["ego_e74"].state.status == "waiting"

        env.step({
            "ego_o3": [1.0, 0.0],
        })

        np.testing.assert_array_equal(
            env.action_history.histories["ego_o3"][0],
            [1.0, 0.0],
        )

        # 等待中的 Ego 不应收到动作
        assert np.all(
            env.action_history.histories["ego_e74"] == 0.0
        )

        # 等待第二辆 Ego 实际出发
        while env.egos["ego_e74"].state.status == "waiting":
            env.step({
                "ego_o3": [0.0, 0.0],
            })

        # 两辆车提交不同动作
        env.step({
            "ego_o3": [2.0, 0.0],
            "ego_e74": [-1.0, 0.0],
        })

        np.testing.assert_array_equal(
            env.action_history.histories["ego_o3"][0],
            [2.0, 0.0],
        )

        np.testing.assert_array_equal(
            env.action_history.histories["ego_e74"][0],
            [-1.0, 0.0],
        )

        assert env.action_history.histories["ego_o3"].shape == (20, 2)
        assert env.action_history.histories["ego_e74"].shape == (20, 2)

        # 重新开始回合，历史必须清零
        env.reset(seed=42)

        for ego_id in env.AGENTS:
            assert np.all(
                env.action_history.histories[ego_id] == 0.0
            )

    finally:
        env.close()


def test_delayed_policy_observation_integration():
    """真实 SUMO 步进下，验证双 Ego 延迟观测。"""

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        observations, infos = env.reset(seed=42)

        # 第一辆车已出发，第二辆仍在等待
        assert infos["ego_o3"]["observation_ready"]
        assert not infos["ego_e74"]["observation_ready"]

        assert env.policy_observations["ego_o3"].shape == (134,)
        assert env.policy_observations["ego_e74"].shape == (134,)

        assert np.all(
            env.policy_observations["ego_e74"] == 0
        )

        for _ in range(30):
            active_ids = [
                ego_id
                for ego_id in env.AGENTS
                if env.egos[ego_id].state.status == "active"
                and not env.terminations[ego_id]
            ]

            actions = {
                ego_id: [0.0, 0.0]
                for ego_id in active_ids
            }

            observations, rewards, dones, _, infos = (
                env.step(actions)
            )

            for ego_id in env.AGENTS:
                raw = observations[ego_id]
                policy = env.policy_observations[ego_id]

                assert raw.shape == (31, 3)
                assert policy.shape == (134,)
                assert np.isfinite(policy).all()

                if infos[ego_id]["status"] == "waiting":
                    assert not infos[ego_id]["observation_ready"]
                    assert np.all(policy == 0)
                    continue

                assert infos[ego_id]["observation_ready"]

                age = infos[ego_id]["observation_age"]
                assert 0 <= age <= 20

                # Ego 自身状态必须来自当前 SUMO 状态
                np.testing.assert_allclose(
                    policy[:3],
                    raw[0],
                    rtol=0,
                    atol=1e-5,
                )

                # 邻车状态应来自该 Ego 的延迟缓存
                np.testing.assert_array_equal(
                    policy[:93].reshape(31, 3)[1:],
                    env.delay_buffer.last_observations[
                        ego_id
                    ][1:],
                )

                # 掩码动作历史：只保留最近 age 步
                history = policy[93:133].reshape(20, 2)
                assert np.all(history[age:] == 0)

                # 最后一维保存的是实际观测年龄
                assert policy[-1] == age

        # 两辆车都已经初始化自己的延迟缓存
        assert env.delay_buffer.active == {
            "ego_o3",
            "ego_e74",
        }

        # 它们的队列和随机数生成器互相独立
        assert (
            env.delay_buffer.queues["ego_o3"]
            is not env.delay_buffer.queues["ego_e74"]
        )

        assert (
            env.delay_buffer.rng["ego_o3"]
            is not env.delay_buffer.rng["ego_e74"]
        )

        # 重新 reset 后，等待中的 Ego 策略输入为零
        env.reset(seed=42)

        assert np.all(
            env.policy_observations["ego_e74"] == 0
        )

        print("\nDelayed policy observation integration PASS")

    finally:
        env.close()
