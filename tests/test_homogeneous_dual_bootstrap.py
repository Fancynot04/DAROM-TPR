
import numpy as np

from src.envs.multi_ego_env import MultiEgoEnv


def test_homogeneous_dual_bootstrap():
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    departed = {}
    overlap_steps = 0

    try:
        observations, infos = env.reset(seed=42)
        conn = env.world.conn

        assert tuple(env.AGENTS) == ("ego_1", "ego_2")

        def inspect_agents():
            for ego_id in env.AGENTS:
                state = env.egos[ego_id].state

                if state.status != "active":
                    continue

                # 首次实际出发时记录
                if ego_id not in departed:
                    assert ego_id in conn.vehicle.getIDList()

                    road = conn.vehicle.getRoadID(ego_id)
                    route = list(conn.vehicle.getRoute(ego_id))

                    departed[ego_id] = {
                        "time": state.depart_time,
                        "road": road,
                        "route": route,
                    }

                    print(
                        f"{ego_id}: "
                        f"depart={state.depart_time:.1f}s, "
                        f"road={road}, "
                        f"route={route}"
                    )

                    assert road == "O3"
                    assert route[0] == "O3"
                    assert route[-1] == "D3"

                    assert (
                        conn.vehicle.getLaneChangeMode(ego_id)
                        == 0
                    )
                    # 保持原 DAROM 的 speedMode 请求值
                    assert env.controllers[ego_id].speed_mode == 96

                    # SUMO 1.20.0 尚不支持 bit 6，
                    # 因此设置 96 后实际读取为 32
                    assert conn.vehicle.getSpeedMode(ego_id) == 32

        inspect_agents()

        # 从 reset 时刻继续推进最多 300 步
        for _ in range(300):

            if all(env.terminations.values()):
                break

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

            old_time = conn.simulation.getTime()

            observations, rewards, dones, _, infos = (
                env.step(actions)
            )

            new_time = conn.simulation.getTime()

            # 一个全局步骤只能推进 0.1 秒
            assert np.isclose(
                new_time - old_time,
                0.1,
                atol=1e-6,
            )

            inspect_agents()

            current_active = [
                ego_id
                for ego_id in env.AGENTS
                if env.egos[ego_id].state.status == "active"
                and not env.terminations[ego_id]
            ]

            if len(current_active) == 2:
                overlap_steps += 1

            for ego_id in env.AGENTS:
                assert observations[ego_id].shape == (31, 3)
                assert env.policy_observations[ego_id].shape == (134,)

                assert np.isfinite(
                    observations[ego_id]
                ).all()

                assert np.isfinite(
                    env.policy_observations[ego_id]
                ).all()

        print("\n===== Verification Summary =====")
        print("Departed agents:", departed)
        print("Overlap steps:", overlap_steps)
        print("Simulation time:", conn.simulation.getTime())
        print("Terminal reasons:", env.terminal_reasons)

        assert set(departed) == {"ego_1", "ego_2"}
        assert overlap_steps > 0

        print("HOMOGENEOUS DUAL EGO PASS")

    finally:
        env.close()



def test_homogeneous_independent_actions():
    """验证两个 Ego 同时独立执行动作。"""

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        env.reset(seed=42)
        conn = env.world.conn

        # 等待两辆 Ego 同时 Active
        for _ in range(300):
            active = [
                ego_id
                for ego_id in env.AGENTS
                if env.egos[ego_id].state.status == "active"
                and not env.terminations[ego_id]
            ]

            if len(active) == 2:
                break

            env.step({
                ego_id: [0.0, 0.0]
                for ego_id in active
            })
        else:
            raise AssertionError(
                "Two Ego vehicles were never active together"
            )

        # 控制器必须是两个独立对象
        assert (
            env.controllers["ego_1"]
            is not env.controllers["ego_2"]
        )

        # 两辆车提交不同纵向动作
        actions = {
            "ego_1": [1.0, 0.0],
            "ego_2": [-1.0, 0.0],
        }

        before = conn.simulation.getTime()

        observations, rewards, dones, _, infos = (
            env.step(actions)
        )

        after = conn.simulation.getTime()

        # 两辆车提交动作，SUMO 仍只推进一次
        assert np.isclose(after - before, 0.1, atol=1e-6)

        # 验证动作被分别存入各自历史
        np.testing.assert_allclose(
            env.action_history.histories["ego_1"][0],
            [1.0, 0.0],
        )

        np.testing.assert_allclose(
            env.action_history.histories["ego_2"][0],
            [-1.0, 0.0],
        )

        # 两辆车的动作历史不能共用数组
        assert (
            env.action_history.histories["ego_1"]
            is not env.action_history.histories["ego_2"]
        )

        for ego_id in env.AGENTS:
            assert observations[ego_id].shape == (31, 3)
            assert env.policy_observations[ego_id].shape == (134,)

        print("\n===== Independent Actions =====")
        print("ego_1 action:", actions["ego_1"])
        print("ego_2 action:", actions["ego_2"])
        print("SUMO step:", after - before)
        print("Rewards:", rewards)
        print("Done:", dones)
        print("INDEPENDENT ACTIONS PASS")

    finally:
        env.close()


def test_homogeneous_active_masks():
    """验证 Waiting、Active、Finished 对应的 Mask。"""

    from train_masac import snapshot

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        observations, _ = env.reset(seed=42)

        # 阶段 A：ego_1 Active，ego_2 Waiting
        _, _, mask = snapshot(env, observations)

        np.testing.assert_array_equal(
            mask, [1.0, 0.0]
        )
        print("Stage A:", mask)

        # 等待第二辆车实际出发
        for _ in range(300):
            if env.egos["ego_2"].state.status == "active":
                break

            active = [
                ego_id
                for ego_id in env.AGENTS
                if env.egos[ego_id].state.status == "active"
                and not env.terminations[ego_id]
            ]

            observations, _, _, _, _ = env.step({
                ego_id: [0.0, 0.0]
                for ego_id in active
            })
        else:
            raise AssertionError("ego_2 did not depart")

        # 阶段 B：两辆车均 Active
        _, _, mask = snapshot(env, observations)

        np.testing.assert_array_equal(
            mask, [1.0, 1.0]
        )
        print("Stage B:", mask)

        # 人为将 ego_1 设置到超时前一控制步
        # 仅用于验证终止状态，不代表实际驾驶表现
        env.active_steps["ego_1"] = 3999

        before_mask = mask.copy()

        observations, rewards, dones, _, infos = env.step({
            "ego_1": [0.0, 0.0],
            "ego_2": [0.0, 0.0],
        })

        # 本步没有 Pre-step 提前终止
        pre_step_mask = np.array([
            float(infos[ego_id]["pre_step_terminal"])
            for ego_id in env.AGENTS
        ], dtype=np.float32)

        np.testing.assert_array_equal(
            pre_step_mask, [0.0, 0.0]
        )

        acted_mask = before_mask * (1.0 - pre_step_mask)

        _, _, next_mask = snapshot(env, observations)

        # 阶段 C：ego_1 超时，ego_2 继续运行
        np.testing.assert_array_equal(
            acted_mask, [1.0, 1.0]
        )
        np.testing.assert_array_equal(
            next_mask, [0.0, 1.0]
        )

        assert dones["ego_1"]
        assert not dones["ego_2"]
        assert rewards["ego_1"] == -100.0
        assert rewards["ego_2"] == 0.0

        print("Stage C:")
        print("  Active:", before_mask)
        print("  Acted:", acted_mask)
        print("  Next Active:", next_mask)
        print("  Rewards:", rewards)

        # 阶段 D：只允许 ego_2 执行动作
        observations, rewards, dones, _, _ = env.step({
            "ego_2": [0.0, 0.0]
        })

        _, _, final_mask = snapshot(env, observations)

        np.testing.assert_array_equal(
            final_mask, [0.0, 1.0]
        )

        # ego_1 的终止奖励不得重复发放
        assert rewards["ego_1"] == 0.0
        assert dones["ego_1"]

        print("Stage D:", final_mask)
        print("HOMOGENEOUS MASK TEST PASS")

    finally:
        env.close()
