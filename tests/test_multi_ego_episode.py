
"""验证双 Ego 完整运行及独立终止。"""

import numpy as np

from src.envs.multi_ego_env import MultiEgoEnv


def test_full_dual_ego_episode():
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        observations, infos = env.reset(seed=42)

        previous_time = env.world.conn.simulation.getTime()

        last_done = {
            ego_id: False for ego_id in env.AGENTS
        }

        outcomes = {}
        independent_finish_seen = False

        print("\n===== Dual Ego full episode =====")

        for shared_step in range(1, 5001):

            # 仅向当前活跃、尚未终止的 Ego 提交动作
            actions = {
                ego_id: [0.0, 0.0]
                for ego_id in env.AGENTS
                if (
                    env.egos[ego_id].state.status == "active"
                    and not env.terminations[ego_id]
                )
            }

            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            current_time = env.world.conn.simulation.getTime()

            # 无论有几辆活跃 Ego，全局只推进 0.1 秒
            assert np.isclose(
                current_time - previous_time,
                0.1,
                atol=1e-6,
            )
            previous_time = current_time

            assert env.step_count == shared_step

            for ego_id in env.AGENTS:
                obs = observations[ego_id]

                assert obs.shape == (31, 3)
                assert np.isfinite(obs).all()
                assert not truncations[ego_id]

                # 已结束的 Ego 不允许重新变成未结束
                if last_done[ego_id]:
                    assert terminations[ego_id]

                # 新出现的终止事件
                just_finished = (
                    terminations[ego_id]
                    and not last_done[ego_id]
                )

                if just_finished:
                    reason = infos[ego_id]["terminal_reason"]
                    reward = rewards[ego_id]

                    assert reason in (
                        "Arrived",
                        "Collision",
                        "Ego Teleported",
                        "Missed Exit",
                        "Timeout",
                    )

                    expected_reward = (
                        50.0
                        if reason == "Arrived"
                        else -100.0
                    )

                    assert reward == expected_reward

                    assert (
                        infos[ego_id]["status"] == "finished"
                    )

                    outcomes[ego_id] = {
                        "reason": reason,
                        "reward": reward,
                        "active_steps":
                            infos[ego_id]["active_steps"],
                        "finish_time": current_time,
                    }

                    print(
                        f"{ego_id}: "
                        f"{reason}, "
                        f"reward={reward}, "
                        f"active_steps="
                        f"{infos[ego_id]['active_steps']}, "
                        f"time={current_time:.1f}s"
                    )

                else:
                    # 没有新终止事件时，B0 奖励必须为零
                    assert rewards[ego_id] == 0.0

                last_done[ego_id] = terminations[ego_id]

            # 出现一辆完成、另一辆未完成的状态
            if sum(terminations.values()) == 1:
                independent_finish_seen = True

            if all(terminations.values()):
                break

        else:
            raise AssertionError(
                "Episode did not finish within "
                "5000 shared steps"
            )

        # 两辆 Ego 最终都必须获得明确终止事件
        assert set(outcomes) == set(env.AGENTS)

        # 需要实际观察到独立生命周期
        assert independent_finish_seen

        print("\n===== Final summary =====")
        print("Shared steps:", env.step_count)
        print("Simulation time:", current_time)

        for ego_id, result in outcomes.items():
            print(ego_id, result)

        print("\nFull dual Ego episode PASS")

    finally:
        env.close()
