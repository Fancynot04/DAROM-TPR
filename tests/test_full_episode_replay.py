
"""真实 SUMO 完整回合：验证联合经验与奖励归因。"""

import numpy as np

from src.envs.multi_ego_env import MultiEgoEnv
from src.agents.replay_buffer import JointReplayBuffer
from src.agents.transition_collector import JointTransitionCollector


IDS = JointReplayBuffer.AGENT_IDS


def snapshot(env, observations):
    """保存同一个 SUMO 时刻的联合状态。"""

    policy_obs = np.stack([
        env.policy_observations[ego_id]
        for ego_id in IDS
    ]).astype(np.float32)

    critic_obs = np.stack([
        observations[ego_id].reshape(-1)
        for ego_id in IDS
    ]).astype(np.float32)

    active_mask = np.array([
        float(
            env.egos[ego_id].state.status == "active"
            and not env.terminations[ego_id]
        )
        for ego_id in IDS
    ], dtype=np.float32)

    return policy_obs.copy(), critic_obs.copy(), active_mask


def test_full_episode_replay():
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    buffer = JointReplayBuffer(
        capacity=5000,
        seed=42,
    )

    collector = JointTransitionCollector(buffer)

    environment_rewards = np.zeros(2, dtype=np.float64)
    pre_step_counts = np.zeros(2, dtype=int)
    results = {}

    try:
        observations, _ = env.reset(seed=42)

        assert tuple(env.AGENTS) == IDS

        for _ in range(4500):
            policy_obs, critic_obs, active_mask = (
                snapshot(env, observations)
            )

            # 双 Ego 都采用保持车道、零加速度
            actions = {
                ego_id: [0.0, 0.0]
                for i, ego_id in enumerate(IDS)
                if active_mask[i] == 1
            }

            joint_actions = np.zeros(
                (2, 2), dtype=np.float32
            )

            for i, ego_id in enumerate(IDS):
                if ego_id in actions:
                    joint_actions[i] = actions[ego_id]

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            next_policy, next_critic, next_mask = (
                snapshot(env, next_observations)
            )

            reward_array = np.array([
                rewards[ego_id]
                for ego_id in IDS
            ], dtype=np.float32)

            # 注意：这里记录的是当前这一步的新终止事件
            # 不是已经结束车辆的累计终止标志
            done_array = np.array([
                float(
                    active_mask[i] == 1
                    and (
                        terminations[ego_id]
                        or truncations[ego_id]
                    )
                )
                for i, ego_id in enumerate(IDS)
            ], dtype=np.float32)

            pre_step = np.array([
                float(
                    active_mask[i] == 1
                    and infos[ego_id].get(
                        "pre_step_terminal", False
                    )
                )
                for i, ego_id in enumerate(IDS)
            ], dtype=np.float32)

            acted_mask = np.array([
                float(
                    active_mask[i] == 1
                    and pre_step[i] == 0
                )
                for i in range(2)
            ], dtype=np.float32)

            environment_rewards += reward_array
            pre_step_counts += pre_step.astype(int)

            transition = {
                "policy_obs": policy_obs,
                "critic_obs": critic_obs,
                "actions": joint_actions,
                "rewards": reward_array,
                "dones": done_array,
                "active_mask": active_mask,
                "acted_mask": acted_mask,
                "next_policy_obs": next_policy,
                "next_critic_obs": next_critic,
                "next_active_mask": next_mask,
            }

            # 由 Collector 负责归因和延迟提交
            collector.push(
                transition,
                pre_step_terminal=pre_step,
            )

            for ego_id in IDS:
                if ego_id in results:
                    continue

                if (
                    terminations[ego_id]
                    or truncations[ego_id]
                ):
                    results[ego_id] = infos[ego_id][
                        "terminal_reason"
                    ]

            observations = next_observations

            if len(results) == len(IDS):
                break

        else:
            raise AssertionError("Episode did not finish")

        collector.flush()

        n = len(buffer)

        stored_rewards = buffer.data[
            "rewards"
        ][:n].sum(axis=0, dtype=np.float64)

        # 1. 环境奖励与 Replay 奖励严格一致
        np.testing.assert_allclose(
            stored_rewards,
            environment_rewards,
            rtol=0,
            atol=1e-5,
        )

        # 2. 预期终止结果
        
        # 两辆 Ego 都必须完成一次明确的终止结算
        assert set(results) == set(IDS)

        expected_rewards = np.asarray([
            50.0 if results[ego_id] == "Arrived" else -100.0
            for ego_id in IDS
        ])

        np.testing.assert_allclose(
            stored_rewards,
            expected_rewards,
            rtol=0,
            atol=1e-5,
        )

        stored_acted = buffer.data["acted_mask"][:n]
        stored_reward_rows = buffer.data["rewards"][:n]
        stored_dones = buffer.data["dones"][:n]
        stored_next_mask = buffer.data["next_active_mask"][:n]

        # 未执行动作的经验不能承担终止奖励
        assert np.all(
            stored_reward_rows[stored_acted == 0] == 0
        )

        for i, ego_id in enumerate(IDS):
            terminal_rows = np.flatnonzero(
                stored_dones[:, i] == 1
            )

            # 每辆 Ego 恰好结算一次终止事件
            assert len(terminal_rows) == 1

            index = terminal_rows[0]

            assert stored_acted[index, i] == 1
            assert stored_next_mask[index, i] == 0
            assert stored_reward_rows[index, i] == expected_rewards[i]

            # 如果发生 Pre-step 终止，
            # 终止奖励必须归因给之前执行的有效动作
            assert pre_step_counts[i] <= 1

            if pre_step_counts[i] == 1:
                assert stored_acted[index, i] == 1


        print("\n===== Full Episode Replay =====")
        print("Terminal results:", results)
        print("Environment rewards:", environment_rewards)
        print("Replay rewards:", stored_rewards)
        print("Stored transitions:", n)
        print("Pre-step event counts:", pre_step_counts)
        print("Full episode replay PASS")

    finally:
        env.close()
