
"""真实 SUMO 双 Ego 经验回放集成测试。"""

import numpy as np
import torch

from src.envs.multi_ego_env import MultiEgoEnv
from src.agents.replay_buffer import JointReplayBuffer
from src.agents.masac import MultiAgentGaussianActor
from src.agents.centralized_critic import CentralizedTwinQ


def snapshot(env, observations):
    """读取同一个仿真时刻的 Actor/Critic 状态。"""

    ids = JointReplayBuffer.AGENT_IDS

    policy_obs = np.stack([
        env.policy_observations[ego_id]
        for ego_id in ids
    ]).astype(np.float32)

    critic_obs = np.stack([
        observations[ego_id].reshape(-1)
        for ego_id in ids
    ]).astype(np.float32)

    active_mask = np.array([
        float(
            env.egos[ego_id].state.status == "active"
            and not env.terminations[ego_id]
        )
        for ego_id in ids
    ], dtype=np.float32)

    assert policy_obs.shape == (2, 134)
    assert critic_obs.shape == (2, 93)

    return (
        policy_obs.copy(),
        critic_obs.copy(),
        active_mask.copy(),
    )


def test_real_sumo_to_joint_replay():
    torch.manual_seed(42)

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    buffer = JointReplayBuffer(
        capacity=100,
        seed=42,
    )

    try:
        observations, _ = env.reset(seed=42)

        assert tuple(env.AGENTS) == buffer.AGENT_IDS

        mask_history = []

        for _ in range(30):
            # 1. SUMO 推进前：保存当前状态
            policy_obs, critic_obs, active_mask = (
                snapshot(env, observations)
            )

            mask_history.append(active_mask.copy())

            # 本轮仅测试数据链路，使用固定动作
            actions = {
                ego_id: [0.0, 0.0]
                for ego_id, active in zip(
                    buffer.AGENT_IDS, active_mask
                )
                if active == 1
            }

            joint_actions = np.zeros(
                (2, 2), dtype=np.float32
            )

            for i, ego_id in enumerate(buffer.AGENT_IDS):
                if ego_id in actions:
                    joint_actions[i] = actions[ego_id]

            # 2. SUMO 全局推进一次
            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            # 3. 推进后：保存下一状态
            (
                next_policy_obs,
                next_critic_obs,
                next_active_mask,
            ) = snapshot(env, next_observations)

            reward_array = np.array([
                rewards[ego_id]
                for ego_id in buffer.AGENT_IDS
            ], dtype=np.float32)

            done_array = np.array([
                float(
                    terminations[ego_id]
                    or truncations[ego_id]
                )
                for ego_id in buffer.AGENT_IDS
            ], dtype=np.float32)

            # 真正执行了当前动作的 Ego
            acted_mask = np.array([
                float(
                    ego_id in actions
                    and not infos[ego_id].get(
                        "pre_step_terminal", False
                    )
                )
                for ego_id in buffer.AGENT_IDS
            ], dtype=np.float32)

            # 暂时禁止错误记录 pre-step 终止奖励。
            # 这一问题将在完整回合采集前单独解决。
            invalid_reward = (
                (acted_mask == 0)
                & (reward_array != 0)
            )

            assert not np.any(invalid_reward), (
                "Pre-step terminal requires "
                "transition attribution"
            )

            # 4. 写入联合 Replay Buffer
            buffer.add(
                policy_obs=policy_obs,
                critic_obs=critic_obs,
                actions=joint_actions,
                rewards=reward_array,
                dones=done_array,
                active_mask=active_mask,
                acted_mask=acted_mask,
                next_policy_obs=next_policy_obs,
                next_critic_obs=next_critic_obs,
                next_active_mask=next_active_mask,
            )

            observations = next_observations

        # 30 次 SUMO 全局步 = 30 条联合经验
        assert len(buffer) == 30

        # 应该覆盖单车活跃与双车活跃
        assert any(
            np.array_equal(m, [1.0, 0.0])
            for m in mask_history
        )

        assert any(
            np.array_equal(m, [1.0, 1.0])
            for m in mask_history
        )

        # 5. 从真实回放数据采样
        batch = buffer.sample(batch_size=8)

        assert batch["policy_obs"].shape == (
            8, 2, 134
        )

        assert batch["critic_obs"].shape == (
            8, 2, 93
        )

        assert batch["actions"].shape == (
            8, 2, 2
        )

        # 6. 检查真实数据能进入网络
        actors = MultiAgentGaussianActor()
        critic = CentralizedTwinQ()

        with torch.no_grad():
            sampled_actions, log_probs = actors(
                batch["policy_obs"],
                batch["active_mask"],
            )

            q1, q2 = critic(
                batch["critic_obs"],
                batch["actions"],
                batch["active_mask"],
            )

        assert sampled_actions.shape == (8, 2, 2)
        assert log_probs.shape == (8, 2)
        assert q1.shape == (8, 2)
        assert q2.shape == (8, 2)

        assert torch.isfinite(sampled_actions).all()
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(q1).all()
        assert torch.isfinite(q2).all()

        print("\n===== Real SUMO Replay =====")
        print("Stored transitions:", len(buffer))
        print("Sampled batch:", batch["policy_obs"].shape)
        print("Actor forward:", sampled_actions.shape)
        print("Critic forward:", q1.shape)
        print("Real SUMO replay PASS")

    finally:
        env.close()
