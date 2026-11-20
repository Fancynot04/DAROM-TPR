
import numpy as np
import torch

from src.agents.masac import MultiAgentGaussianActor
from src.agents.centralized_critic import CentralizedTwinQ
from src.agents.replay_buffer import JointReplayBuffer
from src.agents.masac_trainer import (
    MASACTrainer,
    compute_td_target,
)


def test_td_target_terminal_mask():
    """已终止或下一步无效的 Ego 不再 Bootstrap。"""

    rewards = torch.tensor([
        [50.0, -100.0],
        [3.0, 7.0],
    ])

    dones = torch.tensor([
        [1.0, 0.0],
        [0.0, 0.0],
    ])

    next_mask = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    q1 = torch.tensor([
        [900.0, 10.0],
        [20.0, 30.0],
    ])

    q2 = torch.tensor([
        [500.0, 8.0],
        [15.0, 25.0],
    ])

    log_probs = torch.tensor([
        [0.5, -1.0],
        [-2.0, -1.5],
    ])

    alpha = torch.tensor([0.2, 0.3])

    target = compute_td_target(
        rewards,
        dones,
        next_mask,
        q1,
        q2,
        log_probs,
        alpha,
        gamma=0.9,
    )

    expected = torch.tensor([
        [50.0, -92.53],
        [16.86, 7.0],
    ])

    torch.testing.assert_close(
        target,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )


def make_batch():
    """使用联合 Replay Buffer 构造训练 Batch。"""

    rng = np.random.default_rng(42)
    buffer = JointReplayBuffer(
        capacity=16,
        seed=42,
    )

    for _ in range(8):
        transition = {
            "policy_obs": rng.normal(
                size=(2, 134)
            ).astype(np.float32),

            "critic_obs": rng.normal(
                size=(2, 93)
            ).astype(np.float32),

            "actions": rng.uniform(
                -5.0, 5.0, size=(2, 2)
            ).astype(np.float32),

            "rewards": np.array(
                [50.0, 0.0],
                dtype=np.float32,
            ),

            "dones": np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),

            "active_mask": np.array(
                [1.0, 1.0],
                dtype=np.float32,
            ),

            "acted_mask": np.array(
                [1.0, 1.0],
                dtype=np.float32,
            ),

            "next_policy_obs": rng.normal(
                size=(2, 134)
            ).astype(np.float32),

            "next_critic_obs": rng.normal(
                size=(2, 93)
            ).astype(np.float32),

            "next_active_mask": np.array(
                [0.0, 1.0],
                dtype=np.float32,
            ),
        }

        buffer.add(**transition)

    return buffer.sample(batch_size=4)


def test_critic_update():
    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()

    trainer = MASACTrainer(
        actors=actors,
        critic=critic,
    )

    batch = make_batch()

    critic_before = [
        p.detach().clone()
        for p in critic.parameters()
    ]

    actor_before = [
        p.detach().clone()
        for p in actors.parameters()
    ]

    result = trainer.update_critic(batch)

    assert np.isfinite(result["critic_loss"])
    assert np.isfinite(result["mean_target"])

    # Batch 有 4 条联合经验，每条 2 个有效 Ego
    assert result["valid_agents"] == 8

    # Critic 参数发生变化
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            critic_before,
            critic.parameters(),
        )
    )

    # Actor 参数不应由 Critic 更新改变
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            actor_before,
            actors.parameters(),
        )
    )

    # Target Critic 不应接收梯度
    assert all(
        p.grad is None
        for p in trainer.target_critic.parameters()
    )

    print("\nCritic update:", result)


def test_critic_update_with_inactive_ego():
    """只更新真正执行动作的 Ego。"""

    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()
    trainer = MASACTrainer(actors, critic)

    batch = make_batch()

    # Ego A 已经结束：不再参与当前损失
    batch["active_mask"][:, 0] = 0
    batch["acted_mask"][:, 0] = 0
    batch["rewards"][:, 0] = 0
    batch["dones"][:, 0] = 0

    result = trainer.update_critic(batch)

    assert result["valid_agents"] == 4
    assert np.isfinite(result["critic_loss"])


def test_actor_update_both_agents():
    """两个 Actor 都能通过 Q 网络获得策略梯度。"""

    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()
    trainer = MASACTrainer(actors, critic)

    batch = make_batch()

    before = {
        ego_id: [
            p.detach().clone()
            for p in actors.actors[ego_id].parameters()
        ]
        for ego_id in actors.ego_ids
    }

    result = trainer.update_actors(batch)

    for ego_id in actors.ego_ids:
        assert result[f"{ego_id}_actor_updates"] == 1
        assert result[f"{ego_id}_valid_samples"] == 4

        assert np.isfinite(
            result[f"{ego_id}_actor_loss"]
        )

        after = list(
            actors.actors[ego_id].parameters()
        )

        assert any(
            not torch.equal(p0, p1)
            for p0, p1 in zip(before[ego_id], after)
        )


def test_actor_gradient_isolation():
    """仅更新 Ego M，不允许 Ego X 和 Critic 改变。"""

    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()
    trainer = MASACTrainer(actors, critic)

    batch = make_batch()

    # Ego X 没有执行有效动作
    batch["acted_mask"][:, 1] = 0

    actor_before = {
        ego_id: [
            p.detach().clone()
            for p in actors.actors[ego_id].parameters()
        ]
        for ego_id in actors.ego_ids
    }

    critic_before = [
        p.detach().clone()
        for p in critic.parameters()
    ]

    result = trainer.update_actors(batch)

    assert result["ego_1_actor_updates"] == 1
    assert result["ego_2_actor_updates"] == 0

    # Ego M 的网络应该更新
    assert any(
        not torch.equal(p0, p1)
        for p0, p1 in zip(
            actor_before["ego_1"],
            actors.actors["ego_1"].parameters(),
        )
    )

    # Ego X 的网络完全不变
    assert all(
        torch.equal(p0, p1)
        for p0, p1 in zip(
            actor_before["ego_2"],
            actors.actors["ego_2"].parameters(),
        )
    )

    # Critic 参数完全不变
    assert all(
        torch.equal(p0, p1)
        for p0, p1 in zip(
            critic_before,
            critic.parameters(),
        )
    )

    # Critic 的梯度开关必须恢复
    assert all(
        p.requires_grad
        for p in critic.parameters()
    )



def test_alpha_update_direction():
    """验证 alpha 随熵偏差调整的方向。"""

    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()
    trainer = MASACTrainer(actors, critic)

    before = trainer.alpha_values().clone()

    # Ego M 概率密度较高，策略熵偏低
    # Ego X 在此批次无有效样本
    log_probs = torch.tensor([
        [10.0, 0.0],
        [10.0, 0.0],
    ])

    valid_mask = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    result = trainer.update_alpha_from_log_probs(
        log_probs,
        valid_mask,
    )

    after = trainer.alpha_values()

    # 熵偏低，alpha 应增大以鼓励探索
    assert after[0] > before[0]

    # 未参与学习的 Ego，alpha 不变
    torch.testing.assert_close(
        after[1], before[1], rtol=0, atol=0
    )

    assert result["ego_1_alpha_updates"] == 1
    assert result["ego_2_alpha_updates"] == 0

    # 独立验证熵偏高时 alpha 减小
    trainer2 = MASACTrainer(
        MultiAgentGaussianActor(),
        CentralizedTwinQ(),
    )

    before2 = trainer2.alpha_values().clone()

    trainer2.update_alpha_from_log_probs(
        torch.tensor([
            [-10.0, 0.0],
            [-10.0, 0.0],
        ]),
        valid_mask,
    )

    after2 = trainer2.alpha_values()

    assert after2[0] < before2[0]


def test_alpha_update_no_actor_change():
    """alpha 更新不能修改 Actor 和 Critic 参数。"""

    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()
    trainer = MASACTrainer(actors, critic)

    batch = make_batch()

    actor_before = [
        p.detach().clone()
        for p in actors.parameters()
    ]

    critic_before = [
        p.detach().clone()
        for p in critic.parameters()
    ]

    result = trainer.update_alpha(batch)

    assert result["ego_1_alpha_updates"] == 1
    assert result["ego_2_alpha_updates"] == 1

    assert all(
        torch.equal(a, b)
        for a, b in zip(
            actor_before, actors.parameters()
        )
    )

    assert all(
        torch.equal(a, b)
        for a, b in zip(
            critic_before, critic.parameters()
        )
    )


def test_team_target_bootstraps_past_early_arrival_and_waiting():
    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()
    team = CentralizedTwinQ(team=True)
    trainer = MASACTrainer(actors, critic, gamma=0.99, team_critic=team)
    with torch.no_grad():
        for param in trainer.target_team_critic.parameters():
            param.zero_()
        trainer.target_team_critic.q1[-1].bias.fill_(20)
        trainer.target_team_critic.q2[-1].bias.fill_(30)
    batch = make_batch()
    # Early arrival, no-action wait, second arrival, first team failure.
    batch["dones"][0] = torch.tensor([1.0, 0.0])
    batch["next_active_mask"][0].zero_()
    batch["active_mask"][1].zero_()
    batch["acted_mask"][1].zero_()
    batch.update(
        team_active=torch.ones(4),
        team_done=torch.tensor([0., 0., 1., 1.]),
        team_reward=torch.tensor([0., 0., 100., 0.]),
        success_mask=torch.tensor([[0., 0.], [1., 0.], [1., 0.], [0., 0.]]),
        next_success_mask=torch.tensor([[1., 0.], [1., 0.], [1., 1.], [0., 0.]]),
    )
    before = team.q1[-1].bias.detach().clone()
    stats = trainer.update_critic(batch)
    assert stats["team_valid_samples"] == 4
    assert abs(stats["team_mean_target"] - (2 * 0.99 * 20 + 100) / 4) < 1e-5
    assert not torch.equal(before, team.q1[-1].bias)


def test_team_q_changes_both_actor_updates():
    from copy import deepcopy
    from torch import nn

    class ActionTeamCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, observations, actions, active_mask, *, success_mask):
            q = (actions[:, :, 0] * active_mask).sum(dim=1) * self.scale
            return q, q

    actors = MultiAgentGaussianActor()
    critic = CentralizedTwinQ()
    baseline = MASACTrainer(deepcopy(actors), deepcopy(critic))
    team = ActionTeamCritic()
    cooperative = MASACTrainer(actors, critic, team_critic=team, team_q_weight=1.0)
    batch = make_batch()

    torch.manual_seed(123)
    baseline.update_actors(batch)
    batch.update(team_active=torch.ones(4), success_mask=torch.zeros(4, 2))
    torch.manual_seed(123)
    stats = cooperative.update_actors(batch)

    for key in actors.ego_ids:
        assert stats[f"{key}_actor_updates"] == 1
        assert f"{key}_team_q" in stats
        assert not torch.equal(
            baseline.actors.actors[key].mean_head.weight,
            cooperative.actors.actors[key].mean_head.weight,
        )
    assert team.scale.grad is None
    assert team.scale.requires_grad
