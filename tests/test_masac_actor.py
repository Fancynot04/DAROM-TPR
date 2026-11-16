
"""CTDE-MASAC Actor 单元测试。"""

import torch

from src.agents.masac import MultiAgentGaussianActor


def test_actor_output():
    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()

    obs = torch.randn(4, 2, 134)
    mask = torch.ones(4, 2)

    actions, log_probs = actors(obs, mask)

    assert actions.shape == (4, 2, 2)
    assert log_probs.shape == (4, 2)

    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()

    assert torch.all(actions <= 5.0)
    assert torch.all(actions >= -5.0)


def test_decentralized_execution():
    """改变 Ego B 观测，不得影响 Ego A 的策略输出。"""

    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()

    obs = torch.randn(1, 2, 134)
    mask = torch.ones(1, 2)

    actions1, _ = actors(
        obs,
        mask,
        deterministic=True,
    )

    changed = obs.clone()
    changed[:, 1, :] += 100.0

    actions2, _ = actors(
        changed,
        mask,
        deterministic=True,
    )

    torch.testing.assert_close(
        actions1[:, 0],
        actions2[:, 0],
    )

    # 已结束 Ego 不应输出有效动作
    mask = torch.tensor([[1.0, 0.0]])

    actions3, log_probs3 = actors(
        obs,
        mask,
        deterministic=True,
    )

    assert torch.all(actions3[:, 1] == 0.0)
    assert torch.all(log_probs3[:, 1] == 0.0)


def test_actor_backward():
    """确认 Actor、GRU 均能参与梯度更新。"""

    torch.manual_seed(42)

    actors = MultiAgentGaussianActor()

    obs = torch.randn(4, 2, 134)
    mask = torch.ones(4, 2)

    actions, log_probs = actors(obs, mask)

    loss = (
        actions.square().mean()
        + log_probs.square().mean()
    )

    loss.backward()

    for ego_id in actors.ego_ids:
        actor = actors.actors[ego_id]

        # 策略网络有梯度
        assert any(
            p.grad is not None
            and p.grad.abs().sum() > 0
            for p in actor.policy.parameters()
        )

        # DAROM-GRU 编码器也有梯度
        assert any(
            p.grad is not None
            and p.grad.abs().sum() > 0
            for p in actor.encoder.parameters()
        )
